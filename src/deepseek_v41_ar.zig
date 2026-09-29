//! The AR token-parity harness (track M, M3): the native model, its routed
//! experts streamed from the bank, against the Python reference's greedy ids
//! (R/exl3/runtime/dump_dsv41_ar_ref.py: the stock trunk + the EXL3 decode
//! lane). Both sides feed the prompt in forwards of <= 8 rows and decode one
//! token per forward, so every routed call is a decode-lane call. Window only;
//! the host dry path is the model test "the AR dry path ...".

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const routes = @import("deepseek_v41_routes.zig");
const engram = @import("deepseek_v41_engram.zig");
const mdl = @import("deepseek_v41_model.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_bank = @import("expert_bank.zig");
const expert_stream = @import("expert_stream.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const ds = @import("deepseek_v41_dspark.zig");
const dss = @import("deepseek_v41_dspark_serve.zig");
const xk = @import("exl3_kernels.zig");
const kernel_set = @import("kernel_set.zig");
const xq = @import("exl3_quant.zig");
const status = @import("status.zig");
const module = @import("deepseek_v41_module.zig");
const cell = @import("deepseek_v41_cell.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_admission = @import("expert_admission.zig");

/// One phase's memory for the bill (C4), printed on its own line: MLX's active bytes now, its
/// high-water mark since the previous probe (then reset), and the process footprint now
/// (`status.getAppMemFootprintMb`). The gap between the footprint and MLX is the host side.
fn memProbe(harness: []const u8, phase: []const u8) void {
    var active: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    const fp_mib: u64 = status.getAppMemFootprintMb();
    std.debug.print("\n{s}: memory {s}: MLX active {d:.2} GB, MLX peak since the last probe {d:.2} GB, footprint {d:.2} GB\n", .{
        harness, phase, @as(f64, @floatFromInt(active)) / 1e9, @as(f64, @floatFromInt(peak)) / 1e9, @as(f64, @floatFromInt(fp_mib << 20)) / 1e9,
    });
    _ = mlx.mlx_reset_peak_memory();
}

/// The load context's kernels on the harness's GPU stream, as the served module takes them
/// (C2, kernels note sec. 19): the kernel set (registry, kernels built), the backend's
/// launcher installed, then the EXL3 quant accepted (its self-check subset judged; its
/// decode GEMV is the stock chain's).
const Kernels = struct {
    set: *kernel_set.Set,
    exl3: *xq.Accepted(ops.MlxOps),

    fn deinit(self: Kernels, g: *ops.MlxOps) void {
        _ = mlx.mlx_synchronize(g.s);
        self.exl3.deinit(g);
        kernel_set.Set.uninstall(ops.MlxOps, g);
        self.set.deinit();
    }
};

fn acceptKernels(gpa: std.mem.Allocator, g: *ops.MlxOps, c: *const v41.Config) !Kernels {
    var diag: xk.Diag = .{};
    errdefer std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
    const set = try kernel_set.Set.init(gpa, .{ .device = .{ .stream = g.s } }, &diag);
    errdefer set.deinit();
    set.install(ops.MlxOps, g);
    errdefer kernel_set.Set.uninstall(ops.MlxOps, g);
    const exl3 = try xq.accept(ops.MlxOps, gpa, g, .{ .kernels = set }, .{
        .hidden = c.hidden_size,
        .inter = c.moe_intermediate_size,
        .top_k = c.n_experts_per_tok,
        .n_layers = c.n_layers,
        .act = .{ .swiglu_clamped = c.swiglu_limit },
        .input = .bfloat16,
    }, &diag);
    return .{ .set = set, .exl3 = exl3 };
}

/// Every bound bank of the hook `ex` against the kernels' signatures (once, after growth).
fn checkBanks(g: *ops.MlxOps, k: Kernels, ex: anytype) !void {
    var diag: xk.Diag = .{};
    errdefer std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
    for (ex.banks) |per| for (per) |maybe| if (maybe) |b| try k.exl3.checkBank(g, b, &diag);
}

pub const reference_format = "mlx-serve-dsv41-ar-ref-v1";

pub const Step = struct { logits_sha256: []const u8, top2: [2]u32, margin: f64 };

pub const Reference = struct {
    format: []const u8,
    prompt_ids: []const u32,
    chunk: u32,
    new_tokens: u32,
    generated_ids: []const u32,
    steps: []const Step,
};

/// sha256 of each evaluated logits row, as the reference records it.
const StepHashes = struct {
    out: [][64]u8,
    n: usize = 0,

    pub fn step(self: *StepHashes, _: *ops.MlxOps, logits: mlx.mlx_array) !void {
        if (self.n == 0) memProbe("dsv41 ar", "prompt (residents bound on first use, the prompt's forwards)");
        const n = mlx.mlx_array_size(logits);
        const p = mlx.mlx_array_data_float32(logits) orelse return error.MlxNoData;
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(p[0..n]), &d, .{});
        self.out[self.n] = std.fmt.bytesToHex(d, .lower);
        self.n += 1;
    }
};

const testing = std.testing;

// Guarded window only (loads the bank): DSV41_AR_REF=<dump_dsv41_ar_ref.py json> DSV41_BANK=<bank>
// DSV41_ENGRAM_TOKEN_MAP=<converter map> _GPU_WINDOW_LOCKED=1 [DSV41_AR_ROWS=<decode rows per layer, default 16>]
test "dsv41 ar: the native path with streamed experts generates the Python reference's tokens" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_AR_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    if (servedSchedule()) return error.SkipZigTest; // the served schedule's own test below
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(16 << 20));
    const ref = try std.json.parseFromSliceLeaky(Reference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, reference_format)) return error.ReferenceFormat;
    if (ref.chunk == 0 or ref.chunk > 8 or ref.generated_ids.len != ref.new_tokens) return error.ReferenceShape;
    const rows: u32 = if (std.c.getenv("DSV41_AR_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 16;

    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 ar: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    // The default device is the GPU (compile availability follows it, as in Python).
    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(gpa, s);
    defer g.deinit();
    // The bound: MLX keeps no freed buffer (the kernels' startup check and each forward's transients go back).
    var prev_cache: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cache, 0);
    defer _ = mlx.mlx_set_cache_limit(&prev_cache, prev_cache);
    memProbe("dsv41 ar", "start");
    const kernels = try acceptKernels(gpa, &g, &c);
    defer kernels.deinit(&g);
    memProbe("dsv41 ar", "kernels accepted (the startup self-check)");

    var weights = try dss.loadResidents(io, gpa, bank_dir, &c);
    defer weights.deinit();
    var src = try engram.RowSource.open(gpa, io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const M = mdl.Model(ops.MlxOps);
    const m = try M.init(gpa, &g, c, try routes.parse(&.{}, &diag), &weights, &src);
    defer m.deinit(&g);
    var st = try m.newState();
    defer st.deinit(&g, gpa);

    // The streamed experts: MLX slot rows, no prefill rows, grown before the first forward.
    var ediag: expert_bank.Diag = .{};
    var ebank = expert_bank.Bank.open(gpa, io, bank_dir, expert_bank.dsv41, &ediag) catch |e| {
        std.debug.print("dsv41 ar: {s}\n", .{ediag.message()});
        return e;
    };
    defer ebank.deinit();
    const nl = c.n_layers;
    const none = try a.alloc(u32, nl);
    @memset(none, 0);
    const grown = try a.alloc(u32, nl);
    @memset(grown, rows);
    const stream = try expert_stream.Stream.init(gpa, &ebank, .{ .rows = none, .slot_memory = .{ .mlx = s } });
    defer stream.deinit();
    var ssrc = xp.StreamSource.init(stream);
    const Chain = xp.EagerChain(ops.MlxOps, *const xq.Gemv(ops.MlxOps));
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(&kernels.exl3.gemv, &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, kernels, &ex);
    memProbe("dsv41 ar", "slots grown (the residents are loaded lazily, at first use)");

    const out = try a.alloc(u32, ref.new_tokens);
    var hashes: StepHashes = .{ .out = try a.alloc([64]u8, ref.new_tokens) };
    const t0 = std.Io.Timestamp.now(io, .boot);
    try m.greedy(&g, &st, ref.prompt_ids, ref.chunk, &ex, out, &hashes);
    const wall_ms = @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms);

    var first: ?usize = null;
    var logits_equal: usize = 0;
    for (out, ref.generated_ids, 0..) |mine, theirs, i| {
        if (mine != theirs and first == null) first = i;
        if (i < ref.steps.len and std.mem.eql(u8, &hashes.out[i], ref.steps[i].logits_sha256)) logits_equal += 1;
    }
    memProbe("dsv41 ar", "decode (the generated tokens)");
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("\ndsv41 ar: {d} prompt tokens in forwards of {d}, {d} generated; ids {s}; logits rows equal {d}/{d}; {d} rows/layer; routes {d}, hits {d}, misses {d}, {d} B read in {d} preadv; {d} ms; MLX peak {d} B\n", .{
        ref.prompt_ids.len,                           ref.chunk,             out.len,
        if (first == null) "IDENTICAL" else "DIFFER", logits_equal,          out.len,
        rows,                                         sst.route_calls,       sst.expert_cache_hits,
        sst.expert_cache_misses,                      sst.expert_bytes_read, sst.preadv_calls,
        wall_ms,                                      peak,
    });
    if (first) |i| std.debug.print("dsv41 ar: first differing step {d}: native {d}, reference {d} (reference top-2 {any}, margin {d})\n", .{ i, out[i], ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin });
    try testing.expectEqualSlices(u32, ref.generated_ids, out);
}

/// `DSV41_AR_SCHEDULE=served`: the "dsv41 ar:" harness runs the server's schedule instead of 8-row chunks.
fn servedSchedule() bool {
    const v = std.c.getenv("DSV41_AR_SCHEDULE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "served");
}

/// When the served schedule's phase change (the embedding fence, the grown slot banks) runs: `late` at
/// the first 1-row extend, as served; `early_fence` the fence before the prompt; `early_grow` the grow
/// (and the decode cache charge) before the prompt, whose forwards then take the decode lane (<= 8 rows).
pub const ArPhase = enum { late, early_grow, early_fence };
pub const ArTier = enum { served, stock };

/// The served schedule's variant (pass3ab): the prompt's first forward of `split` rows, then ONE extend of
/// the rest; the phase change; the numeric tier.
pub const ServedRun = struct { split: u32, phase: ArPhase, tier: ArTier };

/// DSV41_AR_SPLIT / DSV41_AR_PHASE / DSV41_AR_TIER for an `n`-token prompt (null = unset), refused by name.
pub fn parseServedRun(n: u32, split_s: ?[]const u8, phase_s: ?[]const u8, tier_s: ?[]const u8) !ServedRun {
    const split: u32 = if (split_s) |v| std.fmt.parseInt(u32, v, 10) catch return error.ArSplitNotANumber else n - 1;
    if (split < 1 or split > n - 1) return error.ArSplitRange;
    const phase = if (phase_s) |v| std.meta.stringToEnum(ArPhase, v) orelse return error.ArPhaseUnknown else .late;
    const tier = if (tier_s) |v| std.meta.stringToEnum(ArTier, v) orelse return error.ArTierUnknown else .served;
    // The grown stream refuses wide-lane calls: the early grow feeds the prompt in <= 8-row forwards, 63 + 1 only.
    if (phase == .early_grow and split != n - 1) return error.ArEarlyGrowSplit;
    // The stock tier's prompt runs in 8-row forwards anyway: only the last token's forward is positioned.
    if (tier == .stock and split != n - 1) return error.ArStockSplit;
    return .{ .split = split, .phase = phase, .tier = tier };
}

/// One Module call over prompt rows [lo, hi): the first is `prefill` (a fresh request), the rest `extend`.
pub const PromptCall = struct { lo: u32, hi: u32 };

/// The prompt's Module calls, in order: [0, split) then [split, n); under `early_grow` [0, n - 1) in
/// forwards of <= 8 rows (the decode width), then the last token alone.
pub fn promptCalls(a: std.mem.Allocator, n: u32, run: ServedRun) ![]PromptCall {
    var calls: std.ArrayList(PromptCall) = .empty;
    if (run.phase == .early_grow) {
        var lo: u32 = 0;
        const w: u32 = mdl.Model(ops.MlxOps).scratch_rows; // the decode lane's widest forward
        while (lo < n - 1) : (lo += w) try calls.append(a, .{ .lo = lo, .hi = @min(lo + w, n - 1) });
        try calls.append(a, .{ .lo = n - 1, .hi = n });
    } else {
        try calls.append(a, .{ .lo = 0, .hi = run.split });
        try calls.append(a, .{ .lo = run.split, .hi = n });
    }
    return calls.items;
}

/// Every Module call's rows, prompt then generated (the last prompt call yields generated id 0; each later
/// id is fed alone: `new_tokens - 1` 1-row calls).
pub fn forwardRows(a: std.mem.Allocator, calls: []const PromptCall, new_tokens: u32) ![]u32 {
    const rows = try a.alloc(u32, calls.len + new_tokens - 1);
    for (calls, 0..) |c, i| rows[i] = c.hi - c.lo;
    @memset(rows[calls.len..], 1);
    return rows;
}

/// What the served-schedule run records (the ar-ref-v1 fields plus the schedule; chunk 0 = the model's own rule).
const ServedRecord = struct {
    format: []const u8 = reference_format,
    schedule: []const u8 = "served",
    trunk: []const u8 = "deepseek_v41_module.Module, as the server constructs it, at `tier`",
    split: u32,
    phase: []const u8,
    tier: []const u8,
    /// The model's own chunk rule for a multi-row call (null: derived; the stock tier's 8).
    model_prefill_chunk: ?i64,
    /// The rows of every Module call, prompt then generated.
    forwards: []const u32,
    prompt_ids: []const u32,
    chunk: u32 = 0,
    new_tokens: u32,
    generated_ids: []const u32,
    generated_ids_sha256: []const u8,
    steps: []const Step,
    reference_ids_equal: bool,
    first_difference: ?usize,
    wall_ms: i64,
};

// Guarded window only (loads the bank): DSV41_AR_SCHEDULE=served DSV41_AR_REF=<ar-ref json: its prompt and
// token count> DSV41_BANK=<bank> DSV41_AR_OUT=<new json> _GPU_WINDOW_LOCKED=1 [DSV41_AR_BASELINE_GB=<the
// server's --memory-baseline-gb>] [DSV41_AR_ROWS=<the server's --expert-rows>]. The server's schedule through the
// served module itself (mlx-serve Generator: generate.zig step 0 + deepseek_v41_module prefill / extend):
// ONE forward of prompt[0 .. n-1] (Module.prefill, the model's own chunk rule, the wide routed lane), then the
// last prompt token alone (Module.extend: the phase change first, once), then each generated id alone, greedy.
// Records the ids and each step's logits sha256 / top-2 / margin in the ar-ref format; prints the comparison with
// the reference's ids (the harness schedule) without judging it (the wide lane is rounding-class).
test "dsv41 ar: the served schedule through the served module records its greedy ids" {
    if (!servedSchedule()) return error.SkipZigTest;
    const ref_path = std.mem.span(std.c.getenv("DSV41_AR_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const out_path = std.mem.span(std.c.getenv("DSV41_AR_OUT") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(16 << 20));
    const ref = try std.json.parseFromSliceLeaky(Reference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, reference_format)) return error.ReferenceFormat;
    if (ref.prompt_ids.len < 2 or ref.new_tokens == 0 or ref.generated_ids.len != ref.new_tokens) return error.ReferenceShape;

    const n: u32 = @intCast(ref.prompt_ids.len);
    const run = try parseServedRun(n, envStr("DSV41_AR_SPLIT"), envStr("DSV41_AR_PHASE"), envStr("DSV41_AR_TIER"));
    const calls = try promptCalls(a, n, run);
    const forwards = try forwardRows(a, calls, ref.new_tokens);
    var config = try model.parseConfig(io, a, bank_dir);
    if (std.c.getenv("DSV41_AR_BASELINE_GB")) |v| config.memory_baseline_bytes = @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
    if (std.c.getenv("DSV41_AR_ROWS")) |v| config.expert_rows = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    config.numeric_tier = switch (run.tier) {
        .served => .served,
        .stock => .stock,
    };

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    memProbe("dsv41 ar served", "start");
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const m = try module.Module.init(gpa, io, &config, &weights, s);
    defer m.deinit();
    memProbe("dsv41 ar served", "module constructed (kernels, arm, residents, warm-up)");

    const out = try a.alloc(u32, ref.new_tokens);
    const steps = try a.alloc(Step, ref.new_tokens);
    const t0 = std.Io.Timestamp.now(io, .boot);
    // The phase change moved before the prompt (the module's own pieces; `late` leaves it to extend).
    switch (run.phase) {
        .late => {},
        .early_fence => {
            try dss.embeddingFence(ops.MlxOps, &m.g, m.model, &m.embed_rows, m.weights);
            m.fenced = true;
        },
        .early_grow => {
            _ = mlx.mlx_clear_cache();
            var prev_limit: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev_limit, @import("expert_admission.zig").Envelope.dsv41_pass2.decode_cache_bytes);
            switch (m.arm) {
                inline else => |t| try t.arm.grow(&m.g),
            }
        },
    }
    // The prompt's calls; the last one's logits are generated id 0.
    var logits = try m.prefill(ref.prompt_ids[calls[0].lo..calls[0].hi], 0);
    for (calls[1..]) |c| {
        _ = mlx.mlx_array_free(logits);
        logits = try m.extend(ref.prompt_ids[c.lo..c.hi]);
    }
    memProbe("dsv41 ar served", "the prompt's calls");
    for (out, steps, 0..) |*o, *st, i| {
        if (i > 0) {
            _ = mlx.mlx_array_free(logits);
            logits = try m.extend(&.{out[i - 1]});
        }
        st.* = try stepOf(a, logits, s);
        o.* = st.top2[0];
    }
    _ = mlx.mlx_array_free(logits);
    const wall_ms: i64 = @intCast(@divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms));
    memProbe("dsv41 ar served", "decode (the generated tokens)");

    var d: [32]u8 = undefined;
    const le = try a.alloc(u8, 4 * out.len);
    for (out, 0..) |v, i| std.mem.writeInt(u32, le[4 * i ..][0..4], v, .little);
    std.crypto.hash.sha2.Sha256.hash(le, &d, .{});
    const ids_sha = std.fmt.bytesToHex(d, .lower);
    var first: ?usize = null;
    for (out, ref.generated_ids, 0..) |mine, theirs, i| if (mine != theirs) {
        first = i;
        break;
    };
    const rec: ServedRecord = .{
        .split = run.split,
        .phase = @tagName(run.phase),
        .tier = @tagName(run.tier),
        .model_prefill_chunk = module.numericTier(config.numeric_tier.?).prefill_chunk,
        .forwards = forwards,
        .prompt_ids = ref.prompt_ids,
        .new_tokens = ref.new_tokens,
        .generated_ids = out,
        .generated_ids_sha256 = &ids_sha,
        .steps = steps,
        .reference_ids_equal = first == null,
        .first_difference = first,
        .wall_ms = wall_ms,
    };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\ndsv41 ar served: split {d}+{d}, phase {t}, tier {t}; {d} prompt tokens in {d} calls, {d} generated; ids sha256 {s}; step 0 top-2 {any} margin {d}, step 1 top-2 {any} margin {d}; vs the reference's ids (harness schedule): {s}, first difference {?d}; {d} ms; wrote {s}\n", .{
        run.split,     n - run.split,  run.phase,     run.tier,       n,         calls.len, out.len, &ids_sha,
        steps[0].top2, steps[0].margin, steps[1].top2, steps[1].margin, if (first == null) "IDENTICAL" else "DIFFER", first, wall_ms, out_path,
    });
    if (first) |i| std.debug.print("dsv41 ar served: first differing step {d}: served {d} (top-2 {any}, margin {d}), reference {d} (top-2 {any}, margin {d})\n", .{
        i, out[i], steps[i].top2, steps[i].margin, ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin,
    });
}

fn envStr(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

test "dsv41 ar: the served schedule's variants parse by name and plan their Module calls (pass3ab)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Defaults: 63 + 1, late, served.
    const d = try parseServedRun(64, null, null, null);
    try testing.expectEqual(ServedRun{ .split = 63, .phase = .late, .tier = .served }, d);
    // Refusals by name.
    try testing.expectError(error.ArSplitNotANumber, parseServedRun(64, "x", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "64", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "0", null, null));
    try testing.expectError(error.ArPhaseUnknown, parseServedRun(64, null, "early", null));
    try testing.expectError(error.ArTierUnknown, parseServedRun(64, null, null, "exact"));
    try testing.expectError(error.ArEarlyGrowSplit, parseServedRun(64, "56", "early_grow", null));
    try testing.expectError(error.ArStockSplit, parseServedRun(64, "60", null, "stock"));
    // The six runs' calls and every forward's rows (32 generated ids).
    const R = struct { split: ?[]const u8, phase: ?[]const u8, tier: ?[]const u8, want: []const u32 };
    const ones: [31]u32 = @splat(1);
    const runs = [_]R{
        .{ .split = null, .phase = null, .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R1 served late 63
        .{ .split = "56", .phase = null, .tier = null, .want = &([_]u32{ 56, 8 } ++ ones) }, // R2
        .{ .split = "60", .phase = null, .tier = null, .want = &([_]u32{ 60, 4 } ++ ones) }, // R3
        .{ .split = null, .phase = null, .tier = "stock", .want = &([_]u32{ 63, 1 } ++ ones) }, // R4 (the model chunks 63 by 8)
        .{ .split = null, .phase = "early_fence", .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R5
        .{ .split = null, .phase = "early_grow", .tier = null, .want = &([_]u32{ 8, 8, 8, 8, 8, 8, 8, 7, 1 } ++ ones) }, // R6
    };
    for (runs) |r| {
        const run = try parseServedRun(64, r.split, r.phase, r.tier);
        const calls = try promptCalls(a, 64, run);
        try testing.expectEqual(@as(u32, 0), calls[0].lo);
        try testing.expectEqual(@as(u32, 64), calls[calls.len - 1].hi);
        for (calls[1..], calls[0 .. calls.len - 1]) |c, p| try testing.expectEqual(p.hi, c.lo);
        try testing.expectEqualSlices(u32, r.want, try forwardRows(a, calls, 32));
    }
    // An 8-row extend is not a decode-width phase trigger: the phase change runs at the first 1-row call.
    try testing.expect(!module.phaseChangeDue(8, false) and module.phaseChangeDue(1, false) and !module.phaseChangeDue(1, true));
    // The stock tier's model chunks a multi-row call by 8; the served tier derives its chunk.
    try testing.expectEqual(@as(?i64, 8), module.numericTier(.stock).prefill_chunk);
}

/// One step's record from the module's logits (any float dtype; hashed as the f32 row).
fn stepOf(a: std.mem.Allocator, logits: mlx.mlx_array, s: mlx.mlx_stream) !Step {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const row = (mlx.mlx_array_data_float32(f) orelse return error.MlxNoData)[0..n];
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(row), &d, .{});
    const t = top2Of(row);
    return .{ .logits_sha256 = try a.dupe(u8, &std.fmt.bytesToHex(d, .lower)), .top2 = t, .margin = @floatCast(row[t[0]] - row[t[1]]) };
}

/// The two highest entries, ties to the lower id (MLX argmax's pick for the first).
fn top2Of(row: []const f32) [2]u32 {
    var t: [2]u32 = if (row[1] > row[0]) .{ 1, 0 } else .{ 0, 1 };
    for (row[2..], 2..) |v, i| {
        if (v > row[t[0]]) {
            t[1] = t[0]; // (not `t = .{ i, t[0] }`: the result location aliases t)
            t[0] = @intCast(i);
        } else if (v > row[t[1]]) t[1] = @intCast(i);
    }
    return t;
}

test "dsv41 ar: the served schedule's host preconditions: the top-2 rule and, on the bank, the shell config" {
    try testing.expectEqual([2]u32{ 2, 0 }, top2Of(&.{ 1.0, 0.5, 3.0, 1.0 }));
    try testing.expectEqual([2]u32{ 0, 1 }, top2Of(&.{ 2.0, 2.0, 1.0 }));
    try testing.expectEqual([2]u32{ 1, 3 }, top2Of(&.{ 0.0, 5.0, 1.0, 5.0 }));
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // What Module.init reads from the shell's config: the bank dir and the Engram token map beside it.
    const config = try model.parseConfig(testing.io, arena.allocator(), bank_dir);
    try testing.expect(config.expert_bank_dir != null and config.engram_token_map_path != null);
    if (std.c.getenv("DSV41_AR_REF")) |rp| {
        const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, std.mem.span(rp), arena.allocator(), .limited(16 << 20));
        const ref = try std.json.parseFromSliceLeaky(Reference, arena.allocator(), text, .{ .ignore_unknown_fields = true });
        try testing.expectEqualStrings(reference_format, ref.format);
        try testing.expect(ref.prompt_ids.len >= 2 and ref.generated_ids.len == ref.new_tokens);
    }
}

/// The standard cell's prompt (`mtplx-server-cell-prompt-ids-v1`: scripts/fable/server_cell_bench.py's
/// export; the sweep cell at 16,384 templated tokens, seed 20260829).
pub const prompt_ids_schema = "mtplx-server-cell-prompt-ids-v1";
pub const CellPrompt = struct { cell: []const u8, target_tokens: u32 = 0, seed: ?u64 = null, token_ids: []const u32, token_ids_sha256: []const u8 };
pub const PromptIds = struct { schema: []const u8, context_sha256: []const u8 = "", prompts: []const CellPrompt };

/// The standard cell's prompt from `path`: the sweep entry at `target` tokens and `seed`, its ids'
/// digest (Python's `json.dumps`) equal to the file's. Refused by name otherwise.
pub fn standardPrompt(a: std.mem.Allocator, io: std.Io, path: []const u8, target: u32, seed: u64) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(PromptIds, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, f.schema, prompt_ids_schema)) return error.PromptIdsSchema;
    for (f.prompts) |p| {
        if (!std.mem.eql(u8, p.cell, "sweep") or p.target_tokens != target or p.seed != seed) continue;
        if (p.token_ids.len != target) return error.PromptIdsLength;
        const d = try cell.idsSha256(a, p.token_ids);
        if (!std.mem.eql(u8, &d, p.token_ids_sha256)) return error.PromptIdsDigest;
        return p.token_ids;
    }
    return error.PromptIdsNoCell;
}

/// The seeded fixture (`dsv41-seeded-mtp-comparison-v1`): the Python tier's headline cases (the
/// FASTEST prompt of record is case code-20260923, the one grade_typical_case.py grades).
pub const seeded_fixture_schema = "dsv41-seeded-mtp-comparison-v1";
const FixtureCase = struct { id: []const u8, prompt_ids: []const u32, prompt_ids_sha256: []const u8 };
const SeededFixture = struct { schema: []const u8, cases: []const FixtureCase };

/// The cell's prompt: case `case_id` of a seeded fixture (DSV41_CELL_CASE; the headline's fastest
/// prompt), else the standard sweep entry of a prompt-ids file; 16,384 tokens, its json.dumps digest
/// equal to the file's. Refused by name otherwise.
pub fn cellPrompt(a: std.mem.Allocator, io: std.Io, path: []const u8, case_id: ?[]const u8) ![]const u32 {
    const id = case_id orelse return standardPrompt(a, io, path, 16384, 20260829);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(SeededFixture, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, f.schema, seeded_fixture_schema)) return error.PromptIdsSchema;
    for (f.cases) |c| {
        if (!std.mem.eql(u8, c.id, id)) continue;
        if (c.prompt_ids.len != 16384) return error.PromptIdsLength;
        const d = try cell.idsSha256(a, c.prompt_ids);
        if (!std.mem.eql(u8, &d, c.prompt_ids_sha256)) return error.PromptIdsDigest;
        return c.prompt_ids;
    }
    return error.PromptIdsNoCell;
}

/// The typical-tier cell's receipt (`mlx-serve-dsv41-served-cell-v1`): the standard cell's
/// numbers (prefill tok/s, TTFT, decode tok/s, peak GB decimal, wall), the rows admitted, the
/// per-cycle acceptance and the generated ids.
pub const served_cell_format = "mlx-serve-dsv41-served-cell-v1";
const CellCycle = struct { k_eff: u32, accepted: u32, verified: u32 };
const CellReceipt = struct {
    format: []const u8 = served_cell_format,
    tier: []const u8 = "typical (routes.served: C12-C16, A9, C11, C14 woarc; DSpark typical)",
    typical_delta: f64,
    prompt_file: []const u8,
    /// The fixture case (the fastest prompt), or "sweep-16384-20260829" (the standard prompt).
    prompt_source: []const u8,
    prompt_tokens: usize,
    prompt_ids_sha256: []const u8,
    max_tokens: u32,
    finish: []const u8,
    prefill_rows_per_layer: u32,
    decode_rows_per_layer: u32,
    ttft_s: f64,
    prefill_tok_s: f64,
    phase_change_s: f64,
    decode_wall_s: f64,
    decode_tok_s: f64,
    decode_tok_s_with_phase_change: f64,
    wall_s: f64,
    peak_footprint_gb: f64,
    mlx_peak_gb: f64,
    generated_tokens: usize,
    generated_ids: []const u32,
    generated_ids_sha256: []const u8,
    cycles: []const CellCycle,
    accepted_drafts: u32,
    drafted_tokens: u32,
    accept_rate: f64,
    tokens_per_cycle: f64,
};

// Guarded window only (loads the bank and the served module): DSV41_CELL_PROMPT_IDS=<prompt-ids json
// (the standard cell's)> DSV41_BANK=<bank> DSV41_CELL_OUT=<new json> _GPU_WINDOW_LOCKED=1
// [DSV41_CELL_BASELINE_GB=<the box baseline for the admission>] [DSV41_CELL_ROWS=<fixed decode rows>]
// [DSV41_CELL_DELTA=<typical delta, 0.3>] [DSV41_CELL_MAX_TOKENS=<1024>]. The typical tier's timed cell:
// the served module as the server builds it, the standard 16,384-token prompt as ONE prompt pass (the
// model's chunk rule, the wide lane), the phase change, then DSpark cycles (typical acceptance, the greedy
// correction) until 1,024 tokens or an EOS id. Deterministic (the in-process cell's temperature 0).
test "dsv41 served cell: the typical tier's 16K cell through the served module, timed" {
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const out_path = std.mem.span(std.c.getenv("DSV41_CELL_OUT") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, io, prompt_path, case_id, bank_dir);
    const prompt = inputs.prompt;
    var config = inputs.config;
    try cellConfig(&config);
    const delta: f64 = if (std.c.getenv("DSV41_CELL_DELTA")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else 0.3;
    // The cap counts every generated id, the prompt pass's primary included (the Python headline's
    // 1,024 ids = the primary + 1,023; the server's max_tokens counts the same way).
    const max_tokens: u32 = if (std.c.getenv("DSV41_CELL_MAX_TOKENS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 1024;
    if (max_tokens < 2) return error.CellMaxTokens;

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    memProbe("dsv41 served cell", "start");
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const md = try module.Module.init(gpa, io, &config, &weights, s);
    defer md.deinit();
    memProbe("dsv41 served cell", "module constructed (kernels, arm, residents, warm-up)");

    const g = &md.g;
    // The host-waits arm (the served default: no event gates configured).
    const arm = switch (md.arm) {
        .host_waits => |t| t.arm,
        else => return error.CellArmVariant,
    };
    const L = dsl.Loop(ops.MlxOps);
    // The request's bounded lanes: the prompt, the token cap, one verify block (Module.prefill's rule).
    var st = try md.model.newStateWith(md.model.boundedKv(module.Module.maxPositions(prompt.len, prompt.len + max_tokens)));
    defer st.deinit(g, gpa);
    const caches = try a.alloc(L.H.Cache, md.head.nStages());
    for (caches) |*x| x.* = .{};
    defer for (caches) |*x| x.deinit(g);
    var stops: [8]u32 = undefined;
    const n_stop = config.num_eos_tokens;
    @memcpy(stops[0..n_stop], config.eos_token_ids[0..n_stop]);
    var lp = L.init(g, md.model, md.head, &st, caches, .{
        .acceptance = .{ .typical = .{ .delta = @floatCast(delta) } },
        .prompt_chunk = dsl.whole_prompt,
        .max_tokens = max_tokens - 1,
        .stop_ids = stops[0..n_stop],
    });
    defer lp.deinit();

    _ = mlx.mlx_reset_peak_memory();
    const t0 = std.Io.Timestamp.now(io, .boot);
    const primary = try lp.prefill(gpa, &arm.hook, prompt);
    const ttft_s = secondsSince(io, t0);
    memProbe("dsv41 served cell", "prompt (one pass)");
    const t1 = std.Io.Timestamp.now(io, .boot);
    try md.phaseChange();
    const phase_s = secondsSince(io, t1);
    memProbe("dsv41 served cell", "the phase change (embedding fence, slot banks grown)");
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var cycles: std.ArrayList(CellCycle) = .empty;
    const t2 = std.Io.Timestamp.now(io, .boot);
    var finish: dsl.Finish = .stop;
    if (std.mem.indexOfScalar(u32, stops[0..n_stop], primary) == null) while (true) {
        var lg: dsl.CycleLog = .{ .primary = 0 };
        const f = try lp.cycle(&arm.hook, &out, gpa, &lg);
        try cycles.append(a, .{ .k_eff = lg.k_eff, .accepted = lg.accepted, .verified = lg.verified });
        if (f) |x| {
            finish = x;
            break;
        }
    };
    const decode_s = secondsSince(io, t2);
    const wall_s = secondsSince(io, t0);
    memProbe("dsv41 served cell", "cycles");

    const ids = try a.alloc(u32, out.items.len + 1);
    ids[0] = primary;
    @memcpy(ids[1..], out.items);
    var mlx_peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&mlx_peak);
    const fp = arm_mod.footprint();
    const stt = lp.stats;
    const prompt_sha = try cell.idsSha256(a, prompt);
    const ids_sha = try cell.idsSha256(a, ids);
    const rec: CellReceipt = .{
        .typical_delta = delta,
        .prompt_file = prompt_path,
        .prompt_source = case_id orelse "sweep-16384-20260829",
        .prompt_tokens = prompt.len,
        .prompt_ids_sha256 = &prompt_sha,
        .max_tokens = max_tokens,
        .finish = @tagName(finish),
        .prefill_rows_per_layer = arm.prefill_rows[0],
        .decode_rows_per_layer = arm.decode_rows[0],
        .ttft_s = ttft_s,
        .prefill_tok_s = @as(f64, @floatFromInt(prompt.len)) / ttft_s,
        .phase_change_s = phase_s,
        .decode_wall_s = decode_s,
        // The primary token is the prompt pass's; the decode rate counts the cycles' tokens.
        .decode_tok_s = @as(f64, @floatFromInt(out.items.len)) / decode_s,
        .decode_tok_s_with_phase_change = @as(f64, @floatFromInt(out.items.len)) / (decode_s + phase_s),
        .wall_s = wall_s,
        .peak_footprint_gb = @as(f64, @floatFromInt(fp.peak)) / 1e9,
        .mlx_peak_gb = @as(f64, @floatFromInt(mlx_peak)) / 1e9,
        .generated_tokens = ids.len,
        .generated_ids = ids,
        .generated_ids_sha256 = &ids_sha,
        .cycles = cycles.items,
        .accepted_drafts = stt.accepted_drafts,
        .drafted_tokens = stt.drafted_tokens,
        .accept_rate = stt.acceptRate(),
        .tokens_per_cycle = if (cycles.items.len == 0) 0 else @as(f64, @floatFromInt(out.items.len)) / @as(f64, @floatFromInt(cycles.items.len)),
    };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\ndsv41 served cell: typical {d}, {d} prompt tokens, rows {d} prefill / {d} decode per layer; TTFT {d:.2} s = prefill {d:.1} tok/s; phase change {d:.2} s; decode {d} tokens in {d} cycles, {d:.2} s = {d:.2} tok/s ({d:.2} with the phase change); accepted {d}/{d} drafts; wall {d:.2} s; peak footprint {d:.2} GB, MLX peak {d:.2} GB; finish {s}; ids sha256 {s}; wrote {s}\n", .{
        delta,                      prompt.len,             rec.prefill_rows_per_layer, rec.decode_rows_per_layer,
        ttft_s,                     rec.prefill_tok_s,      phase_s,                    out.items.len,
        cycles.items.len,           decode_s,               rec.decode_tok_s,           rec.decode_tok_s_with_phase_change,
        stt.accepted_drafts,        stt.drafted_tokens,     wall_s,                     rec.peak_footprint_gb,
        rec.mlx_peak_gb,            rec.finish,             rec.generated_ids_sha256,   out_path,
    });
}

/// The window's admission inputs on the shell's config, from the environment the runner sets:
/// DSV41_CELL_BASELINE_GB (the guard's box baseline, required), DSV41_CELL_CEILING_GB (the box the
/// admission fits, required: the window and the bill plan the same rows) and DSV41_CELL_ROWS (a
/// forced decode row count; unset = the admission's fill).
fn cellConfig(config: *model.ModelConfig) !void {
    const gb = struct {
        fn of(name: [*:0]const u8) !?u64 {
            const v = std.c.getenv(name) orelse return null;
            return @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
        }
    }.of;
    config.memory_baseline_bytes = (try gb("DSV41_CELL_BASELINE_GB")) orelse return error.CellBaselineMissing;
    config.memory_ceiling_bytes = (try gb("DSV41_CELL_CEILING_GB")) orelse return error.CellCeilingMissing;
    if (std.c.getenv("DSV41_CELL_ROWS")) |v| config.expert_rows = try std.fmt.parseInt(u32, std.mem.span(v), 10);
}

/// The cell's memory bill (decimal bytes), each term by construction from the bank's headers, the
/// admission the module builds with (`Module.armOptions` at the same config) and the arch's prefill
/// bill (`v41.PrefillBill`, its wave pinned by the served 16K trace test): the prompt phase and the
/// decode phase over the box baseline. `processBound` is what the child may hold above the baseline.
pub const CellBill = struct {
    baseline: u64,
    prefill_rows: u32,
    decode_rows: u32,
    /// (layers x rows + the transient bank's max_route_ids rows) x the bank's record.
    slot_prefill: u64,
    slot_decode: u64,
    lookahead_staging: u64,
    /// Every resident tensor the index names (trunk, head, embedding, the DSpark head); the
    /// embedding leaves the device at the prompt fence (decode phase).
    residents: u64,
    embedding: u64,
    /// The Engram sidecar's residents and its row caches (host).
    engram: u64,
    /// The prompt pass's widest wave x 5 / 4 (`PrefillBill.bytes`' margin).
    prefill_wave: u64,
    /// The request's bounded KV (the served ring + the sources' lanes) for prompt + max_tokens + a block.
    kv: u64,
    /// The served tier's prefill allocator cache (4 GiB, D5) and the decode charge.
    prefill_cache: u64,
    decode_cache: u64,
    /// A verify forward's wave (8 rows) with its index chain over every position, and the draft block's.
    decode_wave: u64,
    draft_wave: u64,
    /// The admission's host reserve (pools, tables, the token map, the process).
    host_reserve: u64,

    pub fn prefillTotal(b: CellBill) u64 {
        return b.baseline + b.slot_prefill + b.lookahead_staging + b.residents + b.engram + b.prefill_wave + b.kv + b.prefill_cache + b.host_reserve;
    }

    pub fn decodeTotal(b: CellBill) u64 {
        return b.baseline + b.slot_decode + b.lookahead_staging + b.residents - b.embedding + b.engram + b.kv + b.decode_wave + b.draft_wave + b.decode_cache + b.host_reserve;
    }

    pub fn processBound(b: CellBill) u64 {
        return @max(b.prefillTotal(), b.decodeTotal()) - b.baseline;
    }
};

pub fn cellBill(a: std.mem.Allocator, io: std.Io, config: *const model.ModelConfig, prompt_tokens: u64, max_tokens: u64) !CellBill {
    const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
    var vd: v41.Diag = .{};
    errdefer if (vd.len > 0) std.debug.print("dsv41 served cell bill: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, dir, &vd);
    const ceiling = module.boxCeiling(config.memory_ceiling_bytes orelse return error.CellCeilingMissing, c.n_routed_experts);
    var diag: arm_mod.Diag = .{};
    // The wired bytes the module reads at construction (vm_stat) are the window's: the runner measures
    // them after the guard unloaded the service and passes DSV41_CELL_WIRED_GB (unset: read now).
    var opts = module.armOptions(config, ceiling, .host);
    if (std.c.getenv("DSV41_CELL_WIRED_GB")) |v| opts.wired_bytes = @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
    var p = arm_mod.planRows(a, io, opts, &diag) catch |e| {
        std.debug.print("dsv41 served cell bill: refused: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    defer if (p.draft_subset) |*x| x.deinit();
    const rec = p.inputs.record_bytes;
    const transient: u64 = xp.max_route_ids;
    var ck = try v41.Checkpoint.openIndexed(a, io, dir, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    const epath = try std.fmt.allocPrint(a, "{s}/engram/engram-residents.safetensors", .{dir});
    var eck = try v41.Checkpoint.openFile(a, epath, &vd);
    defer eck.deinit();
    const em = try v41.WeightMap.build(a, try v41.engramSpec(a, &c), &eck, &vd);
    const bill = v41.PrefillBill.of(&c);
    const positions = prompt_tokens + max_tokens + mdl.Model(ops.MlxOps).scratch_rows;
    const rows: u64 = mdl.Model(ops.MlxOps).scratch_rows;
    // A verify forward: the fixed wave at 8 rows plus its index chain over every position (two arrays live).
    const decode_wave = bill.waveBytes(rows, rows, .served) + v41.PrefillBill.chain_copies * rows * bill.index_heads * positions * 4;
    return .{
        .baseline = config.memory_baseline_bytes.?,
        .prefill_rows = p.prefill_rows,
        .decode_rows = p.decode_rows,
        .slot_prefill = (@as(u64, c.n_layers) * p.prefill_rows + transient) * rec,
        .slot_decode = (@as(u64, c.n_layers) * p.decode_rows + transient) * rec,
        .lookahead_staging = p.inputs.lookahead_staging_bytes,
        .residents = m.totalBytes(),
        .embedding = m.bytes_by_module[@backingInt(v41.Module.embed)],
        .engram = em.totalBytes() + engram.row_cache_host_bytes,
        .prefill_wave = bill.waveBytes(bill.chunkRows(prompt_tokens), prompt_tokens, .served) / 4 * 5,
        .kv = bill.window_ring_bytes + positions * bill.kv_source_pos_bytes,
        .prefill_cache = module.prefillCacheLimit(.served),
        .decode_cache = expert_admission.Envelope.dsv41_pass2.decode_cache_bytes,
        .decode_wave = decode_wave,
        .draft_wave = decode_wave,
        .host_reserve = p.inputs.host_reserve_bytes,
    };
}

fn printBill(b: CellBill) void {
    const gb = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e9;
        }
    }.f;
    std.debug.print("\ndsv41 served cell bill (decimal GB; prompt / decode phase):\n", .{});
    const T = struct { name: []const u8, p: u64, d: u64 };
    for ([_]T{
        .{ .name = "box baseline (the guard's)", .p = b.baseline, .d = b.baseline },
        .{ .name = "slot banks (layers x rows + 48) x record", .p = b.slot_prefill, .d = b.slot_decode },
        .{ .name = "lookahead staging", .p = b.lookahead_staging, .d = b.lookahead_staging },
        .{ .name = "residents (the embedding off at the fence)", .p = b.residents, .d = b.residents - b.embedding },
        .{ .name = "Engram residents + row caches", .p = b.engram, .d = b.engram },
        .{ .name = "prompt wave x 5/4 (PrefillBill) / verify + draft waves", .p = b.prefill_wave, .d = b.decode_wave + b.draft_wave },
        .{ .name = "KV (ring + source lanes, bounded)", .p = b.kv, .d = b.kv },
        .{ .name = "MLX allocator cache (the phase's limit)", .p = b.prefill_cache, .d = b.decode_cache },
        .{ .name = "host reserve (pools, tables, process)", .p = b.host_reserve, .d = b.host_reserve },
    }) |t| std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}\n", .{ t.name, gb(t.p), gb(t.d) });
    std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}   rows {d} / {d}; process bound {d:.2}\n", .{ "TOTAL", gb(b.prefillTotal()), gb(b.decodeTotal()), b.prefill_rows, b.decode_rows, gb(b.processBound()) });
    std.debug.print("DSV41_CELL_BILL {{\"baseline_gb\": {d:.3}, \"prefill_rows\": {d}, \"decode_rows\": {d}, \"prefill_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}, \"process_bound_gb\": {d:.3}}}\n", .{ gb(b.baseline), b.prefill_rows, b.decode_rows, gb(b.prefillTotal()), gb(b.decodeTotal()), gb(b.processBound()) });
}

// The runner's --bill mode (host; bank): DSV41_CELL_BILL=1 DSV41_BANK DSV41_CELL_BASELINE_GB
// DSV41_CELL_CEILING_GB [DSV41_CELL_WIRED_GB] [DSV41_CELL_ROWS] [DSV41_CELL_MAX_TOKENS]: the cell's bill at the rows the window
// will admit, printed as a table and one DSV41_CELL_BILL json line.
test "dsv41 served cell: the cell's bill on the host (the window's admission, every term)" {
    if (std.c.getenv("DSV41_CELL_BILL") == null) return error.SkipZigTest;
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    try cellConfig(&config);
    const max_tokens: u64 = if (std.c.getenv("DSV41_CELL_MAX_TOKENS")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else 1024;
    const b = try cellBill(a, testing.io, &config, 16384, max_tokens);
    printBill(b);
    try testing.expect(b.decode_rows >= b.prefill_rows and b.processBound() > 0);
}

fn secondsSince(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.untilNow(io, .boot).nanoseconds)) / 1e9;
}

/// The cell's host inputs: the standard prompt (16,384 tokens, seed 20260829, digest checked) and
/// the shell's config of the bank (its bank / token-map paths, the EOS ids the Generator stops on).
fn cellInputs(a: std.mem.Allocator, io: std.Io, prompt_path: []const u8, case_id: ?[]const u8, bank_dir: []const u8) !struct { prompt: []const u32, config: model.ModelConfig } {
    const prompt = try cellPrompt(a, io, prompt_path, case_id);
    const config = try model.parseConfig(io, a, bank_dir);
    if (config.expert_bank_dir == null or config.engram_token_map_path == null) return error.Dsv41BankDir;
    if (config.num_eos_tokens == 0) return error.NoEosIds;
    return .{ .prompt = prompt, .config = config };
}

// The served cell's preconditions on the real inputs (host; bank mode): DSV41_BANK and
// DSV41_CELL_PROMPT_IDS as the window passes them. The prompt entry, its length and digest, the
// config's paths and EOS ids; the receipt serialises.
test "dsv41 served cell: the window's inputs pass on the host (the standard prompt, the bank's shell config)" {
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Either line: the fastest prompt (a fixture case, DSV41_CELL_CASE) or the standard sweep prompt.
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, testing.io, prompt_path, case_id, bank_dir);
    try testing.expectEqual(@as(usize, 16384), inputs.prompt.len);
    const sha = try cell.idsSha256(a, inputs.prompt);
    const want = if (case_id) |id| (if (std.mem.eql(u8, id, "code-20260923")) "667506d734cce8152f3c42c9a97639a5b466540c70d9798a72e7564e2361bf86" else "") else "1a45b35bae742fae0e26d4f40ee0dc1093a2038e5b514460a4f02e9e56d74565";
    if (want.len > 0) try testing.expectEqualStrings(want, &sha);
    if (case_id != null) {
        try testing.expectError(error.PromptIdsNoCell, cellPrompt(a, testing.io, prompt_path, "no-such-case"));
    } else try testing.expectError(error.PromptIdsNoCell, standardPrompt(a, testing.io, prompt_path, 16384, 1));
    try testing.expectEqual(@as(usize, 16384 + 1024 + mdl.Model(ops.MlxOps).scratch_rows), module.Module.maxPositions(16384, 16384 + 1024));
    // The admission inputs come from the runner's environment (refused by name without them).
    var cfg = inputs.config;
    if (std.c.getenv("DSV41_CELL_BASELINE_GB") == null) try testing.expectError(error.CellBaselineMissing, cellConfig(&cfg));
    const rec: CellReceipt = .{ .typical_delta = 0.3, .prompt_file = prompt_path, .prompt_source = "x", .prompt_tokens = 16384, .prompt_ids_sha256 = "x", .max_tokens = 1024, .finish = "stop", .prefill_rows_per_layer = 1, .decode_rows_per_layer = 2, .ttft_s = 1, .prefill_tok_s = 1, .phase_change_s = 0, .decode_wall_s = 1, .decode_tok_s = 1, .decode_tok_s_with_phase_change = 1, .wall_s = 1, .peak_footprint_gb = 1, .mlx_peak_gb = 1, .generated_tokens = 1, .generated_ids = &.{1}, .generated_ids_sha256 = "y", .cycles = &.{.{ .k_eff = 5, .accepted = 3, .verified = 6 }}, .accepted_drafts = 3, .drafted_tokens = 5, .accept_rate = 0.6, .tokens_per_cycle = 4 };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{});
    try testing.expect(std.mem.indexOf(u8, json, "\"decode_rows_per_layer\":2") != null);
}

pub const dspark_reference_format = "mlx-serve-dsv41-dspark-ref-v1";

pub const RefCycle = struct {
    primary: u32,
    draft_ids: []const u32,
    conf_sigmoid_bits: []const u32,
    k_eff_native: u32,
    drafts: []const u32,
    targets: []const []const u32,
    flags: []const []const bool = &.{},
    verified: u32,
    kept: u32,
};

pub const DsparkReference = struct {
    format: []const u8,
    arm: []const u8,
    delta: ?f64 = null,
    prompt: []const u32,
    tokens: []const u32,
    cycles: []const RefCycle,
};

// Bank mode, host only: DSV41_BANK, DSV41_ENGRAM_TOKEN_MAP and DSV41_ENGRAM_REPLAY_REF=<dump_dsv41_dspark_ref.py
// json>. The reference run's Engram history replayed twice through the native hashing: as the lane of record keeps
// it (each verify's rejected rows trimmed with the KV: deepseek_v41_cache.py trim / rollback) and as the reference
// kept it before the fix (its cache had no engram_state, so no trim reached the history). Names every verify row
// whose Engram rows differ.
test "dsv41 engram: a reference without the Engram trim hashes the verify rows after each trimmed verify apart" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_REPLAY_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(DsparkReference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, dspark_reference_format)) return error.ReferenceFormat;
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 engram: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    var src = try engram.RowSource.open(gpa, io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const per = src.perToken();
    var lane: engram.HashState = .{};
    defer lane.deinit(gpa);
    var untrimmed: engram.HashState = .{};
    defer untrimmed.deinit(gpa);
    // The prompt in forwards of 8 rows, as the reference ran it.
    var i: usize = 0;
    while (i < ref.prompt.len) : (i += 8) {
        const span = ref.prompt[i..@min(i + 8, ref.prompt.len)];
        const rows = try a.alloc(i64, span.len * per);
        try src.advance(gpa, &lane, span, rows);
        try src.advance(gpa, &untrimmed, span, rows);
    }
    var first: ?usize = null;
    var n_cycles: usize = 0;
    var n_rows: usize = 0;
    for (ref.cycles, 0..) |rc, ci| {
        const span = try a.alloc(u32, 1 + rc.drafts.len);
        span[0] = rc.primary;
        @memcpy(span[1..], rc.drafts);
        if (span.len != rc.verified or rc.kept == 0 or rc.kept > rc.verified) return error.ReferenceShape;
        const rl = try a.alloc(i64, span.len * per);
        const ru = try a.alloc(i64, span.len * per);
        try src.advance(gpa, &lane, span, rl);
        try src.advance(gpa, &untrimmed, span, ru);
        var differ: [ds.max_block + 1]u32 = undefined;
        var n: usize = 0;
        for (0..span.len) |j| if (!std.mem.eql(i64, rl[j * per ..][0..per], ru[j * per ..][0..per])) {
            differ[n] = @intCast(j);
            n += 1;
        };
        if (n > 0) {
            if (first == null) first = ci;
            n_cycles += 1;
            n_rows += n;
        }
        if (ci < 6) std.debug.print("dsv41 engram: cycle {d}: verify {any}, kept {d} of {d}; rows whose Engram rows differ {any}\n", .{ ci, span, rc.kept, rc.verified, differ[0..n] });
        lane.trim(rc.verified - rc.kept);
    }
    std.debug.print("dsv41 engram: {d} cycles; the untrimmed history first hashes a verify row apart at cycle {?d}; {d} cycles, {d} rows apart in all\n", .{ ref.cycles.len, first, n_cycles, n_rows });
    // The first verify hashes alike; a trimmed verify leaves the next one's first rows apart.
    try testing.expect(first != null and first.? >= 1);
}

// Guarded window only (loads the bank): DSV41_DSPARK_REF=<dump_dsv41_dspark_ref.py json> DSV41_BANK=<bank>
// DSV41_ENGRAM_TOKEN_MAP=<converter map> _GPU_WINDOW_LOCKED=1 [DSV41_AR_ROWS=<decode rows per layer, default 16>]
// [DSV41_KV_BOUNDED=1: the request's KV lanes bounded to its positions (M5BOUND), else the tier's route]
test "dsv41 ar: the native DSpark loop takes the Python lane's cycle decisions on the real model" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_DSPARK_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(DsparkReference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, dspark_reference_format)) return error.ReferenceFormat;
    const rows: u32 = if (std.c.getenv("DSV41_AR_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 16;

    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 dspark: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(gpa, s);
    defer g.deinit();
    // The bound: MLX keeps no freed buffer (the kernels' startup check and each forward's transients go back).
    var prev_cache: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cache, 0);
    defer _ = mlx.mlx_set_cache_limit(&prev_cache, prev_cache);
    memProbe("dsv41 dspark", "start");
    const kernels = try acceptKernels(gpa, &g, &c);
    defer kernels.deinit(&g);
    memProbe("dsv41 dspark", "kernels accepted (the startup self-check)");
    // The served decode seam's own binding of the residents (`Dspark(A).open`).
    const L = dsl.Loop(ops.MlxOps);
    // The Python reference ran the stock path (every lever unset).
    const res = try dss.Resources(ops.MlxOps).open(gpa, io, &g, bank_dir, c, routes.stock, map_path, null, &diag);
    defer res.deinit(&g);
    const m = res.model;
    const head = res.head;
    // M5BOUND: every KV lane sized once to the run's positions (the prompt, its tokens, one verify block).
    const kv_bound: ?u32 = if (std.c.getenv("DSV41_KV_BOUNDED") != null) @intCast(ref.prompt.len + ref.tokens.len + 8) else null;
    var st = if (kv_bound) |n| try m.newStateWith(m.boundedKv(n)) else try m.newState();
    defer st.deinit(&g, gpa);
    var caches: [8]L.H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);

    var ediag: expert_bank.Diag = .{};
    var ebank = expert_bank.Bank.open(gpa, io, bank_dir, expert_bank.dsv41, &ediag) catch |e| {
        std.debug.print("dsv41 dspark: {s}\n", .{ediag.message()});
        return e;
    };
    defer ebank.deinit();
    const none = try a.alloc(u32, c.n_layers);
    @memset(none, 0);
    const grown = try a.alloc(u32, c.n_layers);
    @memset(grown, rows);
    const stream = try expert_stream.Stream.init(gpa, &ebank, .{ .rows = none, .slot_memory = .{ .mlx = s } });
    defer stream.deinit();
    var ssrc = xp.StreamSource.init(stream);
    const Chain = xp.EagerChain(ops.MlxOps, *const xq.Gemv(ops.MlxOps));
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(&kernels.exl3.gemv, &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, kernels, &ex);
    memProbe("dsv41 dspark", "residents and the draft head built, slots grown");

    const acceptance: ds.Acceptance = if (std.mem.eql(u8, ref.arm, "typical")) .{ .typical = .{ .delta = @floatCast(ref.delta.?) } } else .greedy;
    var lp = L.init(&g, m, head, &st, caches[0..head.nStages()], .{ .acceptance = acceptance, .max_tokens = 1 << 20 });
    defer lp.deinit();
    const primary = try lp.prefill(gpa, &ex, ref.prompt);
    try testing.expectEqual(ref.tokens[0], primary);
    memProbe("dsv41 dspark", "prompt");
    // The served adapter's fence: the embedding table retires to its host rows before the cycles.
    try res.retireEmbedding(&g);
    memProbe("dsv41 dspark", "the prompt fence (the embedding table freed)");
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    // The gate: the generated ids and each cycle's acceptance (drafts proposed,
    // rows verified, drafts accepted). The finer decisions (draft ids, sigmoid
    // bits, verify argmax, typical flags) are reported, first difference named.
    var first_accept: ?usize = null;
    var first_decision: ?usize = null;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var classified = false;
    for (ref.cycles, 0..) |rc, i| {
        var lg: dsl.CycleLog = .{ .primary = 0, .want_top = true };
        _ = try lp.cycle(&ex, &out, gpa, &lg);
        const accept_same = lg.accepted + 1 == rc.kept and lg.verified == rc.verified and lg.k_eff == rc.drafts.len;
        var same = accept_same and lg.primary == rc.primary and lg.k_native == rc.k_eff_native;
        same = same and std.mem.eql(u32, lg.native[0..rc.draft_ids.len], rc.draft_ids);
        same = same and std.mem.eql(u32, lg.drafts[0..lg.k_eff], rc.drafts);
        for (rc.conf_sigmoid_bits, 0..) |bits, j| same = same and @as(u32, @bitCast(lg.conf[j])) == bits;
        var t: usize = 0;
        for (rc.targets) |chunk| for (chunk) |v| {
            const eq = t < lg.n_targets and lg.targets[t] == v;
            same = same and eq;
            // The first verify row that picks another token, by the tie-flip rule: the reference's token
            // is our second and the top two logits lie within 2^-5 of the row's rms.
            if (!eq and !classified and t < lg.n_targets) {
                classified = true;
                const margin = (lg.top_logits[t][0] - lg.top_logits[t][1]) / lg.rms[t];
                const flip = lg.top_ids[t][1] == v and margin <= 1.0 / 32.0;
                std.debug.print("dsv41 dspark: first divergence cycle {d} verify row {d}: ours {d} (logit {d:.6}), second {d} (logit {d:.6}), reference {d}; margin / rms {d:.6}: {s}\n", .{
                    i,                                                                                                                                                                                    t, lg.top_ids[t][0], lg.top_logits[t][0], lg.top_ids[t][1], lg.top_logits[t][1], v, margin,
                    if (flip) "TIE FLIP (within 2^-5 of the row rms)" else if (lg.top_ids[t][1] == v) "NOT a tie flip (margin above 2^-5)" else "NOT a tie flip (the reference token is not our second)",
                });
            }
            t += 1;
        };
        var fl: usize = 0;
        for (rc.flags) |chunk| for (chunk) |v| {
            same = same and fl < lg.n_flags and lg.flags[fl] == v;
            fl += 1;
        };
        if (!accept_same and first_accept == null) {
            first_accept = i;
            std.debug.print("dsv41 dspark: cycle {d} acceptance differs: drafts {d} vs {d}, verified {d} vs {d}, accepted {d} vs {d}\n", .{ i, lg.k_eff, rc.drafts.len, lg.verified, rc.verified, lg.accepted, rc.kept - 1 });
        }
        if (!same and first_decision == null) {
            first_decision = i;
            std.debug.print("\ndsv41 dspark: cycle {d} first finer difference: primary {d} vs {d}, native {any} vs {any}, k {d} vs {d}, conf bits {any} vs {any}, drafts {any} vs {any}, targets {any} vs {any}, flags {any} vs {any}\n", .{
                i,                    lg.primary,
                rc.primary,           lg.native[0..rc.draft_ids.len],
                rc.draft_ids,         lg.k_native,
                rc.k_eff_native,      @as([]const u32, @ptrCast(lg.conf[0..rc.conf_sigmoid_bits.len])),
                rc.conf_sigmoid_bits, lg.drafts[0..lg.k_eff],
                rc.drafts,            lg.targets[0..lg.n_targets],
                rc.targets,           lg.flags[0..lg.n_flags],
                rc.flags,
            });
        }
    }
    const wall_ms = @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms);
    memProbe("dsv41 dspark", "cycles");
    const n = @min(out.items.len, ref.tokens.len - 1);
    const ids_same = std.mem.eql(u32, out.items[0..n], ref.tokens[1..][0..n]);
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("\ndsv41 dspark: {s} arm, kv {s} {d}, {d} cycles; ids {s} ({d}); per-cycle acceptance {s}; decisions {s}; accepted {d}/{d}; {d} rows/layer; routes {d}, {d} B read; {d} ms; MLX peak {d} B\n", .{
        ref.arm,                                             if (kv_bound != null) "bounded" else "tier",
        kv_bound orelse 0,                                   ref.cycles.len,
        if (ids_same) "IDENTICAL" else "DIFFER",             n + 1,
        if (first_accept == null) "IDENTICAL" else "DIFFER", if (first_decision == null) "IDENTICAL" else "DIFFER",
        lp.stats.accepted_drafts,                            lp.stats.drafted_tokens,
        rows,                                                sst.route_calls,
        sst.expert_bytes_read,                               wall_ms,
        peak,
    });
    try testing.expect(first_accept == null);
    try testing.expectEqualSlices(u32, ref.tokens[1..][0..n], out.items[0..n]);
}
