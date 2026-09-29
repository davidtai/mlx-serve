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

/// One kv-source layer's lanes at a point of the run (the short-prompt readout): the entry offset, the
/// window rows visible (offset - drop) and the drop, the compressed rows, the compressor frontier's fed
/// rows; for an index source, n_comp and the selection's valid keys for the last row (min(index_topk,
/// offset / ratio): the selection keeps the top index_topk of the groups the row reaches).
const LayerStateLine = struct {
    point: []const u8,
    layer: u32,
    ratio: u32,
    index_source: bool,
    positions: [2]u32,
    offset: u32,
    window_rows: u32,
    window_drop: u32,
    compressed_rows: u32,
    frontier_rows: u32,
    n_comp: ?u32,
    valid_selected_keys: ?u32,
    index_topk: u32,
};

/// DSV41_AR_PROMPT_IDS=<json with "prompt_ids"> + DSV41_AR_PROMPT_TOKENS=<L>: the first L ids as the prompt
/// (both or neither; refused by name otherwise).
pub fn promptOverride(a: std.mem.Allocator, io: std.Io, path: ?[]const u8, tokens: ?[]const u8) !?[]const u32 {
    if (path == null and tokens == null) return null;
    const p = path orelse return error.ArPromptOverrideHalf;
    const t = tokens orelse return error.ArPromptOverrideHalf;
    const l = std.fmt.parseInt(usize, t, 10) catch return error.ArPromptTokensNotANumber;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(struct { prompt_ids: []const u32 }, a, text, .{ .ignore_unknown_fields = true });
    if (l < 2 or l > f.prompt_ids.len) return error.ArPromptTokensRange;
    return f.prompt_ids[0..l];
}

/// What the served-schedule run records (the ar-ref-v1 fields plus the schedule; chunk 0 = the model's own rule).
const ServedRecord = struct {
    /// "ref" (the reference's prompt) or "p16" (the override's first `prompt_tokens` ids).
    prompt: []const u8,
    prompt_tokens: u32,
    state: []const LayerStateLine,
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

    const override = try promptOverride(a, io, envStr("DSV41_AR_PROMPT_IDS"), envStr("DSV41_AR_PROMPT_TOKENS"));
    const prompt: []const u32 = override orelse ref.prompt_ids;
    const n: u32 = @intCast(prompt.len);
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
    var state: std.ArrayList(LayerStateLine) = .empty;
    const probe = stateProbe(&m.model.c);
    var logits = try m.prefill(prompt[calls[0].lo..calls[0].hi], 0);
    for (calls[1..]) |c| {
        _ = mlx.mlx_array_free(logits);
        logits = try m.extend(prompt[c.lo..c.hi]);
    }
    try readState(a, &state, m, probe, "after_prompt", calls[calls.len - 1].lo);
    memProbe("dsv41 ar served", "the prompt's calls");
    for (out, steps, 0..) |*o, *st, i| {
        if (i > 0) {
            _ = mlx.mlx_array_free(logits);
            const before = m.state.?.offset;
            logits = try m.extend(&.{out[i - 1]});
            if (i <= 2) try readState(a, &state, m, probe, if (i == 1) "after_step1" else "after_step2", before);
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
    if (override == null) for (out, ref.generated_ids, 0..) |mine, theirs, i| if (mine != theirs) {
        first = i;
        break;
    };
    const rec: ServedRecord = .{
        .prompt = if (override != null) "p16" else "ref",
        .prompt_tokens = n,
        .state = state.items,
        .split = run.split,
        .phase = @tagName(run.phase),
        .tier = @tagName(run.tier),
        .model_prefill_chunk = module.numericTier(config.numeric_tier.?).prefill_chunk,
        .forwards = forwards,
        .prompt_ids = prompt,
        .new_tokens = ref.new_tokens,
        .generated_ids = out,
        .generated_ids_sha256 = &ids_sha,
        .steps = steps,
        .reference_ids_equal = override == null and first == null,
        .first_difference = first,
        .wall_ms = wall_ms,
    };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\ndsv41 ar served: split {d}+{d}, phase {t}, tier {t}; {d} prompt tokens in {d} calls, {d} generated; ids sha256 {s}; step 0 top-2 {any} margin {d}, step 1 top-2 {any} margin {d}; vs the reference's ids (harness schedule): {s}, first difference {?d}; {d} ms; wrote {s}\n", .{
        run.split,     n - run.split,  run.phase,     run.tier,       n,         calls.len, out.len, &ids_sha,
        steps[0].top2, steps[0].margin, steps[1].top2, steps[1].margin, if (override != null) "ref: prompt differs" else if (first == null) "IDENTICAL" else "DIFFER", first, wall_ms, out_path,
    });
    if (first) |i| std.debug.print("dsv41 ar served: first differing step {d}: served {d} (top-2 {any}, margin {d}), reference {d} (top-2 {any}, margin {d})\n", .{
        i, out[i], steps[i].top2, steps[i].margin, ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin,
    });
}

/// The readout's layers: the first kv source of each of the first two compression ratios (V4.1's
/// config: ratio 2 on layers 2-19, ratio 1 on 20-39; not V4's 4 / 128).
fn stateProbe(c: *const v41.Config) [2]?u32 {
    var out: [2]?u32 = .{ null, null };
    var ratios: [2]u8 = .{ 0, 0 };
    for (c.layers[0..c.n_layers], 0..) |li, l| {
        if (!li.kv_source or li.ratio == 0) continue;
        for (&out, &ratios) |*o, *r| {
            if (o.* != null and r.* == li.ratio) break;
            if (o.* == null) {
                o.* = @intCast(l);
                r.* = li.ratio;
                break;
            }
        }
    }
    return out;
}

fn readState(a: std.mem.Allocator, lines: *std.ArrayList(LayerStateLine), m: *module.Module, probe: [2]?u32, point: []const u8, lo: u32) !void {
    const st = &m.state.?;
    const c = &m.model.c;
    for (probe) |pl| {
        const l = pl orelse continue;
        const ls = &st.layers[l];
        const li = c.layers[l];
        const drop = ls.window.dropOffset();
        const n_comp = ls.compress.rows();
        const line: LayerStateLine = .{
            .point = point,
            .layer = l,
            .ratio = li.ratio,
            .index_source = li.index_source,
            .positions = .{ lo, ls.offset },
            .offset = ls.offset,
            .window_rows = ls.offset - drop,
            .window_drop = drop,
            .compressed_rows = n_comp,
            .frontier_rows = ls.nFed(),
            .n_comp = if (li.index_source) n_comp else null,
            .valid_selected_keys = if (li.index_source) @min(c.index_topk, ls.offset / li.ratio) else null,
            .index_topk = c.index_topk,
        };
        try lines.append(a, line);
        std.debug.print("dsv41 ar state: {s} layer {d} ratio {d}{s}: positions [{d}, {d}), offset {d}, window rows {d} (drop {d}), compressed rows {d}, frontier rows {d}, n_comp {?d}, valid selected keys {?d} of index_topk {d}\n", .{
            point, l, li.ratio, if (li.index_source) " (index source)" else "", lo, ls.offset, ls.offset, line.window_rows, drop, n_comp, line.frontier_rows, line.n_comp, line.valid_selected_keys, c.index_topk,
        });
    }
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
    // The prompt override: both variables or neither, a number, within the file.
    try testing.expectEqual(@as(?[]const u32, null), try promptOverride(a, testing.io, null, null));
    try testing.expectError(error.ArPromptOverrideHalf, promptOverride(a, testing.io, "x.json", null));
    try testing.expectError(error.ArPromptOverrideHalf, promptOverride(a, testing.io, null, "64"));
    try testing.expectError(error.ArPromptTokensNotANumber, promptOverride(a, testing.io, "x.json", "sixty"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "p.json", .data = "{\"prompt_ids\": [5, 6, 7, 8, 9], \"note\": 1}" });
    var root: [512]u8 = undefined;
    const rp = root[0..try tmp.dir.realPath(testing.io, &root)];
    const path = try std.fmt.allocPrint(a, "{s}/p.json", .{rp});
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, (try promptOverride(a, testing.io, path, "3")).?);
    try testing.expectError(error.ArPromptTokensRange, promptOverride(a, testing.io, path, "6"));
    // L1-L6: the long prompts' plans (late, split L-1).
    for ([_]u32{ 64, 128, 256, 1024 }) |l| {
        const run = try parseServedRun(l, null, null, null);
        const calls = try promptCalls(a, l, run);
        try testing.expectEqual(@as(usize, 2), calls.len);
        try testing.expectEqual(l - 1, calls[0].hi - calls[0].lo);
    }
    // The readout's layers on the real geometry: a ratio-4 and a ratio-128 kv source.
    const json = try v41.testConfigJson(a, .real);
    const rc = try v41.Config.parse(a, json, null);
    const pr = stateProbe(&rc);
    try testing.expect(pr[0] != null and pr[1] != null);
    try testing.expectEqual(@as(u8, 2), rc.layers[pr[0].?].ratio);
    try testing.expectEqual(@as(u8, 1), rc.layers[pr[1].?].ratio);
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
