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
/// (`status.footprint`, the one reader). The gap between the footprint and MLX is the host side.
fn memProbe(harness: []const u8, phase: []const u8) void {
    _ = memProbePeak(harness, phase);
}

/// `memProbe`, returning the MLX peak since the previous probe (the probe resets it).
fn memProbePeak(harness: []const u8, phase: []const u8) usize {
    var active: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    const fp = status.footprint().now;
    std.debug.print("\n{s}: memory {s}: MLX active {d:.2} GB, MLX peak since the last probe {d:.2} GB, footprint {d:.2} GB\n", .{
        harness, phase, @as(f64, @floatFromInt(active)) / 1e9, @as(f64, @floatFromInt(peak)) / 1e9, @as(f64, @floatFromInt(fp)) / 1e9,
    });
    // The box's pages as the guard reads them: what is outside this footprint shows here.
    const v = status.vmBytes();
    std.debug.print("NATIVE vm {s}: physical used {d} B (wired {d}, active {d}, inactive {d}, compressor {d}; purgeable {d}, speculative {d}, file-backed {d}), footprint {d} B\n", .{
        phase, status.physicalUsedBytes(v), v.wired, v.active, v.inactive, v.compressor, v.purgeable, v.speculative, v.external, fp,
    });
    _ = mlx.mlx_reset_peak_memory();
    return peak;
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
/// Every native receipt's runtime label (the Python lanes' receipts carry "python-mtplx").
pub const runtime_native = "native-mlx-serve";

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
/// the rest (none when `split` is the whole prompt: the served shell's shape since dsv41 prefills
/// unchunked, one Module.prefill whose logits yield the first id); the phase change; the numeric tier.
pub const ServedRun = struct { split: u32, phase: ArPhase, tier: ArTier };

/// DSV41_AR_SPLIT / DSV41_AR_PHASE / DSV41_AR_TIER for an `n`-token prompt (null = unset), refused by name.
pub fn parseServedRun(n: u32, split_s: ?[]const u8, phase_s: ?[]const u8, tier_s: ?[]const u8) !ServedRun {
    const phase = if (phase_s) |v| std.meta.stringToEnum(ArPhase, v) orelse return error.ArPhaseUnknown else .late;
    const tier = if (tier_s) |v| std.meta.stringToEnum(ArTier, v) orelse return error.ArTierUnknown else .served;
    // The default is the served shell's: the whole prompt in one prefill (the stock tier and the early
    // grow keep the last token's own 1-row forward).
    const whole = tier == .served and phase != .early_grow;
    const split: u32 = if (split_s) |v| std.fmt.parseInt(u32, v, 10) catch return error.ArSplitNotANumber else if (whole) n else n - 1;
    if (split < 1 or split > n) return error.ArSplitRange;
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
        if (run.split < n) try calls.append(a, .{ .lo = run.split, .hi = n });
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
    runtime: []const u8 = runtime_native,
    /// The prefill routes the module installed (read back from the module, not the settings).
    prefill_routes: module.Installed,
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
    // The box the module's admission fills (the guard's ceiling; unset: the GPU's working set).
    if (std.c.getenv("DSV41_AR_CEILING_GB")) |v| config.memory_ceiling_bytes = @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
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
    status.startFootprintInterval();
    // The step's vm start (before any load): the page cache the step creates is measured from here.
    const vm_start = status.vmBytes();
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const m = try module.Module.init(gpa, io, &config, &weights, s);
    defer m.deinit();
    const constructed = phaseMemory("module constructed", m.bill.constructionTerms(), 0, vm_start.external);
    printPhaseMemory(a, constructed);
    // The window's own proofs (the harness's, never the served path's): no page cache left by the load.
    try checkPageCache(constructed.file_cache_created_bytes);
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
    // The prompt's start: the box's pages outside this footprint, the reference the phase change's box proof uses.
    const prompt_start_outside = module.outsideOf(module.BoundaryMemory.now());
    var logits = try m.prefill(prompt[calls[0].lo..calls[0].hi], 0);
    memProbe("dsv41 ar served", "the prompt's first call (before the phase change)");
    for (calls[1..]) |c| {
        _ = mlx.mlx_array_free(logits);
        logits = try m.extend(prompt[c.lo..c.hi]);
    }
    try readState(a, &state, m, probe, "after_prompt", calls[calls.len - 1].lo);
    printPhaseMemory(a, phaseMemory("prompt pass", m.bill.prefillTerms(), 0, vm_start.external));
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
    // The phase change ran inside the first decode-width extend: this interval spans it and the decode.
    printPhaseMemory(a, phaseMemory("phase change + decode", m.bill.decodeTerms(), 0, vm_start.external));
    if (m.phase_change) |pc| {
        if (std.json.Stringify.valueAlloc(a, pc, .{})) |j| std.debug.print("NATIVE DSV41_PHASE_CHANGE {s}\n", .{j}) else |_| {}
        try checkBoxReclaimed(pc, prompt_start_outside);
    }
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
        .prefill_routes = m.installed,
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
    std.debug.print("\nNATIVE dsv41 ar served: split {d}+{d}, phase {t}, tier {t}; {d} prompt tokens in {d} calls, {d} generated; ids sha256 {s}; step 0 top-2 {any} margin {d}, step 1 top-2 {any} margin {d}; vs the reference's ids (harness schedule): {s}, first difference {?d}; {d} ms; wrote {s}\n", .{
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
    // Defaults: the whole prompt in one prefill (the served shell's shape), late, served.
    const d = try parseServedRun(64, null, null, null);
    try testing.expectEqual(ServedRun{ .split = 64, .phase = .late, .tier = .served }, d);
    // The stock tier and the early grow keep 63 + 1.
    try testing.expectEqual(@as(u32, 63), (try parseServedRun(64, null, null, "stock")).split);
    try testing.expectEqual(@as(u32, 63), (try parseServedRun(64, null, "early_grow", null)).split);
    // Refusals by name.
    try testing.expectError(error.ArSplitNotANumber, parseServedRun(64, "x", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "65", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "0", null, null));
    try testing.expectError(error.ArPhaseUnknown, parseServedRun(64, null, "early", null));
    try testing.expectError(error.ArTierUnknown, parseServedRun(64, null, null, "exact"));
    try testing.expectError(error.ArEarlyGrowSplit, parseServedRun(64, "56", "early_grow", null));
    try testing.expectError(error.ArStockSplit, parseServedRun(64, "60", null, "stock"));
    // The six runs' calls and every forward's rows (32 generated ids).
    const R = struct { split: ?[]const u8, phase: ?[]const u8, tier: ?[]const u8, want: []const u32 };
    const ones: [31]u32 = @splat(1);
    const runs = [_]R{
        .{ .split = null, .phase = null, .tier = null, .want = &([_]u32{64} ++ ones) }, // R0 served: the whole prompt
        .{ .split = "63", .phase = null, .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R1 served late 63
        .{ .split = "56", .phase = null, .tier = null, .want = &([_]u32{ 56, 8 } ++ ones) }, // R2
        .{ .split = "60", .phase = null, .tier = null, .want = &([_]u32{ 60, 4 } ++ ones) }, // R3
        .{ .split = null, .phase = null, .tier = "stock", .want = &([_]u32{ 63, 1 } ++ ones) }, // R4 (the model chunks 63 by 8)
        .{ .split = "63", .phase = "early_fence", .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R5
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
    // L1-L6: the long prompts' plans (late, the whole prompt in one call).
    for ([_]u32{ 64, 128, 256, 1024 }) |l| {
        const run = try parseServedRun(l, null, null, null);
        const calls = try promptCalls(a, l, run);
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqual(l, calls[0].hi - calls[0].lo);
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

/// The expert stream's counters over a phase of the request (two reads of its existing Stats, taken
/// outside the timed ranges: free for the measured cell).
const StreamPhase = struct {
    route_calls: u64,
    hits: u64,
    misses: u64,
    bytes_read: u64,
    preadv_calls: u64,
    read_busy_s: f64,
    read_seconds: f64,

    fn of(a: expert_stream.Stats, b: expert_stream.Stats) StreamPhase {
        return .{
            .route_calls = b.route_calls -| a.route_calls,
            .hits = b.expert_cache_hits -| a.expert_cache_hits,
            .misses = b.expert_cache_misses -| a.expert_cache_misses,
            .bytes_read = b.expert_bytes_read -| a.expert_bytes_read,
            .preadv_calls = b.preadv_calls -| a.preadv_calls,
            .read_busy_s = @as(f64, @floatFromInt(b.read_wall_ns -| a.read_wall_ns)) / 1e9,
            .read_seconds = b.expert_read_seconds - a.expert_read_seconds,
        };
    }
};

/// A decode-profile run's cycle (DSV41_CELL_DECODE_PROFILE): the host time of each phase
/// (`dsl.Phase`, exclusive, from the loop's stamps) and the stream's counters over the cycle.
const ProfCycle = struct { k_eff: u32, accepted: u32, draft_ms: f64, verify_ms: f64, decide_ms: f64, commit_ms: f64, tail_ms: f64, misses: u64, bytes_read: u64, read_busy_ms: f64 };

/// The decode profile's stamper: `mark(p)` charges the host time since the previous mark to `p`.
const Stamper = struct {
    io: std.Io,
    last: std.Io.Timestamp,
    ns: [@intFromEnum(dsl.Phase.tail) + 1]u64 = @splat(0),

    fn begin(self: *Stamper) void {
        self.ns = @splat(0);
        self.last = std.Io.Timestamp.now(self.io, .boot);
    }

    pub fn mark(self: *Stamper, p: dsl.Phase) void {
        self.ns[@intFromEnum(p)] += @intCast(self.last.untilNow(self.io, .boot).nanoseconds);
        self.last = std.Io.Timestamp.now(self.io, .boot);
    }

    fn ms(self: *const Stamper, p: dsl.Phase) f64 {
        return @as(f64, @floatFromInt(self.ns[@intFromEnum(p)])) / 1e6;
    }
};
const CellReceipt = struct {
    runtime: []const u8 = runtime_native,
    format: []const u8 = served_cell_format,
    tier: []const u8 = "typical (routes.served: C12-C16, A9, C11, C14 woarc; DSpark typical)",
    typical_delta: f64,
    /// The decode lane the Module installed (`Module.decodeLane`: "dspark typical 0.3").
    decode_lane: []const u8 = "",
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
    /// The stream over the prompt pass and over the decode (the phase change's grow between them).
    prompt_stream: ?StreamPhase = null,
    decode_stream: ?StreamPhase = null,
    /// Set by a decode-profile run only (not a timed cell: its stamps sit in the loop).
    decode_profile: ?[]const ProfCycle = null,
    /// The prefill ladder's routes the Module was built with (null = the setting's default, off).
    layer_major: ?bool = null,
    event_gates: ?bool = null,
    wide_feed: ?bool = null,
    wide_seed: ?bool = null,
    wide_hot_first: ?bool = null,
    wide_depth: ?u8 = null,
    wide_cold_rows: ?u8 = null,
    /// The attention call sites the Module installed (read back from it).
    prefill_attn: ?bool = null,
    prefill_index: ?bool = null,
    prefill_hc: ?bool = null,
    prefill_combine: ?bool = null,
    prefill_oproj: ?bool = null,
    prefill_host_shared: ?bool = null,
    prefill_joinless: ?bool = null,
    embedding_rows: ?bool = null,
    /// NATIVE per-phase memory: each boundary's billed terms beside the measured footprint (its interval
    /// high-water mark), its task_vm_info split, MLX active / cache / peak, the box's pages, the residuals.
    bill_baseline_bytes: ?u64 = null,
    phase_memory: ?[]const PhaseMemory = null,
    /// The phase change's readings before / after the frees and after the grow, the freed bytes, the reclaim time.
    phase_change: ?module.PhaseChangeRecord = null,
    /// The verify-row routes the Module installed.
    decode_attn_softmax: ?bool = null,
    decode_index_topk: ?bool = null,
    decode_smallm: ?bool = null,
    decode_mxfp8_rows: ?bool = null,
    /// File-backed pages at the step's vm start (each phase record's file_cache_created_bytes is from here).
    file_backed_start_bytes: ?u64 = null,
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
    // The Module's construction / phase-change evidence lines (log.info: the routes installed, the
    // construction check, the phase change's boundary marks) reach the window log; none is per token.
    testing.log_level = .info;
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
    try cellFill(a, io, &config, prompt.len, max_tokens);
    // The bill at the admitted rows (host): the phase records' billed terms.
    const bill = try cellBill(a, io, &config, prompt.len, max_tokens);

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
    status.startFootprintInterval();
    // The step's vm start (before any load): the page cache the step creates is measured from here.
    const vm_start = status.vmBytes();
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const md = try module.Module.init(gpa, io, &config, &weights, s);
    defer md.deinit();
    const constructed = phaseMemory("module constructed", bill.constructionTerms(), 0, vm_start.external);
    // The window's own proof (the harness's): the load left no page cache for the kernel to age in later.
    try checkPageCache(constructed.file_cache_created_bytes);
    printPhaseMemory(a, constructed);
    memProbe("dsv41 served cell", "module constructed (kernels, arm, residents, warm-up)");

    // Either arm the configuration builds: host waits (the served default) or event gates (C6).
    switch (md.arm) {
        inline else => |t| try cellRun(t.arm, .{ .a = a, .gpa = gpa, .io = io, .md = md, .config = &config, .prompt = prompt, .delta = delta, .max_tokens = max_tokens, .case_id = case_id, .prompt_path = prompt_path, .out_path = out_path, .bill = bill, .constructed = constructed, .file_backed_start = vm_start.external }),
    }
}

const CellCtx = struct {
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    md: *module.Module,
    config: *const model.ModelConfig,
    prompt: []const u32,
    delta: f64,
    max_tokens: u32,
    case_id: ?[]const u8,
    prompt_path: []const u8,
    out_path: []const u8,
    bill: CellBill,
    constructed: PhaseMemory,
    /// File-backed pages at the step's vm start (the page cache the step creates is measured from here).
    file_backed_start: u64,
};

/// The timed cell over the Module's arm (`arm` the host-waits or the event-gated one).
fn cellRun(arm: anytype, cx: CellCtx) !void {
    const a = cx.a;
    const gpa = cx.gpa;
    const io = cx.io;
    const md = cx.md;
    const config = cx.config;
    const prompt = cx.prompt;
    const delta = cx.delta;
    const max_tokens = cx.max_tokens;
    const case_id = cx.case_id;
    const prompt_path = cx.prompt_path;
    const out_path = cx.out_path;
    const g = &md.g;
    // The served decode lane (the Module's DSpark strategy): the shell's calls, in the shell's order.
    if (md.draftBlockSize() == 0) return error.CellNeedsDspark;
    var stops: [8]u32 = undefined;
    const n_stop = config.num_eos_tokens;
    @memcpy(stops[0..n_stop], config.eos_token_ids[0..n_stop]);
    const isStop = struct {
        fn f(ss: []const u32, t: u32) bool {
            return std.mem.indexOfScalar(u32, ss, t) != null;
        }
    }.f;
    // The Module's strategy carries the tier's delta; a cell asking for another one is refused.
    if (@as(f32, @floatCast(delta)) != module.dspark_typical_delta) return error.CellDeltaNotTheModules;

    const profile = std.c.getenv("DSV41_CELL_DECODE_PROFILE") != null;
    const s_start = arm.hook.source.stats();
    _ = mlx.mlx_reset_peak_memory();
    // The prompt's start: the phase change's reclaim reference (the loop drives the prompt itself).
    const prompt_start_outside = module.outsideOf(module.BoundaryMemory.now());
    const t0 = std.Io.Timestamp.now(io, .boot);
    // The whole prompt in one Module.prefill (the request's bounded lanes: the prompt, the token cap,
    // one verify block; the strategy seeded from every prompt row); its argmax is the primary (the
    // Generator's greedy pick). A shell that sends the prompt this way (dsv41 prefills unchunked)
    // decodes the cell's ids.
    const pl = try md.prefill(prompt, prompt.len + max_tokens);
    const primary = try g.hostArgmax(pl);
    _ = mlx.mlx_array_free(pl);
    const ttft_s = secondsSince(io, t0);
    const s_prompt = arm.hook.source.stats();
    // The phase records (outside the timed spans' hot paths: at their boundaries).
    var phases: [4]PhaseMemory = undefined;
    phases[0] = cx.constructed;
    phases[1] = phaseMemory("prompt pass", cx.bill.prefillTerms(), 0, cx.file_backed_start);
    printPhaseMemory(a, phases[1]);
    // The MLX peak over the request: each probe reads and resets it, so keep the max of its phases.
    var mlx_peak: usize = @max(phases[1].mlx_peak_bytes, memProbePeak("dsv41 served cell", "prompt (one pass)"));
    const t1 = std.Io.Timestamp.now(io, .boot);
    try md.phaseChange();
    // The window's box proof (the harness's): every release since the prompt began reclaimed in vm_stat too.
    if (md.phase_change) |pc| try checkBoxReclaimed(pc, prompt_start_outside);
    const phase_s = secondsSince(io, t1);
    phases[2] = phaseMemory("phase change", cx.bill.decodeTerms(), 0, cx.file_backed_start);
    if (md.phase_change) |pc| phases[2].settle_ms = pc.settle_ms;
    printPhaseMemory(a, phases[2]);
    mlx_peak = @max(mlx_peak, @max(phases[2].mlx_peak_bytes, memProbePeak("dsv41 served cell", "the phase change (embedding fence, slot banks grown)")));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var cycles: std.ArrayList(CellCycle) = .empty;
    const t2 = std.Io.Timestamp.now(io, .boot);
    var finish: dsl.Finish = .stop;
    var prof: std.ArrayList(ProfCycle) = .empty;
    // `out` holds the tokens after the primary (the cycles'); `next` is the token not yet emitted.
    var next = primary;
    const budget = max_tokens - 1;
    var ended = isStop(stops[0..n_stop], primary);
    var n_rounds: usize = 0;
    while (!ended) {
        if (out.items.len >= budget) {
            finish = .length;
            break;
        }
        var lg: dsl.CycleLog = .{ .primary = 0 };
        // The first token of a round is the previous round's next token (the primary emitted apart).
        const first = n_rounds == 0;
        n_rounds += 1;
        const cap: u32 = @intCast(budget - out.items.len - @intFromBool(!first));
        var r = if (!profile) try md.dsparkRoundLogged(gpa, next, cap, &lg, {}) else blk: {
            var sp: Stamper = .{ .io = io, .last = undefined };
            const c0 = arm.hook.source.stats();
            sp.begin();
            const rr = try md.dsparkRoundLogged(gpa, next, cap, &lg, &sp);
            const c1 = arm.hook.source.stats();
            const sph = StreamPhase.of(c0, c1);
            try prof.append(a, .{ .k_eff = lg.k_eff, .accepted = lg.accepted, .draft_ms = sp.ms(.draft), .verify_ms = sp.ms(.verify), .decide_ms = sp.ms(.decide), .commit_ms = sp.ms(.commit), .tail_ms = sp.ms(.tail), .misses = sph.misses, .bytes_read = sph.bytes_read, .read_busy_ms = sph.read_busy_s * 1e3 });
            break :blk rr;
        };
        defer r.deinit(gpa);
        try cycles.append(a, .{ .k_eff = lg.k_eff, .accepted = lg.accepted, .verified = lg.verified });
        // The round's tokens: [t1, kept drafts]; t1 of the first round is the primary (already counted).
        for (r.tokens[@intFromBool(first)..]) |tok| {
            if (out.items.len >= budget) break;
            try out.append(gpa, tok);
            if (isStop(stops[0..n_stop], tok)) {
                ended = true;
                finish = .stop;
                break;
            }
        }
        next = r.next_token;
        if (!ended and out.items.len < budget and out.items.len + 1 == budget) {
            // One token left: the next token is known without another round.
            try out.append(gpa, next);
            ended = true;
            finish = if (isStop(stops[0..n_stop], next)) .stop else .length;
        }
    }
    const decode_s = secondsSince(io, t2);
    const s_end = arm.hook.source.stats();
    const wall_s = secondsSince(io, t0);
    phases[3] = phaseMemory("decode", cx.bill.decodeTerms(), 0, cx.file_backed_start);
    printPhaseMemory(a, phases[3]);
    mlx_peak = @max(mlx_peak, @max(phases[3].mlx_peak_bytes, memProbePeak("dsv41 served cell", "cycles")));

    const ids = try a.alloc(u32, out.items.len + 1);
    ids[0] = primary;
    @memcpy(ids[1..], out.items);
    const fp = status.footprint();
    const stt = md.dsparkStats() orelse return error.CellNeedsDspark;
    const prompt_sha = try cell.idsSha256(a, prompt);
    const ids_sha = try cell.idsSha256(a, ids);
    const rec: CellReceipt = .{
        .typical_delta = module.dspark_typical_delta,
        .decode_lane = md.decodeLane(),
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
        .prompt_stream = StreamPhase.of(s_start, s_prompt),
        // The decode window starts at the prompt's end: it includes the phase change's grow.
        .decode_stream = StreamPhase.of(s_prompt, s_end),
        .decode_profile = if (profile) prof.items else null,
        // The routes the module installed (read back from it, not from the settings).
        .layer_major = md.installed.layer_major,
        .event_gates = config.expert_event_gates,
        .wide_feed = md.installed.wide.seed and md.installed.wide.hot_first,
        .wide_seed = md.installed.wide.seed,
        .wide_hot_first = md.installed.wide.hot_first,
        .wide_depth = md.installed.wide.depth,
        .wide_cold_rows = md.installed.wide.cold_rows,
        .prefill_attn = md.installed.prefill_attn,
        .prefill_index = md.installed.prefill_index,
        .prefill_hc = md.installed.prefill_hc,
        .prefill_combine = md.installed.prefill_combine,
        .prefill_oproj = md.installed.prefill_oproj,
        .prefill_host_shared = md.installed.prefill_host_shared,
        .prefill_joinless = md.installed.prefill_joinless,
        .embedding_rows = md.installed.embedding_rows,
        .bill_baseline_bytes = cx.bill.baseline,
        .phase_memory = &phases,
        .phase_change = md.phase_change,
        .decode_attn_softmax = md.installed.decode_attn_softmax,
        .decode_index_topk = md.installed.decode_index_topk,
        .decode_smallm = md.installed.decode_smallm,
        .decode_mxfp8_rows = md.installed.decode_mxfp8_rows,
        .file_backed_start_bytes = cx.file_backed_start,
    };
    if (profile) printDecodeProfile(prof.items);
    const json = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\nNATIVE dsv41 served cell: typical {d}, {d} prompt tokens, rows {d} prefill / {d} decode per layer; TTFT {d:.2} s = prefill {d:.1} tok/s; phase change {d:.2} s; decode {d} tokens in {d} cycles, {d:.2} s = {d:.2} tok/s ({d:.2} with the phase change); accepted {d}/{d} drafts; wall {d:.2} s; peak footprint {d:.2} GB, MLX peak {d:.2} GB; finish {s}; ids sha256 {s}; wrote {s}\n", .{
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
    // The box baseline the guard's stop compares against: its non-file start (the guard exports it as
    // _GPU_WINDOW_USED_START_NONFILE_BYTES in nonfile mode; its accounting is non-file start + the
    // step's footprint), else the runner's DSV41_CELL_BASELINE_GB. In-run file-cache growth has no
    // bill term: every resident and record read bypasses the page cache.
    config.memory_baseline_bytes = if (std.c.getenv("_GPU_WINDOW_USED_START_NONFILE_BYTES")) |v|
        std.fmt.parseInt(u64, std.mem.span(v), 10) catch return error.CellBaselineValue
    else
        (try gb("DSV41_CELL_BASELINE_GB")) orelse return error.CellBaselineMissing;
    config.memory_ceiling_bytes = (try gb("DSV41_CELL_CEILING_GB")) orelse return error.CellCeilingMissing;
    if (std.c.getenv("DSV41_CELL_ROWS")) |v| config.expert_rows = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    // The prefill ladder's routes (all off by default; the Module refuses what it cannot build):
    // K16 layer-major, the wide read schedule's feed and depth, the cold rows on the decode GEMV.
    if (envStr("DSV41_CELL_LAYER_MAJOR")) |v| config.layer_major_prefill = try cellBool("DSV41_CELL_LAYER_MAJOR", v);
    // C6: the typical tier's event-gated waves (the Module builds the gated arm; default host waits).
    if (envStr("DSV41_CELL_EVENT_GATES")) |v| config.expert_event_gates = try cellBool("DSV41_CELL_EVENT_GATES", v);
    if (envStr("DSV41_CELL_WIDE_FEED")) |v| config.expert_wide_feed = try cellBool("DSV41_CELL_WIDE_FEED", v);
    // The feed's halves on their own (each overrides the feed's value for its half).
    if (envStr("DSV41_CELL_WIDE_SEED")) |v| config.expert_wide_seed = try cellBool("DSV41_CELL_WIDE_SEED", v);
    if (envStr("DSV41_CELL_WIDE_HOT_FIRST")) |v| config.expert_wide_hot_first = try cellBool("DSV41_CELL_WIDE_HOT_FIRST", v);
    // The attention call sites (the served tier's routes by default; 0 = the stock chain).
    if (envStr("DSV41_CELL_PREFILL_ATTN")) |v| config.prefill_attn = try cellBool("DSV41_CELL_PREFILL_ATTN", v);
    if (envStr("DSV41_CELL_PREFILL_INDEX")) |v| config.prefill_index = try cellBool("DSV41_CELL_PREFILL_INDEX", v);
    if (envStr("DSV41_CELL_PREFILL_HC")) |v| config.prefill_hc = try cellBool("DSV41_CELL_PREFILL_HC", v);
    if (envStr("DSV41_CELL_PREFILL_COMBINE")) |v| config.prefill_combine = try cellBool("DSV41_CELL_PREFILL_COMBINE", v);
    if (envStr("DSV41_CELL_PREFILL_OPROJ")) |v| config.prefill_oproj = try cellBool("DSV41_CELL_PREFILL_OPROJ", v);
    if (envStr("DSV41_CELL_PREFILL_HOST_SHARED")) |v| config.prefill_host_shared = try cellBool("DSV41_CELL_PREFILL_HOST_SHARED", v);
    if (envStr("DSV41_CELL_PREFILL_JOINLESS")) |v| config.prefill_joinless = try cellBool("DSV41_CELL_PREFILL_JOINLESS", v);
    if (envStr("DSV41_CELL_EMBEDDING_ROWS")) |v| config.embedding_host_rows = try cellBool("DSV41_CELL_EMBEDDING_ROWS", v);
    if (envStr("DSV41_CELL_DECODE_ATTN_SOFTMAX")) |v| config.decode_attn_softmax = try cellBool("DSV41_CELL_DECODE_ATTN_SOFTMAX", v);
    if (envStr("DSV41_CELL_DECODE_INDEX_TOPK")) |v| config.decode_index_topk = try cellBool("DSV41_CELL_DECODE_INDEX_TOPK", v);
    if (envStr("DSV41_CELL_DECODE_SMALLM")) |v| config.decode_smallm = try cellBool("DSV41_CELL_DECODE_SMALLM", v);
    if (envStr("DSV41_CELL_DECODE_MXFP8_ROWS")) |v| config.decode_mxfp8_rows = try cellBool("DSV41_CELL_DECODE_MXFP8_ROWS", v);
    if (envStr("DSV41_CELL_WIDE_DEPTH")) |v| {
        const d = std.fmt.parseInt(u8, v, 10) catch return error.CellWideDepth;
        if (d < 1 or d > 2) return error.CellWideDepth;
        config.expert_wide_depth = d;
    }
    if (envStr("DSV41_CELL_WIDE_COLD_ROWS")) |v| {
        const r = std.fmt.parseInt(u8, v, 10) catch return error.CellWideColdRows;
        if (r > 8) return error.CellWideColdRows;
        config.expert_wide_cold_rows = r;
    }
    // A combination the bills do not cover is refused here, by name, before any window work
    // (the Module's own construction check: K16 only on the served tier, with its request bill).
    _ = try module.layerMajor(config);
}

/// The native admission's fill: the cell's own bill at the envelope's rows gives each phase's rows-free
/// total and `module.fillRows` takes ONE row count up to the binding phase's target (no grow at the phase
/// change); the config then carries it as both row counts (the stream's, the bill's). DSV41_CELL_ROWS + DSV41_CELL_PREFILL_ROWS force both (a ladder's
/// later lines at its first line's rows): billed, and refused by name above the target. DSV41_CELL_ROWS
/// alone keeps the envelope's forced-rows admission. DSV41_CELL_FILL_LADDER=1 fills at the prefill
/// ladder's widest admission (two wide windows and the larger of the chunk-major and layer-major prompt
/// waves; feed and cold rows bill nothing), so every ladder line admits the same rows at one baseline.
fn cellFill(a: std.mem.Allocator, io: std.Io, config: *model.ModelConfig, prompt_tokens: u64, max_tokens: u64) !void {
    const target = config.memory_ceiling_bytes.? -| module.ceiling_stop_bytes;
    if (std.c.getenv("DSV41_CELL_PREFILL_ROWS")) |v| {
        const decode = config.expert_rows orelse return error.CellPrefillRowsWithoutRows;
        config.expert_prefill_rows = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.CellPrefillRowsValue;
        const b = try cellBill(a, io, config, prompt_tokens, max_tokens);
        std.debug.print("DSV41_CELL_FILL {{\"baseline_gb\": {d:.3}, \"target_gb\": {d:.3}, \"forced_rows\": [{d}, {d}], \"prefill_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}}}\n", .{
            gbOf(b.baseline), gbOf(target), config.expert_prefill_rows.?, decode, gbOf(b.prefillTotal()), gbOf(b.decodeTotal()),
        });
        if (b.prefillTotal() > target or b.decodeTotal() > target) return error.CellForcedRowsOverTarget;
        return;
    }
    if (config.expert_rows != null) return;
    var nr = try fillAt(a, io, config.*, prompt_tokens, max_tokens);
    if (std.c.getenv("DSV41_CELL_FILL_LADDER") != null) {
        for ([_]bool{ false, true }) |lm| {
            var wide = config.*;
            wide.expert_wide_depth = 2;
            wide.layer_major_prefill = lm;
            const r = try fillAt(a, io, wide, prompt_tokens, max_tokens);
            nr = .{ .prefill = @min(nr.prefill, r.prefill), .decode = @min(nr.decode, r.decode) };
        }
    }
    std.debug.print("DSV41_CELL_FILL {{\"baseline_gb\": {d:.3}, \"target_gb\": {d:.3}, \"ladder\": {}, \"filled_rows\": [{d}, {d}]}}\n", .{
        gbOf(config.memory_baseline_bytes.?), gbOf(target), std.c.getenv("DSV41_CELL_FILL_LADDER") != null, nr.prefill, nr.decode,
    });
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
}

fn gbOf(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

/// The fill for `config`'s routes (its bill at the envelope's rows).
pub fn fillAt(a: std.mem.Allocator, io: std.Io, config: model.ModelConfig, prompt_tokens: u64, max_tokens: u64) !arm_mod.NativeRows {
    return fillAtWired(a, io, config, prompt_tokens, max_tokens, null);
}

/// `fillAt` at pinned wired bytes (null: DSV41_CELL_WIRED_GB, else read now): the fill and the bill
/// that later checks it plan through the same `planRows` inputs only when they see the same wired bytes.
pub fn fillAtWired(a: std.mem.Allocator, io: std.Io, config: model.ModelConfig, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64) !arm_mod.NativeRows {
    var c = config;
    c.expert_rows = null;
    c.expert_prefill_rows = null;
    const b0 = try cellBillWired(a, io, &c, prompt_tokens, max_tokens, wired_bytes);
    const rec = b0.slot_decode / (@as(u64, b0.layers) * b0.decode_rows + b0.transient_rows);
    const per_row = @as(u64, b0.layers) * rec;
    return module.fillRows(.{
        .prefill_fixed = b0.prefillTotal() - b0.prefill_rows * per_row,
        .decode_fixed = b0.decodeTotal() - b0.decode_rows * per_row,
        .per_row = per_row,
    }, config.memory_ceiling_bytes.?, b0.n_experts) catch |e| {
        std.debug.print("DSV41_CELL_REFUSED {s}: the native bill does not fit the ceiling's target at the floor rows\n", .{@errorName(e)});
        return e;
    };
}

fn cellBool(comptime name: []const u8, v: []const u8) !bool {
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    _ = name;
    return error.CellBoolValue;
}

/// The window's proofs, the harness's to assert (the served path judges only its own ledgers): the page
/// cache the step created by the end of construction, and the box's pages at the phase change. A guarded
/// window's guard counts the whole box (other processes included), so these hold the harness's run to it.
pub const page_cache_tolerance_bytes: u64 = 500_000_000;
pub const box_tolerance_bytes: u64 = 500_000_000;

/// The load left no page cache to be aged into the guard's count later (v6c2: 15 GB of speculative pages from
/// unaligned F_NOCACHE reads, 7.7 GB aged in at the grow).
pub fn checkPageCache(created: i64) error{ConstructionLeftPageCache}!void {
    if (created > @as(i64, @intCast(page_cache_tolerance_bytes))) return error.ConstructionLeftPageCache;
}

/// At the phase change: the box's physical pages (vm_stat, the guard's metric) down by the freed bytes, and
/// nothing outside this footprint beyond the prompt's start (SERVED7: the pages stayed counted after they
/// had left the footprint).
pub fn checkBoxReclaimed(r: module.PhaseChangeRecord, prompt_start_outside: u64) error{PhaseChangeNotReclaimed}!void {
    if (r.after.physical + r.freed_bytes > r.before.physical + box_tolerance_bytes) return error.PhaseChangeNotReclaimed;
    if (module.outsideOf(r.after) > prompt_start_outside + box_tolerance_bytes) return error.PhaseChangeNotReclaimed;
}

/// ASSUMPTION the bill rests on: the step creates no page cache. The guard credits only the file cache present
/// at its start and does not count speculative pages until the kernel ages them into inactive, so page cache
/// the step creates is unbilled memory that can land at any later allocation (v6c2: 15 GB of it from
/// construction, 7.7 GB aged in at the grow). The harnesses assert it (`checkPageCache`), and each
/// phase record carries `file_cache_created_bytes` and `box_speculative_bytes`.
///
/// Every prompt pass is billed at the prompt rows: the first one before the phase change grows the banks,
/// every later one after the served path returned them to the prompt rows (the arm's shrink, proven by
/// `Module.reclaimShrink` before the prompt allocates), so max(prompt total, decode total) bounds every request.
///
/// The cell's memory bill (decimal bytes), each term by construction from the bank's headers, the
/// admission the module builds with (`Module.armOptions` at the same config) and the arch's prefill
/// bill (`v41.PrefillBill`, its wave pinned by the served 16K trace test): the prompt phase and the
/// decode phase over the box baseline. `processBound` is what the child may hold above the baseline.
pub const CellBill = struct {
    baseline: u64,
    /// The slot banks' geometry: routed layers, the transient rows (max_route_ids x wide depth), the
    /// layer's experts (the rows' cap).
    layers: u32 = 0,
    transient_rows: u64 = 0,
    n_experts: u32 = 0,
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
    /// The prompt pass's transient: K16's layer-major wave + the wide lane's routed-output copy
    /// (`PrefillBill.layerMajorBilledBytes`), or the chunk-major widest wave x 5 / 4.
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
    /// The wide read schedule's depth window (the admission's `wide_window_bytes`: process lifetime).
    wide_window: u64 = 0,
    /// The process overhead no term above names (`unbilled_process_overhead_bytes`), in the prompt phase.
    unbilled_overhead: u64 = unbilled_process_overhead_bytes,
    /// The input embedding reads its host rows from construction (`embedding_host_rows`, default on): the
    /// device table is freed after the install warm-up, so no phase holds it.
    embedding_host_rows: bool = false,
    /// What the prompt pass leaves alive through decode beyond the KV: the DSpark seed keeps a view of
    /// the whole prompt's main taps (`main_h` slices the concat of every row's `main_hidden`, f32
    /// [seq, n_main x hidden]) and each draft stage's window a view of its whole-prompt main KV
    /// ([seq, head_dim] f32); a view keeps its parent's buffer (v6b: +1.30 GB persistent after the prompt,
    /// 1.11 GB of it these). Decode phase only (inside the prompt wave's kept state during the pass).
    prompt_state: u64 = 0,

    pub fn prefillTotal(b: CellBill) u64 {
        return b.baseline + b.prefillTerms().sum();
    }

    pub fn decodeTotal(b: CellBill) u64 {
        return b.baseline + b.decodeTerms().sum();
    }

    /// The prompt phase's process terms (the prompt pass's peak: every term live at once).
    pub fn prefillTerms(b: CellBill) PhaseTerms {
        return .{ .slot_banks = b.slot_prefill, .lookahead_staging = b.lookahead_staging, .residents = if (b.embedding_host_rows) b.residents - b.embedding else b.residents, .engram = b.engram, .waves = b.prefill_wave, .kv = b.kv, .mlx_cache = b.prefill_cache, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .unbilled_overhead = b.unbilled_overhead };
    }

    /// The decode phase's process terms (the embedding off at the fence; the verify and draft waves).
    pub fn decodeTerms(b: CellBill) PhaseTerms {
        return .{ .slot_banks = b.slot_decode, .lookahead_staging = b.lookahead_staging, .residents = b.residents - b.embedding, .engram = b.engram, .waves = b.decode_wave + b.draft_wave, .kv = b.kv, .mlx_cache = b.decode_cache, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .prompt_state = b.prompt_state };
    }

    /// What the constructed module holds before any request (after the install warm-up released its
    /// buffers and the allocator cache): the prompt phase's persistent terms, no wave, no KV, no cache.
    pub fn constructionTerms(b: CellBill) PhaseTerms {
        var t = b.prefillTerms();
        t.waves = 0;
        t.kv = 0;
        t.mlx_cache = 0;
        return t;
    }

    pub fn processBound(b: CellBill) u64 {
        return @max(b.prefillTotal(), b.decodeTotal()) - b.baseline;
    }
};

/// One phase's billed process terms (decimal bytes; the box baseline apart), as the receipt records them.
pub const PhaseTerms = struct {
    slot_banks: u64 = 0,
    lookahead_staging: u64 = 0,
    residents: u64 = 0,
    engram: u64 = 0,
    /// The prompt wave (prompt phase) or the verify + draft waves (decode phase).
    waves: u64 = 0,
    kv: u64 = 0,
    mlx_cache: u64 = 0,
    host_reserve: u64 = 0,
    wide_window: u64 = 0,
    unbilled_overhead: u64 = 0,
    /// The retained prompt state (decode phase).
    prompt_state: u64 = 0,

    pub fn sum(t: PhaseTerms) u64 {
        var n: u64 = 0;
        inline for (@typeInfo(PhaseTerms).@"struct".field_names) |name| n += @field(t, name);
        return n;
    }
};

/// One phase boundary's memory record (NATIVE; probes at the four boundaries only: module constructed,
/// end of the prompt pass, after the phase change, end of decode): the phase's billed terms, the
/// process ledgers (the guard's footprint and its split), MLX's allocator (active, cache, the peak since
/// the previous boundary), the box's pages as the guard reads them, and billed minus measured.
pub const PhaseMemory = struct {
    phase: []const u8,
    billed: PhaseTerms,
    billed_process_bytes: u64,
    process: status.ProcessMemory,
    mlx_active_bytes: u64,
    mlx_cache_bytes: u64,
    mlx_peak_bytes: u64,
    box_physical_used_bytes: u64,
    box_file_backed_bytes: u64,
    /// Speculative (read-ahead) pages: not in the guard's used count until the kernel ages them into
    /// inactive, which it can do at any later allocation (the v6c2 / SERVED kills: 7.7 GB at the grow).
    box_speculative_bytes: u64 = 0,
    /// The page cache the step created: file-backed pages now less at the step's vm start (the bill assumes 0).
    file_cache_created_bytes: i64 = 0,
    /// The phase's billed process bytes less its measured footprint high-water mark (negative: over the bill).
    residual_bytes: i64,
    /// Billed MLX-device terms (slots, residents, Engram residents, waves, KV) less MLX's peak over the phase.
    mlx_residual_bytes: i64,
    /// The phase change's boundary only: how long the driver took to reclaim the frees before the grow.
    settle_ms: ?u32 = null,
};

/// The boundary's record from what the kernel and MLX already track (no new counter): reads the
/// ledgers and MLX's allocator, then restarts both high-water marks for the next phase.
/// `engram_host_bytes`: the part of the billed Engram term that is host memory (0 since the host side is billed
/// as measured: the Engram term holds its device residents only).
pub fn phaseMemory(phase: []const u8, billed: PhaseTerms, engram_host_bytes: u64, file_backed_start: u64) PhaseMemory {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    const pm = status.processMemory();
    const v = status.vmBytes();
    _ = mlx.mlx_reset_peak_memory();
    status.startFootprintInterval();
    return recordOf(phase, billed, pm, active, cache, peak, status.physicalUsedBytes(v), v.external, v.speculative, file_backed_start, engram_host_bytes);
}

/// `phaseMemory`'s arithmetic (host-testable): the MLX-device share of the bill is every term but the
/// host ones (lookahead staging, host reserve, the wide window's host records, the overhead, the
/// Engram row caches) and the allocator cache.
pub fn recordOf(phase: []const u8, billed: PhaseTerms, pm: status.ProcessMemory, active: u64, cache: u64, peak: u64, physical: u64, file_backed: u64, speculative: u64, file_backed_start: u64, engram_host_bytes: u64) PhaseMemory {
    const process = billed.sum();
    const measured = @max(pm.footprint_interval_peak, pm.footprint);
    const device = billed.slot_banks + billed.residents + (billed.engram -| engram_host_bytes) + billed.waves + billed.kv;
    return .{
        .phase = phase,
        .billed = billed,
        .billed_process_bytes = process,
        .process = pm,
        .mlx_active_bytes = active,
        .mlx_cache_bytes = cache,
        .mlx_peak_bytes = @max(peak, active),
        .box_physical_used_bytes = physical,
        .box_file_backed_bytes = file_backed,
        .box_speculative_bytes = speculative,
        .file_cache_created_bytes = @as(i64, @intCast(file_backed)) - @as(i64, @intCast(file_backed_start)),
        .residual_bytes = @as(i64, @intCast(process)) - @as(i64, @intCast(measured)),
        .mlx_residual_bytes = @as(i64, @intCast(device)) - @as(i64, @intCast(@max(peak, active))),
    };
}

pub fn printPhaseMemory(a: std.mem.Allocator, r: PhaseMemory) void {
    const json = std.json.Stringify.valueAlloc(a, r, .{}) catch return;
    std.debug.print("NATIVE DSV41_PHASE_MEMORY {s}\n", .{json});
}

/// The measured process overhead the named terms do not cover, PROMPT PHASE ONLY: calibrated from the
/// served cells' peak phys_footprint over their own bill's bound (fastest 20260929-152450: 78.294 vs 77.657
/// GB = 0.637; standard 20260929-153540: 77.139 vs 76.591 = 0.548), the larger, rounded up. Those peaks were
/// decode peaks, and what they measured there is now attributed: the retained prompt state (`prompt_state`,
/// 1.11 GB at 16K; pass3ak's decode: MLX 1.46 GB above the constructed module after the prompt, the host side
/// 0.38-0.59 GB against 1.26 billed with this term), so the decode phase no longer carries it (pass3ak's decode
/// residual was +1.30 GB with both). The prompt phase keeps it: its host side measured 1.64-1.87 GB against
/// 1.26 billed without it.
pub const unbilled_process_overhead_bytes: u64 = 640_000_000;

/// The process's host side (its footprint less MLX's active and cache: the read pool and its staging, the
/// lookahead staging, the Engram row caches, the tables, the process itself), billed as measured with a
/// 0.3 GB margin in place of the named host terms (lookahead staging, row caches, host reserve, the wide
/// window's second transient window, the prompt phase's unattributed overhead: 1.90 GB together). pass3am (v7,
/// served-cell-typical-fastest-20260930-065643): 0.31 GB after construction; host and cache together 0.59 GB
/// at the prompt pass's footprint peak (MLX 104.90, footprint 105.49 GB); 0.50-0.59 GB in decode; 1.93 GB at the
/// prompt's end, after its waves were freed (footprint 96.1 GB, far under the peak). MLX active equals the
/// device terms without the wide window at every boundary, so that window is no device memory either.
pub const measured_host_side_bytes: u64 = 900_000_000;

pub fn cellBill(a: std.mem.Allocator, io: std.Io, config: *const model.ModelConfig, prompt_tokens: u64, max_tokens: u64) !CellBill {
    return cellBillWired(a, io, config, prompt_tokens, max_tokens, null);
}

/// `cellBill` at pinned wired bytes (the envelope admission inside `planRows` reads them). A bill taken
/// after construction must pass the wired bytes the arm was planned with (`arm.inputs.wired_bytes`), never
/// a live read: the constructed module's own banks and residents are wired by then (v6 211422 refused
/// PrefillDoesNotFit that way at the fill's own rows).
pub fn cellBillWired(a: std.mem.Allocator, io: std.Io, config: *const model.ModelConfig, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64) !CellBill {
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
    if (wired_bytes) |w| opts.wired_bytes = w;
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
    const bill = v41.PrefillBill.of(&c).withIndexLaunch(try module.prefillIndexRoute(config));
    const positions = prompt_tokens + max_tokens + mdl.Model(ops.MlxOps).scratch_rows;
    const rows: u64 = mdl.Model(ops.MlxOps).scratch_rows;
    // A verify forward: the fixed wave at 8 rows plus its index chain over every position (two arrays live).
    const decode_wave = bill.waveBytes(rows, rows, .served) + v41.PrefillBill.chain_copies * rows * bill.index_heads * positions * 4;
    return .{
        // Unset (a shell without a box baseline): the process terms alone.
        .baseline = config.memory_baseline_bytes orelse 0,
        .layers = c.n_layers,
        .transient_rows = transient,
        .n_experts = c.n_routed_experts,
        .prefill_rows = p.prefill_rows,
        .decode_rows = p.decode_rows,
        .slot_prefill = (@as(u64, c.n_layers) * p.prefill_rows + transient) * rec,
        .slot_decode = (@as(u64, c.n_layers) * p.decode_rows + transient) * rec,
        // The host side is billed as measured (`measured_host_side_bytes`, in host_reserve).
        .lookahead_staging = 0,
        .residents = m.totalBytes(),
        .embedding = m.bytes_by_module[@backingInt(v41.Module.embed)],
        .engram = em.totalBytes(),
        // K16 (the layer-major route) bills its own wave (every chunk's kept state + one sub-wave). With
        // JOINLESS (the served default) the combine reads the DIG-X waves' own outputs: no joined copy, and
        // the wave alone covers the pass (v6b: 13.54 GB measured incl. KV against 14.40 + 0.16 billed);
        // without it, the wide lane's routed-output copy. The chunk-major wave keeps its x 5/4 margin.
        .prefill_wave = if (config.dsv41LayerMajor())
            (if (config.prefill_joinless orelse module.numericTier(.served).routes.prefill_joinless) bill.layerMajorWaveBytes(prompt_tokens, .served) else bill.layerMajorBilledBytes(prompt_tokens, .served))
        else
            bill.waveBytes(bill.chunkRows(prompt_tokens), prompt_tokens, .served) / 4 * 5,
        .kv = bill.window_ring_bytes + positions * bill.kv_source_pos_bytes,
        .prefill_cache = module.prefillCacheLimit(.served),
        .decode_cache = expert_admission.Envelope.dsv41_pass2.decode_cache_bytes,
        .decode_wave = decode_wave,
        .draft_wave = decode_wave,
        .host_reserve = measured_host_side_bytes,
        .wide_window = 0,
        .unbilled_overhead = 0,
        .embedding_host_rows = config.embedding_host_rows orelse true,
        .prompt_state = prompt_tokens * (bill.n_main * bill.hidden * 4 + @as(u64, c.dspark.n_stages) * c.head_dim * 4),
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
        .{ .name = "residents (the embedding: host rows, else off at the fence)", .p = b.prefillTerms().residents, .d = b.residents - b.embedding },
        .{ .name = "Engram residents (row caches: host side)", .p = b.engram, .d = b.engram },
        .{ .name = "prompt wave (K16 + wide lane; chunk-major x 5/4) / verify + draft", .p = b.prefill_wave, .d = b.decode_wave + b.draft_wave },
        .{ .name = "KV (ring + source lanes, bounded)", .p = b.kv, .d = b.kv },
        .{ .name = "MLX allocator cache (the phase's limit)", .p = b.prefill_cache, .d = b.decode_cache },
        .{ .name = "host side, measured (pools, staging, caches, process)", .p = b.host_reserve, .d = b.host_reserve },
        .{ .name = "wide read window (depth 2)", .p = b.wide_window, .d = b.wide_window },
        .{ .name = "retained prompt state (seed views; decode)", .p = 0, .d = b.prompt_state },
        .{ .name = "page cache created by the step (assumed 0; enforced)", .p = 0, .d = 0 },
        .{ .name = "unbilled process overhead (prompt phase; decode's is prompt_state)", .p = b.unbilled_overhead, .d = 0 },
    }) |t| std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}\n", .{ t.name, gb(t.p), gb(t.d) });
    std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}   rows {d} / {d}; process bound {d:.2}\n", .{ "TOTAL", gb(b.prefillTotal()), gb(b.decodeTotal()), b.prefill_rows, b.decode_rows, gb(b.processBound()) });
    std.debug.print("DSV41_CELL_BILL {{\"baseline_gb\": {d:.3}, \"prefill_rows\": {d}, \"decode_rows\": {d}, \"prefill_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}, \"process_bound_gb\": {d:.3}}}\n", .{ gb(b.baseline), b.prefill_rows, b.decode_rows, gb(b.prefillTotal()), gb(b.decodeTotal()), gb(b.processBound()) });
}

/// cell4's bill (served-cell-typical-fastest-20260929-195452: 106 / 148 rows, the 8.716 GB non-file
/// baseline), term by term in bytes as `cellBill` built it on the bank.
/// `cell4Bill` for the module's tests.
pub fn cell4BillForTests() CellBill {
    return cell4Bill();
}

fn cell4Bill() CellBill {
    const rec: u64 = 13_315_584;
    return .{
        .baseline = 8_716_419_072,
        .layers = 40,
        .transient_rows = 48,
        .n_experts = 384,
        .prefill_rows = 106,
        .decode_rows = 148,
        .slot_prefill = (40 * 106 + 48) * rec,
        .slot_decode = (40 * 148 + 48) * rec,
        .lookahead_staging = 54_460_416,
        .residents = 17_680_000_000,
        .embedding = 1_323_827_200,
        .engram = 480_000_000,
        .prefill_wave = 17_995_900_000,
        .kv = 160_000_000,
        .prefill_cache = 4_294_967_296,
        .decode_cache = 270_000_000,
        .decode_wave = 350_000_000,
        .draft_wave = 350_000_000,
        .host_reserve = 408_944_640,
        .wide_window = 48 * rec,
    };
}

test "dsv41 memory: a phase's total is the baseline plus its terms; the construction terms drop the wave, the KV and the cache" {
    const b = cell4Bill();
    try testing.expectEqual(b.baseline + b.slot_prefill + b.lookahead_staging + b.residents + b.engram + b.prefill_wave + b.kv + b.prefill_cache + b.host_reserve + b.unbilled_overhead + b.wide_window, b.prefillTotal());
    // The decode phase carries no unbilled overhead (what it covered there is the retained prompt state).
    try testing.expectEqual(b.baseline + b.slot_decode + b.lookahead_staging + b.residents - b.embedding + b.engram + b.kv + b.decode_wave + b.draft_wave + b.decode_cache + b.host_reserve + b.wide_window + b.prompt_state, b.decodeTotal());
    try testing.expectEqual(@as(u64, 0), b.decodeTerms().unbilled_overhead);
    const c = b.constructionTerms();
    try testing.expectEqual(b.prefillTerms().sum() - b.prefill_wave - b.kv - b.prefill_cache, c.sum());
    // cell4's constructed footprint (76.41 GB) sits under its construction terms (77.00 GB).
    try testing.expect(c.sum() > 76_410_000_000 and c.sum() < 77_100_000_000);
}

test "dsv41 memory: with the embedding on its host rows no phase bills the device table, and the one-count fill gains 2 rows" {
    var dev = cell4Bill();
    dev.embedding_host_rows = false;
    var host = dev;
    host.embedding_host_rows = true;
    try testing.expectEqual(dev.prefillTotal() - dev.embedding, host.prefillTotal());
    try testing.expectEqual(dev.decodeTotal(), host.decodeTotal());
    try testing.expectEqual(dev.constructionTerms().sum() - dev.embedding, host.constructionTerms().sum());
    const per_row = @as(u64, dev.layers) * 13_315_584;
    const at = struct {
        fn f(b: CellBill, base: u64, pr: u64) module.FillBill {
            return .{ .prefill_fixed = b.prefillTotal() - b.baseline + base - @as(u64, b.prefill_rows) * pr, .decode_fixed = b.decodeTotal() - b.baseline + base - @as(u64, b.decode_rows) * pr, .per_row = pr };
        }
    }.f;
    const r_dev = try module.fillRows(at(dev, 9_200_000_000, per_row), 120_259_084_288, 384);
    const r_host = try module.fillRows(at(host, 9_200_000_000, per_row), 120_259_084_288, 384);
    try testing.expectEqual(r_dev.prefill + 2, r_host.prefill);
}

test "dsv41 memory: the phase record's residuals: billed less the interval peak, billed device terms less MLX's peak" {
    const b = cell4Bill();
    // cell4's prompt boundary: footprint 83.03 GB now; MLX active 76.56, peak 91.35 GB.
    const pm: status.ProcessMemory = .{ .footprint = 83_030_000_000, .footprint_interval_peak = 97_000_000_000, .footprint_lifetime_peak = 97_000_000_000 };
    const r = recordOf("prompt pass", b.prefillTerms(), pm, 76_560_000_000, 5_000_000_000, 91_350_000_000, 110_000_000_000, 3_000_000_000, 850_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // The page cache the step created (file-backed now less at its vm start) and the speculative pages, recorded.
    try testing.expectEqual(@as(i64, 3_000_000_000 - 4_870_000_000), r.file_cache_created_bytes);
    try testing.expectEqual(@as(u64, 850_000_000), r.box_speculative_bytes);
    try testing.expectEqual(b.prefillTotal() - b.baseline, r.billed_process_bytes);
    try testing.expectEqual(@as(i64, @intCast(r.billed_process_bytes)) - 97_000_000_000, r.residual_bytes);
    const device = b.slot_prefill + b.residents + (b.engram - engram.row_cache_host_bytes) + b.prefill_wave + b.kv;
    try testing.expectEqual(@as(i64, @intCast(device)) - 91_350_000_000, r.mlx_residual_bytes);
    // A footprint above its (stale) interval peak counts as the measurement; a peak below active reads active.
    const late: status.ProcessMemory = .{ .footprint = 99_000_000_000, .footprint_interval_peak = 0 };
    const r2 = recordOf("decode", b.decodeTerms(), late, 97_000_000_000, 0, 0, 0, 19_950_000_000, 15_940_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // v6c2's construction: 15.08 GB of page cache created, 15.94 GB of it speculative.
    try testing.expectEqual(@as(i64, 15_080_000_000), r2.file_cache_created_bytes);
    try testing.expectEqual(@as(i64, @intCast(b.decodeTotal() - b.baseline)) - 99_000_000_000, r2.residual_bytes);
    try testing.expectEqual(@as(u64, 97_000_000_000), r2.mlx_peak_bytes);
    // The record serialises for the receipt.
    const json = try std.json.Stringify.valueAlloc(testing.allocator, r, .{});
    defer testing.allocator.free(json);
    try testing.expect(std.mem.indexOf(u8, json, "\"footprint_interval_peak\":97000000000") != null);
}

// DSV41_BANK=<bank> (host): the fill and the bill that checks it agree at the same inputs (v6 211422's:
// non-file 8.5487616 GB, box 119.259 GB, wired 3.380 GB), and a bill re-read with the constructed module's
// wired bytes (live, +85 GB) is what refused v6: the check must take the arm's planned wired bytes.
test "dsv41 memory: the fill and its admission agree at the same inputs (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 8_548_761_600;
    config.memory_ceiling_bytes = 119_259_000_000;
    const wired: u64 = 3_380_379_648;
    const nr = try fillAtWired(a, testing.io, config, module.fill_prompt_tokens, module.fill_max_tokens, wired);
    try testing.expect(nr.prefill <= nr.decode);
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const b = try cellBillWired(a, testing.io, &config, module.fill_prompt_tokens, module.fill_max_tokens, wired);
    try testing.expectEqual(nr.prefill, b.prefill_rows);
    try testing.expectEqual(nr.decode, b.decode_rows);
    try testing.expect(b.prefillTotal() <= config.memory_ceiling_bytes.? - module.ceiling_stop_bytes);
    try testing.expect(b.decodeTotal() <= config.memory_ceiling_bytes.? - module.ceiling_stop_bytes);
    // The prompt phase charges the served tier's cache limit exactly (the limit it sets).
    try testing.expectEqual(@as(u64, module.prefillCacheLimit(.served)), b.prefill_cache);
    try testing.expectEqual(@as(u64, 2 << 30), b.prefill_cache);
    // The host side billed as measured, the named host terms folded into it.
    try testing.expectEqual(measured_host_side_bytes, b.host_reserve);
    try testing.expectEqual(@as(u64, 0), b.lookahead_staging + b.wide_window + b.unbilled_overhead);
    std.debug.print("\nfill and admission at v6's inputs: {d} / {d} rows, prompt total {d} B\n", .{ nr.prefill, nr.decode, b.prefillTotal() });
    // The failure mode: the same bill with the constructed module's wired bytes read live.
    try testing.expectError(error.PrefillDoesNotFit, cellBillWired(a, testing.io, &config, module.fill_prompt_tokens, module.fill_max_tokens, wired + 85_000_000_000));
}

test "dsv41 memory: the harness's window proofs: page cache left by the load, the box's pages at the phase change" {
    // v6c2's construction: file-backed 4.87 -> 19.95 GB: refused; configs and the metallib's pages: within it.
    try testing.expectError(error.ConstructionLeftPageCache, checkPageCache(19_950_000_000 - 4_870_000_000));
    try checkPageCache(200_000_000);
    try checkPageCache(-300_000_000);
    const ref: u64 = 13_933_000_000;
    const before: module.BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000, .physical = 91_915_000_000 + ref };
    const freed: module.BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache, .physical = before.physical - before.cache };
    const ok: module.PhaseChangeRecord = .{ .before = before, .after = freed, .freed_bytes = before.cache, .settle_ms = 250 };
    try checkBoxReclaimed(ok, ref);
    // SERVED7's shape: the footprint dropped, the box's pages did not.
    var served7 = ok;
    served7.after.physical = before.physical;
    try testing.expectError(error.PhaseChangeNotReclaimed, checkBoxReclaimed(served7, ref));
    // A short prompt's older releases still counted at the boundary: outside the footprint above the prompt's start.
    const lag: u64 = 2_100_000_000;
    var lagged = ok;
    lagged.before.physical += lag;
    lagged.after.physical += lag;
    try testing.expectError(error.PhaseChangeNotReclaimed, checkBoxReclaimed(lagged, ref));
    // Within the tolerance (other processes' movement): passes.
    var noisy = ok;
    noisy.after.physical += box_tolerance_bytes;
    try checkBoxReclaimed(noisy, ref);
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
    try cellFill(a, testing.io, &config, 16384, max_tokens);
    const b = try cellBill(a, testing.io, &config, 16384, max_tokens);
    printBill(b);
    try testing.expect(b.decode_rows >= b.prefill_rows and b.processBound() > 0);
}

/// The prompt pass's stage profile: the model's probe points (`p.put` in the graph: attn.x ... out.h),
/// each one evaluated where the model publishes it, the host clock charged to the stage that ends
/// there (so a segment is everything the graph built and ran since the previous point), per stage
/// name and per chunk. The routed call's segment ("moe.routed") also carries the expert stream's
/// read counters. The syncs serialize the pass: the profile's wall exceeds the unprobed TTFT; the
/// split, not the sum, is the reading.
const PrefillProbe = struct {
    const n_max = 40;
    g: *ops.MlxOps,
    io: std.Io,
    stats_of: *const fn (*anyopaque) expert_stream.Stats,
    stats_ctx: *anyopaque,
    n_layers: u32,
    last: std.Io.Timestamp,
    names: [n_max][]const u8 = undefined,
    ns: [n_max]u64 = @splat(0),
    n: usize = 0,
    layers_done: u64 = 0,
    chunk_ns: [64]u64 = @splat(0),
    chunk_rows: [64]u32 = @splat(0),
    read_wall_ns: u64 = 0,
    read_bytes: u64 = 0,
    misses: u64 = 0,
    before: expert_stream.Stats = .{},

    fn slot(self: *PrefillProbe, name: []const u8) usize {
        for (self.names[0..self.n], 0..) |x, i| if (std.mem.eql(u8, x, name)) return i;
        self.names[self.n] = name;
        self.n += 1;
        return self.n - 1;
    }

    pub fn put(self: *PrefillProbe, name: []const u8, x: anytype) !void {
        if (@TypeOf(x) != ops.MlxOps.T) return;
        const is_routed = std.mem.eql(u8, name, "moe.routed");
        if (std.mem.eql(u8, name, "gate.weights")) self.before = self.stats_of(self.stats_ctx);
        try self.g.evalAll(&.{x});
        const d: u64 = @intCast(self.last.untilNow(self.io, .boot).nanoseconds);
        self.last = std.Io.Timestamp.now(self.io, .boot);
        self.ns[self.slot(name)] += d;
        const chunk: usize = @min(self.layers_done / self.n_layers, self.chunk_ns.len - 1);
        self.chunk_ns[chunk] += d;
        if (is_routed) {
            const after = self.stats_of(self.stats_ctx);
            self.read_wall_ns += after.read_wall_ns -| self.before.read_wall_ns;
            self.read_bytes += after.expert_bytes_read -| self.before.expert_bytes_read;
            self.misses += after.expert_cache_misses -| self.before.expert_cache_misses;
            if (self.chunk_rows[chunk] == 0) self.chunk_rows[chunk] = @intCast(self.g.shapeOf(x).dim(0));
        }
        if (std.mem.eql(u8, name, "out.h")) self.layers_done += 1;
    }
};

// Profiling window only (the prompt pass, no decode): DSV41_CELL_PROFILE=1 plus the cell's window env
// (DSV41_CELL_PROMPT_IDS [DSV41_CELL_CASE] DSV41_BANK DSV41_CELL_BASELINE_GB DSV41_CELL_CEILING_GB
// _GPU_WINDOW_LOCKED). The served module as the cell builds it; the prompt as ONE model forward at the
// model's chunk rule through the served hook (the cell's prompt pass without the draft seed), probed.
// Prints PREFILL_PROFILE lines: per stage (seconds, share), per chunk (rows, seconds), the stream's
// reads (bytes, read-busy wall, misses), the probed wall. Writes nothing.
test "dsv41 served cell: the prompt pass profiled by stage and chunk (profiling window)" {
    if (std.c.getenv("DSV41_CELL_PROFILE") == null) return error.SkipZigTest;
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, io, prompt_path, case_id, bank_dir);
    var config = inputs.config;
    try cellConfig(&config);
    try cellFill(a, io, &config, inputs.prompt.len, 1024);
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
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const md = try module.Module.init(gpa, io, &config, &weights, s);
    defer md.deinit();
    const arm = switch (md.arm) {
        .host_waits => |t| t.arm,
        else => return error.CellArmVariant,
    };
    const Hook = @TypeOf(arm.hook);
    const stats_of = struct {
        fn f(ctx: *anyopaque) expert_stream.Stats {
            const h: *Hook = @ptrCast(@alignCast(ctx));
            return h.source.stats();
        }
    }.f;
    const g = &md.g;
    var st = try md.model.newStateWith(md.model.boundedKv(module.Module.maxPositions(inputs.prompt.len, inputs.prompt.len + 1024)));
    defer st.deinit(g, gpa);
    var probe: PrefillProbe = .{ .g = g, .io = io, .stats_of = stats_of, .stats_ctx = @ptrCast(&arm.hook), .n_layers = md.model.c.n_layers, .last = undefined };
    const s0 = stats_of(@ptrCast(&arm.hook));
    const t0 = std.Io.Timestamp.now(io, .boot);
    probe.last = t0;
    const r = try md.model.forward(g, &st, inputs.prompt, .{ .logits = .last, .main_hidden = true }, &arm.hook, &probe);
    try g.evalAll(&.{r.logits.?});
    const wall_s = secondsSince(io, t0);
    const s1 = stats_of(@ptrCast(&arm.hook));
    const secs = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.f;
    var total: u64 = 0;
    for (probe.ns[0..probe.n]) |x| total += x;
    std.debug.print("\nPREFILL_PROFILE {{\"prompt_tokens\": {d}, \"probed_wall_s\": {d:.3}, \"stage_sum_s\": {d:.3}, \"chunks\": {d}, \"read_bytes\": {d}, \"read_busy_s\": {d:.3}, \"misses\": {d}, \"routed_read_busy_s\": {d:.3}}}\n", .{
        inputs.prompt.len, wall_s, secs(total), probe.layers_done / probe.n_layers, s1.expert_bytes_read - s0.expert_bytes_read, secs(s1.read_wall_ns - s0.read_wall_ns), s1.expert_cache_misses - s0.expert_cache_misses, secs(probe.read_wall_ns),
    });
    for (probe.names[0..probe.n], probe.ns[0..probe.n]) |name, ns| std.debug.print("PREFILL_PROFILE_STAGE {{\"stage\": \"{s}\", \"s\": {d:.3}, \"share\": {d:.4}}}\n", .{ name, secs(ns), @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(@max(total, 1))) });
    const n_chunks: usize = @intCast(@min((probe.layers_done + probe.n_layers - 1) / probe.n_layers, probe.chunk_ns.len));
    for (0..n_chunks) |i| std.debug.print("PREFILL_PROFILE_CHUNK {{\"chunk\": {d}, \"rows\": {d}, \"s\": {d:.3}}}\n", .{ i, probe.chunk_rows[i], secs(probe.chunk_ns[i]) });
}

/// DECODE_PROFILE lines: the per-phase means over the cycles (ms per cycle) and the stream's.
fn printDecodeProfile(p: []const ProfCycle) void {
    if (p.len == 0) return;
    var sum: ProfCycle = .{ .k_eff = 0, .accepted = 0, .draft_ms = 0, .verify_ms = 0, .decide_ms = 0, .commit_ms = 0, .tail_ms = 0, .misses = 0, .bytes_read = 0, .read_busy_ms = 0 };
    for (p) |c| {
        sum.k_eff += c.k_eff;
        sum.accepted += c.accepted;
        sum.draft_ms += c.draft_ms;
        sum.verify_ms += c.verify_ms;
        sum.decide_ms += c.decide_ms;
        sum.commit_ms += c.commit_ms;
        sum.tail_ms += c.tail_ms;
        sum.misses += c.misses;
        sum.bytes_read += c.bytes_read;
        sum.read_busy_ms += c.read_busy_ms;
    }
    const n: f64 = @floatFromInt(p.len);
    std.debug.print("\nDECODE_PROFILE {{\"cycles\": {d}, \"draft_ms\": {d:.2}, \"verify_ms\": {d:.2}, \"decide_ms\": {d:.2}, \"commit_ms\": {d:.2}, \"tail_ms\": {d:.2}, \"cycle_ms\": {d:.2}, \"misses_per_cycle\": {d:.1}, \"mb_read_per_cycle\": {d:.1}, \"read_busy_ms\": {d:.2}, \"k_eff\": {d:.2}, \"accepted\": {d:.2}}}\n", .{
        p.len,                     sum.draft_ms / n,       sum.verify_ms / n,  sum.decide_ms / n,
        sum.commit_ms / n,         sum.tail_ms / n,        (sum.draft_ms + sum.verify_ms + sum.decide_ms + sum.commit_ms + sum.tail_ms) / n,
        @as(f64, @floatFromInt(sum.misses)) / n, @as(f64, @floatFromInt(sum.bytes_read)) / n / 1e6, sum.read_busy_ms / n,
        @as(f64, @floatFromInt(sum.k_eff)) / n, @as(f64, @floatFromInt(sum.accepted)) / n,
    });
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
    // The fastest line's prompt is pinned here; either file's own digest was checked by the loader.
    if (case_id) |id| if (std.mem.eql(u8, id, "code-20260923")) try testing.expectEqualStrings("667506d734cce8152f3c42c9a97639a5b466540c70d9798a72e7564e2361bf86", &sha);
    if (case_id != null) {
        try testing.expectError(error.PromptIdsNoCell, cellPrompt(a, testing.io, prompt_path, "no-such-case"));
    } else try testing.expectError(error.PromptIdsNoCell, standardPrompt(a, testing.io, prompt_path, 16384, 1));
    try testing.expectEqual(@as(usize, 16384 + 1024 + mdl.Model(ops.MlxOps).scratch_rows), module.Module.maxPositions(16384, 16384 + 1024));
    // The admission inputs come from the runner's environment (refused by name without them).
    var cfg = inputs.config;
    if (std.c.getenv("DSV41_CELL_BASELINE_GB") == null) try testing.expectError(error.CellBaselineMissing, cellConfig(&cfg));
    try testing.expectError(error.CellBoolValue, cellBool("X", "yes"));
    try testing.expect(try cellBool("X", "1") and !try cellBool("X", "0"));
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
