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
const xko = @import("exl3_kernel_ops.zig");

/// The kernels lane's startup acceptance on the harness's GPU stream (registry,
/// kernels built, the device self-check plan judged, the backend's launcher set),
/// as the served arm runs it before the model loads.
fn acceptKernels(gpa: std.mem.Allocator, g: *ops.MlxOps) !*xko.Accepted(ops.MlxOps) {
    var diag: xk.Diag = .{};
    return xko.acceptAtStartup(ops.MlxOps, gpa, g, .{ .device = .{ .stream = g.s } }, &diag) catch |e| {
        std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
        return e;
    };
}

/// Every bound bank of the hook `ex` against the kernels' signatures (once, after growth).
fn checkBanks(g: *ops.MlxOps, reg: *const xk.Registry, ex: anytype) !void {
    var diag: xk.Diag = .{};
    errdefer std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
    for (ex.banks) |per| for (per) |maybe| if (maybe) |b| {
        inline for (.{ .{ xko.Proj.gate, "gate" }, .{ xko.Proj.up, "up" }, .{ xko.Proj.down, "down" } }) |pp| {
            const arr = @field(b, pp[1]);
            try xko.checkBank(ops.MlxOps, g, reg, pp[0], .{ .code = arr.code, .rout = arr.rout, .rin = arr.rin }, &diag);
        }
    };
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
    const kernels = try acceptKernels(gpa, &g);
    defer kernels.deinit(&g);

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
    const Chain = xp.EagerChain(ops.MlxOps, xp.MlxGemv);
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(kernels.gemvRoute(xp.MlxGemv), &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, &kernels.reg, &ex);

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
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("dsv41 ar: {d} prompt tokens in forwards of {d}, {d} generated; ids {s}; logits rows equal {d}/{d}; {d} rows/layer; routes {d}, hits {d}, misses {d}, {d} B read in {d} preadv; {d} ms; MLX peak {d} B\n", .{
        ref.prompt_ids.len,       ref.chunk,               out.len,
        if (first == null) "IDENTICAL" else "DIFFER", logits_equal, out.len, rows,
        sst.route_calls,          sst.expert_cache_hits,   sst.expert_cache_misses,
        sst.expert_bytes_read,    sst.preadv_calls,        wall_ms,
        peak,
    });
    if (first) |i| std.debug.print("dsv41 ar: first differing step {d}: native {d}, reference {d} (reference top-2 {any}, margin {d})\n", .{ i, out[i], ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin });
    try testing.expectEqualSlices(u32, ref.generated_ids, out);
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
    const kernels = try acceptKernels(gpa, &g);
    defer kernels.deinit(&g);
    // The served decode seam's own binding of the residents (`Dspark(A).open`).
    const L = dsl.Loop(ops.MlxOps);
    const res = try dss.Resources(ops.MlxOps).open(gpa, io, &g, bank_dir, c, map_path, null, &diag);
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
    const Chain = xp.EagerChain(ops.MlxOps, xp.MlxGemv);
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(kernels.gemvRoute(xp.MlxGemv), &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, &kernels.reg, &ex);

    const acceptance: ds.Acceptance = if (std.mem.eql(u8, ref.arm, "typical")) .{ .typical = .{ .delta = @floatCast(ref.delta.?) } } else .greedy;
    var lp = L.init(&g, m, head, &st, caches[0..head.nStages()], .{ .acceptance = acceptance, .max_tokens = 1 << 20 });
    defer lp.deinit();
    const primary = try lp.prefill(gpa, &ex, ref.prompt);
    try testing.expectEqual(ref.tokens[0], primary);
    // The served adapter's fence: the embedding table retires to its host rows before the cycles.
    try res.retireEmbedding(&g);
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
                    i, t, lg.top_ids[t][0], lg.top_logits[t][0], lg.top_ids[t][1], lg.top_logits[t][1], v, margin,
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
            std.debug.print("dsv41 dspark: cycle {d} first finer difference: primary {d} vs {d}, native {any} vs {any}, k {d} vs {d}, conf bits {any} vs {any}\n", .{ i, lg.primary, rc.primary, lg.native[0..rc.draft_ids.len], rc.draft_ids, lg.k_native, rc.k_eff_native, @as([]const u32, @ptrCast(lg.conf[0..rc.conf_sigmoid_bits.len])), rc.conf_sigmoid_bits });
        }
    }
    const wall_ms = @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms);
    const n = @min(out.items.len, ref.tokens.len - 1);
    const ids_same = std.mem.eql(u32, out.items[0..n], ref.tokens[1..][0..n]);
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("dsv41 dspark: {s} arm, kv {s} {d}, {d} cycles; ids {s} ({d}); per-cycle acceptance {s}; decisions {s}; accepted {d}/{d}; {d} rows/layer; routes {d}, {d} B read; {d} ms; MLX peak {d} B\n", .{
        ref.arm,                                                   if (kv_bound != null) "bounded" else "tier",
        kv_bound orelse 0,                                         ref.cycles.len,
        if (ids_same) "IDENTICAL" else "DIFFER",                   n + 1,
        if (first_accept == null) "IDENTICAL" else "DIFFER",       if (first_decision == null) "IDENTICAL" else "DIFFER",
        lp.stats.accepted_drafts,                                  lp.stats.drafted_tokens,
        rows,                                                      sst.route_calls,
        sst.expert_bytes_read,                                     wall_ms,
        peak,
    });
    try testing.expect(first_accept == null);
    try testing.expectEqualSlices(u32, ref.tokens[1..][0..n], out.items[0..n]);
}
