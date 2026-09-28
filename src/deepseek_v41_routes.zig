//! The Python runtime's `MTPLX_DSV41_*` levers as the Zig construction-time
//! configuration: trunk routes (`graph.Routes`), the KV backing
//! (`cache.Geometry`) and the prefill schedule. One lever set drives both
//! sides of a parity run. A lever this build cannot run the same way (a Metal
//! kernel, an unimplemented precision) is refused by name, never approximated;
//! levers owned by the draft or the expert streamer are carried, not applied.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const graph = @import("deepseek_v41_graph.zig");
const kvc = @import("deepseek_v41_cache.zig");

pub const Refusal = error{ UnknownLever, LeverValue, LeverNeedsKernel };

pub const Tier = struct {
    routes: graph.Routes = .{},
    kv: kvc.Geometry = .{},
    /// `MTPLX_DSV41_PREFILL_CHUNK`: explicit query chunk (<= 0 one shot); null = derived.
    prefill_chunk: ?i64 = null,
    chunk_target_bytes: f64 = kvc.default_chunk_target_bytes,
    /// K16: every layer over all chunks before the next (one routed-bank read).
    layer_major: bool = false,
    /// Levers of the DSpark draft (M2) and the expert streamer (phase 1 / 2),
    /// accepted here and applied by their owners.
    deferred: [max_deferred][]const u8 = undefined,
    n_deferred: u8 = 0,

    pub const max_deferred = 48;

    pub fn deferredLevers(self: *const Tier) []const []const u8 {
        return self.deferred[0..self.n_deferred];
    }
};

const Kind = enum {
    route,
    /// Byte-identical by construction in the Zig design (or a diagnostic).
    by_design,
    kv,
    prefill,
    /// A Metal kernel: off is accepted, on refuses.
    kernel,
    deferred,
};

const Lever = struct { name: []const u8, kind: Kind };

/// Every lever of the arm presets and the model modules (`ab_decode_env_levers.py`,
/// `deepseek_v41*.py`) plus the streamer / bank levers a tier cell sets.
const levers = [_]Lever{
    .{ .name = "SELECTED_KEYS", .kind = .route },
    .{ .name = "ATTN_CORE_COMPILE", .kind = .route },
    .{ .name = "PREFILL_SCORE_PATH", .kind = .route },
    .{ .name = "PREFILL_SCORE_DTYPE", .kind = .route },
    .{ .name = "PREFILL_SCORE_KEY_CHUNK", .kind = .route },
    .{ .name = "ATTN_COMPILE", .kind = .route },
    .{ .name = "HC_COMPILE", .kind = .route },
    .{ .name = "SMALL_STAGES_FUSED", .kind = .route },
    .{ .name = "ATTN_WO_A_CACHE", .kind = .route },
    .{ .name = "HEAD_MODE", .kind = .route },
    .{ .name = "ATTN_LEAN_CASTS", .kind = .by_design },
    .{ .name = "ATTN_WIN_MEMO", .kind = .by_design },
    .{ .name = "ATTN_SHAPE_STABLE", .kind = .by_design },
    .{ .name = "SELECT_FENCE", .kind = .by_design },
    .{ .name = "VERIFY_RECORD_HASHES", .kind = .by_design },
    .{ .name = "STAGE_TIMING", .kind = .by_design },
    .{ .name = "WINDOW_RING", .kind = .kv },
    .{ .name = "WINDOW_RING_MAX_VERIFY", .kind = .kv },
    .{ .name = "WINDOW_RING_SLACK", .kind = .kv },
    .{ .name = "WINDOW_RING_HEADROOM", .kind = .kv },
    .{ .name = "WINDOW_RING_MAXKV", .kind = .kv },
    .{ .name = "KV_BOUNDED", .kind = .kv },
    .{ .name = "KV_BOUNDED_MAXKV", .kind = .kv },
    .{ .name = "KV_CHUNK_GROW", .kind = .kv },
    .{ .name = "PREFILL_CHUNK", .kind = .prefill },
    .{ .name = "PREFILL_CHUNK_TARGET_GB", .kind = .prefill },
    .{ .name = "PREFILL_LAYER_MAJOR", .kind = .prefill },
    .{ .name = "PREFILL_MOE_TARGET_GB", .kind = .prefill },
    .{ .name = "SINKHORN_METAL", .kind = .kernel },
    .{ .name = "HC_PREMIX_KERNEL", .kind = .kernel },
    .{ .name = "PREFILL_SOFTMAX_KERNEL", .kind = .kernel },
    .{ .name = "DECODE_ATTN_KERNEL", .kind = .kernel },
    .{ .name = "ATTN_FUSED_PROJ", .kind = .kernel },
    .{ .name = "ATTN_WO_A_DIRECT", .kind = .kernel },
    .{ .name = "DSPARK_VERIFY_K29", .kind = .kernel },
    .{ .name = "DSPARK_DECODE_KERNELS", .kind = .kernel },
    .{ .name = "DRAFT_COMPILE", .kind = .deferred },
    .{ .name = "DRAFT_HEAD_BF16", .kind = .deferred },
    .{ .name = "MTP", .kind = .deferred },
    .{ .name = "DSPARK_CONF_THRESHOLD", .kind = .deferred },
    .{ .name = "DSPARK_VERIFY_DECODE_PHASE", .kind = .deferred },
    .{ .name = "DEVICE_SAMPLE", .kind = .deferred },
    .{ .name = "DIVERGENCE_TIE_ULPS", .kind = .deferred },
    .{ .name = "RUNNER", .kind = .deferred },
    .{ .name = "GATE_PREFETCH", .kind = .deferred },
    .{ .name = "GATE_PREFETCH_MIN_LAYER", .kind = .deferred },
    .{ .name = "GATE_PREFETCH_MARGIN", .kind = .deferred },
    .{ .name = "SWITCH_FASTPATH", .kind = .deferred },
    .{ .name = "SWITCH_SUBMIT", .kind = .deferred },
    .{ .name = "SHARED_OVERLAP", .kind = .deferred },
    .{ .name = "DEVICE_ROUTE", .kind = .deferred },
    .{ .name = "DEVICE_ROUTE_PINNED", .kind = .deferred },
    .{ .name = "PIN_WORKING_SET", .kind = .deferred },
    .{ .name = "PIN_REFRESH_TOKENS", .kind = .deferred },
    .{ .name = "SINGLE_SLOT_POOL", .kind = .deferred },
    .{ .name = "VERIFY_SINGLE_BARRIER", .kind = .deferred },
    .{ .name = "MLX_LIMIT_HEADROOM_GIB", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_EXPERTS", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_MIN_ROWS", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_BATCH", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_MATMUL_DTYPE", .kind = .deferred },
    .{ .name = "LAYOUT_FIX", .kind = .deferred },
    .{ .name = "DOWN_K_PAD", .kind = .deferred },
    .{ .name = "IO_READ_FANOUT", .kind = .deferred },
    .{ .name = "TCQ3", .kind = .deferred },
    .{ .name = "TCQ3_ALLOCATION", .kind = .deferred },
    .{ .name = "TCQ3_CONFIDENCE", .kind = .deferred },
    .{ .name = "TCQ3_ENGRAM", .kind = .deferred },
    .{ .name = "TCQ3_PIPELINE", .kind = .deferred },
    .{ .name = "TCQ_BANK", .kind = .deferred },
};

const prefix = "MTPLX_DSV41_";

fn refuse(diag: ?*v41.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

fn oneOf(v: []const u8, words: []const []const u8) bool {
    for (words) |w| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, v, " "), w)) return true;
    return false;
}

const off_words = [_][]const u8{ "", "0", "false", "no", "off", "auto", "none", "default" };
const on_words = [_][]const u8{ "1", "true", "yes", "on" };

fn truthy(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!bool {
    var buf: [16]u8 = undefined;
    const v = std.ascii.lowerString(buf[0..@min(raw.len, buf.len)], raw[0..@min(raw.len, buf.len)]);
    for (off_words) |w| if (std.mem.eql(u8, v, w)) return false;
    for (on_words) |w| if (std.mem.eql(u8, v, w)) return true;
    return refuse(diag, error.LeverValue, "{s}{s}={s}: not a boolean", .{ prefix, name, raw });
}

fn uint(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, raw, " "), 10) catch refuse(diag, error.LeverValue, "{s}{s}={s}: not a non-negative integer", .{ prefix, name, raw });
}

fn gb(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!f64 {
    const v = std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " ")) catch return refuse(diag, error.LeverValue, "{s}{s}={s}: not a number", .{ prefix, name, raw });
    return @max(1.0, v) * 1e9;
}

/// The levers of `pairs` (full `MTPLX_DSV41_*` names; other variables are
/// ignored) as a tier. Precedence as the Python cache: KV_BOUNDED over
/// WINDOW_RING over KV_CHUNK_GROW.
pub fn parse(pairs: []const [2][]const u8, diag: ?*v41.Diag) Refusal!Tier {
    var t: Tier = .{};
    var ring = false;
    var bounded = false;
    var chunk_grow = false;
    var ring_maxkv: ?u32 = null;
    var bounded_maxkv: ?u32 = null;
    for (pairs) |kv| {
        if (!std.mem.startsWith(u8, kv[0], prefix)) continue;
        const name = kv[0][prefix.len..];
        const val = kv[1];
        const lever = for (levers) |l| {
            if (std.mem.eql(u8, l.name, name)) break l;
        } else return refuse(diag, error.UnknownLever, "{s}: not a lever this build knows (refused rather than ignored)", .{kv[0]});
        switch (lever.kind) {
            .route => {
                const r = &t.routes;
                if (std.mem.eql(u8, name, "SELECTED_KEYS")) {
                    r.selected_keys = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "ATTN_CORE_COMPILE")) {
                    r.attn_core_compile = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "ATTN_COMPILE")) {
                    r.attn_compile = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "HC_COMPILE")) {
                    r.hc_compile = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "SMALL_STAGES_FUSED")) {
                    r.small_stages = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "ATTN_WO_A_CACHE")) {
                    r.wo_a_f32 = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_PATH")) {
                    if (oneOf(val, &.{ "lean", "fused", "passcut", "pass_cut" })) {
                        r.lean_prefill_score = true;
                    } else if (!oneOf(val, &.{ "", "default", "off", "none", "control", "oneshot", "one_shot" })) {
                        return refuse(diag, error.LeverValue, "{s}={s}: not oneshot or lean", .{ kv[0], val });
                    }
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_DTYPE")) {
                    if (!oneOf(val, &.{ "", "default", "off", "none", "control", "f32", "fp32", "float32" })) return refuse(diag, error.LeverValue, "{s}={s}: only the f32 score path is ported", .{ kv[0], val });
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_KEY_CHUNK")) {
                    if (!oneOf(val, &.{ "", "0", "off", "none", "default" })) return refuse(diag, error.LeverValue, "{s}={s}: the split-K score path is not ported", .{ kv[0], val });
                } else if (std.mem.eql(u8, name, "HEAD_MODE")) {
                    if (std.mem.eql(u8, val, "bf16")) {
                        r.head = .bf16;
                    } else if (std.mem.eql(u8, val, "mxfp8")) {
                        r.head = .mxfp8;
                    } else if (oneOf(val, &.{ "", "default", "off", "none", "control", "0" })) {
                        r.head = .f32;
                    } else return refuse(diag, error.LeverValue, "{s}={s}: the head codecs ported are bf16 and mxfp8", .{ kv[0], val });
                } else unreachable;
            },
            .by_design => _ = try truthy(name, val, diag),
            .kv => {
                if (std.mem.eql(u8, name, "WINDOW_RING")) {
                    ring = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "KV_BOUNDED")) {
                    bounded = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "KV_CHUNK_GROW")) {
                    chunk_grow = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_MAX_VERIFY")) {
                    t.kv.max_verify = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_SLACK")) {
                    t.kv.slack = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_HEADROOM")) {
                    t.kv.headroom = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_MAXKV")) {
                    const m = try uint(name, val, diag);
                    ring_maxkv = if (m > 0) m else null;
                } else if (std.mem.eql(u8, name, "KV_BOUNDED_MAXKV")) {
                    const m = try uint(name, val, diag);
                    bounded_maxkv = if (m > 0) m else null;
                } else unreachable;
            },
            .prefill => {
                if (std.mem.eql(u8, name, "PREFILL_CHUNK")) {
                    const v = std.mem.trim(u8, val, " ");
                    if (v.len > 0 and !std.ascii.eqlIgnoreCase(v, "auto")) {
                        t.prefill_chunk = std.fmt.parseInt(i64, v, 10) catch return refuse(diag, error.LeverValue, "{s}={s}: not an integer or auto", .{ kv[0], val });
                    }
                } else if (std.mem.eql(u8, name, "PREFILL_CHUNK_TARGET_GB")) {
                    t.chunk_target_bytes = try gb(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_LAYER_MAJOR")) {
                    t.layer_major = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_MOE_TARGET_GB")) {
                    _ = try gb(name, val, diag);
                } else unreachable;
            },
            .kernel => if (try truthy(name, val, diag)) return refuse(diag, error.LeverNeedsKernel, "{s}={s}: a Metal kernel route, not in this build (the kernels lane owns it)", .{ kv[0], val }),
            .deferred => {
                if (t.n_deferred == Tier.max_deferred) return refuse(diag, error.LeverValue, "too many deferred levers", .{});
                t.deferred[t.n_deferred] = kv[0];
                t.n_deferred += 1;
            },
        }
    }
    if (bounded) {
        t.kv.route = .bounded;
        t.kv.max_kv = bounded_maxkv orelse ring_maxkv;
    } else if (ring) {
        t.kv.route = .window_ring;
        t.kv.max_kv = ring_maxkv;
    } else if (chunk_grow) t.kv.route = .chunk_grow;
    return t;
}

/// `K=V K=V ...` (whitespace separated) as pairs borrowing `text`.
pub fn splitPairs(a: std.mem.Allocator, text: []const u8) ![][2][]const u8 {
    var out: std.ArrayList([2][]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\n");
    while (it.next()) |tok| {
        const eq = std.mem.indexOfScalar(u8, tok, '=') orelse return error.LeverSyntax;
        try out.append(a, .{ tok[0..eq], tok[eq + 1 ..] });
    }
    return out.toOwnedSlice(a);
}

const testing = std.testing;

/// The tier arm `cell16k_ring_v2_draft_attn_pf0` (ab_decode_env_levers.py) plus
/// the streamer levers a tier cell sets.
const tier_arm =
    "MTPLX_DSV41_PREFILL_LAYER_MAJOR=1 MTPLX_DSV41_PREFILL_DENSE_EXPERTS=1 MTPLX_DSV41_PREFILL_SCORE_PATH=lean " ++
    "MTPLX_DSV41_SELECTED_KEYS=1 MTPLX_DSV41_WINDOW_RING=1 MTPLX_DSV41_LAYOUT_FIX=1 MTPLX_DSV41_HEAD_MODE=bf16 " ++
    "MTPLX_DSV41_SINKHORN_METAL=1 MTPLX_DSV41_ATTN_COMPILE=1 MTPLX_DSV41_ATTN_WIN_MEMO=1 MTPLX_DSV41_RUNNER=v2 " ++
    "MTPLX_DSV41_DRAFT_COMPILE=1 MTPLX_DSV41_DRAFT_HEAD_BF16=1 MTPLX_DSV41_ATTN_WO_A_CACHE=1 MTPLX_DSV41_ATTN_LEAN_CASTS=1 " ++
    "MTPLX_DSV41_ATTN_FUSED_PROJ=1 MTPLX_DSV41_DECODE_ATTN_KERNEL=0 MTPLX_DSV41_DSPARK_VERIFY_K29=0 MTPLX_DSV41_GATE_PREFETCH=0 " ++
    "MTPLX_DSV41_IO_READ_FANOUT=4 MTPLX_DSV41_TCQ3=1 MTPLX_DSV41_TCQ3_PIPELINE=pipeline";

test "dsv41 routes: the tier arm refuses only for its Metal kernels, and parses without them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    try testing.expectError(error.LeverNeedsKernel, parse(try splitPairs(a, tier_arm), &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "SINKHORN_METAL") != null);
    // Without the kernel levers the pure-MLX remainder is the tier's trunk.
    var pairs: std.ArrayList([2][]const u8) = .empty;
    for (try splitPairs(a, tier_arm)) |p| {
        if (std.mem.endsWith(u8, p[0], "SINKHORN_METAL") or std.mem.endsWith(u8, p[0], "ATTN_FUSED_PROJ")) continue;
        try pairs.append(a, p);
    }
    const t = try parse(pairs.items, &diag);
    const r = t.routes;
    try testing.expect(r.selected_keys and r.lean_prefill_score and r.attn_compile and r.wo_a_f32);
    try testing.expect(!r.hc_compile and !r.small_stages and !r.attn_core_compile);
    try testing.expectEqual(graph.Routes.Head.bf16, r.head);
    try testing.expectEqual(kvc.Route.window_ring, t.kv.route);
    try testing.expectEqual(@as(?u32, null), t.kv.max_kv);
    try testing.expect(t.layer_major);
    try testing.expectEqual(@as(usize, 9), t.deferredLevers().len);
}

test "dsv41 routes: every lever the build cannot run the same way refuses, by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { text: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .text = "MTPLX_DSV41_SELECTED_KEY=1", .err = error.UnknownLever },
        .{ .text = "MTPLX_DSV41_HEAD_MODE=q8", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_DTYPE=bf16", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK=512", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_PATH=fast", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_WINDOW_RING=2", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL=1", .err = error.LeverNeedsKernel },
        .{ .text = "MTPLX_DSV41_HC_PREMIX_KERNEL=on", .err = error.LeverNeedsKernel },
        .{ .text = "MTPLX_DSV41_ATTN_WO_A_DIRECT=1", .err = error.LeverNeedsKernel },
    };
    for (cases, 0..) |cs, i| {
        var diag: v41.Diag = .{};
        if (parse(try splitPairs(a, cs.text), &diag)) |_| {
            std.debug.print("case {d}: parsed, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            try testing.expectEqual(cs.err, e);
            try testing.expect(diag.message().len > 0);
        }
    }
    // KV precedence and caps: KV_BOUNDED over WINDOW_RING; the ring's MAXKV stands in.
    const t = try parse(try splitPairs(a, "MTPLX_DSV41_WINDOW_RING=1 MTPLX_DSV41_KV_BOUNDED=1 MTPLX_DSV41_WINDOW_RING_MAXKV=17664 MTPLX_DSV41_PREFILL_CHUNK=32 PATH=/bin"), null);
    try testing.expectEqual(kvc.Route.bounded, t.kv.route);
    try testing.expectEqual(@as(?u32, 17664), t.kv.max_kv);
    try testing.expectEqual(@as(?i64, 32), t.prefill_chunk);
    const stock = try parse(&.{}, null);
    try testing.expectEqual(kvc.Route.full_history, stock.kv.route);
    try testing.expectEqual(graph.Routes{}, stock.routes);
}
