//! Host half of the expert streamer's lookahead (the lookahead4 lane's
//! Selector and pre-read list). The router barrier also evaluates the next
//! routed layer's own gate on this layer's router input, giving M rows of
//! per-expert scores; per row the top-K experts by score are kept iff their
//! score is at least the row's sixth score minus tau, the union is ordered by
//! (best rank over rows, then score), residents are dropped, and the first
//! `budget` become speculative whole-record reads. A score tie breaks by expert
//! id (the Python lane leaves tied experts in numpy's argpartition order).
//! Pure: it chooses records and reads nothing.

const std = @import("std");
const mlx = @import("mlx.zig");
const expert_policy = @import("expert_policy.zig");

const LayerPolicy = expert_policy.LayerPolicy;

/// The routed top-k: tau is measured from each row's k-th score.
pub const routed_top_k = 6;
pub const min_k = 6;
pub const max_k = 12;
pub const max_budget = 4;
/// Verify rows per layer call.
pub const max_rows = 8;
/// Certain misses handed to one pre-read; the re-validation candidate cap.
pub const max_candidates = 64;

const Entry = struct { rank: u8, score: f32, expert: u16 };

pub const Selector = struct {
    n_experts: u32,
    k: u32,
    /// inf keeps each row's plain top-K.
    tau: f32,
    budget: u32,
    seen: std.DynamicBitSetUnmanaged,

    pub fn init(a: std.mem.Allocator, n_experts: u32, k: u32, tau: f32, budget: u32) !Selector {
        if (k < min_k or k > max_k or k > n_experts or budget == 0 or budget > max_budget or std.math.isNan(tau) or tau == -std.math.inf(f32))
            return error.InvalidSelector;
        return .{ .n_experts = n_experts, .k = k, .tau = tau, .budget = budget, .seen = try std.DynamicBitSetUnmanaged.initEmpty(a, n_experts) };
    }

    pub fn deinit(self: *Selector, a: std.mem.Allocator) void {
        self.seen.deinit(a);
        self.* = undefined;
    }

    /// The records to read ahead for the next layer: `scores` is rows x
    /// n_experts (row-major f32), `next` that layer's residency. Returns the
    /// first `out.len` non-resident experts of the ordered union.
    pub fn select(self: *Selector, scores: []const f32, next: *const LayerPolicy, out: []u16) []u16 {
        std.debug.assert(scores.len % self.n_experts == 0 and scores.len / self.n_experts <= max_rows);
        var entries: [max_rows * max_k]Entry = undefined;
        const ordered = self.order(scores, &entries);
        var n: usize = 0;
        for (ordered) |en| {
            if (self.seen.isSet(en.expert)) continue;
            self.seen.set(en.expert);
            if (next.slotOf(en.expert) != null) continue;
            out[n] = en.expert;
            n += 1;
            if (n == out.len) break;
        }
        for (ordered) |en| self.seen.unset(en.expert);
        return out[0..n];
    }

    /// Per row the top-k by (score desc, expert asc), kept iff score >= the
    /// row's sixth score - tau (f32); then a stable sort by (rank, score desc),
    /// so equal keys keep row-major order, as the lane's lexsort does.
    fn order(self: *const Selector, scores: []const f32, entries: *[max_rows * max_k]Entry) []Entry {
        const e_n = self.n_experts;
        const rows = scores.len / e_n;
        const finite = std.math.isFinite(self.tau);
        var n: usize = 0;
        for (0..rows) |r| {
            const row = scores[r * e_n ..][0..e_n];
            var top: [max_k]u16 = undefined;
            var cnt: usize = 0;
            for (row, 0..) |v, ei| {
                const e: u16 = @intCast(ei);
                if (cnt == self.k and !better(v, e, row[top[cnt - 1]], top[cnt - 1])) continue;
                var i = if (cnt < self.k) cnt else cnt - 1;
                if (cnt < self.k) cnt += 1;
                while (i > 0 and better(v, e, row[top[i - 1]], top[i - 1])) : (i -= 1) top[i] = top[i - 1];
                top[i] = e;
            }
            const threshold: f32 = if (finite) row[top[routed_top_k - 1]] - self.tau else -std.math.inf(f32);
            for (top[0..cnt], 0..) |e, rank| {
                const v = row[e];
                if (finite and !(v >= threshold)) continue;
                entries[n] = .{ .rank = @intCast(rank), .score = v, .expert = e };
                n += 1;
            }
        }
        const out = entries[0..n];
        std.sort.insertion(Entry, out, {}, struct {
            fn less(_: void, a: Entry, b: Entry) bool {
                return if (a.rank != b.rank) a.rank < b.rank else a.score > b.score;
            }
        }.less);
        return out;
    }

    /// The layer call's certain misses: its unique routed experts not resident
    /// in `layer`, in route order, at most `out.len` (they are its plan's misses).
    pub fn certainMisses(self: *Selector, ids: []const u16, layer: *const LayerPolicy, out: []u16) []u16 {
        var n: usize = 0;
        for (ids) |e| {
            if (self.seen.isSet(e)) continue;
            self.seen.set(e);
            if (layer.slotOf(e) == null and n < out.len) {
                out[n] = e;
                n += 1;
            }
        }
        for (ids) |e| self.seen.unset(e);
        return out[0..n];
    }
};

fn better(v: f32, e: u16, w: f32, f: u16) bool {
    return v > w or (v == w and e < f);
}

/// The predictor, folded by the model into the router barrier's eval: the
/// next routed layer's own gate on this layer's router input,
/// s = sqrt(softplus(f32(x @ w^T))) + bias (x [M, hidden], w [n_experts,
/// hidden], bias [n_experts] f32). The lane compiles this chain; the op chain
/// may differ in the last bits, which only changes what is read ahead.
pub fn nextLayerScores(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, stream: mlx.mlx_stream) !mlx.mlx_array {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, w, stream));
    var z = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_matmul(&z, x, wt, stream));
    var zf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zf);
    try mlx.check(mlx.mlx_astype(&zf, z, .float32, stream));
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var sp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp);
    try mlx.check(mlx.mlx_logaddexp(&sp, zf, zero, stream));
    var root = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(root);
    try mlx.check(mlx.mlx_sqrt(&root, sp, stream));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_add(&scores, root, bias, stream));
    return scores;
}

// ── Tests ──

const testing = std.testing;

/// A policy holding exactly `residents` (prefill-admitted, capacity = their count).
fn policyWith(a: std.mem.Allocator, n_experts: u32, residents: []const u16) !LayerPolicy {
    var p = try LayerPolicy.init(a, n_experts, @intCast(residents.len));
    errdefer p.deinit(a);
    if (residents.len > 0) {
        var plan: expert_policy.Plan = .{};
        p.plan(residents, .prefill, &plan);
    }
    return p;
}

fn rowsOf(comptime n_experts: usize, comptime rows: usize, spec: [rows][]const struct { u16, f32 }) [rows * n_experts]f32 {
    var s: [rows * n_experts]f32 = @splat(0);
    for (spec, 0..) |row, r| for (row) |ev| {
        s[r * n_experts + ev[0]] = ev[1];
    };
    return s;
}

test "dsv41 lookahead: the union is ordered by best rank over rows, then score" {
    const a = testing.allocator;
    var sel = try Selector.init(a, 16, 8, std.math.inf(f32), 4);
    defer sel.deinit(a);
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    const s = rowsOf(16, 2, .{
        &.{ .{ 3, 0.9 }, .{ 7, 0.8 }, .{ 1, 0.7 }, .{ 9, 0.6 }, .{ 4, 0.5 }, .{ 2, 0.4 }, .{ 11, 0.3 }, .{ 5, 0.2 }, .{ 13, 0.1 } },
        &.{ .{ 7, 0.95 }, .{ 5, 0.85 }, .{ 3, 0.75 }, .{ 12, 0.65 }, .{ 1, 0.55 }, .{ 0, 0.45 }, .{ 2, 0.35 }, .{ 8, 0.25 }, .{ 14, 0.15 } },
    });
    // rank 0: 7 (.95), 3 (.9); rank 1: 5 (.85), 7; rank 2: 3, 1; rank 3: 12, 9; rank 4: 1, 4; rank 5: 0, 2;
    // rank 6: 2, 11; rank 7: 5, 8. Expert 13 and 14 sit at rank 8: outside K = 8.
    var out: [max_candidates]u16 = undefined;
    try testing.expectEqualSlices(u16, &.{ 7, 3, 5, 1, 12, 9, 4, 0, 2, 11, 8 }, sel.select(&s, &none, &out));
    try testing.expectEqualSlices(u16, &.{ 7, 3, 5, 1 }, sel.select(&s, &none, out[0..sel.budget]));
    // Residents are skipped, not counted against the budget.
    var res = try policyWith(a, 16, &.{ 3, 5, 12 });
    defer res.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 7, 1, 9, 4 }, sel.select(&s, &res, out[0..sel.budget]));
}

test "dsv41 lookahead: tau keeps a candidate within tau of its row's sixth score" {
    const a = testing.allocator;
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    const s = rowsOf(16, 1, .{&.{ .{ 0, 1.0 }, .{ 1, 0.9 }, .{ 2, 0.8 }, .{ 3, 0.7 }, .{ 4, 0.6 }, .{ 5, 0.5 }, .{ 6, 0.45 }, .{ 7, 0.39 }, .{ 8, 0.38 } }});
    var out: [max_candidates]u16 = undefined;
    var t = try Selector.init(a, 16, 8, 0.1, 4);
    defer t.deinit(a);
    // Sixth score 0.5, threshold 0.4 in f32: 0.45 is kept, 0.39 is not; 0.38 (rank 8) is outside K.
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 5, 6 }, t.select(&s, &none, &out));
    var z = try Selector.init(a, 16, 8, 0.0, 4);
    defer z.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 5 }, z.select(&s, &none, &out));
    var neg = try Selector.init(a, 16, 8, -0.15, 4);
    defer neg.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, neg.select(&s, &none, &out));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 5, 0, 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 13, 0, 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, std.math.nan(f32), 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, -std.math.inf(f32), 2));
    try testing.expectError(error.InvalidSelector, Selector.init(a, 16, 8, 0, 5));
}

test "dsv41 lookahead: a score tie breaks by expert id" {
    const a = testing.allocator;
    var none = try policyWith(a, 16, &.{});
    defer none.deinit(a);
    var sel = try Selector.init(a, 16, 6, std.math.inf(f32), 4);
    defer sel.deinit(a);
    const s = rowsOf(16, 1, .{&.{ .{ 9, 2.0 }, .{ 15, 2.0 }, .{ 4, 2.0 }, .{ 1, 1.0 }, .{ 10, 0.5 }, .{ 3, 0.5 }, .{ 7, 0.5 } }});
    var out: [max_candidates]u16 = undefined;
    // Three tied at the top, three tied for the last two places: the lower ids win both.
    try testing.expectEqualSlices(u16, &.{ 4, 9, 15, 1, 3, 7 }, sel.select(&s, &none, &out));
}

test "dsv41 lookahead: certain misses are the route's unique non-residents in route order" {
    const a = testing.allocator;
    var sel = try Selector.init(a, 16, 8, std.math.inf(f32), 2);
    defer sel.deinit(a);
    var res = try policyWith(a, 16, &.{ 2, 4 });
    defer res.deinit(a);
    var out: [max_candidates]u16 = undefined;
    const ids = [_]u16{ 5, 2, 9, 5, 4, 1, 9, 12 };
    try testing.expectEqualSlices(u16, &.{ 5, 9, 1, 12 }, sel.certainMisses(&ids, &res, &out));
    try testing.expectEqualSlices(u16, &.{ 5, 9 }, sel.certainMisses(&ids, &res, out[0..2]));
    // The plan's misses are the same experts in the same order (decode).
    var plan: expert_policy.Plan = .{};
    try res.grow(3);
    res.plan(&ids, .decode, &plan);
    try testing.expectEqualSlices(u16, &.{ 5, 9, 1, 12 }, plan.missesOf());
}

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

// DSV41_PHASE0B_MLX=1, inside a guarded window (creates MLX arrays).
test "dsv41 lookahead 0b: the predictor graph scores like its host reference" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const m = 3;
    const d = 64;
    const e_n = 16;
    var rng = std.Random.DefaultPrng.init(3);
    const rand = rng.random();
    var xb: [m * d]u16 = undefined;
    var wb: [e_n * d]u16 = undefined;
    var bias: [e_n]f32 = undefined;
    for (&xb) |*v| v.* = bf16Bits(rand.float(f32) * 2 - 1);
    for (&wb) |*v| v.* = bf16Bits((rand.float(f32) * 2 - 1) * 0.5);
    for (&bias) |*v| v.* = rand.float(f32) * 0.2;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    const x = mlx.mlx_array_new_data(&xb, &[_]c_int{ m, d }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const w = mlx.mlx_array_new_data(&wb, &[_]c_int{ e_n, d }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    const b = mlx.mlx_array_new_data(&bias, &[_]c_int{e_n}, 1, .float32);
    defer _ = mlx.mlx_array_free(b);
    const s = try nextLayerScores(x, w, b, stream);
    defer _ = mlx.mlx_array_free(s);
    try mlx.check(mlx.mlx_array_eval(s));
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(s));
    try testing.expectEqual(@as(usize, m * e_n), mlx.mlx_array_size(s));
    const got = (mlx.mlx_array_data_float32(s) orelse return error.MlxNoData)[0 .. m * e_n];
    // Host: the product rounded to bf16 (the matmul's output dtype), then the f32 tail.
    var worst: f32 = 0;
    for (0..m) |r| for (0..e_n) |e| {
        var acc: f32 = 0;
        for (0..d) |k| acc += fromBf16(xb[r * d + k]) * fromBf16(wb[e * d + k]);
        const z = fromBf16(bf16Bits(acc));
        const want = @sqrt(@max(z, 0) + std.math.log1p(@exp(-@abs(z)))) + bias[e];
        worst = @max(worst, @abs(got[r * e_n + e] - want));
    };
    std.debug.print("predictor graph: worst |mlx - host| {d:.6}\n", .{worst});
    try testing.expect(worst < 2e-2);
}

// DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py> (its scores file beside it)
test "dsv41 lookahead: the recorded trace selects exactly like the Python lane" {
    const path = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const Call = struct { ids: []const u16, pre: []const u16, sel: []const []const u16, cand: []const []const u16 };
    const Config = struct { k: u32, tau: ?f32, budget: u32 };
    const TieCall = struct { call: u32, config: u32, sel: []const u16, cand: []const u16 };
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        rows: []const u32,
        configs: []const Config,
        scores_file: []const u8,
        scores_rows: u64,
        tie_calls: []const TieCall,
        calls: []const Call,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var dir_buf: [1024]u8 = undefined;
    const scores_path = try std.fmt.bufPrint(&dir_buf, "{s}/{s}", .{ std.fs.path.dirname(path) orelse ".", f.scores_file });
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, scores_path, a, .limited(64 << 20));
    defer a.free(raw);
    try testing.expectEqual(f.scores_rows * f.experts * 4, raw.len);
    const scores = try a.alloc(f32, raw.len / 4);
    defer a.free(scores);
    for (scores, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));

    const policies = try a.alloc(LayerPolicy, f.layers);
    defer a.free(policies);
    var n_init: usize = 0;
    defer for (policies[0..n_init]) |*p| p.deinit(a);
    var plan: expert_policy.Plan = .{};
    for (0..f.layers) |l| {
        policies[l] = try LayerPolicy.init(a, f.experts, f.prefill_capacity[l]);
        n_init += 1;
        const p = &policies[l];
        p.prepareSeed(f.resident0[l]);
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        var i: usize = 0;
        while (i < sorted.len) : (i += f.transient) p.plan(sorted[i..@min(i + f.transient, sorted.len)], .prefill, &plan);
    }
    for (policies, f.decode_capacity) |*p, cap| try p.grow(cap);

    var sels: [3]Selector = undefined;
    try testing.expectEqual(@as(usize, 3), f.configs.len);
    var n_sel: usize = 0;
    defer for (sels[0..n_sel]) |*s| s.deinit(a);
    for (f.configs, &sels) |cf, *s| {
        s.* = try Selector.init(a, f.experts, cf.k, cf.tau orelse std.math.inf(f32), cf.budget);
        n_sel += 1;
    }
    var out: [max_candidates]u16 = undefined;
    var at: usize = 0;
    var n_pre: usize = 0;
    var n_equal: usize = 0;
    var n_tie: usize = 0;
    var tie_i: usize = 0;
    for (f.calls, 0..) |call, ci| {
        const cycle = ci / f.layers;
        const l = ci % f.layers;
        const pre = sels[0].certainMisses(call.ids, &policies[l], &out);
        try testing.expectEqualSlices(u16, call.pre, pre);
        n_pre += pre.len;
        if (l + 1 < f.layers) {
            const m = f.rows[cycle];
            const s = scores[at * f.experts ..][0 .. m * f.experts];
            at += m;
            for (&sels, 0..) |*sel, k| {
                var want_sel = call.sel[k];
                var want_cand = call.cand[k];
                if (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci and f.tie_calls[tie_i].config == k) {
                    want_sel = f.tie_calls[tie_i].sel;
                    want_cand = f.tie_calls[tie_i].cand;
                    tie_i += 1;
                    n_tie += 1;
                } else n_equal += 1;
                testing.expectEqualSlices(u16, want_sel, sel.select(s, &policies[l + 1], out[0..sel.budget])) catch |e| {
                    std.debug.print("call {d} (cycle {d}, layer {d}) config {d}: selection differs\n", .{ ci, cycle, l, k });
                    return e;
                };
                try testing.expectEqualSlices(u16, want_cand, sel.select(s, &policies[l + 1], &out));
            }
        } else for (call.sel) |w| try testing.expectEqual(@as(usize, 0), w.len);
        policies[l].plan(call.ids, .decode, &plan);
    }
    try testing.expectEqual(f.tie_calls.len, tie_i);
    try testing.expectEqual(f.scores_rows, at);
    std.debug.print("lookahead parity: {d} layer calls, {d} certain misses equal the lane's; {d} x 2 selections equal the lane's, {d} x 2 equal its tie-broken-by-id variant (exact score ties)\n", .{ f.calls.len, n_pre, n_equal, n_tie });
}
