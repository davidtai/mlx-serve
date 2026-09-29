//! Per-layer expert residency policy of the expert streamer. Pure: it plans
//! which slot serves each routed expert and which records must be read, and
//! performs neither. A port of the Python stack's LayerExpertSlotBank in the
//! tier's configuration: one request, one resident pool per layer plus a
//! shared transient scratch, 2Q prefill admission behind the prompt-frequency
//! seed, transition-window decode admission. That class is the oracle
//! (R/exl3/runtime/dump_phase1_route_fixture.py replays a recorded trace
//! through it for the parity test below).

const std = @import("std");

pub const Phase = enum { prefill, decode };

/// Widest route a plan holds: 8 verify rows x top-6 routed experts.
pub const max_route_ids = 48;
pub const no_expert: u16 = std.math.maxInt(u16);
pub const no_slot: u32 = std.math.maxInt(u32);

/// Decode admission: routes remembered by the frequency term, and the weights of
/// the transition prediction, window frequency and recency terms.
const window_limit = 16;
const w_prediction: f32 = 0.7;
const w_frequency: f32 = 0.2;
const w_recency: f32 = 0.1;

pub const Load = struct { expert: u16, slot: u32, persistent: bool };
pub const Eviction = struct { slot: u32, previous: u16, next: u16 };

/// One route's decisions, in the orders of the Python RoutePlan.
pub const Plan = struct {
    phase: Phase = .decode,
    n_ids: u32 = 0,
    /// Per routed id, its slot: persistent slots are [0, capacity), the
    /// transient scratch follows them.
    slots: [max_route_ids]u32 = undefined,
    n_hits: u32 = 0,
    /// Resident experts, in first-appearance order.
    hits: [max_route_ids]u16 = undefined,
    n_misses: u32 = 0,
    /// Non-resident experts, in admission order.
    misses: [max_route_ids]u16 = undefined,
    n_loads: u32 = 0,
    /// Persistent loads in admission order, then transient loads.
    loads: [max_route_ids]Load = undefined,
    /// The persistent loads: `loads[0..n_persistent]`.
    n_persistent: u32 = 0,
    n_evictions: u32 = 0,
    evictions: [max_route_ids]Eviction = undefined,

    pub fn slotsOf(p: *const Plan) []const u32 {
        return p.slots[0..p.n_ids];
    }
    pub fn hitsOf(p: *const Plan) []const u16 {
        return p.hits[0..p.n_hits];
    }
    pub fn missesOf(p: *const Plan) []const u16 {
        return p.misses[0..p.n_misses];
    }
    pub fn loadsOf(p: *const Plan) []const Load {
        return p.loads[0..p.n_loads];
    }
    pub fn evictionsOf(p: *const Plan) []const Eviction {
        return p.evictions[0..p.n_evictions];
    }
};

pub const LayerPolicy = struct {
    n_experts: u32,
    /// Persistent slots; raised once, at the decode transition.
    capacity: u32,
    occupancy: u32 = 0,
    /// [n_experts]: slots beyond `capacity` stay empty.
    slot_to_expert: []u16,
    expert_to_slot: []u32,
    // Prefill: 2Q single pool behind the prompt-frequency seed.
    prefill_freq: []u32,
    seed: std.DynamicBitSetUnmanaged,
    protected: std.DynamicBitSetUnmanaged,
    /// Pool clock stamp of each resident expert (0 = none).
    recency: []u64,
    clock: u64 = 0,
    // Decode: one-step transitions + a bounded route window.
    epoch: i64 = 0,
    last_used: []i64,
    /// [n_experts * n_experts]: transitions previous route -> current route.
    counts: []f32,
    denominators: []f32,
    window_freq: []f32,
    window: [window_limit][max_route_ids]u16 = undefined,
    window_len: [window_limit]u8 = undefined,
    window_head: u32 = 0,
    window_count: u32 = 0,
    // Per-plan scratch.
    in_route: std.DynamicBitSetUnmanaged,
    retained: std.DynamicBitSetUnmanaged,
    scores: []f32,
    candidates: []u16,
    victims: []u32,
    available: []u32,
    admission: []u32,

    pub fn init(a: std.mem.Allocator, n_experts: u32, capacity: u32) !LayerPolicy {
        if (n_experts == 0 or n_experts >= no_expert or capacity > n_experts) return error.InvalidCapacity;
        const n: usize = n_experts;
        var p: LayerPolicy = undefined;
        p = .{
            .n_experts = n_experts,
            .capacity = capacity,
            .slot_to_expert = try a.alloc(u16, n),
            .expert_to_slot = undefined,
            .prefill_freq = undefined,
            .seed = undefined,
            .protected = undefined,
            .recency = undefined,
            .last_used = undefined,
            .counts = undefined,
            .denominators = undefined,
            .window_freq = undefined,
            .in_route = undefined,
            .retained = undefined,
            .scores = undefined,
            .candidates = undefined,
            .victims = undefined,
            .available = undefined,
            .admission = undefined,
        };
        errdefer a.free(p.slot_to_expert);
        p.expert_to_slot = try a.alloc(u32, n);
        errdefer a.free(p.expert_to_slot);
        p.prefill_freq = try a.alloc(u32, n);
        errdefer a.free(p.prefill_freq);
        p.seed = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.seed.deinit(a);
        p.protected = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.protected.deinit(a);
        p.recency = try a.alloc(u64, n);
        errdefer a.free(p.recency);
        p.last_used = try a.alloc(i64, n);
        errdefer a.free(p.last_used);
        p.counts = try a.alloc(f32, n * n);
        errdefer a.free(p.counts);
        p.denominators = try a.alloc(f32, n);
        errdefer a.free(p.denominators);
        p.window_freq = try a.alloc(f32, n);
        errdefer a.free(p.window_freq);
        p.in_route = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.in_route.deinit(a);
        p.retained = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
        errdefer p.retained.deinit(a);
        p.scores = try a.alloc(f32, n);
        errdefer a.free(p.scores);
        p.candidates = try a.alloc(u16, n + max_route_ids);
        errdefer a.free(p.candidates);
        p.victims = try a.alloc(u32, n);
        errdefer a.free(p.victims);
        p.available = try a.alloc(u32, n);
        errdefer a.free(p.available);
        p.admission = try a.alloc(u32, n);
        @memset(p.slot_to_expert, no_expert);
        @memset(p.expert_to_slot, no_slot);
        @memset(p.prefill_freq, 0);
        @memset(p.recency, 0);
        @memset(p.last_used, -1);
        @memset(p.counts, 0);
        @memset(p.denominators, 0);
        @memset(p.window_freq, 0);
        @memset(p.admission, no_slot);
        return p;
    }

    pub fn deinit(p: *LayerPolicy, a: std.mem.Allocator) void {
        a.free(p.slot_to_expert);
        a.free(p.expert_to_slot);
        a.free(p.prefill_freq);
        p.seed.deinit(a);
        p.protected.deinit(a);
        a.free(p.recency);
        a.free(p.last_used);
        a.free(p.counts);
        a.free(p.denominators);
        a.free(p.window_freq);
        p.in_route.deinit(a);
        p.retained.deinit(a);
        a.free(p.scores);
        a.free(p.candidates);
        a.free(p.victims);
        a.free(p.available);
        a.free(p.admission);
        p.* = undefined;
    }

    pub fn slotOf(p: *const LayerPolicy, expert: u16) ?u32 {
        const s = p.expert_to_slot[expert];
        return if (s == no_slot) null else s;
    }

    /// prepare_prefill_seed: counts the prompt's routed ids, then re-protects
    /// the resident part of its top-(capacity - protected) and marks the rest to
    /// be admitted first, protected, by the prefill routes.
    pub fn prepareSeed(p: *LayerPolicy, a: std.mem.Allocator, ids: []const u16) !void {
        for (ids) |e| p.prefill_freq[e] += 1;
        const empty: i64 = @as(i64, p.capacity) - @as(i64, @intCast(p.protected.count()));
        p.seed.unsetAll();
        if (empty <= 0) return;
        const counts = try a.alloc(u32, p.n_experts);
        defer a.free(counts);
        @memset(counts, 0);
        var n_unique: usize = 0;
        for (ids) |e| {
            if (counts[e] == 0) {
                p.candidates[n_unique] = e;
                n_unique += 1;
            }
            counts[e] += 1;
        }
        const ranked = p.candidates[0..n_unique];
        std.sort.pdq(u16, ranked, @as([]const u32, counts), struct {
            fn lessThan(c: []const u32, x: u16, y: u16) bool {
                return if (c[x] != c[y]) c[x] > c[y] else x < y;
            }
        }.lessThan);
        const chosen = ranked[0..@min(ranked.len, @as(usize, @intCast(empty)))];
        // Resident choices: re-protected in ascending count order (stable).
        var n_res: usize = 0;
        for (chosen) |e| if (p.expert_to_slot[e] != no_slot) {
            p.victims[n_res] = e;
            n_res += 1;
        };
        const res = p.victims[0..n_res];
        std.sort.insertion(u32, res, @as([]const u32, counts), struct {
            fn lessThan(c: []const u32, x: u32, y: u32) bool {
                return c[x] < c[y];
            }
        }.lessThan);
        for (res) |e| {
            p.protected.set(e);
            p.clock += 1;
            p.recency[e] = p.clock;
        }
        for (chosen) |e| if (p.expert_to_slot[e] == no_slot) p.seed.set(e);
    }

    /// The one phase change: `capacity` persistent slots from now on, the new
    /// ones empty.
    pub fn grow(p: *LayerPolicy, capacity: u32) !void {
        if (capacity < p.capacity or capacity > p.n_experts) return error.InvalidCapacity;
        p.capacity = capacity;
    }

    /// Forgets a resident expert (its record failed to load).
    pub fn invalidate(p: *LayerPolicy, expert: u16) void {
        const s = p.expert_to_slot[expert];
        if (s == no_slot) return;
        p.slot_to_expert[s] = no_expert;
        p.expert_to_slot[expert] = no_slot;
        p.occupancy -= 1;
        p.protected.unset(expert);
        p.recency[expert] = 0;
    }

    /// Resolves one route. `ids` holds at most `max_route_ids` expert ids
    /// (< n_experts, repeats allowed: M rows x top-k); the transient scratch
    /// must hold `max_route_ids` slots.
    pub fn plan(p: *LayerPolicy, ids: []const u16, phase: Phase, out: *Plan) void {
        std.debug.assert(ids.len > 0 and ids.len <= max_route_ids);
        out.* = .{ .phase = phase, .n_ids = @intCast(ids.len) };
        var unique_buf: [max_route_ids]u16 = undefined;
        var n_unique: usize = 0;
        for (ids) |e| {
            std.debug.assert(e < p.n_experts);
            if (p.in_route.isSet(e)) continue;
            p.in_route.set(e);
            unique_buf[n_unique] = e;
            n_unique += 1;
        }
        const unique = unique_buf[0..n_unique];
        defer for (unique) |e| p.in_route.unset(e);

        if (phase == .decode) {
            p.epoch += 1;
            for (ids) |e| p.last_used[e] = p.epoch;
            p.observe(unique);
        }
        for (unique) |e| {
            if (p.expert_to_slot[e] != no_slot) {
                out.hits[out.n_hits] = e;
                out.n_hits += 1;
            } else {
                out.misses[out.n_misses] = e;
                out.n_misses += 1;
            }
        }
        // Prefill hits refresh recency; decode hits touch only transition state.
        // (Python walks a set here: the order among one wave's hits can differ.)
        if (phase == .prefill) for (out.hitsOf()) |e| {
            p.clock += 1;
            p.recency[e] = p.clock;
        };

        var transient_buf: [max_route_ids]u16 = undefined;
        var n_transient: usize = 0;
        switch (phase) {
            .decode => {
                const admissions = p.transitionAdmissions(out);
                for (out.missesOf()) |e| {
                    const slot = admissions[e];
                    if (slot == no_slot) {
                        transient_buf[n_transient] = e;
                        n_transient += 1;
                        continue;
                    }
                    p.assign(slot, e, out);
                    out.loads[out.n_loads] = .{ .expert = e, .slot = slot, .persistent = true };
                    out.n_loads += 1;
                }
                for (out.missesOf()) |e| admissions[e] = no_slot;
            },
            .prefill => {
                p.seedFirst(out);
                // `in_route` holds this route's experts: its hits and each miss
                // once admitted are never victims.
                for (out.missesOf()) |e| {
                    const is_seed = p.seed.isSet(e);
                    const slot = p.emptySlot() orelse p.probationVictim() orelse no_slot;
                    if (is_seed) p.seed.unset(e);
                    if (slot == no_slot) {
                        transient_buf[n_transient] = e;
                        n_transient += 1;
                        continue;
                    }
                    const victim = p.slot_to_expert[slot];
                    if (victim != no_expert) {
                        p.protected.unset(victim);
                        p.recency[victim] = 0;
                    }
                    p.assign(slot, e, out);
                    p.clock += 1;
                    p.recency[e] = p.clock;
                    p.protected.setValue(e, is_seed);
                    out.loads[out.n_loads] = .{ .expert = e, .slot = slot, .persistent = true };
                    out.n_loads += 1;
                }
            },
        }
        out.n_persistent = out.n_loads;
        for (transient_buf[0..n_transient], 0..) |e, k| {
            out.loads[out.n_loads] = .{ .expert = e, .slot = p.capacity + @as(u32, @intCast(k)), .persistent = false };
            out.n_loads += 1;
        }
        for (ids, 0..) |e, i| out.slots[i] = p.slotFor(e, out);
    }

    fn slotFor(p: *const LayerPolicy, e: u16, out: *const Plan) u32 {
        if (p.expert_to_slot[e] != no_slot) return p.expert_to_slot[e];
        for (out.loadsOf()) |l| if (l.expert == e) return l.slot;
        unreachable;
    }

    fn assign(p: *LayerPolicy, slot: u32, e: u16, out: *Plan) void {
        const previous = p.slot_to_expert[slot];
        if (previous != no_expert) {
            p.expert_to_slot[previous] = no_slot;
            out.evictions[out.n_evictions] = .{ .slot = slot, .previous = previous, .next = e };
            out.n_evictions += 1;
        } else p.occupancy += 1;
        p.slot_to_expert[slot] = e;
        p.expert_to_slot[e] = slot;
    }

    fn emptySlot(p: *const LayerPolicy) ?u32 {
        if (p.occupancy >= p.capacity) return null;
        for (p.slot_to_expert[0..p.capacity], 0..) |e, s| if (e == no_expert) return @intCast(s);
        return null;
    }

    /// The coldest probationary resident (lowest (recency, slot)) that this
    /// route does not hold; prefill never evicts a protected expert.
    fn probationVictim(p: *const LayerPolicy) ?u32 {
        var best: ?u32 = null;
        var best_recency: u64 = 0;
        for (p.slot_to_expert[0..p.capacity], 0..) |e, s| {
            if (e == no_expert or p.in_route.isSet(e) or p.protected.isSet(e)) continue;
            if (best == null or p.recency[e] < best_recency) {
                best = @intCast(s);
                best_recency = p.recency[e];
            }
        }
        return best;
    }

    /// Seed misses first, least frequent first (stable), then the rest.
    fn seedFirst(p: *const LayerPolicy, out: *Plan) void {
        var seeds: [max_route_ids]u16 = undefined;
        var rest: [max_route_ids]u16 = undefined;
        var ns: usize = 0;
        var nr: usize = 0;
        for (out.missesOf()) |e| {
            if (p.seed.isSet(e)) {
                seeds[ns] = e;
                ns += 1;
            } else {
                rest[nr] = e;
                nr += 1;
            }
        }
        if (ns == 0) return;
        std.sort.insertion(u16, seeds[0..ns], @as([]const u32, p.prefill_freq), struct {
            fn lessThan(f: []const u32, x: u16, y: u16) bool {
                return f[x] < f[y];
            }
        }.lessThan);
        @memcpy(out.misses[0..ns], seeds[0..ns]);
        @memcpy(out.misses[ns..][0..nr], rest[0..nr]);
    }

    /// Publishes one decode route (its unique experts) into the transition
    /// counts and the route window.
    fn observe(p: *LayerPolicy, current: []const u16) void {
        const n = p.n_experts;
        if (p.window_count > 0) {
            const prev_i = (p.window_head + p.window_count - 1) % window_limit;
            const prev = p.window[prev_i][0..p.window_len[prev_i]];
            for (prev) |r| {
                for (current) |c| p.counts[@as(usize, r) * n + c] += 1.0;
                p.denominators[r] += @floatFromInt(current.len);
            }
        }
        if (p.window_count == window_limit) {
            const old = p.window[p.window_head][0..p.window_len[p.window_head]];
            for (old) |e| p.window_freq[e] -= 1.0;
            p.window_head = (p.window_head + 1) % window_limit;
            p.window_count -= 1;
        }
        const at = (p.window_head + p.window_count) % window_limit;
        @memcpy(p.window[at][0..current.len], current);
        p.window_len[at] = @intCast(current.len);
        p.window_count += 1;
        for (current) |e| p.window_freq[e] += 1.0;
    }

    /// The causal decode scores (float32, in the Python evaluation order):
    /// 0.7 * transition prediction from the current route + 0.2 * window
    /// frequency / its max + 0.1 * 1 / (1 + epoch - last use).
    fn computeScores(p: *LayerPolicy) void {
        const n = p.n_experts;
        const cur_i = (p.window_head + p.window_count - 1) % window_limit;
        const current = p.window[cur_i][0..p.window_len[cur_i]];
        @memset(p.scores, 0);
        for (current) |r| {
            const d = p.denominators[r];
            if (!(d > 0)) continue;
            const row = p.counts[@as(usize, r) * n ..][0..n];
            for (p.scores, row) |*s, c| s.* += c / d;
        }
        var max_window: f32 = 1.0;
        for (p.window_freq) |f| max_window = @max(max_window, f);
        for (p.scores, 0..) |*s, e| {
            const lu = p.last_used[e];
            const recency: f32 = if (lu >= 0) @floatCast(1.0 / (1.0 + @as(f64, @floatFromInt(p.epoch - lu)))) else 0;
            s.* = w_prediction * s.* + w_frequency * (p.window_freq[e] / max_window) + w_recency * recency;
        }
    }

    /// (score, last use, prompt frequency, -expert): Python's rank tuple.
    fn rankLess(p: *const LayerPolicy, a: u16, b: u16) bool {
        if (p.scores[a] != p.scores[b]) return p.scores[a] < p.scores[b];
        if (p.last_used[a] != p.last_used[b]) return p.last_used[a] < p.last_used[b];
        if (p.prefill_freq[a] != p.prefill_freq[b]) return p.prefill_freq[a] < p.prefill_freq[b];
        return a > b;
    }

    /// One bounded cut over residents and misses: of everything that may
    /// change (empty slots + residents this route does not hold), keep the
    /// highest ranks. Returns `admission[expert]` = its persistent slot, or
    /// no_slot (served transient); callers reset the misses' entries.
    fn transitionAdmissions(p: *LayerPolicy, out: *const Plan) []u32 {
        const admission = p.admission;
        if (out.n_misses == 0) return admission;
        p.computeScores();
        const cap = p.capacity;
        var n_cand: usize = 0;
        var n_evictable: usize = 0;
        for (p.slot_to_expert[0..cap]) |e| {
            if (e == no_expert or p.in_route.isSet(e)) continue;
            p.candidates[n_cand] = e;
            n_cand += 1;
            n_evictable += 1;
        }
        const free_budget = cap - p.occupancy;
        var n_empty: usize = 0;
        for (p.slot_to_expert[0..cap], 0..) |e, s| {
            if (n_empty == free_budget) break;
            if (e == no_expert) {
                p.available[n_empty] = @intCast(s);
                n_empty += 1;
            }
        }
        const adjustable = n_empty + n_evictable;
        if (adjustable == 0) return admission;
        for (out.missesOf()) |e| {
            p.candidates[n_cand] = e;
            n_cand += 1;
        }
        const keep = @min(adjustable, n_cand);
        const cands = p.candidates[0..n_cand];
        std.sort.pdq(u16, cands, @as(*const LayerPolicy, p), struct {
            fn greater(pp: *const LayerPolicy, a: u16, b: u16) bool {
                return pp.rankLess(b, a);
            }
        }.greater);
        for (cands[0..keep]) |e| p.retained.set(e);
        defer for (cands[0..keep]) |e| p.retained.unset(e);

        var n_victims: usize = 0;
        for (p.slot_to_expert[0..cap], 0..) |e, s| {
            if (e == no_expert or p.in_route.isSet(e) or p.retained.isSet(e)) continue;
            p.victims[n_victims] = @intCast(s);
            n_victims += 1;
        }
        std.sort.pdq(u32, p.victims[0..n_victims], @as(*const LayerPolicy, p), struct {
            fn less(pp: *const LayerPolicy, a: u32, b: u32) bool {
                return pp.rankLess(pp.slot_to_expert[a], pp.slot_to_expert[b]);
            }
        }.less);
        @memcpy(p.available[n_empty..][0..n_victims], p.victims[0..n_victims]);
        var next: usize = 0;
        for (out.missesOf()) |e| {
            if (!p.retained.isSet(e)) continue;
            admission[e] = p.available[next];
            next += 1;
        }
        return admission;
    }
};

/// Python's _bounded_decode_miss_route_parts over records already in
/// placement order: parts of at most `per_part`, each cut moved back to the
/// last physical gap inside its window so a contiguous run stays whole.
/// Writes each part's end index; returns them.
pub fn boundedParts(offsets: []const u64, lengths: []const u64, per_part: u32, ends: []u32) []u32 {
    const n = offsets.len;
    var n_parts: usize = 0;
    var start: usize = 0;
    while (start < n) {
        var end = @min(start + per_part, n);
        if (end < n) {
            var gap: ?usize = null;
            var i = start + 1;
            while (i <= end) : (i += 1) {
                if (offsets[i - 1] + lengths[i - 1] != offsets[i]) gap = i;
            }
            if (gap) |g| end = g;
        }
        ends[n_parts] = @intCast(end);
        n_parts += 1;
        start = end;
    }
    return ends[0..n_parts];
}

// ── Tests ──

const testing = std.testing;
const expert_bank = @import("expert_bank.zig");

/// Every plan leaves a bijective residency map and serves each id exactly as
/// its hit or load says.
fn checkPlan(p: *const LayerPolicy, ids: []const u16, out: *const Plan, transient: u32) !void {
    var occupied: u32 = 0;
    for (p.slot_to_expert[0..p.n_experts], 0..) |e, s| {
        if (e == no_expert) continue;
        try testing.expect(s < p.capacity);
        try testing.expectEqual(@as(u32, @intCast(s)), p.expert_to_slot[e]);
        occupied += 1;
    }
    try testing.expectEqual(p.occupancy, occupied);
    try testing.expect(p.occupancy <= p.capacity);
    for (out.hitsOf()) |h| try testing.expect(p.expert_to_slot[h] != no_slot);
    var transient_seen: [max_route_ids]bool = @splat(false);
    for (out.loadsOf()) |l| {
        if (l.persistent) {
            try testing.expectEqual(l.slot, p.expert_to_slot[l.expert]);
        } else {
            try testing.expect(l.slot >= p.capacity and l.slot < p.capacity + transient);
            try testing.expect(!transient_seen[l.slot - p.capacity]);
            transient_seen[l.slot - p.capacity] = true;
            try testing.expectEqual(no_slot, p.expert_to_slot[l.expert]);
        }
    }
    for (out.evictionsOf()) |ev| {
        try testing.expectEqual(no_slot, p.expert_to_slot[ev.previous]);
        for (out.hitsOf()) |h| try testing.expect(h != ev.previous);
    }
    try testing.expectEqual(out.n_hits + out.n_misses, @as(u32, @intCast(countUnique(ids))));
    try testing.expectEqual(out.n_misses, out.n_loads);
    for (ids, out.slotsOf()) |e, s| {
        if (p.expert_to_slot[e] != no_slot) {
            try testing.expectEqual(p.expert_to_slot[e], s);
        } else {
            var found = false;
            for (out.loadsOf()) |l| if (l.expert == e) {
                try testing.expectEqual(l.slot, s);
                found = true;
            };
            try testing.expect(found);
        }
    }
}

fn countUnique(ids: []const u16) usize {
    var n: usize = 0;
    for (ids, 0..) |e, i| {
        if (std.mem.indexOfScalar(u16, ids[0..i], e) == null) n += 1;
    }
    return n;
}

test "dsv41 policy: decode keeps what the transitions predict" {
    // One persistent slot. After 1 -> 2 has been seen, a route of [1] finds 2
    // predicted next and serves 1 from the transient scratch instead.
    var p = try LayerPolicy.init(testing.allocator, 8, 1);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{1}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 1, .slot = 0, .persistent = true }}, out.loadsOf());
    // [2]: prediction 0 for both; window 0.2 each; recency 1/2 vs 1 -> 2 wins.
    p.plan(&.{2}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 2, .slot = 0, .persistent = true }}, out.loadsOf());
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 0, .previous = 1, .next = 2 }}, out.evictionsOf());
    // [1]: 2 scores 0.7 + 0.1 + 0.05, 1 scores 0.2 + 0.1 -> 1 goes transient.
    p.plan(&.{ 1, 1 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 1, .slot = 1, .persistent = false }}, out.loadsOf());
    try testing.expectEqualSlices(u32, &.{ 1, 1 }, out.slotsOf());
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
    try testing.expectEqual(@as(?u32, 0), p.slotOf(2));
}

test "dsv41 policy: decode fills empty slots in slot order and never evicts a hit" {
    var p = try LayerPolicy.init(testing.allocator, 32, 4);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{ 7, 3, 7 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 7, .slot = 0, .persistent = true },
        .{ .expert = 3, .slot = 1, .persistent = true },
    }, out.loadsOf());
    try testing.expectEqualSlices(u32, &.{ 0, 1, 0 }, out.slotsOf());
    // A deterministic pseudo-random trace: every plan keeps the invariants.
    var rng = std.Random.DefaultPrng.init(41);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    for (0..400) |step| {
        const n = r.intRangeAtMost(usize, 1, 18);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, if (step % 3 == 0) 32 else 12));
        const before = p.slot_to_expert[0..4].*;
        p.plan(ids[0..n], .decode, &out);
        try checkPlan(&p, ids[0..n], &out, max_route_ids);
        for (out.hitsOf()) |h| try testing.expect(std.mem.indexOfScalar(u16, &before, h) != null);
    }
}

test "dsv41 policy: prefill admits the seed first and never evicts it" {
    var p = try LayerPolicy.init(testing.allocator, 16, 2);
    defer p.deinit(testing.allocator);
    // Prompt frequency: 5 x3, 9 x2, 1 x1 -> seed = {5, 9}.
    try p.prepareSeed(testing.allocator, &.{ 5, 9, 1, 5, 9, 5 });
    var out: Plan = .{};
    p.plan(&.{ 1, 9, 5 }, .prefill, &out);
    // Seed first, least frequent first; the pool is then full of protected
    // experts, so 1 overflows to the transient scratch.
    try testing.expectEqualSlices(u16, &.{ 9, 5, 1 }, out.missesOf());
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 9, .slot = 0, .persistent = true },
        .{ .expert = 5, .slot = 1, .persistent = true },
        .{ .expert = 1, .slot = 2, .persistent = false },
    }, out.loadsOf());
    try checkPlan(&p, &.{ 1, 9, 5 }, &out, max_route_ids);
    // A probationary resident is evicted before any protected one.
    var q = try LayerPolicy.init(testing.allocator, 16, 2);
    defer q.deinit(testing.allocator);
    try q.prepareSeed(testing.allocator, &.{3});
    q.plan(&.{ 3, 4 }, .prefill, &out);
    q.plan(&.{6}, .prefill, &out);
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 1, .previous = 4, .next = 6 }}, out.evictionsOf());
}

test "dsv41 policy: growth adds empty slots used before any eviction" {
    var p = try LayerPolicy.init(testing.allocator, 16, 2);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{ 1, 2 }, .prefill, &out);
    try p.grow(4);
    try testing.expectError(error.InvalidCapacity, p.grow(3));
    p.plan(&.{ 1, 3, 4 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 3, .slot = 2, .persistent = true },
        .{ .expert = 4, .slot = 3, .persistent = true },
    }, out.loadsOf());
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
    try testing.expectEqual(@as(?u32, 1), p.slotOf(2));
}

test "dsv41 policy: bounded parts cut at the last physical gap" {
    var ends: [8]u32 = undefined;
    // 0-10-20 contiguous | 35-45 | 60-70.
    const off = [_]u64{ 0, 10, 20, 35, 45, 60, 70 };
    const len: [7]u64 = @splat(10);
    try testing.expectEqualSlices(u32, &.{ 3, 5, 7 }, boundedParts(&off, &len, 3, &ends));
    // Padded records never touch: plain runs of three.
    const padded: [7]u64 = @splat(9);
    try testing.expectEqualSlices(u32, &.{ 3, 6, 7 }, boundedParts(&off, &padded, 3, &ends));
    // One contiguous run longer than the bound is cut at the bound.
    const run = [_]u64{ 0, 10, 20, 30, 40 };
    try testing.expectEqualSlices(u32, &.{ 3, 5 }, boundedParts(&run, len[0..5], 3, &ends));
    try testing.expectEqual(@as(usize, 0), boundedParts(&.{}, &.{}, 3, &ends).len);
}

/// One plan as R/exl3/runtime/dump_phase1_route_fixture.py records it from the
/// Python bank. loads: [expert, slot, persistent, physical skip];
/// evictions: [slot, previous, next]; parts: decode miss parts by expert.
pub const FixPlan = struct {
    ids: []const u16,
    slots: []const u32,
    hits: []const u16,
    misses: []const u16,
    loads: []const [4]u32,
    evictions: []const [3]u32,
    parts: []const []const u16,
};

/// A one-layer replay for the real-bank test: small rows, per-row routes.
pub const BankTrace = struct {
    layer: u32,
    seed: []const u16,
    prefill_rows: u32,
    decode_rows: u32,
    transient: u32,
    prefill: []const FixPlan,
    routes: []const FixPlan,
};

pub fn expectPlan(p: *const Plan, want: FixPlan) !void {
    try testing.expectEqualSlices(u32, want.slots, p.slotsOf());
    try testing.expectEqualSlices(u16, want.hits, p.hitsOf());
    try testing.expectEqualSlices(u16, want.misses, p.missesOf());
    try testing.expectEqual(want.loads.len, p.n_loads);
    for (want.loads, p.loadsOf()) |w, l| {
        try testing.expectEqual(w[0], l.expert);
        try testing.expectEqual(w[1], l.slot);
        try testing.expectEqual(w[2] != 0, l.persistent);
    }
    try testing.expectEqual(want.evictions.len, p.n_evictions);
    for (want.evictions, p.evictionsOf()) |w, ev| {
        try testing.expectEqual(w[0], ev.slot);
        try testing.expectEqual(w[1], ev.previous);
        try testing.expectEqual(w[2], ev.next);
    }
}

/// The decode miss parts of `p` (by expert), in the real bank's placement.
fn partsOf(p: *const Plan, table: []const expert_bank.Layer, layer: u32, out_experts: *[max_route_ids]u16, ends: *[max_route_ids]u32) []u32 {
    const n = p.n_loads;
    for (p.loadsOf(), 0..) |l, i| out_experts[i] = l.expert;
    std.sort.insertion(u16, out_experts[0..n], {}, std.sort.asc(u16));
    var offsets: [max_route_ids]u64 = undefined;
    var lengths: [max_route_ids]u64 = undefined;
    const t = table[layer];
    for (out_experts[0..n], 0..) |e, i| {
        offsets[i] = t.base_offset + @as(u64, e) * t.record_bytes;
        lengths[i] = t.logical_bytes;
    }
    return boundedParts(offsets[0..n], lengths[0..n], 3, ends);
}

/// ExpertSlotPool._prepare_load's reuse rule over physical rows: a load whose
/// row already holds (layer, expert) is not read.
const Physical = struct {
    persistent: []u16,
    experts: u32,
    transient: [max_route_ids]?[2]u32 = @splat(null),

    fn skip(ph: *Physical, layer: u32, capacity: u32, w: [4]u32) bool {
        if (w[2] != 0) {
            const o = &ph.persistent[layer * ph.experts + w[1]];
            const held = o.* == w[0];
            o.* = @intCast(w[0]);
            return held;
        }
        const o = &ph.transient[w[1] - capacity];
        const held = if (o.*) |h| h[0] == layer and h[1] == w[0] else false;
        o.* = .{ layer, w[0] };
        return held;
    }
};

// DSV41_PHASE1_ROUTE_FIXTURE=<json from R/exl3/runtime/dump_phase1_route_fixture.py>
test "dsv41 policy: the recorded trace plans exactly like the Python bank" {
    const path = std.mem.span(std.c.getenv("DSV41_PHASE1_ROUTE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        records_per_part: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        seed_plans: []const []const FixPlan,
        routes: []const FixPlan,
        final_slot_to_expert: []const []const i32,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    try testing.expectEqual(@as(u32, max_route_ids), f.transient);
    try testing.expectEqual(@as(u32, 3), f.records_per_part);

    var table: [40]expert_bank.Layer = undefined;
    const k3: [40]u32 = @splat(3);
    try testing.expectEqual(@as(u32, 40), f.layers);
    _ = expert_bank.layerTable(&k3, 5120, 2304, f.experts, &table).?;

    const policies = try a.alloc(LayerPolicy, f.layers);
    defer a.free(policies);
    var n_init: usize = 0;
    defer for (policies[0..n_init]) |*p| p.deinit(a);
    // The Python pool's physical owners, for its read skips.
    var phys: Physical = .{ .persistent = try a.alloc(u16, f.layers * f.experts), .experts = f.experts };
    defer a.free(phys.persistent);
    @memset(phys.persistent, no_expert);

    var out: Plan = .{};
    var n_plans: usize = 0;
    var n_reads: usize = 0;
    var n_skips: usize = 0;
    for (0..f.layers) |l| {
        policies[l] = try LayerPolicy.init(a, f.experts, f.prefill_capacity[l]);
        n_init += 1;
        const p = &policies[l];
        try p.prepareSeed(a, f.resident0[l]);
        for (f.seed_plans[l]) |want| {
            p.plan(want.ids, .prefill, &out);
            try expectPlan(&out, want);
            for (want.loads) |w| try testing.expectEqual(w[3] != 0, phys.skip(@intCast(l), p.capacity, w));
            n_plans += 1;
        }
    }
    for (policies, f.decode_capacity) |*p, cap| try p.grow(cap);

    var parts_experts: [max_route_ids]u16 = undefined;
    var ends: [max_route_ids]u32 = undefined;
    for (f.routes, 0..) |want, i| {
        const l: u32 = @intCast(i % f.layers);
        const p = &policies[l];
        p.plan(want.ids, .decode, &out);
        expectPlan(&out, want) catch |e| {
            std.debug.print("route {d} (cycle {d}, layer {d}) differs\n", .{ i, i / f.layers, l });
            return e;
        };
        for (want.loads) |w| {
            const skip = phys.skip(l, p.capacity, w);
            try testing.expectEqual(w[3] != 0, skip);
            if (skip) n_skips += 1 else n_reads += 1;
        }
        const got = partsOf(&out, &table, l, &parts_experts, &ends);
        try testing.expectEqual(want.parts.len, got.len);
        var start: u32 = 0;
        for (want.parts, got) |wp, end| {
            try testing.expectEqualSlices(u16, wp, parts_experts[start..end]);
            start = end;
        }
        n_plans += 1;
    }
    for (policies, f.final_slot_to_expert) |*p, want| {
        try testing.expectEqual(want.len, p.capacity);
        for (want, p.slot_to_expert[0..p.capacity]) |w, e| {
            try testing.expectEqual(w, if (e == no_expert) @as(i32, -1) else @as(i32, e));
        }
    }
    std.debug.print("policy parity: {d} plans ({d} decode routes) equal the Python bank's; {d} reads, {d} skipped\n", .{ n_plans, f.routes.len, n_reads, n_skips });
}
