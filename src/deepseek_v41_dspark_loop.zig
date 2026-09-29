//! The DSpark-direct decode loop on the native path (Python
//! `dspark_generate` + `_decode_cycles` as the lane of record runs it: the
//! hybrid causal lookup to depth 7 in one verify chunk, greedy or #475 typical
//! acceptance). The prompt runs in forwards of at most `prompt_chunk` rows
//! (every routed call a decode-lane call), its main hiddens seed the draft
//! head once; each cycle drafts, verifies `[primary, drafts]`, accepts, and
//! commits by trimming the target and seeding the draft windows.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const mdl = @import("deepseek_v41_model.zig");
const ds = @import("deepseek_v41_dspark.zig");
const dh = @import("deepseek_v41_dspark_head.zig");

pub const Config = struct {
    /// Requested native draft depth (capped by the head's block size).
    k_request: u32 = 5,
    /// The confidence early stop (the runner's 0.5); null keeps every draft.
    confidence_threshold: ?f64 = 0.5,
    /// `hybrid_install`'s causal lookup: a full native proposal extended from the history.
    lookup: ?struct { minimum_context: u32 = 2, extra_tokens: u32 = 2 } = .{},
    acceptance: ds.Acceptance = .greedy,
    /// Rows per prompt forward; `whole_prompt` runs it as one forward, chunked
    /// by the model's own rule (every chunk wider than a route takes runs
    /// through the wide lane: the served prompt pass).
    prompt_chunk: u32 = 8,
    max_tokens: u32,
    stop_ids: []const u32 = &.{},
};

pub const Finish = enum { length, stop };

/// `Config.prompt_chunk`: the whole prompt in one forward.
pub const whole_prompt: u32 = std.math.maxInt(u32);

/// One cycle's decisions, as the Python oracle fixture records them.
pub const CycleLog = struct {
    primary: u32,
    native: [ds.max_block]u32 = undefined,
    n_native: u32 = 0,
    /// Drafts kept by the confidence early stop, before the lookup.
    k_native: u32 = 0,
    /// The host values the cycle read: sigmoid confidences, the verify rows'
    /// argmax (all chunks) and the typical flags.
    conf: [ds.max_block]f32 = undefined,
    targets: [ds.max_block + 1]u32 = undefined,
    n_targets: u32 = 0,
    flags: [ds.max_block + 1]bool = undefined,
    n_flags: u32 = 0,
    drafts: [ds.max_block]u32 = undefined,
    k_eff: u32 = 0,
    accepted: u32 = 0,
    correction: u32 = 0,
    verified: u32 = 0,
    trimmed: u32 = 0,
    /// Set by a caller that classifies divergences (the window harness): each
    /// verify row's top two ids and f32 logits and its rms, indexed like `targets`.
    want_top: bool = false,
    top_ids: [ds.max_block + 1][2]u32 = undefined,
    top_logits: [ds.max_block + 1][2]f32 = undefined,
    rms: [ds.max_block + 1]f32 = undefined,
};

pub fn Loop(comptime G: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const M = mdl.Model(G);
        pub const H = dh.Head(G);

        g: *G,
        model: *M,
        head: *const H,
        st: *M.State,
        caches: []H.Cache,
        cfg: Config,
        lookup: ?ds.Lookup = null,
        stats: ds.Stats = .{},
        k_cap: u32,
        max_rows: u32,
        /// `main_hidden` of the next draft's main token (kept across resets).
        main_h: ?T = null,
        primary: u32 = 0,

        pub fn init(g: *G, model: *M, head: *const H, st: *M.State, caches: []H.Cache, cfg: Config) Self {
            const k_cap = @min(cfg.k_request, head.blockSize());
            const extra: u32 = if (cfg.lookup) |l| (if (k_cap == ds.Lookup.key_len) l.extra_tokens else 0) else 0;
            var self: Self = .{ .g = g, .model = model, .head = head, .st = st, .caches = caches, .cfg = cfg, .k_cap = k_cap, .max_rows = k_cap + extra + 1 };
            self.stats.speculative_depth = k_cap + extra;
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.main_h) |x| self.g.release(x);
            if (self.lookup) |*l| l.deinit();
        }

        fn setMain(self: *Self, x: T) void {
            const k = self.g.keep(x);
            if (self.main_h) |old| self.g.release(old);
            self.main_h = k;
        }

        fn sliceRows(g: *G, x: T, lo: c_int, hi: c_int) !T {
            const s = g.shapeOf(x);
            return g.slice(x, &.{ 0, lo, 0 }, &.{ s.d[0], hi, s.d[2] }, &.{ 1, 1, 1 });
        }

        fn evalWindows(self: *Self) !void {
            var ws: [8]T = undefined;
            var n: usize = 0;
            for (self.caches) |c| if (c.window) |w| {
                ws[n] = w;
                n += 1;
            };
            try self.g.evalAll(ws[0..n]);
        }

        fn isStop(self: *const Self, tok: u32) bool {
            return std.mem.indexOfScalar(u32, self.cfg.stop_ids, tok) != null;
        }

        /// `dspark_generate`'s prompt pass: the forwards, the primary pick,
        /// `_seed_prefill_state` (one seed over every prompt row).
        pub fn prefill(self: *Self, a: std.mem.Allocator, ex: anytype, prompt: []const u32) !u32 {
            const g = self.g;
            const chunk = self.cfg.prompt_chunk;
            var mains: std.ArrayList(T) = .empty;
            defer {
                for (mains.items) |x| g.release(x);
                mains.deinit(a);
            }
            var i: usize = 0;
            while (i < prompt.len) {
                const end = @min(i + chunk, prompt.len);
                const last = end == prompt.len;
                const r = try self.model.forward(g, self.st, prompt[i..end], .{ .logits = if (last) .last else .none, .main_hidden = true }, ex, graph.NoProbe{});
                try M.fence(g, self.st, &.{ if (last) r.logits.? else r.hidden, r.main_hidden.? });
                try ex.flush();
                try mains.append(a, g.keep(r.main_hidden.?));
                if (last) self.primary = try g.hostArgmax(r.logits.?);
                g.reset();
                i = end;
            }
            const all = if (mains.items.len == 1) mains.items[0] else try g.concat(mains.items, 1);
            try self.head.seedMain(g, all, self.caches);
            const n: c_int = @intCast(prompt.len);
            self.setMain(try sliceRows(g, all, n - 1, n));
            try self.evalWindows();
            try g.evalAll(&.{self.main_h.?});
            g.reset();
            if (self.cfg.lookup) |l| {
                self.lookup = try ds.Lookup.init(a, prompt, l.minimum_context, l.extra_tokens);
                try self.lookup.?.appendCommitted(&.{self.primary});
            }
            return self.primary;
        }

        /// The target's decision on a verify chunk: its rows' argmax and, for the
        /// typical tier, `_TYPICAL_DECIDE`'s flags on the drafted rows (one sync).
        fn decide(self: *Self, logits: T, drafts: []const u32, rows: [2]u32, k_eff: u32, target: []u32, flags: []bool) !?[]const bool {
            const g = self.g;
            const s = g.shapeOf(logits);
            const vocab = s.dim(-1);
            const width: u32 = @intCast(s.dim(1));
            const row2 = try g.reshape(logits, &.{ @intCast(width), vocab });
            const tt = try g.argmax(row2, -1);
            const typ = switch (self.cfg.acceptance) {
                .greedy => {
                    _ = try g.hostU32(tt, target[0..width]);
                    return null;
                },
                .typical => |t| t,
            };
            const drafted: u32 = @min(k_eff -| rows[0], width);
            if (drafted == 0) {
                _ = try g.hostU32(tt, target[0..width]);
                return flags[0..0];
            }
            const lg = try g.astype(try g.slice(row2, &.{ 0, 0 }, &.{ @intCast(drafted), vocab }, &.{ 1, 1 }), .float32);
            const log_p = try g.sub(lg, try g.logsumexp(lg, -1, true));
            const p = try g.exp(log_p);
            const entropy = try g.neg(try g.sum(try g.mul(p, log_p), -1, false));
            const floor = try g.minimum(try g.scalar(typ.eps, .float32), try g.mul(try g.scalar(typ.delta, .float32), try g.exp(try g.neg(entropy))));
            var idb: [ds.max_block]i32 = undefined;
            for (idb[0..drafted], drafts[rows[0]..][0..drafted]) |*d, v| d.* = @intCast(v);
            const ids = try g.hostArray(std.mem.sliceAsBytes(idb[0..drafted]), &.{ @intCast(drafted), 1 }, .int32);
            const typical = try g.greater(try g.reshape(try g.takeAlongAxis(p, ids, -1), &.{-1}), floor);
            try g.evalAll(&.{ tt, typical });
            _ = try g.hostU32(tt, target[0..width]);
            return try g.hostBool(typical, flags[0..drafted]);
        }

        /// A logged verify chunk's rows, for the tie-flip rule: the top two ids
        /// and f32 logits of each row and the row's rms (one extra sync, only
        /// when the caller asks: `CycleLog.want_top`).
        fn topTwo(self: *Self, logits: T, width: u32, ids: [][2]u32, vals: [][2]f32, rms: []f32) !void {
            const g = self.g;
            const vocab = g.shapeOf(logits).dim(-1);
            const w: c_int = @intCast(width);
            const rows = try g.astype(try g.reshape(logits, &.{ w, vocab }), .float32);
            const first_id = try g.argmax(rows, -1);
            const m1 = try g.max(rows, -1, false);
            const col = try g.reshape(try g.arange(0, @floatFromInt(vocab), 1, .uint32), &.{ 1, vocab });
            const masked = try g.where(try g.equal(col, try g.reshape(first_id, &.{ w, 1 })), try g.scalar(-std.math.inf(f64), .float32), rows);
            const second_id = try g.argmax(masked, -1);
            const m2 = try g.max(masked, -1, false);
            const r = try g.sqrt(try g.mean(try g.square(rows), -1, false));
            try g.evalAll(&.{ first_id, m1, second_id, m2, r });
            var a1: [ds.max_block + 1]u32 = undefined;
            var a2: [ds.max_block + 1]u32 = undefined;
            var v1: [ds.max_block + 1]f32 = undefined;
            var v2: [ds.max_block + 1]f32 = undefined;
            _ = try g.hostU32(first_id, a1[0..width]);
            _ = try g.hostU32(second_id, a2[0..width]);
            _ = try g.hostF32(m1, v1[0..width]);
            _ = try g.hostF32(m2, v2[0..width]);
            _ = try g.hostF32(r, rms[0..width]);
            for (ids[0..width], vals[0..width], 0..) |*id, *v, i| {
                id.* = .{ a1[i], a2[i] };
                v.* = .{ v1[i], v2[i] };
            }
        }

        /// One cycle; returns null to continue, or how the run finished.
        pub fn cycle(self: *Self, ex: anytype, out: *std.ArrayList(u32), a: std.mem.Allocator, log: ?*CycleLog) !?Finish {
            const g = self.g;
            const st = &self.stats;
            var drafts_buf: [ds.max_block]u32 = undefined;
            var drafts: []const u32 = &.{};
            var k_eff: u32 = 0;
            var native: [ds.max_block]u32 = undefined;
            if (self.k_cap > 0) {
                const d = try self.head.draftBlock(g, self.main_h.?, self.primary, self.caches, self.model.embed, self.model.head);
                const bs = self.head.blockSize();
                try g.evalAll(&.{ d.ids, d.conf });
                _ = try g.hostU32(d.ids, native[0..bs]);
                var conf: [ds.max_block]f32 = undefined;
                _ = try g.hostF32(try g.sigmoid(try g.astype(d.conf, .float32)), conf[0..bs]);
                k_eff = ds.effectiveDraftLen(conf[0..bs], self.k_cap, self.cfg.confidence_threshold);
                if (log) |lg| {
                    lg.k_native = k_eff;
                    @memcpy(lg.conf[0..bs], conf[0..bs]);
                }
                drafts = if (self.lookup) |*l| l.extend(native[0..k_eff], &drafts_buf) else blk: {
                    @memcpy(drafts_buf[0..k_eff], native[0..k_eff]);
                    break :blk drafts_buf[0..k_eff];
                };
                if (log) |lg| {
                    lg.n_native = bs;
                    @memcpy(lg.native[0..bs], native[0..bs]);
                }
                k_eff = @intCast(drafts.len);
            }
            // Verify [primary, drafts] in chunks of max_rows, stopping at the correction.
            var block: [ds.max_block + 1]u32 = undefined;
            block[0] = self.primary;
            @memcpy(block[1..][0..drafts.len], drafts);
            const n_block: u32 = @intCast(drafts.len + 1);
            var o: ds.Outcome = .{};
            var hiddens: [ds.max_block + 1]T = undefined;
            var n_hidden: usize = 0;
            var start: u32 = 0;
            while (start < n_block) {
                const end = @min(start + self.max_rows, n_block);
                const r = try self.model.forward(g, self.st, block[start..end], .{ .logits = .all, .main_hidden = true }, ex, graph.NoProbe{});
                try g.evalAll(&.{ r.logits.?, r.main_hidden.? });
                st.verify_calls += 1;
                hiddens[n_hidden] = r.main_hidden.?;
                n_hidden += 1;
                var target: [ds.max_block + 1]u32 = undefined;
                var flags: [ds.max_block + 1]bool = undefined;
                const typ = try self.decide(r.logits.?, drafts, .{ start, end }, k_eff, &target, &flags);
                if (log) |lg| {
                    if (lg.want_top) try self.topTwo(r.logits.?, end - start, lg.top_ids[lg.n_targets..], lg.top_logits[lg.n_targets..], lg.rms[lg.n_targets..]);
                    @memcpy(lg.targets[lg.n_targets..][0 .. end - start], target[0 .. end - start]);
                    lg.n_targets += end - start;
                    if (typ) |ty| {
                        @memcpy(lg.flags[lg.n_flags..][0..ty.len], ty);
                        lg.n_flags += @intCast(ty.len);
                    }
                }
                const done = ds.acceptChunk(&o, st, drafts, k_eff, .{ start, end }, target[0 .. end - start], typ);
                start = end;
                if (done) break;
            }
            const correction = o.correction.?; // acceptChunk sets it on the chunk that ends the verify
            st.endCycle(o, k_eff);
            const verify_hidden = if (n_hidden == 1) hiddens[0] else try g.concat(hiddens[0..n_hidden], 1);
            // Commit: keep [primary, d1 .. d_accepted] in the target, seed the draft windows.
            try self.model.trim(g, self.st, o.trimRows());
            try self.head.seedMain(g, try sliceRows(g, verify_hidden, 0, @intCast(o.accepted + 1)), self.caches);
            try self.evalWindows();
            if (log) |lg| {
                lg.primary = self.primary;
                lg.k_eff = k_eff;
                lg.accepted = o.accepted;
                lg.correction = correction;
                lg.verified = o.verified;
                lg.trimmed = o.trimRows();
                @memcpy(lg.drafts[0..drafts.len], drafts);
            }
            // Emit up to max_tokens; the stop token is emitted, then the run ends.
            var emitted: [ds.max_block + 1]u32 = undefined;
            @memcpy(emitted[0..o.accepted], drafts[0..o.accepted]);
            emitted[o.accepted] = correction;
            const base = out.items.len;
            var finish: ?Finish = null;
            for (emitted[0 .. o.accepted + 1]) |tok| {
                if (out.items.len >= self.cfg.max_tokens) break;
                try out.append(a, tok);
                if (self.isStop(tok)) {
                    finish = .stop;
                    break;
                }
            }
            st.generated_tokens = @intCast(out.items.len);
            if (self.lookup) |*l| try l.appendCommitted(out.items[base..]);
            if (finish == null and out.items.len >= self.cfg.max_tokens) finish = .length;
            if (finish == null) {
                self.primary = correction;
                self.setMain(try sliceRows(g, verify_hidden, @intCast(o.accepted), @intCast(o.accepted + 1)));
                try g.evalAll(&.{self.main_h.?});
            }
            try ex.flush();
            g.reset();
            return finish;
        }

        /// The run after `prefill`: cycles until the token cap or a stop id.
        pub fn run(self: *Self, ex: anytype, out: *std.ArrayList(u32), a: std.mem.Allocator) !Finish {
            if (self.isStop(self.primary)) return .stop;
            while (true) {
                if (try self.cycle(ex, out, a, null)) |f| return f;
            }
        }
    };
}

// ── Tests ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const routes = @import("deepseek_v41_routes.zig");
const xp = @import("deepseek_v41_experts.zig");

/// Host reads of a scripted run: routed ids from a PRNG, the prompt's pick,
/// and per cycle the draft ids, their sigmoid confidences, the verify
/// targets and (typical) the flags, in the loop's read order.
const Script = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260928),
    n_experts: u16,
    pick: u32,
    u32s: []const []const u32,
    f32s: []const []const f32,
    bools: []const []const bool = &.{},
    nu: usize = 0,
    nf: usize = 0,
    nb: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax, .u32s = u32s_, .f32s = f32s_, .bools = bools_ };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return s.pick;
    }
    fn next(comptime E: type, list: []const []const E, i: *usize, out: []E) !void {
        if (i.* >= list.len) return error.ScriptExhausted;
        if (list[i.*].len != out.len) return error.ScriptShape;
        @memcpy(out, list[i.*]);
        i.* += 1;
    }
    fn u32s_(ctx: *anyopaque, out: []u32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(u32, s.u32s, &s.nu, out);
    }
    fn f32s_(ctx: *anyopaque, out: []f32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(f32, s.f32s, &s.nf, out);
    }
    fn bools_(ctx: *anyopaque, out: []bool) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(bool, s.bools, &s.nb, out);
    }
};

const Rig = struct {
    m: *mdl.Mini,
    g: TraceOps,
    lookup: mdl.SpecLookup,
    model: *Loop(TraceOps).M,
    head: *Loop(TraceOps).H,
    st: Loop(TraceOps).M.State,
    caches: [4]Loop(TraceOps).H.Cache = @splat(.{}),
    src: xp.FakeSource,
    ex: xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath),

    fn init(rig: *Rig) !void {
        const a = testing.allocator;
        rig.m = try mdl.Mini.init();
        rig.g = TraceOps.init(a);
        rig.caches = @splat(.{});
        rig.lookup = .{ .g = &rig.g, .spec = rig.m.spec };
        const c = &rig.m.c;
        rig.model = try Loop(TraceOps).M.init(a, &rig.g, rig.m.c, try routes.parse(&.{}, null), &rig.lookup, &rig.m.src);
        rig.head = try Loop(TraceOps).H.init(a, &rig.g, rig.m.c, .{}, &rig.lookup);
        rig.st = try rig.model.newState();
        var rows: [8]u32 = @splat(0);
        var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
        rig.src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
        rig.ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &rig.g, &rig.src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, c);
        try rig.ex.grow(&rig.g, grown[0..c.n_layers]);
    }

    fn deinit(rig: *Rig) void {
        for (rig.caches[0..rig.head.nStages()]) |*c| c.deinit(&rig.g);
        rig.ex.deinit();
        rig.src.deinit();
        rig.st.deinit(&rig.g, testing.allocator);
        rig.head.deinit(&rig.g);
        rig.model.deinit(&rig.g);
        rig.g.deinit();
        rig.m.deinit();
    }
};

test "dsv41 dspark loop: the mini model's cycles draft, verify, accept, trim and seed as _decode_cycles does" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    const c = &rig.m.c;
    try testing.expectEqual(@as(u32, 2), rig.head.blockSize());
    // Four cycles (block 2, confidence 0.5, greedy): a reject at depth 1; an early
    // stop to one draft and its bonus; a stop to one draft rejected; both accepted
    // (the bonus is cut by max_tokens 6).
    var script: Script = .{
        .n_experts = @intCast(c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 2), lp.k_cap);
    try testing.expectEqual(@as(u32, 3), lp.max_rows);
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    try testing.expectEqual(@as(u32, 3), try lp.prefill(a, &rig.ex, &prompt));
    try testing.expectEqual(@as(u32, 20), rig.caches[0].offset);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var logs: [4]CycleLog = @splat(.{ .primary = 0 });
    for (&logs, 0..) |*lg, i| {
        const f = try lp.cycle(&rig.ex, &out, a, lg);
        try testing.expectEqual(@as(?Finish, if (i == 3) .length else null), f);
    }
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    const want = [_][4]u32{ .{ 2, 1, 9, 1 }, .{ 1, 1, 12, 0 }, .{ 1, 0, 20, 1 }, .{ 2, 2, 40, 0 } };
    for (logs, want) |lg, w| {
        try testing.expectEqual(w[0], lg.k_eff);
        try testing.expectEqual(w[1], lg.accepted);
        try testing.expectEqual(w[2], lg.correction);
        try testing.expectEqual(w[3], lg.trimmed);
    }
    const st = lp.stats;
    try testing.expectEqualSlices(u32, &.{ 4, 2 }, st.drafted_by_depth[0..2]);
    try testing.expectEqualSlices(u32, &.{ 3, 1 }, st.accepted_by_depth[0..2]);
    try testing.expectEqual(@as(u32, 4), st.cycles);
    try testing.expectEqual(@as(u32, 4), st.verify_calls);
    try testing.expectEqual(@as(u32, 2), st.correction_tokens);
    try testing.expectEqual(@as(u32, 2), st.bonus_tokens);
    try testing.expectEqual(@as(u32, 6), st.generated_tokens);
    // The target and the draft windows both hold the prompt plus every committed row.
    try testing.expectEqual(@as(u32, 28), rig.st.offset);
    try testing.expectEqual(@as(u32, 28), rig.caches[0].offset);
    try testing.expectEqual(script.u32s.len, script.nu);
    try testing.expectEqual(script.f32s.len, script.nf);
    // Every verify forward routed through the source: 3 prompt forwards + 4 verifies per layer.
    try testing.expectEqual(@as(u64, 7 * c.n_layers), rig.src.stats().route_calls);
}

test "dsv41 dspark loop: the K33 draft block replays the eager one from regions built at construction" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    const n_st = rig.head.nStages();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    // A K33 head over the same residents (its regions are built here, at construction).
    const k33 = try Loop(TraceOps).H.init(a, &rig.g, rig.m.c, .{ .draft_rows = graph.draft_compile_max_rows }, &rig.lookup);
    defer k33.deinit(&rig.g);
    const e0 = rig.g.nodes.items.len;
    const de = try rig.head.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e1 = rig.g.nodes.items.len;
    const dk = try k33.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e2 = rig.g.nodes.items.len;
    inline for (.{ "ids", "logits", "conf" }) |f| {
        try testing.expect(rig.g.shapeOf(@field(de, f)).eql(rig.g.shapeOf(@field(dk, f))));
        try testing.expectEqual(rig.g.dtypeOf(@field(de, f)), rig.g.dtypeOf(@field(dk, f)));
    }
    const count = struct {
        fn f(g: *const TraceOps, from: usize, to: usize, op: ops.Op) usize {
            var n: usize = 0;
            for (g.nodes.items[from..to]) |nd| n += @intFromBool(nd.op == op);
            return n;
        }
    }.f;
    // Eager: no region. K33: per stage the HC attention prep, the main KV, the QKV and
    // output prep, the HC ffn prep, the gate prefix, the MoE combine and the HC post;
    // then a markov step per draft and the confidence.
    try testing.expectEqual(@as(usize, 0), count(&rig.g, e0, e1, .tape_begin));
    try testing.expectEqual(8 * n_st + rig.head.blockSize() + 1, count(&rig.g, e1, e2, .tape_begin));
    // The regions hold the eager body's heavy ops.
    inline for (.{ ops.Op.qmm, ops.Op.gather_qmm, ops.Op.softmax, ops.Op.argmax, ops.Op.matmul }) |op| {
        try testing.expectEqual(count(&rig.g, e0, e1, op), count(&rig.g, e1, e2, op));
    }
}

/// A lookup that records every name asked for and every name dropped.
const DropLookup = struct {
    inner: *const mdl.SpecLookup,
    got: std.ArrayList([]u8) = .empty,
    dropped: std.ArrayList([]u8) = .empty,

    pub fn get(self: *DropLookup, name: []const u8) ?u32 {
        self.got.append(testing.allocator, testing.allocator.dupe(u8, name) catch @panic("oom")) catch @panic("oom");
        return self.inner.get(name);
    }

    pub fn drop(self: *DropLookup, name: []const u8) void {
        self.dropped.append(testing.allocator, testing.allocator.dupe(u8, name) catch @panic("oom")) catch @panic("oom");
    }

    fn has(list: []const []u8, name: []const u8) bool {
        for (list) |x| if (std.mem.eql(u8, x, name)) return true;
        return false;
    }

    fn deinit(self: *DropLookup) void {
        for (self.got.items) |x| testing.allocator.free(x);
        for (self.dropped.items) |x| testing.allocator.free(x);
        self.got.deinit(testing.allocator);
        self.dropped.deinit(testing.allocator);
    }
};

test "dsv41 dspark loop: a pinned subset head keeps only its experts, maps every routed id through its lut and otherwise drafts as the full head" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    // The mini head (one stage) at 4 experts, so a subset can keep some, leave some out and
    // move a kept expert to another slot: the full and the compact head over the same residents.
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var c4 = rig.m.c;
    c4.dspark.n_routed_experts = 4;
    const c = &c4;
    const lookup4: mdl.SpecLookup = .{ .g = &rig.g, .spec = try v41.residentSpec(arena.allocator(), c) };
    const full = try Loop(TraceOps).H.init(a, &rig.g, c4, .{}, &lookup4);
    defer full.deinit(&rig.g);
    const n_st = full.nStages();
    try testing.expectEqual(@as(usize, 1), n_st);
    const n: u16 = 4;
    // Stage 0 keeps {1, 3}: 1 -> slot 0, 3 -> slot 1, the left-out 0 and 2 -> slot 0.
    const text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"kind\": \"test\", \"n_experts\": 4, \"selected\": [[1, 3]]}";
    try rig.m.tmp.dir.writeFile(testing.io, .{ .sub_path = "subset.json", .data = text });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/subset.json", .{root[0..try rig.m.tmp.dir.realPath(testing.io, &root)]});
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    var sub = try dh.Subset.load(a, testing.io, .{ .path = path, .sha256 = &hex }, null);
    defer sub.deinit();
    var dl: DropLookup = .{ .inner = &lookup4 };
    defer dl.deinit();
    rig.g.record_host = true;
    const compact = try Loop(TraceOps).H.initWith(a, &rig.g, c4, .{}, &dl, .{ .subset = &sub });
    defer compact.deinit(&rig.g);
    // Only the kept experts' arrays are asked for (in ascending order: slot i is kept
    // expert i); every per-expert array is dropped, a left-out one before anything read it.
    var nb: [96]u8 = undefined;
    for (sub.selected, 0..) |kept, st| {
        var slot: usize = 0;
        for (0..n) |e| inline for (.{ "w1", "w3", "w2" }) |w| inline for (.{ "weight", "scales" }) |part| {
            const name = try std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}." ++ w ++ "." ++ part, .{ st, e });
            const is_kept = std.mem.indexOfScalar(u16, kept, @intCast(e)) != null;
            try testing.expectEqual(is_kept, DropLookup.has(dl.got.items, name));
            try testing.expect(DropLookup.has(dl.dropped.items, name));
        };
        for (dl.got.items) |name| {
            var buf2: [96]u8 = undefined;
            const want = try std.fmt.bufPrint(&buf2, "mtp.{d}.ffn.experts.", .{st});
            if (!std.mem.startsWith(u8, name, want) or !std.mem.endsWith(u8, name, ".w1.weight")) continue;
            const e = try std.fmt.parseInt(u16, name[want.len .. std.mem.indexOfScalarPos(u8, name, want.len, '.').?], 10);
            try testing.expectEqual(kept[slot], e);
            slot += 1;
        }
        try testing.expectEqual(kept.len, slot);
        // The compact banks and the lut (`positions.get(expert, 0)`, run_full.py:92).
        const stg = compact.stages[st];
        inline for (.{ "w1", "w3", "w2" }) |w| try testing.expectEqual(@as(c_int, @intCast(kept.len)), rig.g.shapeOf(@field(stg.experts, w).w).dim(0));
        const lut = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, rig.g.hostBytesOf(stg.lut.?).?));
        try testing.expectEqual(@as(usize, n), lut.len);
        for (0..n) |e| {
            const want: i32 = if (std.mem.indexOfScalar(u16, kept, @intCast(e))) |i| @intCast(i) else 0;
            try testing.expectEqual(want, lut[e]);
        }
    }
    try testing.expect(compact.identity == .compact);
    try testing.expectEqualSlices(u8, &sub.sha256, &compact.identity.compact);
    var lut4: [4]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&lut4), rig.g.hostBytesOf(compact.stages[0].lut.?).?);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1 }, &lut4);
    try testing.expect(full.identity == .full and full.pruned_bytes == 0 and full.stages[0].lut == null);
    // The credit: the left-out experts' bytes (2 of them) at the head's per-expert bytes.
    try testing.expectEqual(@as(u64, 2), sub.pruned());
    try testing.expectEqual(sub.pruned() * dh.expertBytes(c), compact.pruned_bytes);
    // One expert's six arrays, as bound, are `expertBytes`.
    var per_expert: u64 = 0;
    inline for (.{ "w1", "w3", "w2" }) |w| inline for (.{ "weight", "scales" }) |part| {
        const x = lookup4.get("mtp.0.ffn.experts.0." ++ w ++ "." ++ part).?;
        per_expert += @intCast(rig.g.shapeOf(x).numel() * @as(i64, @intCast(ops.dtypeSize(rig.g.dtypeOf(x)))));
    };
    try testing.expectEqual(per_expert, dh.expertBytes(c));
    // A draft block: the same graph as the full head's, plus each stage MoE's lut gather.
    var script: Script = .{ .n_experts = @intCast(c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    const e0 = rig.g.nodes.items.len;
    _ = try full.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e1 = rig.g.nodes.items.len;
    _ = try compact.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e2 = rig.g.nodes.items.len;
    try testing.expectEqual(e1 - e0 + n_st, e2 - e1);
    inline for (@typeInfo(ops.Op).@"enum".field_names) |name| {
        const op = @field(ops.Op, name);
        var nf: usize = 0;
        var nc: usize = 0;
        for (rig.g.nodes.items[e0..e1]) |nd| nf += @intFromBool(nd.op == op);
        for (rig.g.nodes.items[e1..e2]) |nd| nc += @intFromBool(nd.op == op);
        if (op == .take) {
            try testing.expectEqual(nf + n_st, nc);
        } else try testing.expectEqual(nf, nc);
    }
}

test "dsv41 dspark loop: a logged cycle that wants the tie-flip rule reads each verify row's top two and rms" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    // One cycle (block 2, greedy): drafts 5, 6; verify rows [3, 5, 6] -> targets 5, 9, 7 (accepts 1).
    // Then the logged top two: ids [5, 9, 7] / [8, 2, 4], logits [2, 1, 3] / [1.5, 0.5, 2.5], rms [4, 4, 4].
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 5, 9, 7 }, &.{ 8, 2, 4 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 2, 1, 3 }, &.{ 1.5, 0.5, 2.5 }, &.{ 4, 4, 4 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var lg: CycleLog = .{ .primary = 0, .want_top = true };
    const e0 = rig.g.nodes.items.len;
    _ = try lp.cycle(&rig.ex, &out, a, &lg);
    try testing.expectEqual(@as(usize, 4), script.nu);
    try testing.expectEqual(@as(usize, 4), script.nf);
    try testing.expectEqualSlices(u32, &.{ 5, 9 }, out.items);
    try testing.expectEqual([2]u32{ 9, 2 }, lg.top_ids[1]);
    try testing.expectEqual([2]f32{ 1, 0.5 }, lg.top_logits[1]);
    try testing.expectEqual(@as(f32, 4), lg.rms[2]);
    // The verify's own argmax, plus the logged first and second (masked) argmax.
    var n_argmax: usize = 0;
    var n_where: usize = 0;
    for (rig.g.nodes.items[e0..]) |nd| {
        n_argmax += @intFromBool(nd.op == .argmax);
        n_where += @intFromBool(nd.op == .where);
    }
    try testing.expect(n_argmax >= 3 and n_where >= 1);
}

/// A wide route that records each call: its layer (routes are built in layer
/// order), rows, act rows and slots, in the order the forward makes them.
const WideLog = struct {
    const xk = @import("exl3_kernels.zig");
    const xko = @import("exl3_kernel_ops.zig");
    var next_layer: u32 = 0;
    var order: [512]u32 = undefined;
    var n_order: usize = 0;
    layer: u32,
    calls: u32 = 0,
    rows: u32 = 0,
    finishes: u32 = 0,
    ok: bool = true,

    pub fn init(_: std.mem.Allocator, _: *const xk.Registry, _: xko.PrefillShape, _: ?*xk.Diag) !WideLog {
        next_layer += 1;
        return .{ .layer = next_layer - 1 };
    }
    pub fn deinit(_: *WideLog, _: *TraceOps) void {}
    pub fn call(self: *WideLog, g: *TraceOps, act: u32, r: xko.PrefillRows, bank: xko.BankArrays(u32)) !u32 {
        self.calls += 1;
        self.rows += @intCast(r.slot.len);
        order[n_order] = self.layer;
        n_order += 1;
        const tokens: u32 = @intCast(g.shapeOf(act).d[0]);
        const cap: u32 = @intCast(g.shapeOf(bank.gate.code).d[0]);
        const act_row = r.act_row.?;
        self.ok = self.ok and g.dtypeOf(act) == .bfloat16 and act_row.len == r.slot.len;
        for (r.slot, act_row) |slot, row| self.ok = self.ok and slot < cap and row < tokens;
        return g.input(&.{ @intCast(r.slot.len), g.shapeOf(act).d[1] }, .float32);
    }
    pub fn finish(self: *WideLog, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "dsv41 dspark loop: the served prompt pass is one forward the model chunks; its wide chunks run through the wide lane, chunk-major" {
    const a = testing.allocator;
    const xk = @import("exl3_kernels.zig");
    const m = try mdl.Mini.init();
    defer m.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = m.spec };
    // The model's own chunk rule, pinned small (30 rows) so a mini prompt spans several chunks.
    const model = try Loop(TraceOps).M.init(a, &g, m.c, try routes.parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "30" }}, null), &lookup, &m.src);
    defer model.deinit(&g);
    const head = try Loop(TraceOps).H.init(a, &g, m.c, .{}, &lookup);
    defer head.deinit(&g);
    const nl = m.c.n_layers;
    const k = m.c.n_experts_per_tok;
    var rows0: [8]u32 = @splat(@intCast(m.c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = m.c.hidden_size, .inter = m.c.moe_intermediate_size, .n_experts = m.c.n_routed_experts, .rows = rows0[0..nl] });
    defer src.deinit();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    WideLog.next_layer = 0;
    WideLog.n_order = 0;
    const Ex = xp.ExpertsWith(TraceOps, xp.FakeSource, xp.TraceMath, .{ .prefill = WideLog });
    var ex = try Ex.initWith(a, &g, &src, .{ .hidden = @intCast(m.c.hidden_size), .inter = @intCast(m.c.moe_intermediate_size) }, &m.c, .{ .prefill = .{ .reg = &reg } });
    defer ex.deinit();
    var script: Script = .{ .n_experts = @intCast(m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    g.host_values = script.values();
    // 70 prompt rows: spans 30 / 30 / 10. The first two are wider than a route takes (30 x k > 48), the last is not.
    const n_prompt = 70;
    try testing.expect(30 * k > xp.max_route_ids and 10 * k <= xp.max_route_ids);
    var prompt: [n_prompt]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(1 + i % 50);
    // The request's KV bounded to its positions: the prompt, 4 tokens and one verify block.
    var st = try model.newStateWith(model.boundedKv(n_prompt + 4 + 8));
    defer st.deinit(&g, a);
    const max_len = st.max_len orelse return error.TestUnexpectedResult;
    try testing.expect(max_len >= n_prompt + 4 + 8);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var lp = Loop(TraceOps).init(&g, model, head, &st, caches[0..head.nStages()], .{ .lookup = null, .max_tokens = 4, .prompt_chunk = whole_prompt });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 3), try lp.prefill(a, &ex, &prompt));
    try testing.expectEqual(@as(u32, n_prompt), st.offset);
    try testing.expectEqual(@as(u32, n_prompt), caches[0].offset);
    // Every layer's wide route took the two wide chunks' rows (60 x k), act rows indexing the chunk,
    // slots inside the bank, bf16 act; one finish per call.
    for (ex.wide_routes) |r| {
        try testing.expect(r.ok);
        try testing.expectEqual(@as(u32, 60 * k), r.rows);
        try testing.expectEqual(r.calls, r.finishes);
    }
    // Chunk-major: chunk 0 through layers 0 .. L-1, then chunk 1 (a layer's calls in a chunk adjacent).
    var runs: [64]u32 = undefined;
    var n_runs: usize = 0;
    for (WideLog.order[0..WideLog.n_order]) |l| {
        if (n_runs == 0 or runs[n_runs - 1] != l) {
            runs[n_runs] = l;
            n_runs += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2 * nl), n_runs);
    for (runs[0..n_runs], 0..) |l, i| try testing.expectEqual(@as(u32, @intCast(i % nl)), l);
    // The narrow last chunk takes the decode lane: one route per layer per chunk, wide or not.
    try testing.expectEqual(@as(u64, 3 * nl), src.stats().route_calls);
    // A forward past the request's positions is refused before any lane is written.
    const room: usize = max_len - st.offset;
    const too_many = try a.alloc(u32, room + 1);
    defer a.free(too_many);
    @memset(too_many, 1);
    try testing.expectError(error.BoundedLaneFull, model.forward(&g, &st, too_many, .{ .logits = .none }, &ex, graph.NoProbe{}));
    try testing.expectEqual(@as(u32, n_prompt), st.offset);
}

test "dsv41 dspark loop: typical flags accept what the argmax rejects; the correction stays the argmax" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 7, 8, 9 }, &.{ 11, 12 }, &.{ 13, 14, 15 } },
        .f32s = &.{ &.{ 0.9, 0.9 }, &.{ 0.9, 0.9 } },
        // cycle 1: both drafts typical though the argmax disagrees: bonus 9;
        // cycle 2: depth 0 typical, depth 1 not: the correction is that row's argmax, 14.
        .bools = &.{ &.{ true, true }, &.{ true, false } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .lookup = null, .acceptance = .{ .typical = .{ .delta = 0.5 } }, .max_tokens = 5 });
    defer lp.deinit();
    var prompt: [9]u32 = @splat(4);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    try testing.expectEqual(Finish.length, try lp.run(&rig.ex, &out, a));
    try testing.expectEqualSlices(u32, &.{ 5, 6, 9, 11, 14 }, out.items);
    try testing.expectEqual(@as(u32, 2), script.nb);
    // The typical decision ran on the device: logsumexp over the drafted rows, one flag per draft.
    var n_lse: usize = 0;
    for (rig.g.nodes.items) |nd| n_lse += @intFromBool(nd.op == .logsumexp);
    try testing.expectEqual(@as(usize, 2), n_lse);
}

// DSV41_DSPARK_CYCLES_FIXTURE=<json from R/exl3/runtime/dump_dsv41_dspark_cycles.py>
test "dsv41 dspark loop: the lane's recorded cycles replay decision for decision" {
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_CYCLES_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const Cycle = struct {
        primary: u32,
        draft_ids: []const u32,
        conf_sigmoid: []const f32,
        k_eff_native: u32,
        native: []const u32,
        drafts: []const u32,
        targets: []const []const u32,
        flags: []const []const bool = &.{},
        verified: u32,
        kept: u32,
        emitted: []const u32 = &.{},
    };
    const Fixture = struct {
        format: []const u8,
        arm: []const u8,
        delta: ?f64 = null,
        confidence_threshold: f64,
        config: std.json.Value,
        prompt: []const u32,
        max_tokens: u32,
        tokens: []const u32,
        cycles: []const Cycle,
        stats: struct { drafted_by_depth: []const u32, accepted_by_depth: []const u32, cycles: u32, bonus_tokens: u32, correction_tokens: u32 },
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    try testing.expectEqualStrings("mlx-serve-dsv41-dspark-cycles-v1", f.format);
    const cfg_json = try std.json.Stringify.valueAlloc(a, f.config, .{});
    defer a.free(cfg_json);
    var diag: v41.Diag = .{};
    const c = v41.Config.parse(a, cfg_json, &diag) catch |e| {
        std.debug.print("fixture config refused: {s}\n", .{diag.message()});
        return e;
    };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const M = Loop(TraceOps).M;
    const model = try M.init(a, &g, c, try routes.parse(&.{}, null), &lookup, null);
    defer model.deinit(&g);
    const head = try Loop(TraceOps).H.init(a, &g, c, .{}, &lookup);
    defer head.deinit(&g);
    var st = try model.newState();
    defer st.deinit(&g, a);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var rows: [8]u32 = @splat(0);
    var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
    defer src.deinit();
    var ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &g, &src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, &c);
    defer ex.deinit();
    try ex.grow(&g, grown[0..c.n_layers]);
    // The loop's host reads, in its order: per cycle the draft ids, their sigmoid
    // confidences, then per verify chunk the argmax rows (and the typical flags).
    var u32s: std.ArrayList([]const u32) = .empty;
    defer u32s.deinit(a);
    var f32s: std.ArrayList([]const f32) = .empty;
    defer f32s.deinit(a);
    var bools: std.ArrayList([]const bool) = .empty;
    defer bools.deinit(a);
    for (f.cycles) |cy| {
        try u32s.append(a, cy.draft_ids);
        try f32s.append(a, cy.conf_sigmoid);
        for (cy.targets, 0..) |t, i| {
            try u32s.append(a, t);
            if (i < cy.flags.len) try bools.append(a, cy.flags[i]);
        }
    }
    var script: Script = .{ .n_experts = @intCast(c.n_routed_experts), .pick = f.tokens[0], .u32s = u32s.items, .f32s = f32s.items, .bools = bools.items };
    g.host_values = script.values();
    const acceptance: ds.Acceptance = if (std.mem.eql(u8, f.arm, "typical")) .{ .typical = .{ .delta = @floatCast(f.delta.?) } } else .greedy;
    var lp = Loop(TraceOps).init(&g, model, head, &st, caches[0..head.nStages()], .{ .confidence_threshold = f.confidence_threshold, .acceptance = acceptance, .max_tokens = f.max_tokens - 1 });
    defer lp.deinit();
    try testing.expectEqual(f.tokens[0], try lp.prefill(a, &ex, f.prompt));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    for (f.cycles, 0..) |cy, i| {
        var lg: CycleLog = .{ .primary = 0 };
        const fin = try lp.cycle(&ex, &out, a, &lg);
        errdefer std.debug.print("cycle {d} differs\n", .{i});
        try testing.expectEqual(cy.primary, lg.primary);
        try testing.expectEqual(cy.k_eff_native, lg.k_native);
        try testing.expectEqualSlices(u32, cy.drafts, lg.drafts[0..lg.k_eff]);
        try testing.expectEqual(cy.kept - 1, lg.accepted);
        try testing.expectEqual(cy.verified, lg.verified);
        try testing.expectEqual(cy.verified - cy.kept, lg.trimmed);
        // Uncut cycles emit the accepted drafts and their correction / bonus.
        if (cy.emitted.len == lg.accepted + 1) try testing.expectEqual(cy.emitted[lg.accepted], lg.correction);
        try testing.expectEqual(i + 1 == f.cycles.len, fin != null);
    }
    try testing.expectEqualSlices(u32, f.tokens[1..], out.items);
    try testing.expectEqualSlices(u32, f.stats.drafted_by_depth, lp.stats.drafted_by_depth[0..f.stats.drafted_by_depth.len]);
    try testing.expectEqualSlices(u32, f.stats.accepted_by_depth, lp.stats.accepted_by_depth[0..f.stats.accepted_by_depth.len]);
    try testing.expectEqual(f.stats.cycles, lp.stats.cycles);
    try testing.expectEqual(f.stats.bonus_tokens, lp.stats.bonus_tokens);
    try testing.expectEqual(f.stats.correction_tokens, lp.stats.correction_tokens);
    std.debug.print("dsv41 dspark loop: {d} recorded {s} cycles replay decision for decision ({d} tokens)\n", .{ f.cycles.len, f.arm, f.tokens.len });
}

// DSV41_DSPARK_MINI_CONFIG=<dump_dsv41_dspark_cycles.py --dump-config json>
test "dsv41 dspark loop: the fixture's mini config builds the model and head, and a full proposal takes the lookup" {
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_MINI_CONFIG") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    defer a.free(text);
    var diag: v41.Diag = .{};
    const c = v41.Config.parse(a, text, &diag) catch |e| {
        std.debug.print("mini config refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(u32, 5), c.dspark.block_size);
    try testing.expectEqual(@as(u32, 0), c.engram.n_layers);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const L = Loop(TraceOps);
    const model = try L.M.init(a, &g, c, try routes.parse(&.{}, null), &lookup, null);
    defer model.deinit(&g);
    const head = try L.H.init(a, &g, c, .{}, &lookup);
    defer head.deinit(&g);
    try testing.expectEqual(@as(usize, 2), head.nStages());
    var st = try model.newState();
    defer st.deinit(&g, a);
    var caches: [4]L.H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var rows: [8]u32 = @splat(0);
    var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
    defer src.deinit();
    var ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &g, &src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, &c);
    defer ex.deinit();
    try ex.grow(&g, grown[0..c.n_layers]);
    // Prompt "... 9 | 1 2 3 4 5 6 7 | ... 9": after the pick 9, the draft 1 2 3 4 5 is extended by 6 7.
    const prompt = [_]u32{ 30, 31, 8, 9, 1, 2, 3, 4, 5, 6, 7, 40, 41, 8 };
    var script: Script = .{
        .n_experts = @intCast(c.n_routed_experts),
        .pick = 9,
        .u32s = &.{ &.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 3, 4, 5, 6, 50, 51 } },
        .f32s = &.{&.{ 0.9, 0.9, 0.9, 0.9, 0.9 }},
    };
    g.host_values = script.values();
    var lp = L.init(&g, model, head, &st, caches[0..head.nStages()], .{ .max_tokens = 64 });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 8), lp.max_rows);
    try testing.expectEqual(@as(u32, 7), lp.stats.speculative_depth);
    _ = try lp.prefill(a, &ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var lg: CycleLog = .{ .primary = 0 };
    try testing.expect(try lp.cycle(&ex, &out, a, &lg) == null);
    try testing.expectEqual(@as(u32, 5), lg.k_native);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 7 }, lg.drafts[0..lg.k_eff]);
    // The target keeps 1..6 and says 50 at the second lookup depth.
    try testing.expectEqual(@as(u32, 6), lg.accepted);
    try testing.expectEqual(@as(u32, 50), lg.correction);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 50 }, out.items);
    try testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 1, 1, 1 }, lp.stats.drafted_by_depth[0..7]);
    try testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 1, 1, 0 }, lp.stats.accepted_by_depth[0..7]);
}
