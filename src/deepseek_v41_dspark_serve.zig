//! The served arm's decode seam over the native DSpark loop: the `D` of
//! `deepseek_v41_serve.Session` (`init(seed, rows, max_cycles)`, `begin(cfg)`,
//! `prefill(arm, g, prompt)`, `cycle(arm, g, a, out)`, `stats()`), bound as
//! `ServingDecode` once `arm.serving_decode` is `.dspark`.
//!
//! The loop's model and draft head are bound once, before the first request:
//! `open` binds the bank's residents on MLX (the text trunk at the stock tier,
//! the draft head's stages, the Engram rows); `attach` takes a model and head
//! built elsewhere (host tests). `deinit` releases them. The routed experts
//! are the arm's hook (`arm.hook`): its math carries the kernels' GEMV.
//!
//! Per request: `begin` maps the request (`depth` = the native draft depth the
//! head proposes, capped at its block; the lane's lookup extends a full
//! 5-token proposal to depth 7; `typical_delta` = #475 typical acceptance,
//! else greedy); `prefill` starts from a fresh target state and fresh draft
//! windows; every `cycle` appends the tokens the cycle committed. The Session
//! owns stop ids and `max_tokens`, so the loop runs uncapped (it never ends a
//! request itself) and `stats().generated_tokens` counts every committed token
//! (the primary included), as the stand-in's does.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const routes = @import("deepseek_v41_routes.zig");
const eng = @import("deepseek_v41_engram.zig");
const model_io = @import("model.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");

/// The loop's model and draft head over a bank's residents, owned: bound once
/// (every refusal named, before any request), released by `deinit` after the
/// last forward.
pub fn Resources(comptime G: type) type {
    const L = dsl.Loop(G);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        weights: model_io.Weights,
        engram: eng.RowSource,
        model: *L.M,
        head: *L.H,

        /// The text trunk at the stock tier, the draft head's stages and the
        /// Engram row source over `token_map` (the tokenizer's exported map).
        pub fn open(a: std.mem.Allocator, io: std.Io, g: *G, model_dir: []const u8, c: v41.Config, token_map: []const u8, diag: *v41.Diag) !*Self {
            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.a = a;
            self.weights = try model_io.loadWeights(io, a, model_dir);
            errdefer self.weights.deinit();
            self.engram = try eng.RowSource.open(a, io, model_dir, token_map, &c, diag);
            errdefer self.engram.deinit();
            self.model = try L.M.init(a, g, c, try routes.parse(&.{}, diag), &self.weights, &self.engram);
            errdefer self.model.deinit(g);
            self.head = try L.H.init(a, g, c, .{}, &self.weights);
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.head.deinit(g);
            self.model.deinit(g);
            self.engram.deinit();
            self.weights.deinit();
            self.a.destroy(self);
        }
    };
}

/// The DSpark decode seam of arm `A` (`A.Backend` is the graph backend).
pub fn Dspark(comptime A: type) type {
    const G = A.Backend;
    return struct {
        const Self = @This();
        pub const Loop = dsl.Loop(G);
        pub const Res = Resources(G);

        /// The seam's construction values (the stand-in's); `rows` follows each
        /// request's `depth + 1`.
        seed: u64,
        rows: u32,
        max_cycles: u32,
        model: *Loop.M = undefined,
        head: *const Loop.H = undefined,
        /// Set by `open` (released by `deinit`); null after `attach`.
        owned: ?*Res = null,
        /// The request's target state, draft windows and loop (set by `attach`).
        req: ?*Req = null,
        cfg: dsl.Config = .{ .max_tokens = std.math.maxInt(u32) },
        st: arm_mod.Stats = .{},

        const Req = struct {
            a: std.mem.Allocator,
            state: Loop.M.State = undefined,
            caches: []Loop.H.Cache,
            loop: Loop = undefined,
            live: bool = false,

            /// Releases the previous request's arrays (finished, cancelled or failed).
            fn drop(self: *Req, g: *G) void {
                if (!self.live) return;
                self.loop.deinit();
                for (self.caches) |*c| c.deinit(g);
                self.state.deinit(g, self.a);
                self.live = false;
            }
        };

        pub fn init(seed: u64, rows: u32, max_cycles: u32) Self {
            return .{ .seed = seed, .rows = rows, .max_cycles = max_cycles };
        }

        /// Construction on the served backend: the residents of `arm`'s model
        /// directory under its config, then `attach`.
        pub fn open(self: *Self, a: std.mem.Allocator, io: std.Io, arm: *const A, g: *G, token_map: []const u8, diag: *v41.Diag) !void {
            const res = try Res.open(a, io, g, arm.model_dir, arm.config, token_map, diag);
            errdefer res.deinit(g);
            try self.attach(a, res.model, res.head);
            self.owned = res;
        }

        /// Construction over a model and head the caller owns (they outlive the seam).
        pub fn attach(self: *Self, a: std.mem.Allocator, model: *Loop.M, head: *const Loop.H) !void {
            const r = try a.create(Req);
            errdefer a.destroy(r);
            r.* = .{ .a = a, .caches = try a.alloc(Loop.H.Cache, head.nStages()) };
            self.model = model;
            self.head = head;
            self.req = r;
        }

        pub fn deinit(self: *Self, g: *G) void {
            if (self.req) |r| {
                r.drop(g);
                r.a.free(r.caches);
                r.a.destroy(r);
                self.req = null;
            }
            if (self.owned) |res| res.deinit(g);
            self.owned = null;
        }

        /// A served request: its draft depth and acceptance; counters reset.
        pub fn begin(self: *Self, cfg: arm_mod.DecodeConfig) !void {
            self.cfg = .{
                .k_request = cfg.depth,
                .acceptance = if (cfg.typical_delta) |d| .{ .typical = .{ .delta = d } } else .greedy,
                .max_tokens = std.math.maxInt(u32),
            };
            self.rows = cfg.depth + 1;
            self.st = .{};
        }

        /// The prompt from a fresh target state and fresh draft windows; the primary token.
        pub fn prefill(self: *Self, arm: *A, g: *G, prompt: []const u32) !u32 {
            const r = self.req.?;
            r.drop(g);
            r.state = try self.model.newState();
            @memset(r.caches, .{});
            r.loop = Loop.init(g, self.model, self.head, &r.state, r.caches, self.cfg);
            r.live = true;
            const primary = try r.loop.prefill(arm.a, &arm.hook, prompt);
            self.st.generated_tokens = 1;
            return primary;
        }

        /// One DSpark cycle: appends its committed tokens (the accepted drafts
        /// and the correction); the Session ends the request.
        pub fn cycle(self: *Self, arm: *A, _: *G, a: std.mem.Allocator, out: *std.ArrayList(u32)) !bool {
            const r = self.req.?;
            const base = out.items.len;
            _ = try r.loop.cycle(&arm.hook, out, a, null);
            self.st.cycles = r.loop.stats.cycles;
            self.st.verify_calls = r.loop.stats.verify_calls;
            self.st.generated_tokens += @intCast(out.items.len - base);
            return false;
        }

        pub fn stats(self: *const Self) arm_mod.Stats {
            return self.st;
        }
    };
}

// ── Tests (host: the arm's synthetic bank, the mini model on the trace backend) ──

const testing = std.testing;
const ops = @import("deepseek_v41_ops.zig");
const mdl = @import("deepseek_v41_model.zig");
const serve = @import("deepseek_v41_serve.zig");
const TraceOps = ops.TraceOps;
const TraceArm = arm_mod.Arm(TraceOps, arm_mod.StandInMath(TraceOps));
const D = Dspark(TraceArm);
const TraceSession = serve.Session(TraceArm, D);

/// The host reads of a scripted run, in the loop's read order: each routing
/// barrier's ids (k distinct experts per row), the prompt's pick, per cycle
/// the draft ids, their sigmoid confidences, the verify targets and (typical)
/// the flags.
const Script = struct {
    n_experts: u16,
    k: u16,
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
        for (out, 0..) |*o, i| o.* = @intCast((i / s.k + i % s.k) % s.n_experts);
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

/// The served engine on the trace backend: the arm over the synthetic bank
/// (its hook routes every forward), the mini model and draft head attached.
const Rig = struct {
    tm: *arm_mod.TestModel,
    mini: *mdl.Mini,
    g: TraceOps,
    lookup: mdl.SpecLookup,
    arm: *TraceArm,
    model: *D.Loop.M,
    head: *D.Loop.H,
    session: TraceSession,

    fn create() !*Rig {
        const a = testing.allocator;
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.tm = try arm_mod.TestModel.create(true);
        errdefer r.tm.destroy();
        r.mini = try mdl.Mini.init();
        errdefer r.mini.deinit();
        r.g = TraceOps.init(a);
        errdefer r.g.deinit();
        var diag: arm_mod.Diag = .{};
        r.arm = TraceArm.init(a, testing.io, &r.g, {}, r.tm.options(), &diag) catch |e| {
            std.debug.print("dsv41 dspark serve: {s}\n", .{diag.message()});
            return e;
        };
        errdefer r.arm.deinit();
        r.lookup = .{ .g = &r.g, .spec = r.mini.spec };
        r.model = try D.Loop.M.init(a, &r.g, r.mini.c, try routes.parse(&.{}, null), &r.lookup, &r.mini.src);
        errdefer r.model.deinit(&r.g);
        r.head = try D.Loop.H.init(a, &r.g, r.mini.c, .{}, &r.lookup);
        errdefer r.head.deinit(&r.g);
        r.session = .{ .a = a, .io = testing.io, .arm = r.arm, .g = &r.g, .decode = D.init(0, 6, 0), .opts = .{}, .release = noRelease };
        try r.session.decode.attach(a, r.model, r.head);
        return r;
    }

    fn noRelease(_: *TraceSession) void {}

    fn script(r: *Rig, s: *Script) void {
        s.n_experts = @intCast(r.mini.c.n_routed_experts);
        s.k = @intCast(r.mini.c.n_experts_per_tok);
        r.g.host_values = s.values();
    }

    fn destroy(r: *Rig) void {
        r.session.engine().deinit();
        r.session.decode.deinit(&r.g);
        r.head.deinit(&r.g);
        r.model.deinit(&r.g);
        r.arm.deinit();
        r.g.deinit();
        r.mini.deinit();
        r.tm.destroy();
        testing.allocator.destroy(r);
    }
};

const Collect = struct {
    tokens: std.ArrayList(u32) = .empty,

    fn push(ctx: *anyopaque, t: u32) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.tokens.append(testing.allocator, t) catch @panic("oom");
    }

    fn sink(self: *Collect) serve.Sink {
        return .{ .ctx = self, .push = push };
    }
};

/// One request as the scheduler drives it: begin, then step until it finishes.
fn drive(e: serve.Engine, r: serve.Request, out: *Collect) !struct { finish: serve.Finish, steps: u32 } {
    try e.begin(r);
    var steps: u32 = 0;
    while (true) {
        steps += 1;
        if (try e.step(out.sink())) |f| return .{ .finish = f, .steps = steps };
    }
}

test "dsv41 dspark serve: a request runs the DSpark loop through the engine until max_tokens" {
    const rig = try Rig.create();
    defer rig.destroy();
    // The mini head drafts blocks of 2 (depth 5 capped); greedy. Cycle 1 accepts
    // both drafts (bonus 12); cycle 2 rejects at depth 2 (correction 22); cycle 3
    // stops early at one draft, accepted, whose correction 40 max_tokens cuts.
    var s: Script = .{
        .n_experts = 0,
        .k = 0,
        .pick = 3,
        .u32s = &.{ &.{ 10, 11 }, &.{ 10, 11, 12 }, &.{ 20, 21 }, &.{ 20, 22, 23 }, &.{ 30, 31 }, &.{ 30, 40 } },
        .f32s = &.{ &.{ 0.9, 0.9 }, &.{ 0.9, 0.9 }, &.{ 0.9, 0.2 } },
    };
    rig.script(&s);
    const e = rig.session.engine();
    var out: Collect = .{};
    defer out.tokens.deinit(testing.allocator);
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const res = try drive(e, .{ .prompt = &prompt, .max_tokens = 7, .stop_ids = &.{}, .acceptance = .greedy }, &out);
    try testing.expectEqual(serve.Finish.length, res.finish);
    try testing.expectEqual(@as(u32, 4), res.steps);
    try testing.expectEqualSlices(u32, &.{ 3, 10, 11, 12, 20, 22, 30 }, out.tokens.items);
    var run = try e.end();
    defer run.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, out.tokens.items, run.generated);
    try testing.expectEqual(@as(u32, 3), run.stats.cycles);
    try testing.expectEqual(@as(u32, 3), run.stats.verify_calls);
    // Every committed token: the primary, 3 + 2 + 2 (the Session cut the last).
    try testing.expectEqual(@as(u32, 8), run.stats.generated_tokens);
    // Prompt forwards of 8 + 4 rows, then 3 verify forwards: every layer routed through the arm's hook.
    try testing.expectEqual(@as(u64, 5 * rig.mini.c.n_layers), run.io_end.route_calls);
    try testing.expectEqual(s.u32s.len, s.nu);
    try testing.expectEqual(s.f32s.len, s.nf);
}

test "dsv41 dspark serve: a stop id ends the request unemitted; the next request starts from a fresh state" {
    const rig = try Rig.create();
    defer rig.destroy();
    var s: Script = .{
        .n_experts = 0,
        .k = 0,
        .pick = 3,
        .u32s = &.{ &.{ 10, 11 }, &.{ 10, 11, 12 } },
        .f32s = &.{&.{ 0.9, 0.9 }},
    };
    rig.script(&s);
    const e = rig.session.engine();
    var first: Collect = .{};
    defer first.tokens.deinit(testing.allocator);
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const r1 = try drive(e, .{ .prompt = &long, .max_tokens = 20, .stop_ids = &.{11}, .acceptance = .greedy }, &first);
    try testing.expectEqual(serve.Finish.stop, r1.finish);
    try testing.expectEqualSlices(u32, &.{ 3, 10 }, first.tokens.items);
    var run1 = try e.end();
    run1.deinit(testing.allocator);
    // The first request committed 12 + 3 rows; the second starts at 0 in the target and the draft windows.
    var second: Collect = .{};
    defer second.tokens.deinit(testing.allocator);
    const short = [_]u32{ 7, 8, 9 };
    const r2 = try drive(e, .{ .prompt = &short, .max_tokens = 1, .stop_ids = &.{}, .acceptance = .greedy }, &second);
    try testing.expectEqual(serve.Finish.length, r2.finish);
    try testing.expectEqualSlices(u32, &.{3}, second.tokens.items);
    const req = rig.session.decode.req.?;
    try testing.expectEqual(@as(u32, 3), req.state.offset);
    try testing.expectEqual(@as(u32, 3), req.caches[0].offset);
    var run2 = try e.end();
    defer run2.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 0), run2.stats.cycles);
    try testing.expectEqual(@as(u32, 1), run2.stats.generated_tokens);
}

test "dsv41 dspark serve: the request's depth and typical acceptance reach the loop" {
    const rig = try Rig.create();
    defer rig.destroy();
    // Depth 1: one draft kept of the head's block of 2, a 2-row verify. The
    // typical flag accepts draft 10 though the argmax says 50; the correction
    // stays that row's argmax, 51.
    var s: Script = .{
        .n_experts = 0,
        .k = 0,
        .pick = 3,
        .u32s = &.{ &.{ 10, 11 }, &.{ 50, 51 } },
        .f32s = &.{&.{ 0.9, 0.9 }},
        .bools = &.{&.{true}},
    };
    rig.script(&s);
    const e = rig.session.engine();
    var out: Collect = .{};
    defer out.tokens.deinit(testing.allocator);
    const prompt = [_]u32{ 5, 6, 7, 8, 9, 10 };
    const res = try drive(e, .{ .prompt = &prompt, .max_tokens = 3, .stop_ids = &.{}, .depth = 1, .acceptance = .{ .typical = 0.5 } }, &out);
    try testing.expectEqual(serve.Finish.length, res.finish);
    try testing.expectEqualSlices(u32, &.{ 3, 10, 51 }, out.tokens.items);
    const lp = &rig.session.decode.req.?.loop;
    try testing.expectEqual(@as(u32, 1), lp.k_cap);
    try testing.expectEqual(@as(u32, 2), lp.max_rows);
    try testing.expectEqual(@as(f32, 0.5), lp.cfg.acceptance.typical.delta);
    try testing.expectEqual(@as(usize, 1), s.nb);
    var run = try e.end();
    defer run.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 1), run.stats.cycles);
}
