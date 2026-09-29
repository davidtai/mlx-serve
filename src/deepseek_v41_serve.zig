//! The request path of the native deepseek_v41 arm in the server: the
//! scheduler hands a request's prompt ids to `Engine.begin`, then calls
//! `Engine.step` once per tick (the first step is the prefill and yields the
//! primary token, every later one is a decode cycle) and pushes what it emits;
//! stop ids end the request unemitted, `max_tokens` bounds it. One request at
//! a time: the scheduler refuses a second concurrent one by name. Every call
//! runs on the thread that built the arm's stream (the inference thread) or is
//! refused. `end` returns the request's run in the bench cell's shape, so the
//! cell's receipt writer serves per-request stats too.
//!
//! The decode loop is the arm's seam: `begin(cfg) !void`, `prefill(arm, g,
//! prompt) !u32`, `cycle(arm, g, a, out) !bool`, `stats()`. Which loop, math
//! and residents the server constructs is `deepseek_v41_bind`'s (`serving`,
//! `openServing`); until the DSpark loop binds, the server refuses by name.

const std = @import("std");
const mlx = @import("mlx.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const cell = @import("deepseek_v41_cell.zig");
const expert_lookahead = @import("expert_lookahead.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");

pub const Acceptance = union(enum) { greedy, typical: f32 };

/// Serve-level defaults: the lane of record (typical at delta 0.3, DSpark depth 5).
pub const Options = struct {
    acceptance: Acceptance = .{ .typical = 0.3 },
    depth: u32 = 5,
};

/// The serve options from the model's acceptance setting (`mtp_acceptance`:
/// exact = greedy, typical = its delta; none = the lane of record) and the
/// `--mtp-depth` cap (0 = the default depth).
pub fn optionsFrom(mode: ?mtp_acceptance.Mode, depth: u32) !Options {
    var o: Options = .{};
    if (mode) |m| o.acceptance = switch (m) {
        .exact => .greedy,
        .typical => |t| .{ .typical = t.delta },
        .tokenv3 => return error.Dsv41AcceptanceNotServed,
    };
    if (depth > max_depth) return error.DsparkDepthOutOfRange;
    if (depth > 0) o.depth = depth;
    return o;
}

/// A cycle verifies depth + 1 rows; the decode lane takes at most 8.
pub const max_depth: u32 = expert_lookahead.max_rows - 1;

pub const Request = struct {
    prompt: []const u32,
    max_tokens: u32,
    /// EOS and stop ids: never emitted, the request ends there.
    stop_ids: []const u32,
    /// 0 takes the serve default.
    depth: u32 = 0,
    acceptance: ?Acceptance = null,
};

pub const Finish = enum { stop, length };

/// Where emitted tokens go (the scheduler's slot).
pub const Sink = struct {
    ctx: *anyopaque,
    push: *const fn (ctx: *anyopaque, token: u32) void,
};

pub const Engine = struct {
    ptr: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        begin: *const fn (*anyopaque, Request) anyerror!void,
        step: *const fn (*anyopaque, Sink) anyerror!?Finish,
        end: *const fn (*anyopaque) anyerror!cell.Run,
        receipt: *const fn (*anyopaque, *const cell.Run, []const u8, *std.Io.Writer) anyerror!void,
        deinit: *const fn (*anyopaque) void,
    };

    pub fn begin(e: Engine, r: Request) !void {
        return e.vt.begin(e.ptr, r);
    }
    pub fn step(e: Engine, s: Sink) !?Finish {
        return e.vt.step(e.ptr, s);
    }
    /// The request's run (the caller frees it with the engine's allocator).
    pub fn end(e: Engine) !cell.Run {
        return e.vt.end(e.ptr);
    }
    /// The run as the cell's receipt at `path`, plus its log lines.
    pub fn receipt(e: Engine, run: *const cell.Run, path: []const u8, log: *std.Io.Writer) !void {
        return e.vt.receipt(e.ptr, run, path, log);
    }
    pub fn deinit(e: Engine) void {
        e.vt.deinit(e.ptr);
    }
};

/// The engine over arm `A` with decode loop `D`.
pub fn Session(comptime A: type, comptime D: type) type {
    const G = A.Backend;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        io: std.Io,
        arm: *A,
        g: *G,
        decode: D,
        opts: Options,
        /// Called at deinit (the owner of `arm` / `g`), or null when borrowed.
        release: ?*const fn (*Self) void = null,
        active: bool = false,
        done: bool = false,
        primed: bool = false,
        prompt: []u32 = &.{},
        stop_ids: []u32 = &.{},
        max_tokens: u32 = 0,
        depth: u32 = 0,
        generated: std.ArrayList(u32) = .empty,
        t0: std.Io.Timestamp = undefined,
        t1: std.Io.Timestamp = undefined,
        prompt_eval_s: f64 = 0,
        io_after_prefill: @import("expert_stream.zig").Stats = .{},
        requests: u64 = 0,

        pub fn engine(self: *Self) Engine {
            return .{ .ptr = self, .vt = &vt };
        }

        const vt: Engine.VTable = .{ .begin = beginFn, .step = stepFn, .end = endFn, .receipt = receiptFn, .deinit = deinitFn };

        fn of(p: *anyopaque) *Self {
            return @ptrCast(@alignCast(p));
        }

        fn onOwner(self: *const Self) !void {
            if (std.Thread.getCurrentId() != self.arm.stream.owner) return error.NotInferenceThread;
        }

        fn beginFn(p: *anyopaque, r: Request) anyerror!void {
            const self = of(p);
            try self.onOwner();
            if (r.prompt.len == 0) return error.EmptyPrompt;
            if (r.max_tokens == 0) return error.ZeroMaxTokens;
            const depth = if (r.depth == 0) self.opts.depth else r.depth;
            if (depth > max_depth) return error.DsparkDepthOutOfRange;
            const acc = r.acceptance orelse self.opts.acceptance;
            if (acc == .typical and !(acc.typical > 0 and acc.typical <= 1)) return error.TypicalThresholdOutOfRange;
            // A request left unfinished (cancelled, failed) is dropped here:
            // the scheduler runs one request at a time.
            self.clear();
            self.prompt = try self.a.dupe(u32, r.prompt);
            self.stop_ids = try self.a.dupe(u32, r.stop_ids);
            self.max_tokens = r.max_tokens;
            self.depth = depth;
            try self.decode.begin(.{
                .depth = depth,
                .typical_delta = if (acc == .typical) acc.typical else null,
                .seed = std.hash.Wyhash.hash(self.requests, std.mem.sliceAsBytes(r.prompt)),
            });
            if (G == ops.MlxOps) _ = mlx.mlx_reset_peak_memory();
            self.t0 = std.Io.Timestamp.now(self.io, .boot);
            self.active = true;
            self.requests += 1;
        }

        fn stepFn(p: *anyopaque, sink: Sink) anyerror!?Finish {
            const self = of(p);
            try self.onOwner();
            if (!self.active or self.done) return error.NoActiveRequest;
            var toks: std.ArrayList(u32) = .empty;
            defer toks.deinit(self.a);
            var loop_done = false;
            if (!self.primed) {
                try toks.append(self.a, try self.decode.prefill(self.arm, self.g, self.prompt));
                self.prompt_eval_s = seconds(self.t0.untilNow(self.io, .boot));
                self.io_after_prefill = self.arm.stream.stats();
                self.t1 = std.Io.Timestamp.now(self.io, .boot);
                if (!self.arm.grown) try self.arm.grow(self.g);
                self.primed = true;
            } else {
                loop_done = try self.decode.cycle(self.arm, self.g, self.a, &toks);
            }
            for (toks.items) |t| {
                if (std.mem.indexOfScalar(u32, self.stop_ids, t) != null) {
                    self.done = true;
                    return .stop;
                }
                sink.push(sink.ctx, t);
                try self.generated.append(self.a, t);
                if (self.generated.items.len >= self.max_tokens) {
                    self.done = true;
                    return .length;
                }
            }
            if (loop_done) {
                self.done = true;
                return .stop;
            }
            return null;
        }

        fn endFn(p: *anyopaque) anyerror!cell.Run {
            const self = of(p);
            try self.onOwner();
            if (!self.active) return error.NoActiveRequest;
            var mlx_peak: ?u64 = null;
            if (G == ops.MlxOps) {
                var peak: usize = 0;
                _ = mlx.mlx_get_peak_memory(&peak);
                mlx_peak = peak;
            }
            const run: cell.Run = .{
                .prompt = try self.a.dupe(u32, self.prompt),
                .generated = try self.a.dupe(u32, self.generated.items),
                .prompt_eval_s = self.prompt_eval_s,
                .decode_wall_s = if (self.primed) seconds(self.t1.untilNow(self.io, .boot)) else 0,
                .pass_wall_s = seconds(self.t0.untilNow(self.io, .boot)),
                .stats = self.decode.stats(),
                .io_after_prefill = self.io_after_prefill,
                .io_end = self.arm.stream.stats(),
                .footprint = arm_mod.footprint(),
                .mlx_peak_bytes = mlx_peak,
            };
            self.clear();
            return run;
        }

        fn receiptFn(p: *anyopaque, run: *const cell.Run, path: []const u8, log: *std.Io.Writer) anyerror!void {
            const self = of(p);
            const a = self.a;
            const spec: cell.Spec = .{ .prompt_tokens = @intCast(run.prompt.len), .cycles = run.stats.cycles, .rows = self.verifyRows(), .seed = 0 };
            const prompt_sha = try cell.idsSha256(a, run.prompt);
            const ids_sha = try cell.idsSha256(a, run.generated);
            const binding: arm_mod.DecodeBinding = if (D == arm_mod.StandIn(A)) .stand_in else .dspark;
            const r = cell.receiptOf(run, spec, binding, self.arm.admissionRecord(), .{ .DSV41_EXL3_BANK = self.arm.model_dir }, &prompt_sha, &ids_sha);
            try cell.publish(a, self.io, &r, path, log);
        }

        /// The rows a cycle verifies at most: the stand-in's depth + 1; the
        /// DSpark loop's own bound (its lookup extends a full 5-draft proposal
        /// by 2, so 8 at depth 5), set by the request's prefill.
        fn verifyRows(self: *const Self) u32 {
            if (D == arm_mod.StandIn(A)) return self.depth + 1;
            const r = self.decode.req orelse return self.depth + 1;
            return if (r.live) r.loop.max_rows else self.depth + 1;
        }

        fn deinitFn(p: *anyopaque) void {
            const self = of(p);
            self.clear();
            self.generated.deinit(self.a);
            if (self.release) |f| f(self) else self.a.destroy(self);
        }

        fn clear(self: *Self) void {
            self.a.free(self.prompt);
            self.a.free(self.stop_ids);
            self.prompt = &.{};
            self.stop_ids = &.{};
            self.generated.clearRetainingCapacity();
            self.active = false;
            self.done = false;
            self.primed = false;
        }
    };
}

/// A finished request's one log line (the scheduler's, and the served path's gate's).
pub fn writeRequestLine(run: *const cell.Run, w: *std.Io.Writer) std.Io.Writer.Error!void {
    const n = run.generated.len;
    try w.print("[dsv41] request: {d} prompt, {d} generated in {d} cycles ({d:.2} tok/cycle), prefill {d:.3} s, decode {d:.2} tok/s\n", .{
        run.prompt.len,
        n,
        run.stats.cycles,
        if (run.stats.cycles > 0) @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(run.stats.cycles)) else 0,
        run.prompt_eval_s,
        if (run.decode_wall_s > 0) @as(f64, @floatFromInt(n -| 1)) / run.decode_wall_s else 0,
    });
}

fn seconds(d: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e9;
}

// ── Tests ──

const testing = std.testing;
const TraceArm = arm_mod.Arm(ops.TraceOps, arm_mod.StandInMath(ops.TraceOps));
const TraceSession = Session(TraceArm, arm_mod.StandIn(TraceArm));

const Collect = struct {
    tokens: std.ArrayList(u32) = .empty,

    fn push(ctx: *anyopaque, t: u32) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.tokens.append(testing.allocator, t) catch @panic("oom");
    }

    fn sink(self: *Collect) Sink {
        return .{ .ctx = self, .push = push };
    }
};

/// A stand-in engine on the mini model over a synthetic bank (host rows, no MLX).
const HostEngine = struct {
    tm: *arm_mod.TestModel,
    g: ops.TraceOps,
    arm: *TraceArm,
    session: TraceSession,

    fn create() !*HostEngine {
        const a = testing.allocator;
        const self = try a.create(HostEngine);
        errdefer a.destroy(self);
        self.tm = try arm_mod.TestModel.create(true);
        errdefer self.tm.destroy();
        self.g = ops.TraceOps.init(a);
        errdefer self.g.deinit();
        var diag: arm_mod.Diag = .{};
        self.arm = try TraceArm.init(a, std.testing.io, &self.g, {}, self.tm.options(), &diag);
        self.session = .{ .a = a, .io = std.testing.io, .arm = self.arm, .g = &self.g, .decode = arm_mod.StandIn(TraceArm).init(0, 6, 0), .opts = .{}, .release = noRelease };
        self.session.decode.bind(&self.g);
        return self;
    }

    fn noRelease(_: *TraceSession) void {}

    fn destroy(self: *HostEngine) void {
        self.session.engine().deinit();
        self.arm.deinit();
        self.g.deinit();
        self.tm.destroy();
        testing.allocator.destroy(self);
    }
};

/// Drives one request as the scheduler does: begin, then step until it finishes.
fn drive(e: Engine, r: Request, out: *Collect) !struct { finish: Finish, steps: u32 } {
    try e.begin(r);
    var steps: u32 = 0;
    while (true) {
        steps += 1;
        if (try e.step(out.sink())) |f| return .{ .finish = f, .steps = steps };
    }
}

test "dsv41 serve: a request runs prefill then cycles until max_tokens, the primary token first" {
    const h = try HostEngine.create();
    defer h.destroy();
    const e = h.session.engine();
    var out: Collect = .{};
    defer out.tokens.deinit(testing.allocator);
    const res = try drive(e, .{ .prompt = &.{ 1, 2, 3, 4, 5, 6, 7 }, .max_tokens = 9, .stop_ids = &.{} }, &out);
    try testing.expectEqual(Finish.length, res.finish);
    try testing.expectEqual(@as(usize, 9), out.tokens.items.len);
    try testing.expect(res.steps >= 3);
    var run = try e.end();
    defer run.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, out.tokens.items, run.generated);
    try testing.expectEqual(res.steps - 1, run.stats.cycles);
    // Depth 5 by default: verify forwards of 6 rows; prefill of 7 tokens = 2 forwards.
    try testing.expectEqual(@as(u64, (2 + res.steps - 1) * 5), run.io_end.route_calls);
    try testing.expectError(error.NoActiveRequest, e.end());
}

test "dsv41 serve: a stop id ends the request unemitted" {
    const h = try HostEngine.create();
    defer h.destroy();
    const e = h.session.engine();
    // The stand-in's token stream for this prompt, then the same request stopping at its 4th token.
    var first: Collect = .{};
    defer first.tokens.deinit(testing.allocator);
    _ = try drive(e, .{ .prompt = &.{ 9, 8, 7 }, .max_tokens = 12, .stop_ids = &.{} }, &first);
    var r0 = try e.end();
    r0.deinit(testing.allocator);
    const stop = first.tokens.items[3];
    const want = std.mem.indexOfScalar(u32, first.tokens.items, stop).?;
    h.session.requests = 0;
    var second: Collect = .{};
    defer second.tokens.deinit(testing.allocator);
    const res = try drive(e, .{ .prompt = &.{ 9, 8, 7 }, .max_tokens = 12, .stop_ids = &.{stop} }, &second);
    try testing.expectEqual(Finish.stop, res.finish);
    try testing.expectEqualSlices(u32, first.tokens.items[0..want], second.tokens.items);
    var r1 = try e.end();
    r1.deinit(testing.allocator);
}

test "dsv41 serve: request options are checked before anything runs, and a dropped request is recycled" {
    const h = try HostEngine.create();
    defer h.destroy();
    const e = h.session.engine();
    try testing.expectError(error.EmptyPrompt, e.begin(.{ .prompt = &.{}, .max_tokens = 4, .stop_ids = &.{} }));
    try testing.expectError(error.ZeroMaxTokens, e.begin(.{ .prompt = &.{1}, .max_tokens = 0, .stop_ids = &.{} }));
    try testing.expectError(error.DsparkDepthOutOfRange, e.begin(.{ .prompt = &.{1}, .max_tokens = 4, .stop_ids = &.{}, .depth = max_depth + 1 }));
    try testing.expectError(error.TypicalThresholdOutOfRange, e.begin(.{ .prompt = &.{1}, .max_tokens = 4, .stop_ids = &.{}, .acceptance = .{ .typical = 0 } }));
    var s: Collect = .{};
    defer s.tokens.deinit(testing.allocator);
    try testing.expectError(error.NoActiveRequest, e.step(s.sink()));
    // Depth 2: 3-row cycles. A request dropped after its prefill is replaced by the next begin.
    try e.begin(.{ .prompt = &.{ 1, 2 }, .max_tokens = 50, .stop_ids = &.{}, .depth = 2 });
    _ = try e.step(s.sink());
    try testing.expectEqual(@as(u32, 3), h.session.decode.rows);
    const res = try drive(e, .{ .prompt = &.{3}, .max_tokens = 2, .stop_ids = &.{}, .acceptance = .greedy }, &s);
    try testing.expectEqual(Finish.length, res.finish);
    try testing.expectEqual(@as(u32, 6), h.session.decode.rows);
    var run = try e.end();
    defer run.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), run.generated.len);
    try testing.expectEqual(@as(u64, 2), h.session.requests);
}

test "dsv41 serve: every engine call from a thread other than the inference thread is refused" {
    const h = try HostEngine.create();
    defer h.destroy();
    const e = h.session.engine();
    const Helper = struct {
        fn run(eng: Engine, out: *[3]?anyerror) void {
            out[0] = if (eng.begin(.{ .prompt = &.{1}, .max_tokens = 1, .stop_ids = &.{} })) null else |err| err;
            var c: Collect = .{};
            out[1] = if (eng.step(c.sink())) |_| null else |err| err;
            out[2] = if (eng.end()) |_| null else |err| err;
        }
    };
    var got: [3]?anyerror = .{ null, null, null };
    const t = try std.Thread.spawn(.{}, Helper.run, .{ e, &got });
    t.join();
    for (got) |g| try testing.expectEqual(@as(?anyerror, error.NotInferenceThread), g);
}

test "dsv41 serve: a served request's run writes the cell's receipt" {
    const h = try HostEngine.create();
    defer h.destroy();
    const e = h.session.engine();
    var out: Collect = .{};
    defer out.tokens.deinit(testing.allocator);
    _ = try drive(e, .{ .prompt = &.{ 4, 5, 6 }, .max_tokens = 7, .stop_ids = &.{} }, &out);
    var run = try e.end();
    defer run.deinit(testing.allocator);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/serve-1.comparison.json", .{h.tm.root});
    defer testing.allocator.free(path);
    var log: std.Io.Writer.Allocating = .init(testing.allocator);
    defer log.deinit();
    try e.receipt(&run, path, &log.writer);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(text);
    const Back = struct { kind: []const u8, decode_binding: []const u8, generated_ids: []const u32, stats: struct { cycles: u32 }, selection: struct { policy: struct { prompt_tokens: u32, verify_rows: u32 } } };
    const back = try std.json.parseFromSlice(Back, testing.allocator, text, .{ .ignore_unknown_fields = true });
    defer back.deinit();
    try testing.expectEqualStrings(cell.receipt_kind, back.value.kind);
    try testing.expectEqualStrings("stand_in", back.value.decode_binding);
    try testing.expectEqualSlices(u32, out.tokens.items, back.value.generated_ids);
    try testing.expectEqual(run.stats.cycles, back.value.stats.cycles);
    try testing.expectEqual(@as(u32, 3), back.value.selection.policy.prompt_tokens);
    try testing.expectEqual(@as(u32, 6), back.value.selection.policy.verify_rows);
    try testing.expect(std.mem.indexOf(u8, log.written(), "COMPARISON_COMPLETE") != null);
}

test "dsv41 serve: the serve options follow the model's acceptance setting and the depth cap" {
    const d = try optionsFrom(null, 0);
    try testing.expectEqual(@as(f32, 0.3), d.acceptance.typical);
    try testing.expectEqual(@as(u32, 5), d.depth);
    try testing.expect((try optionsFrom(.exact, 3)).acceptance == .greedy);
    try testing.expectEqual(@as(u32, 3), (try optionsFrom(.exact, 3)).depth);
    try testing.expectEqual(@as(f32, 0.2), (try optionsFrom(.{ .typical = .{ .delta = 0.2 } }, 0)).acceptance.typical);
    try testing.expectError(error.Dsv41AcceptanceNotServed, optionsFrom(.{ .tokenv3 = 0.5 }, 0));
    try testing.expectError(error.DsparkDepthOutOfRange, optionsFrom(null, max_depth + 1));
}
