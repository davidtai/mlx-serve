//! The routed experts of the native DeepSeek-V4.1 model over the expert
//! streamer (track M, M3). Oracle: the EXL3 decode lane of the Python tier,
//! `plane_lane.PackedDecode.run` driving `exl3_lane.Exl3PackedOps`.
//!
//! Per routed-layer call the trunk hands the hook `xf [n, hidden]` and the
//! router's `indices [n, k]`. The hook reads the ids on the host (the routing
//! barrier), asks an expert source for their slots (`route`), runs the
//! residents' wave at once, then each miss part's gate/up once its gate/up
//! bytes landed (`waitGu`) and its down once the rest landed (`waitDown`),
//! and hands the call back (`release`: the rows stay pinned until the next
//! route's flush, after the eval that consumed them). The outputs come back
//! in the router's order as `[n, k, hidden]` f32; the trunk combines them.
//! `grow` is the one phase change; `flush` follows the forward's last eval.
//!
//! Sources: `StreamSource` (the streamer's `Stream`) and `FakeSource` (the
//! streamer's residency policy with no reads, for host tests). The math of a
//! group of rows sharing a bank is a seam: `EagerChain` is the exact tier's
//! op chain around the EXL3 decode GEMV, which the kernels lane binds
//! (`MlxGemv`); `TraceGemv` / `TraceMath` stand in on the trace backend.

const std = @import("std");
const mlx = @import("mlx.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("expert_io.zig");
const expert_policy = @import("expert_policy.zig");
const expert_stream = @import("expert_stream.zig");
const expert_lookahead = @import("expert_lookahead.zig");
const expert_event = @import("expert_event.zig");

pub const max_route_ids = expert_policy.max_route_ids;
pub const BankKind = expert_stream.BankKind;
pub const SlotRef = expert_stream.SlotRef;
pub const Stats = expert_stream.Stats;
pub const Error = expert_stream.Error;
const Load = expert_policy.Load;
const n_banks = std.meta.fieldNames(BankKind).len;

/// What one routed-layer call serves, valid until the call is released. Per
/// routed id, in the router's flat order: the slot's bank and row, and the
/// wave that computes it (0 = resident at the call; p + 1 = miss part p,
/// whose gate/up may run after `waitGu(p)` and its down after `waitDown(p)`).
pub const Served = struct {
    refs: []const SlotRef,
    waves: []const u8,
    n_parts: u32,
};

/// One projection's slot arrays in a bank (the streamer's `ProjArrays` over
/// any backend): code int16 [rows, in/16, out/16, 16K], rout f16 [rows, out],
/// rin f16 [rows, in].
pub fn ProjOf(comptime T: type) type {
    return struct { code: T, rout: T, rin: T };
}

/// A bank's nine arrays by projection (the streamer's `BankArrays`).
pub fn BankArraysOf(comptime T: type) type {
    return struct { gate: ProjOf(T), up: ProjOf(T), down: ProjOf(T) };
}

// ── The source contract ──

/// Compile-time check that `S` is an expert source:
///   `Call` (a live route);
///   `route(*S, layer, ids, scores) Error!*Call` (scores: the next routed
///       layer's gate scores for the lookahead, or empty);
///   `served(*S, *const Call) Served`;
///   `waitGu(*S, *Call, part) Error!void`, `waitDown(*S, *Call, part) Error!void`;
///   `release(*S, *Call) void` (after the call's waves are built);
///   `flush(*S) Error!void` (after the forward's last eval);
///   `grow(*S, decode_rows) !void` (the one phase change);
///   `stats(*S) Stats`;
///   `bankRows(*S, layer, BankKind) u32` and `bankArrays(*S, g, layer, BankKind)
///       !?BankArraysOf(G.T)` (what the math binds; the latter generic over the backend).
pub fn assertSource(comptime S: type) void {
    comptime {
        if (!@hasDecl(S, "Call")) @compileError(@typeName(S) ++ " is not an expert source: no Call");
        expectMethod(S, "route", &.{ *S, u32, []const u16, []const f32 }, *S.Call);
        expectMethod(S, "served", &.{ *S, *const S.Call }, Served);
        expectMethod(S, "waitGu", &.{ *S, *S.Call, u32 }, void);
        expectMethod(S, "waitDown", &.{ *S, *S.Call, u32 }, void);
        expectMethod(S, "release", &.{ *S, *S.Call }, void);
        expectMethod(S, "flush", &.{*S}, void);
        expectMethod(S, "grow", &.{ *S, []const u32 }, void);
        expectMethod(S, "stats", &.{*S}, Stats);
        expectMethod(S, "bankRows", &.{ *S, u32, BankKind }, u32);
        if (!@hasDecl(S, "bankArrays")) @compileError(@typeName(S) ++ " is not an expert source: no bankArrays");
    }
}

fn expectMethod(comptime S: type, comptime name: []const u8, comptime params: []const type, comptime Payload: type) void {
    const where = @typeName(S) ++ "." ++ name;
    if (!@hasDecl(S, name)) @compileError(@typeName(S) ++ " is not an expert source: no " ++ name);
    const info = @typeInfo(@TypeOf(@field(S, name))).@"fn";
    if (info.param_types.len != params.len) @compileError(where ++ ": the contract takes a different parameter count");
    for (info.param_types, params) |p, t| {
        if (p.? != t) @compileError(where ++ ": parameter " ++ @typeName(p.?) ++ " where the contract has " ++ @typeName(t));
    }
    const R = info.return_type.?;
    const P = switch (@typeInfo(R)) {
        .error_union => |eu| eu.payload,
        else => R,
    };
    if (P != Payload) @compileError(where ++ ": returns " ++ @typeName(P) ++ " where the contract has " ++ @typeName(Payload));
}

/// Per routed id of `r`, its wave: 0 for a slot resident at the call (a hit),
/// p + 1 for a slot part p loads.
fn wavesOf(plan: *const expert_policy.Plan, hit_slots: []const u32, part_loads: []const []const Load, out: []u8) void {
    var keys: [max_route_ids]u32 = undefined;
    var vals: [max_route_ids]u8 = undefined;
    var n: usize = 0;
    for (hit_slots) |s| {
        keys[n] = s;
        vals[n] = 0;
        n += 1;
    }
    for (part_loads, 1..) |loads, w| for (loads) |l| {
        keys[n] = l.slot;
        vals[n] = @intCast(w);
        n += 1;
    };
    for (plan.slotsOf(), out) |s, *w| w.* = vals[std.mem.indexOfScalar(u32, keys[0..n], s).?];
}

/// Trace arrays in one record's geometry, `rows` rows.
fn traceBank(g: *ops.TraceOps, geom: *const expert_bank.Layer, rows: u32) !BankArraysOf(u32) {
    var a: [expert_bank.n_components]u32 = undefined;
    for (&a, geom.segments) |*x, seg| {
        var shape: [4]c_int = undefined;
        shape[0] = @intCast(rows);
        for (seg.shape[0..seg.rank], 1..) |d, i| shape[i] = @intCast(d);
        x.* = try g.input(shape[0 .. seg.rank + 1], switch (seg.dtype) {
            .I16 => .int16,
            .F16 => .float16,
        });
    }
    return .{
        .gate = .{ .code = a[0], .rout = a[1], .rin = a[2] },
        .up = .{ .code = a[3], .rout = a[4], .rin = a[5] },
        .down = .{ .code = a[6], .rout = a[7], .rin = a[8] },
    };
}

// ── StreamSource: the streamer's Stream ──

/// The streamer as an expert source. Routes, waits, release, flush and growth
/// are the Stream's own; the call view comes from `refsOf` and the parts'
/// loads (`Route.partLoads`); the MLX arrays from `bankArrays` (slot memory
/// `.mlx`: with host rows the MLX executor refuses at construction).
pub const StreamSource = struct {
    stream: *expert_stream.Stream,
    calls: [n_routes]Call = @splat(.{}),

    const n_routes = @typeInfo(@FieldType(expert_stream.Stream, "routes")).array.len;

    pub const Call = struct {
        route: ?*expert_stream.Route = null,
        n_ids: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub fn init(stream: *expert_stream.Stream) StreamSource {
        return .{ .stream = stream };
    }

    /// Calls mirror the Stream's route ring one to one.
    fn callOf(self: *StreamSource, r: *expert_stream.Route) *Call {
        const i = (@intFromPtr(r) - @intFromPtr(&self.stream.routes[0])) / @sizeOf(expert_stream.Route);
        return &self.calls[i];
    }

    pub fn route(self: *StreamSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        const r = try self.stream.route(layer, ids, scores);
        const call = self.callOf(r);
        call.* = .{ .route = r, .n_ids = @intCast(ids.len) };
        _ = self.stream.refsOf(r, &call.refs);
        var bufs: [max_route_ids][max_route_ids]Load = undefined;
        var parts: [max_route_ids][]const Load = undefined;
        for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
        wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], call.waves[0..ids.len]);
        return call;
    }

    pub fn served(_: *StreamSource, call: *const Call) Served {
        return .{ .refs = call.refs[0..call.n_ids], .waves = call.waves[0..call.n_ids], .n_parts = call.route.?.n_parts };
    }

    pub fn waitGu(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitGu(call.route.?, part);
    }

    pub fn waitDown(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitDown(call.route.?, part);
    }

    pub fn release(self: *StreamSource, call: *Call) void {
        self.stream.release(call.route.?);
    }

    pub fn flush(self: *StreamSource) Error!void {
        return self.stream.flush();
    }

    pub fn grow(self: *StreamSource, decode_rows: []const u32) !void {
        return self.stream.grow(decode_rows);
    }

    pub fn stats(self: *StreamSource) Stats {
        return self.stream.stats();
    }

    /// Event gates of the call's reads (a stream built with `event`).
    pub fn gate(self: *StreamSource, call: *Call) Error!?expert_stream.Gates {
        return self.stream.gate(call.route.?);
    }

    pub fn bankRows(self: *StreamSource, layer: u32, kind: BankKind) u32 {
        const ls = &self.stream.layers[layer];
        return switch (kind) {
            .base => ls.base.rows,
            .ext => if (ls.ext) |e| e.rows else 0,
            .transient => self.stream.transient.rows,
        };
    }

    /// MLX: the stream's own arrays (never freed here); null for host rows or
    /// a bank without rows. Trace: inputs in the bank's geometry.
    pub fn bankArrays(self: *StreamSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        const G = @TypeOf(g.*);
        if (G == ops.MlxOps) {
            const b = self.stream.bankArrays(layer, kind) orelse return null;
            return .{
                .gate = .{ .code = b.gate.code, .rout = b.gate.rout, .rin = b.gate.rin },
                .up = .{ .code = b.up.code, .rout = b.up.rout, .rin = b.up.rin },
                .down = .{ .code = b.down.code, .rout = b.down.rout, .rin = b.down.rin },
            };
        } else if (G == ops.TraceOps) {
            const rows = self.bankRows(layer, kind);
            if (rows == 0) return null;
            return try traceBank(g, &self.stream.bank.layers[layer], rows);
        } else @compileError("StreamSource binds MlxOps or TraceOps arrays");
    }
};

// ── FakeSource: the streamer's residency with no reads (host tests) ──

/// The streamer's residency policy (one `LayerPolicy` per layer, the Stream's
/// part cutting and slot arithmetic) with every load landed at once and no
/// memory: the fixture-backed stand-in for host tests. It keeps the Stream's
/// call rules (a route flushes released calls first; four calls at most;
/// `grow` once, with every call released) and logs every call it gets.
pub const FakeSource = struct {
    a: std.mem.Allocator,
    geom: expert_bank.Layer,
    n_experts: u32,
    policies: []expert_policy.LayerPolicy,
    base_rows: []u32,
    ext_rows: []u32,
    transient_rows: u32,
    records_per_part: u32,
    phase: expert_policy.Phase = .prefill,
    calls: [4]Call = @splat(.{}),
    log: std.ArrayList(Event) = .empty,
    counters: Stats = .{},
    /// When set, events are stamped with its node count (op order checks).
    trace: ?*const ops.TraceOps = null,
    /// The lookahead's selection on the next layer's gate scores (the Stream's
    /// `speculate`), logged per call that passes scores.
    selector: ?expert_lookahead.Selector = null,
    picks: std.ArrayList(Pick) = .empty,
    gate_value: u64 = 0,

    pub const Pick = struct { layer: u32, n: u8 = 0, experts: [expert_lookahead.max_budget]u16 = undefined };

    pub const Call = struct {
        state: enum { free, live, released } = .free,
        layer: u32 = 0,
        plan: expert_policy.Plan = .{},
        n_parts: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub const Event = struct {
        pub const Kind = enum { route, wait_gu, wait_down, release, flush, grow, gate };
        kind: Kind,
        layer: u32 = 0,
        part: u32 = 0,
        /// Trace nodes recorded before the event (0 without `trace`).
        at: usize = 0,
    };

    pub const Options = struct {
        hidden: u64,
        inter: u64,
        n_experts: u32,
        /// Persistent rows per layer before `grow`.
        rows: []const u32,
        transient_rows: u32 = max_route_ids,
        records_per_part: u32 = 3,
    };

    pub fn init(a: std.mem.Allocator, opt: Options) !FakeSource {
        const geom = expert_bank.layerSegments(3, opt.hidden, opt.inter) orelse return error.InvalidGeometry;
        const n = opt.rows.len;
        const policies = try a.alloc(expert_policy.LayerPolicy, n);
        errdefer a.free(policies);
        var n_init: usize = 0;
        errdefer for (policies[0..n_init]) |*p| p.deinit(a);
        for (policies, opt.rows) |*p, rows| {
            p.* = try expert_policy.LayerPolicy.init(a, opt.n_experts, rows);
            n_init += 1;
        }
        const base = try a.dupe(u32, opt.rows);
        errdefer a.free(base);
        const ext = try a.alloc(u32, n);
        @memset(ext, 0);
        return .{
            .a = a,
            .geom = geom,
            .n_experts = opt.n_experts,
            .policies = policies,
            .base_rows = base,
            .ext_rows = ext,
            .transient_rows = opt.transient_rows,
            .records_per_part = opt.records_per_part,
        };
    }

    pub fn deinit(self: *FakeSource) void {
        if (self.selector) |*sel| sel.deinit(self.a);
        self.picks.deinit(self.a);
        for (self.policies) |*p| p.deinit(self.a);
        self.a.free(self.policies);
        self.a.free(self.base_rows);
        self.a.free(self.ext_rows);
        self.log.deinit(self.a);
        self.* = undefined;
    }

    fn note(self: *FakeSource, ev: Event) void {
        var e = ev;
        if (self.trace) |t| e.at = t.nodes.items.len;
        self.log.append(self.a, e) catch @panic("fake source log: out of memory");
    }

    pub fn seedPrefill(self: *FakeSource, layer: u32, ids: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        try self.policies[layer].prepareSeed(self.a, ids);
    }

    fn slotRef(self: *const FakeSource, layer: u32, slot: u32) SlotRef {
        const cap = self.policies[layer].capacity;
        if (slot < self.base_rows[layer]) return .{ .bank = .base, .row = slot };
        if (slot < cap) return .{ .bank = .ext, .row = slot - self.base_rows[layer] };
        return .{ .bank = .transient, .row = slot - cap };
    }

    pub fn route(self: *FakeSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        std.debug.assert(ids.len > 0 and ids.len <= max_route_ids);
        std.debug.assert(scores.len == 0 or (self.selector != null and self.phase == .decode));
        try self.flush();
        const call = for (&self.calls) |*c| {
            if (c.state == .free) break c;
        } else return error.RoutesExhausted;
        call.* = .{ .layer = layer };
        const plan = &call.plan;
        self.policies[layer].plan(ids, self.phase, plan);
        // A row a live call still serves from is never refilled.
        for (plan.loadsOf()) |l| for (&self.calls) |*o| {
            if (o == call or o.state != .live or o.layer != layer) continue;
            if (std.mem.indexOfScalar(u32, o.plan.slotsOf(), l.slot) != null) return error.SlotStillPinned;
        };
        // The Stream's parts: loads in file (expert) order; decode = bounded
        // parts of `records_per_part` (no two records touch), prefill = pool-job chunks.
        var order: [max_route_ids]Load = undefined;
        const n = plan.n_loads;
        @memcpy(order[0..n], plan.loadsOf());
        std.sort.insertion(Load, order[0..n], {}, struct {
            fn less(_: void, x: Load, y: Load) bool {
                return x.expert < y.expert;
            }
        }.less);
        var offsets: [max_route_ids]u64 = undefined;
        var lengths: [max_route_ids]u64 = undefined;
        for (order[0..n], 0..) |l, i| {
            offsets[i] = @as(u64, l.expert) * self.geom.record_bytes;
            lengths[i] = self.geom.logical_bytes;
        }
        var ends_buf: [max_route_ids]u32 = undefined;
        const ends = if (self.phase == .decode)
            expert_policy.boundedParts(offsets[0..n], lengths[0..n], self.records_per_part, &ends_buf)
        else blk: {
            var k: u32 = 0;
            var e: u32 = 0;
            while (e < n) : (k += 1) {
                e = @min(e + expert_io.max_items, n);
                ends_buf[k] = e;
            }
            break :blk ends_buf[0..k];
        };
        var parts: [max_route_ids][]const Load = undefined;
        var start: u32 = 0;
        for (ends, 0..) |end, p| {
            parts[p] = order[start..end];
            start = end;
        }
        call.n_parts = @intCast(ends.len);
        var hit_slots: [max_route_ids]u32 = undefined;
        for (plan.hitsOf(), hit_slots[0..plan.n_hits]) |e, *s| s.* = self.policies[layer].slotOf(e).?;
        wavesOf(plan, hit_slots[0..plan.n_hits], parts[0..ends.len], call.waves[0..ids.len]);
        for (plan.slotsOf(), call.refs[0..ids.len]) |s, *r| r.* = self.slotRef(layer, s);
        const c = &self.counters;
        c.route_calls += 1;
        c.expert_cache_hits += plan.n_hits;
        c.expert_cache_misses += plan.n_misses;
        c.expert_cache_evictions += plan.n_evictions;
        for (plan.loadsOf()) |l| {
            if (l.persistent) c.persistent_loads += 1 else c.transient_loads += 1;
        }
        if (scores.len > 0 and layer + 1 < self.policies.len) {
            const sel = &self.selector.?;
            var pick: Pick = .{ .layer = layer + 1 };
            pick.n = @intCast(sel.select(scores, &self.policies[layer + 1], pick.experts[0..sel.budget]).len);
            self.picks.append(self.a, pick) catch @panic("fake source picks: out of memory");
        }
        call.state = .live;
        self.note(.{ .kind = .route, .layer = layer });
        return call;
    }

    /// The Stream's gate values: the call's gate/up wave at `gu`, part p's down at `down_first + p`.
    pub fn gate(self: *FakeSource, call: *Call) Error!?expert_stream.Gates {
        std.debug.assert(call.state == .live);
        if (call.n_parts == 0) return null;
        const lo = self.gate_value;
        self.gate_value = lo + 1 + call.n_parts;
        self.note(.{ .kind = .gate, .layer = call.layer, .part = call.n_parts });
        return .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = call.n_parts };
    }

    pub fn served(_: *FakeSource, call: *const Call) Served {
        const n = call.plan.n_ids;
        return .{ .refs = call.refs[0..n], .waves = call.waves[0..n], .n_parts = call.n_parts };
    }

    pub fn waitGu(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_gu, .layer = call.layer, .part = part });
    }

    pub fn waitDown(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_down, .layer = call.layer, .part = part });
    }

    pub fn release(self: *FakeSource, call: *Call) void {
        std.debug.assert(call.state == .live);
        call.state = .released;
        self.note(.{ .kind = .release, .layer = call.layer });
    }

    pub fn flush(self: *FakeSource) Error!void {
        var any = false;
        for (&self.calls) |*c| if (c.state == .released) {
            c.state = .free;
            any = true;
        };
        if (any) self.note(.{ .kind = .flush });
    }

    pub fn grow(self: *FakeSource, decode_rows: []const u32) !void {
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (decode_rows.len != self.policies.len) return error.InvalidRows;
        try self.flush();
        for (&self.calls) |*c| if (c.state != .free) return error.RoutesLive;
        for (self.policies, decode_rows) |*p, rows| {
            if (rows < p.capacity or rows > self.n_experts) return error.InvalidRows;
        }
        for (self.policies, decode_rows, self.ext_rows) |*p, rows, *ext| {
            ext.* = rows - p.capacity;
            p.grow(rows) catch unreachable;
        }
        self.phase = .decode;
        self.note(.{ .kind = .grow });
    }

    pub fn stats(self: *FakeSource) Stats {
        return self.counters;
    }

    pub fn bankRows(self: *FakeSource, layer: u32, kind: BankKind) u32 {
        return switch (kind) {
            .base => self.base_rows[layer],
            .ext => self.ext_rows[layer],
            .transient => self.transient_rows,
        };
    }

    pub fn bankArrays(self: *FakeSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        if (@TypeOf(g.*) != ops.TraceOps) @compileError("FakeSource binds trace arrays only");
        const rows = self.bankRows(layer, kind);
        if (rows == 0) return null;
        return try traceBank(g, &self.geom, rows);
    }

    /// Calls that are neither free nor released (tests).
    pub fn liveCalls(self: *const FakeSource) usize {
        var n: usize = 0;
        for (self.calls) |c| n += @intFromBool(c.state == .live);
        return n;
    }
};

// ── The math seam ──

/// `128 ** -0.5` as the Python float rounds to the f32 MLX takes.
const t128_scale: f32 = @bitCast(@as(u32, 0x3db504f3));

/// The exact tier's routed-expert math, `exl3_lane.Exl3PackedOps` op for op:
/// per projection `t128(gemv(t128(x * rin[slot])) ) * rout[slot]` (f32), the
/// clamped SwiGLU between gate/up and down. `Gemv.project(g, k, out_dim, xh,
/// ids, code)` is the EXL3 decode GEMV (`z = xh @ W_hat[slot]` in the trellis
/// domain, f32 [rows, out_dim]), which the kernels lane binds.
pub fn EagerChain(comptime G: type, comptime Gemv: type) type {
    return struct {
        const Self = @This();
        const T = G.T;

        gemv: Gemv,
        hidden: u32,
        inter: u32,
        /// Trellis bits per weight (3 on every layer of the bank of record).
        k: u32 = 3,
        swiglu_limit: f64,

        pub fn init(gemv: Gemv, c: *const v41.Config) Self {
            return .{ .gemv = gemv, .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .swiglu_limit = c.swiglu_limit };
        }

        /// `tcq_runtime._t128`: the normalised 128-block Walsh-Hadamard over the last axis.
        fn t128(g: *G, v: T) !T {
            const sh = g.shapeOf(v);
            var blocks = sh;
            blocks.d[sh.n - 1] = @divExact(sh.d[sh.n - 1], 128);
            blocks.d[sh.n] = 128;
            blocks.n += 1;
            const h = try g.hadamard(try g.reshape(try g.astype(v, .float32), blocks.slice()), t128_scale);
            return g.reshape(h, sh.slice());
        }

        /// `Exl3PackedOps._project`.
        fn project(self: *const Self, g: *G, x: T, ids: T, p: ProjOf(T), out_dim: u32) !T {
            const rin = try g.astype(try g.take(p.rin, ids, 0), .float32);
            const xh = try t128(g, try g.mul(try g.astype(x, .float32), rin));
            const z = try self.gemv.project(g, self.k, out_dim, xh, ids, p.code);
            const rout = try g.astype(try g.take(p.rout, ids, 0), .float32);
            return g.mul(try t128(g, z), rout);
        }

        /// Gate and up of rows `x` [rows, hidden] at slot rows `ids` [rows]
        /// (uint32), then `_clamped_swiglu(g, u, limit)`: f32 [rows, inter].
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const gg = try self.project(g, x, ids, gate, self.inter);
            const uu = try self.project(g, x, ids, up, self.inter);
            const lim = self.swiglu_limit;
            const uc = try g.clip(uu, try g.scalar(-lim, .float32), try g.scalar(lim, .float32));
            const gc = try g.minimum(gg, try g.scalar(lim, .float32));
            return g.mul(try g.silu(gc), uc);
        }

        /// Down of the SwiGLU rows `h` [rows, inter]: f32 [rows, hidden].
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.project(g, h, ids, d, self.hidden);
        }
    };
}

/// The EXL3 decode GEMV on MLX, bound by the kernels lane (phase-3 ops): an
/// output the backend owns (`g.adopt`), f32 [rows, out_dim]. Construction-time
/// binding; nothing is looked up per call.
pub const MlxGemv = struct {
    ctx: *const anyopaque,
    project_fn: *const fn (ctx: *const anyopaque, g: *ops.MlxOps, k: u32, out_dim: u32, xh: mlx.mlx_array, ids: mlx.mlx_array, code: mlx.mlx_array) anyerror!mlx.mlx_array,

    pub fn project(self: MlxGemv, g: *ops.MlxOps, k: u32, out_dim: u32, xh: mlx.mlx_array, ids: mlx.mlx_array, code: mlx.mlx_array) !mlx.mlx_array {
        return self.project_fn(self.ctx, g, k, out_dim, xh, ids, code);
    }
};

/// The GEMV's launch contract on the trace backend (the kernel manifest's
/// inputs: xh f32 [rows, in], ids uint32 [rows], code int16 [cap, in/16,
/// out/16, 16K]); the output is a kernel node f32 [rows, out_dim].
pub const TraceGemv = struct {
    pub fn project(_: TraceGemv, g: *ops.TraceOps, k: u32, out_dim: u32, xh: u32, ids: u32, code: u32) !u32 {
        const sx = g.shapeOf(xh);
        const si = g.shapeOf(ids);
        const sc = g.shapeOf(code);
        if (g.dtypeOf(xh) != .float32 or g.dtypeOf(ids) != .uint32 or g.dtypeOf(code) != .int16) return error.GemvDtype;
        if (sx.n != 2 or si.n != 1 or sc.n != 4 or si.d[0] != sx.d[0]) return error.GemvShape;
        const out: c_int = @intCast(out_dim);
        if (sc.d[1] * 16 != sx.d[1] or sc.d[2] * 16 != out or sc.d[3] != 16 * @as(c_int, @intCast(k))) return error.GemvShape;
        return g.kernel(&.{ sx.d[0], out }, .float32);
    }
};

/// Shape-level math for trace models whose widths no EXL3 bank has (the mini
/// config's hidden 64): one kernel node per projection pair.
pub const TraceMath = struct {
    hidden: c_int,
    inter: c_int,

    pub fn gateUp(self: *const TraceMath, g: *ops.TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        _ = gate;
        _ = up;
        if (g.shapeOf(x).d[1] != self.hidden or g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .float32);
    }

    pub fn down(self: *const TraceMath, g: *ops.TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        _ = d;
        if (g.shapeOf(h).d[1] != self.inter or g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .float32);
    }
};

// ── The executor: the model's routed hook over a source ──

/// Construction-time routes of the executor.
pub const Routes = struct {
    /// Pass `route` the next routed layer's gate scores (the streamer's
    /// lookahead predictor, evaluated with the routing barrier).
    lookahead: bool = false,
    /// Event gates instead of host waits: every wave is built at once over
    /// event-wait aliases of the bank arrays (the typical tier's gate).
    gated: bool = false,
};

/// The routed-expert hook of `Model(G)` over source `S` with math `M`
/// (`gateUp(g, x, ids, gate, up)`, `down(g, h, ids, d)`). Bank arrays are
/// bound once at `init` (base, transient) and at `grow` (the grown rows).
/// `at(layer)` is the per-layer hook the trunk calls.
pub fn Experts(comptime G: type, comptime S: type, comptime M: type) type {
    return ExpertsWith(G, S, M, .{});
}

pub fn ExpertsWith(comptime G: type, comptime S: type, comptime M: type, comptime routes: Routes) type {
    comptime {
        assertSource(S);
        if (routes.gated) expectMethod(S, "gate", &.{ *S, *S.Call }, ?expert_stream.Gates);
    }
    return struct {
        const Self = @This();
        const T = G.T;
        pub const Arrays = BankArraysOf(T);

        /// A routed layer's gate (the lookahead predictor reads the next layer's).
        pub const Gate = struct { w: T, bias: T };

        a: std.mem.Allocator,
        source: *S,
        math: M,
        hidden: c_int,
        n_experts: u32,
        /// Per layer, per bank kind: what the math binds (null: no rows).
        banks: [][n_banks]?Arrays,
        /// Lookahead: every routed layer's gate, by layer.
        gates: []const Gate = &.{},
        /// Gated: the event the stream signals (an MLX backend's MTLSharedEvent).
        event: expert_event.Event = .{ .id = 0, .object = 0 },

        pub const Options = struct { gates: []const Gate = &.{}, event: ?expert_event.Event = null };

        pub fn init(a: std.mem.Allocator, g: *G, source: *S, math: M, c: *const v41.Config) !Self {
            return initWith(a, g, source, math, c, .{});
        }

        pub fn initWith(a: std.mem.Allocator, g: *G, source: *S, math: M, c: *const v41.Config, opt: Options) !Self {
            if (routes.lookahead and opt.gates.len != c.n_layers) return error.LookaheadNeedsGates;
            if (routes.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            const banks = try a.alloc([n_banks]?Arrays, c.n_layers);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@intFromEnum(kind)] = try bind(g, source, @intCast(l), kind);
            }
            var self: Self = .{ .a = a, .source = source, .math = math, .hidden = @intCast(c.hidden_size), .n_experts = c.n_routed_experts, .banks = banks, .gates = opt.gates };
            if (opt.event) |e| self.event = e;
            return self;
        }

        /// The router that scores layer `layer`'s read-ahead: the NEXT layer's
        /// (none after the last layer).
        pub fn predictorGate(self: *const Self, layer: u32) ?Gate {
            return if (layer + 1 < self.gates.len) self.gates[layer + 1] else null;
        }

        /// `expert_lookahead.nextLayerScores`: sqrt(softplus(f32(x @ w^T))) + bias.
        fn nextScores(g: *G, xf: T, gate: Gate) !T {
            const z = try g.astype(try g.matmul(xf, try g.transpose(gate.w)), .float32);
            return g.add(try g.sqrt(try g.logaddexp(z, try g.scalar(0, .float32))), gate.bias);
        }

        /// An event wait's aliases of `xs` (the GPU reads them after `value`).
        fn eventWait(self: *Self, g: *G, xs: []const T, value: u64, deps: []const T, outs: []T) !void {
            if (G == ops.MlxOps) {
                try expert_event.wait(xs, self.event, value, deps, false, g.s, outs);
                for (outs) |*o| o.* = try g.adopt(o.*);
            } else {
                for (xs, outs) |x, *o| o.* = try g.eventAlias(x, value, deps.len);
            }
        }

        fn waitProj(self: *Self, g: *G, p: ProjOf(T), value: u64, deps: []const T) !ProjOf(T) {
            var out: [3]T = undefined;
            try self.eventWait(g, &.{ p.code, p.rout, p.rin }, value, deps, &out);
            return .{ .code = out[0], .rout = out[1], .rin = out[2] };
        }

        pub fn deinit(self: *Self) void {
            self.a.free(self.banks);
            self.* = undefined;
        }

        /// A bank with rows must have arrays the math can bind (MLX slot memory).
        fn bind(g: *G, source: *S, layer: u32, kind: BankKind) !?Arrays {
            const arrays = try source.bankArrays(g, layer, kind);
            if (arrays == null and source.bankRows(layer, kind) > 0) return error.SlotArraysUnbound;
            return arrays;
        }

        /// The one phase change: the source grows, the grown rows are bound.
        pub fn grow(self: *Self, g: *G, decode_rows: []const u32) !void {
            try self.source.grow(decode_rows);
            for (self.banks, 0..) |*b, l| b[@intFromEnum(BankKind.ext)] = try bind(g, self.source, @intCast(l), .ext);
        }

        /// After the forward's last eval: settles and unpins released calls.
        pub fn flush(self: *Self) !void {
            try self.source.flush();
        }

        pub const Hook = struct {
            ex: *Self,
            layer: u32,

            pub fn routed(h: Hook, g: *G, xf: T, indices: T) !T {
                return h.ex.run(g, h.layer, xf, indices);
            }
        };

        pub fn at(self: *Self, layer: u32) Hook {
            return .{ .ex = self, .layer = layer };
        }

        const Group = struct { bank: BankKind, n: u32 = 0, pos: [max_route_ids]u8 = undefined };

        /// Outputs in wave order and the routed position of each output row.
        const Acc = struct {
            outs: [max_route_ids]T = undefined,
            n_outs: usize = 0,
            pos: [max_route_ids]u32 = undefined,
            n_pos: usize = 0,
        };

        const Wave = struct {
            groups: [n_banks]Group = undefined,
            n: usize = 0,
            ids: [n_banks]T = undefined,
            h: [n_banks]T = undefined,
        };

        /// The positions of wave `w`, grouped by bank in first-appearance order
        /// (`Exl3PackedOps.gate_up`'s groups), each group's gate/up + SwiGLU.
        fn gateUpWave(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) !Wave {
            var wave: Wave = .{};
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                const gi = for (wave.groups[0..wave.n], 0..) |gr, i| {
                    if (gr.bank == ref.bank) break i;
                } else blk: {
                    wave.groups[wave.n] = .{ .bank = ref.bank };
                    wave.n += 1;
                    break :blk wave.n - 1;
                };
                const gr = &wave.groups[gi];
                gr.pos[gr.n] = @intCast(pos);
                gr.n += 1;
            }
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@intFromEnum(gr.bank)] else self.banks[layer][@intFromEnum(gr.bank)]).?;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |pos, *t, *r| {
                    t.* = @intCast(pos / k);
                    r.* = sv.refs[pos].row;
                }
                const n: c_int = @intCast(gr.n);
                const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{n}, .int32), 0);
                wave.ids[i] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{n}, .uint32);
                wave.h[i] = try self.math.gateUp(g, x, wave.ids[i], arrays.gate, arrays.up);
            }
            return wave;
        }

        /// Each group's down; the outputs join the call's accumulator.
        fn downWave(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) ![]const T {
            const first = acc.n_outs;
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@intFromEnum(gr.bank)] else self.banks[layer][@intFromEnum(gr.bank)]).?;
                acc.outs[acc.n_outs] = try self.math.down(g, wave.h[i], wave.ids[i], arrays.down);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |pos| {
                    acc.pos[acc.n_pos] = pos;
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// The gated waves (the lookahead4 lane's event gate): every part's
        /// gate/up over bank arrays waited at `gu`, then part p's down over
        /// arrays waited at `down_first + p`, ordered after every gate/up and
        /// the previous part's down. Nothing waits on the host.
        fn gatedParts(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) !void {
            var gu: [n_banks]?Arrays = @splat(null);
            for (sv.waves, sv.refs) |w, ref| {
                const b = @intFromEnum(ref.bank);
                if (w == 0 or gu[b] != null) continue;
                const arrays = self.banks[layer][b].?;
                gu[b] = .{ .gate = try self.waitProj(g, arrays.gate, gates.gu, &.{}), .up = try self.waitProj(g, arrays.up, gates.gu, &.{}), .down = arrays.down };
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gateUpWave(g, layer, xf, k, sv, @intCast(p + 1), &gu);
                for (waves[p].h[0..waves[p].n]) |h| {
                    hs[n_hs] = h;
                    n_hs += 1;
                }
            }
            var prev: []const T = &.{};
            for (0..sv.n_parts) |p| {
                var deps: [2 * max_route_ids]T = undefined;
                @memcpy(deps[0..n_hs], hs[0..n_hs]);
                @memcpy(deps[n_hs..][0..prev.len], prev);
                var dn = gu;
                for (&dn) |*d| if (d.*) |*arr| {
                    arr.down = try self.waitProj(g, arr.down, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                };
                prev = try self.downWave(g, layer, &waves[p], acc, &dn);
            }
        }

        /// `PackedDecode.run` (gate/up publish before down): the routing
        /// barrier, the residents' wave, then per miss part gate/up after
        /// `waitGu` and down after `waitDown`, each wave started on the GPU;
        /// release; the outputs in the router's order, `[n, k, hidden]` f32.
        pub fn run(self: *Self, g: *G, layer: u32, xf: T, indices: T) !T {
            const n: u32 = @intCast(g.shapeOf(xf).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            const n_ids = n * k;
            // A route decision on M: calls of at most `max_route_ids` ids (decode
            // and verify, <= 8 rows of top-6) are the decode lane; wider calls
            // are the prefill lane (seed waves + the DIG kernels), not ported.
            if (n_ids > max_route_ids) return error.PrefillLaneNotPorted;
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [expert_lookahead.max_rows * 512]f32 = undefined;
            var scores: []const f32 = &.{};
            if (if (routes.lookahead) self.predictorGate(layer) else null) |gate| {
                // The predictor joins the routing barrier's eval (the last layer predicts nothing).
                const sc = try nextScores(g, xf, gate);
                try g.evalAll(&.{ indices, sc });
                scores = try g.hostF32(sc, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const call = try self.source.route(layer, ids, scores);
            var released = false;
            errdefer if (!released) self.source.release(call);
            const sv = self.source.served(call);
            var acc: Acc = .{};
            // Residents: gate/up and down at once.
            const hits = try self.gateUpWave(g, layer, xf, k, sv, 0, null);
            if (hits.n > 0) try g.asyncEval(try self.downWave(g, layer, &hits, &acc, null));
            if (routes.gated) {
                if (try self.source.gate(call)) |gates| try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.source.waitGu(call, part);
                const wave = try self.gateUpWave(g, layer, xf, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                try self.source.waitDown(call, part);
                try g.asyncEval(try self.downWave(g, layer, &wave, &acc, null));
            }
            self.source.release(call);
            released = true;
            // `take(concatenate(outputs), argsort(positions))`: the inverse
            // permutation of the (unique) positions, made on the host.
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            for (acc.pos[0..acc.n_pos], 0..) |pos, j| order[pos] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..acc.n_pos]), &.{@intCast(acc.n_pos)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }
    };
}

// ── Tests ──

const testing = std.testing;
const TraceOps = ops.TraceOps;

comptime {
    assertSource(StreamSource);
    assertSource(FakeSource);
}

/// Scripted host reads for the trace backend: the routed ids of each call.
const Script = struct {
    calls: []const []const u16,
    next: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }

    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *Script = @ptrCast(@alignCast(ctx));
        if (self.next >= self.calls.len) return error.ScriptExhausted;
        const c = self.calls[self.next];
        self.next += 1;
        if (c.len != out.len) return error.ScriptShape;
        @memcpy(out, c);
    }

    fn argmax(_: *anyopaque) anyerror!u32 {
        return error.NoPicks;
    }
};

fn kindsOf(log: []const FakeSource.Event, buf: []u8) []const u8 {
    for (log, buf[0..log.len]) |e, *c| c.* = switch (e.kind) {
        .route => 'R',
        .wait_gu => 'g',
        .wait_down => 'd',
        .release => 'r',
        .flush => 'f',
        .grow => 'G',
        .gate => 'E',
    };
    return buf[0..log.len];
}

/// Distinct banks among the positions of wave `w` (the groups the wave runs).
fn groupsOf(sv: Served, w: u8) usize {
    var seen: [n_banks]bool = @splat(false);
    var n: usize = 0;
    for (sv.waves, sv.refs) |wv, ref| {
        if (wv != w or seen[@intFromEnum(ref.bank)]) continue;
        seen[@intFromEnum(ref.bank)] = true;
        n += 1;
    }
    return n;
}

/// Trace nodes of kind `op` between consecutive log events: [0] before the
/// first event, [j] between events j - 1 and j, [len] after the last.
fn opsBetween(g: *const TraceOps, log: []const FakeSource.Event, first: usize, op: ops.Op, out: []usize) void {
    @memset(out, 0);
    for (g.nodes.items[first..], first..) |nd, i| if (nd.op == op) {
        const j = for (log, 0..) |e, jj| {
            if (e.at > i) break jj;
        } else log.len;
        out[j] += 1;
    };
}

fn testConfig(hidden: u32, inter: u32, n_layers: u8) v41.Config {
    var c: v41.Config = undefined;
    c.hidden_size = hidden;
    c.moe_intermediate_size = inter;
    c.n_layers = n_layers;
    c.swiglu_limit = 10.0;
    return c;
}

test "dsv41 experts: the fake and the stream adapter are expert sources" {
    comptime assertSource(StreamSource);
    comptime assertSource(FakeSource);
    try testing.expectEqual(@as(usize, 4), StreamSource.n_routes);
}

test "dsv41 experts: a decode call runs the residents, then each part's gate/up after waitGu and its down after waitDown" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2); // widths an EXL3 bank has (t128 blocks of 128)
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = Experts(TraceOps, FakeSource, Chain);
    var ex = try Ex.init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try testing.expect(ex.banks[1][@intFromEnum(BankKind.ext)] == null);
    try ex.grow(&g, &.{ 8, 8 });
    try testing.expect(ex.banks[1][@intFromEnum(BankKind.ext)] != null);
    src.log.clearRetainingCapacity();
    src.trace = &g;

    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 2, 256 }, .bfloat16);
    const idx = try g.input(&.{ 2, 3 }, .int32);

    // Call 1 (layer 0, 2 rows x top-3): six misses, two parts of three in expert order.
    const first = g.nodes.items.len;
    const y = try ex.at(0).routed(&g, xf, idx);
    try testing.expect(g.shapeOf(y).eql(ops.Shape.of(&.{ 2, 3, 256 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(y));
    var kb: [64]u8 = undefined;
    try testing.expectEqualStrings("Rgdgdr", kindsOf(src.log.items, &kb));
    const sv = src.served(&src.calls[0]);
    try testing.expectEqual(@as(u32, 2), sv.n_parts);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 2, 2, 2 }, sv.waves);
    const g1 = groupsOf(sv, 1);
    const g2 = groupsOf(sv, 2);
    // Kernels per interval [<R, R..g0, g0..d0, d0..g1, g1..d1, d1..r, >r]: none
    // before a publish, gate + up per group after waitGu, down per group after waitDown.
    var k: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .kernel, &k);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 2 * g1, g1, 2 * g2, g2, 0 }, &k);
    // The routing barrier comes before the route; each wave starts on the GPU.
    var hr: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .host_read, &hr);
    try testing.expectEqualSlices(usize, &.{ 1, 0, 0, 0, 0, 0, 0 }, &hr);
    var ae: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .async_eval, &ae);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1, 1, 1, 0 }, &ae);
    // Two t128 (reshape, hadamard, reshape) around each of the three GEMVs of a group.
    var hd: [7]usize = undefined;
    opsBetween(&g, src.log.items, first, .hadamard, &hd);
    try testing.expectEqual(4 * g1 + 2 * g1 + 4 * g2 + 2 * g2, hd[2] + hd[3] + hd[4] + hd[5]);

    // Call 2 (same layer): the released call is flushed by the route; 1, 2, 3
    // are resident (wave 0, at once), 7 and 9 load in one part.
    const mark = src.log.items.len;
    const second = g.nodes.items.len;
    _ = try ex.at(0).routed(&g, xf, idx);
    try testing.expectEqualStrings("fRgdr", kindsOf(src.log.items[mark..], &kb));
    const sv2 = src.served(&src.calls[0]);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 1, 0 }, sv2.waves);
    const h0 = groupsOf(sv2, 0);
    const h1 = groupsOf(sv2, 1);
    var k2: [6]usize = undefined;
    opsBetween(&g, src.log.items[mark..], second, .kernel, &k2);
    // [<f, f..R, R..g0 (the residents' gate/up + down), g0..d0, d0..r, >r]
    try testing.expectEqualSlices(usize, &.{ 0, 0, 3 * h0, 2 * h1, h1, 0 }, &k2);
    const st = src.stats();
    try testing.expectEqual(@as(u64, 2), st.route_calls);
    try testing.expectEqual(@as(u64, 3), st.expert_cache_hits);
    try testing.expectEqual(@as(u64, 8), st.expert_cache_misses);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    try testing.expectEqual(FakeSource.Event.Kind.flush, src.log.items[src.log.items.len - 1].kind);
}

test "dsv41 experts: a wider call is the prefill lane's (not ported), refused before any route" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 1);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{4} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    const xf = try g.input(&.{ 9, 256 }, .bfloat16);
    const idx = try g.input(&.{ 9, 6 }, .int32);
    try testing.expectError(error.PrefillLaneNotPorted, ex.at(0).routed(&g, xf, idx));
    try testing.expectEqual(@as(usize, 0), src.log.items.len);
}

const SynthBank = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    bank: expert_bank.Bank,

    fn open(n_experts: u32) !SynthBank {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try expert_bank.writeSynth(testing.allocator, &tmp, .{ .n_experts = n_experts });
        errdefer testing.allocator.free(image);
        var rbuf: [512]u8 = undefined;
        const implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = n_experts, .n_layers = 2 };
        const bank = try expert_bank.Bank.open(testing.allocator, std.testing.io, try expert_bank.tmpRoot(&tmp, &rbuf), implemented, null);
        return .{ .tmp = tmp, .image = image, .bank = bank };
    }

    fn close(self: *SynthBank) void {
        self.bank.deinit();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

/// Every routed id's slot row holds its record's bytes (the rows the hook bound).
fn expectRowsHold(s: *expert_stream.Stream, sb: *const SynthBank, layer: u32, ids: []const u16, sv: Served) !void {
    const geom = &sb.bank.layers[layer];
    const r = for (&s.routes) |*rr| {
        if (rr.state != .free and rr.layer == layer and rr.plan.n_ids == ids.len) break rr;
    } else return error.NoRoute;
    for (ids, r.plan.slotsOf(), sv.refs) |e, slot, ref| {
        try testing.expectEqual(s.slotRef(layer, slot), ref);
        const off = sb.bank.recordOffset(layer, e);
        for (geom.segments, 0..) |seg, comp| {
            try testing.expectEqualSlices(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(layer, slot, @enumFromInt(comp))[0..seg.length]);
        }
    }
}

test "dsv41 experts: the stream adapter hands the hook every routed record's rows, parts after their reads" {
    const a = testing.allocator;
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
    defer s.deinit();
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(64, 32, 2);
    const Ex = Experts(TraceOps, StreamSource, TraceMath);
    var ex = try Ex.init(a, &g, &src, .{ .hidden = 64, .inter = 32 }, &c);
    defer ex.deinit();
    // Base rows and the shared transient bank are bound in the stream's geometry.
    const base = ex.banks[0][@intFromEnum(BankKind.base)].?;
    try testing.expect(g.shapeOf(base.gate.code).eql(ops.Shape.of(&.{ 4, 4, 2, 48 })));
    try testing.expect(g.shapeOf(ex.banks[1][@intFromEnum(BankKind.transient)].?.down.rin).eql(ops.Shape.of(&.{ 12, 32 })));

    var script: Script = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 9, 1, 12, 2, 20, 21 }, &.{ 20, 12, 9, 22, 23, 1 } } };
    g.host_values = script.values();
    const xf = try g.input(&.{ 3, 64 }, .bfloat16);
    const idx = try g.input(&.{ 3, 2 }, .int32);
    // Prefill route (one part: the pool's job chunk); every row it serves holds its record.
    _ = try ex.at(0).routed(&g, xf, idx);
    try expectRowsHold(s, &sb, 0, script.calls[0], src.served(&src.calls[0]));
    try testing.expectEqual(@as(u32, 1), src.served(&src.calls[0]).n_parts);
    // Growth binds layer 1's grown rows; decode routes cut parts of <= 3 records.
    try ex.grow(&g, &.{ 4, 8 });
    try testing.expect(g.shapeOf(ex.banks[1][@intFromEnum(BankKind.ext)].?.up.rout).eql(ops.Shape.of(&.{ 4, 32 })));
    try testing.expect(ex.banks[0][@intFromEnum(BankKind.ext)] == null);
    _ = try ex.at(1).routed(&g, xf, idx);
    const r1 = src.calls[for (src.calls, 0..) |cl, i| {
        if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1) break i;
    } else unreachable];
    try expectRowsHold(s, &sb, 1, script.calls[1], src.served(&r1));
    try testing.expectEqual(@as(u32, 2), src.served(&r1).n_parts); // six misses: 1, 2, 9 | 12, 20, 21
    _ = try ex.at(1).routed(&g, xf, idx);
    const r2 = src.calls[for (src.calls, 0..) |cl, i| {
        if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1 and cl.route.?.plan.n_hits > 0) break i;
    } else unreachable];
    const sv2 = src.served(&r2);
    try expectRowsHold(s, &sb, 1, script.calls[2], sv2);
    // 20, 12, 9, 1 are resident (wave 0); 22 and 23 load in one part.
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 0 }, sv2.waves);
    try ex.flush();
    const st = src.stats();
    try testing.expectEqual(@as(u64, 3), st.route_calls);
    try testing.expectEqual(@as(u64, 4), st.expert_cache_hits);
}

// DSV41_PHASE1_ROUTE_FIXTURE=<json from R/exl3/runtime/dump_phase1_route_fixture.py>
test "dsv41 experts: the recorded trace's 3,600 decode calls run through the hook as the Python bank serves them" {
    const path = std.mem.span(std.c.getenv("DSV41_PHASE1_ROUTE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const FixPlan = expert_policy.FixPlan;
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
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = f.experts, .rows = f.prefill_capacity, .transient_rows = f.transient, .records_per_part = f.records_per_part });
    defer src.deinit();
    // Prefill: the seeds and their waves, as the Python bank admitted them.
    for (0..f.layers) |l| {
        try src.seedPrefill(@intCast(l), f.resident0[l]);
        for (f.seed_plans[l]) |want| {
            const call = try src.route(@intCast(l), want.ids, &.{});
            try expert_policy.expectPlan(&call.plan, want);
            src.release(call);
        }
    }
    var g = TraceOps.init(a);
    defer g.deinit();
    var c = testConfig(5120, 2304, @intCast(f.layers));
    c.swiglu_limit = 10.0;
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try Experts(TraceOps, FakeSource, Chain).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, f.decode_capacity);
    const Replay = struct {
        routes: []const FixPlan,
        next: usize = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.routes[self.next].ids);
            self.next += 1;
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };
    var replay: Replay = .{ .routes = f.routes };
    g.host_values = .{ .ctx = &replay, .ids = Replay.ids, .argmax = Replay.argmax };
    var xs: [9]u32 = undefined;
    var is: [9]u32 = undefined;
    for (1..9) |rows| {
        xs[rows] = try g.input(&.{ @intCast(rows), 5120 }, .bfloat16);
        is[rows] = try g.input(&.{ @intCast(rows), 6 }, .int32);
    }
    var n_parts: usize = 0;
    var n_groups: usize = 0;
    for (f.routes, 0..) |want, i| {
        const l: u32 = @intCast(i % f.layers);
        const rows = want.ids.len / 6;
        const y = try ex.at(l).routed(&g, xs[rows], is[rows]);
        try testing.expect(g.shapeOf(y).eql(ops.Shape.of(&.{ @intCast(rows), 6, 5120 })));
        const call = for (&src.calls) |*cl| {
            if (cl.state == .released) break cl;
        } else unreachable;
        expert_policy.expectPlan(&call.plan, want) catch |e| {
            std.debug.print("route {d} (cycle {d}, layer {d}) differs\n", .{ i, i / f.layers, l });
            return e;
        };
        // The waves are the recorded parts: part p's experts are the ids of wave p + 1.
        const sv = src.served(call);
        try testing.expectEqual(want.parts.len, sv.n_parts);
        for (want.parts, 1..) |part, w| {
            for (want.ids, sv.waves) |e, wv| {
                const in_part = std.mem.indexOfScalar(u16, part, e) != null;
                try testing.expectEqual(in_part, wv == w);
            }
        }
        for (want.ids, sv.waves) |e, wv| try testing.expectEqual(std.mem.indexOfScalar(u16, want.hits, e) != null, wv == 0);
        n_parts += sv.n_parts;
        for (0..sv.n_parts + 1) |w| n_groups += groupsOf(sv, @intCast(w));
    }
    try testing.expectEqual(f.routes.len, replay.next);
    try ex.flush();
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    std.debug.print("dsv41 experts: {d} recorded decode calls through the hook, {d} parts, {d} bank groups; plans, parts and waves equal the Python bank's\n", .{ f.routes.len, n_parts, n_groups });
}

test "dsv41 experts: the gated route builds every wave at once over event-wait aliases, no host waits" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 2);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    var ex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, &.{ 8, 8 });
    src.log.clearRetainingCapacity();
    src.trace = &g;
    var script: Script = .{ .calls = &.{&.{ 1, 2, 3, 4, 5, 6 }} };
    g.host_values = script.values();
    const first = g.nodes.items.len;
    _ = try ex.at(0).routed(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 3 }, .int32));
    var kb: [16]u8 = undefined;
    try testing.expectEqualStrings("REr", kindsOf(src.log.items, &kb));
    const sv = src.served(&src.calls[0]);
    // Gate/up arrays waited once per miss bank (6 each); each part's down arrays (3 per bank) at its own value.
    var kinds: [n_banks]bool = @splat(false);
    for (sv.waves, sv.refs) |w, r| {
        if (w > 0) kinds[@intFromEnum(r.bank)] = true;
    }
    var n_kinds: usize = 0;
    for (kinds) |k| n_kinds += @intFromBool(k);
    var n_alias: usize = 0;
    var n_kernel: usize = 0;
    for (g.nodes.items[first..]) |nd| {
        n_alias += @intFromBool(nd.op == .event_wait);
        n_kernel += @intFromBool(nd.op == .kernel);
    }
    try testing.expectEqual(6 * n_kinds + 3 * n_kinds * sv.n_parts, n_alias);
    try testing.expectEqual(3 * (groupsOf(sv, 1) + groupsOf(sv, 2)), n_kernel);
    // One gate per call: its gate/up value and one down value per part.
    try testing.expectEqual(@as(u64, 1 + sv.n_parts), src.gate_value);
    // In build order: every gate/up array at the gate/up value with no deps, then
    // part p's down arrays at down_first + p after all gate/up outputs (+ part p-1's downs).
    const gv: expert_stream.Gates = .{ .gu = 1, .down_first = 2, .n_parts = sv.n_parts }; // the FakeSource's first gate
    const ws = g.waits.items;
    try testing.expectEqual(n_alias, ws.len);
    for (ws[0 .. 6 * n_kinds]) |w| try testing.expectEqual(TraceOps.Wait{ .value = gv.gu, .n_deps = 0 }, w);
    var prev_deps: u32 = 0;
    for (0..sv.n_parts) |p| {
        const part = ws[6 * n_kinds + 3 * n_kinds * p ..][0 .. 3 * n_kinds];
        for (part) |w| {
            try testing.expectEqual(gv.down_first + p, w.value);
            try testing.expectEqual(part[0].n_deps, w.n_deps);
        }
        if (p > 0) try testing.expect(part[0].n_deps > prev_deps);
        prev_deps = part[0].n_deps;
    }
}

test "dsv41 experts: the read-ahead of layer l is scored by layer l + 1's router, none after the last" {
    const a = testing.allocator;
    const c = testConfig(256, 128, 3);
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4, 4 } });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = ExpertsWith(TraceOps, FakeSource, Chain, .{ .lookahead = true });
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
    var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &c), &c, .{ .gates = &gates });
    defer ex.deinit();
    try testing.expectEqual(gates[1].w, ex.predictorGate(0).?.w);
    try testing.expectEqual(gates[2].w, ex.predictorGate(1).?.w);
    try testing.expect(ex.predictorGate(2) == null);
}

// DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py> (its scores file beside it)
test "dsv41 experts: the scores the hook passes reproduce the streamer's read-ahead picks on the recorded trace" {
    const path = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const Call = struct { ids: []const u16, sel: []const []const u16 };
    const Cfg = struct { k: u32, tau: ?f32, budget: u32 };
    const TieCall = struct { call: u32, config: u32, sel: []const u16 };
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        rows: []const u32,
        configs: []const Cfg,
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
    const scores = try a.alloc(f32, raw.len / 4);
    defer a.free(scores);
    for (scores, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));

    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = f.experts, .rows = f.prefill_capacity, .transient_rows = f.transient });
    defer src.deinit();
    const cf = f.configs[0];
    src.selector = try expert_lookahead.Selector.init(a, f.experts, cf.k, cf.tau orelse std.math.inf(f32), cf.budget);
    // The prefill residency the fixture replays: the seed, admitted in waves of the transient width.
    for (0..f.layers) |l| {
        try src.seedPrefill(@intCast(l), f.resident0[l]);
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        var i: usize = 0;
        while (i < sorted.len) : (i += f.transient) {
            const call = try src.route(@intCast(l), sorted[i..@min(i + f.transient, sorted.len)], &.{});
            src.release(call);
        }
    }
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(5120, 2304, @intCast(f.layers));
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Ex = ExpertsWith(TraceOps, FakeSource, Chain, .{ .lookahead = true });
    const gates = try a.alloc(Ex.Gate, f.layers);
    defer a.free(gates);
    for (gates) |*gt| gt.* = .{ .w = try g.input(&.{ @intCast(f.experts), 5120 }, .bfloat16), .bias = try g.input(&.{@intCast(f.experts)}, .float32) };
    var cc = c;
    cc.n_routed_experts = f.experts;
    var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &cc), &cc, .{ .gates = gates });
    defer ex.deinit();
    try ex.grow(&g, f.decode_capacity);
    const Replay = struct {
        calls: []const Call,
        scores: []const f32,
        experts: usize,
        next: usize = 0,
        at: usize = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.calls[self.next].ids);
            self.next += 1;
        }
        fn f32s(ctx: *anyopaque, out: []f32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.scores[self.at..][0..out.len]);
            self.at += out.len;
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };
    var replay: Replay = .{ .calls = f.calls, .scores = scores, .experts = f.experts };
    g.host_values = .{ .ctx = &replay, .ids = Replay.ids, .argmax = Replay.argmax, .f32s = Replay.f32s };
    var xs: [9]u32 = undefined;
    var is: [9]u32 = undefined;
    for (1..9) |rows| {
        xs[rows] = try g.input(&.{ @intCast(rows), 5120 }, .bfloat16);
        is[rows] = try g.input(&.{ @intCast(rows), 6 }, .int32);
    }
    var tie_i: usize = 0;
    var n_picks: usize = 0;
    for (f.calls, 0..) |call, ci| {
        const l: u32 = @intCast(ci % f.layers);
        const m = f.rows[ci / f.layers];
        const before = src.picks.items.len;
        _ = try ex.at(l).routed(&g, xs[m], is[m]);
        if (l + 1 < f.layers) {
            try testing.expectEqual(before + 1, src.picks.items.len);
            const pick = src.picks.items[before];
            try testing.expectEqual(l + 1, pick.layer);
            var want = call.sel[0];
            if (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci) {
                if (f.tie_calls[tie_i].config == 0) want = f.tie_calls[tie_i].sel;
                while (tie_i < f.tie_calls.len and f.tie_calls[tie_i].call == ci) tie_i += 1;
            }
            testing.expectEqualSlices(u16, want, pick.experts[0..pick.n]) catch |e| {
                std.debug.print("call {d} (layer {d}): read-ahead picks differ\n", .{ ci, l });
                return e;
            };
            n_picks += 1;
        } else try testing.expectEqual(before, src.picks.items.len);
    }
    try testing.expectEqual(f.scores_rows * f.experts, replay.at);
    try ex.flush();
    std.debug.print("dsv41 experts: {d} layer calls through the hook with lookahead scores; {d} read-ahead picks equal the streamer's\n", .{ f.calls.len, n_picks });
}
