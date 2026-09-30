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
//! A call of more rows than a route takes (a prefill chunk) is the wide lane:
//! the math's `prefill` over the slots its experts are served in.
//!
//! Sources: `StreamSource` (the streamer's `Stream`) and `FakeSource` (the
//! streamer's residency policy with no reads, for host tests). The math of a
//! group of rows sharing a bank is the C2 `quant` seam (quant.zig): `QuantMath`
//! over an accepted quant (the EXL3 quant: PREP=rin + the decode GEMV, DIG-X
//! prefill waves) is the served math; `EagerChain` is the stock tier's op chain
//! around the quant's decode GEMV (the parity harnesses); `TraceGemv` /
//! `TraceMath` stand in on the trace backend.

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
const xk = @import("exl3_kernels.zig");
const quant = @import("quant.zig");
const xq = @import("exl3_quant.zig");

pub const max_route_ids = expert_policy.max_route_ids;
/// The widest decode / verify forward (the RC routes' and the verify block's rows): at most
/// `decode_forward_rows * top_k` routed ids, which construction proves fit one route (`max_route_ids`),
/// so a forward of <= 8 rows never reaches the wide lane.
pub const decode_forward_rows: u32 = 8;
/// The fewest routed ids a wide-lane call takes (one more than a route): refused below, by name.
pub const wide_min_ids: u32 = max_route_ids + 1;
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
pub const ProjOf = xq.ProjArrays;

/// A bank's nine arrays by projection (the streamer's `BankArrays`; the quant seam's).
pub fn BankArraysOf(comptime T: type) type {
    return quant.BankArrays(ProjOf(T));
}

// ── The source contract ──

/// Compile-time check that `S` is an expert source:
///   `Call` (a live route);
///   `route(*S, layer, ids, scores) Error!*Call` (scores: the next routed
///       layer's gate scores for the lookahead, or empty; a source reads
///       them only in the decode phase and ignores them before it);
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

    pub fn seedPrefill(self: *StreamSource, layer: u32, ids: []const u16) !void {
        return self.stream.seedPrefill(layer, ids);
    }

    /// Prefill routes one layer may hold live at once (`Stream.Options.wide_depth`).
    pub fn wideDepth(self: *const StreamSource) u8 {
        return self.stream.wide_depth;
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
        std.debug.assert(scores.len == 0 or self.selector != null);
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
        if (scores.len > 0 and self.phase == .decode and layer + 1 < self.policies.len) {
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

/// The stock tier's routed-expert math, `exl3_lane.Exl3PackedOps` op for op:
/// per projection `t128(gemv(t128(x * rin[slot])) ) * rout[slot]` (f32), the
/// clamped SwiGLU between gate/up and down. `Gemv.project(g, proj, xh, ids,
/// code)` is the EXL3 decode GEMV (`z = xh @ W_hat[slot]` in the trellis domain,
/// f32 [rows, out]): the accepted EXL3 quant's (`*const exl3_quant.Gemv(G)`).
/// The parity harnesses' math (the stock path reads f32 routed rows).
pub fn EagerChain(comptime G: type, comptime Gemv: type) type {
    return struct {
        const Self = @This();
        const T = G.T;

        gemv: Gemv,
        swiglu_limit: f64,

        pub fn init(gemv: Gemv, c: *const v41.Config) Self {
            return .{ .gemv = gemv, .swiglu_limit = c.swiglu_limit };
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
        fn project(self: *const Self, g: *G, x: T, ids: T, p: ProjOf(T), proj: xq.Proj) !T {
            const rin = try g.astype(try g.take(p.rin, ids, 0), .float32);
            const xh = try t128(g, try g.mul(try g.astype(x, .float32), rin));
            const z = try self.gemv.project(g, proj, xh, ids, p.code);
            const rout = try g.astype(try g.take(p.rout, ids, 0), .float32);
            return g.mul(try t128(g, z), rout);
        }

        /// Gate and up of rows `x` [rows, hidden] at slot rows `ids` [rows]
        /// (uint32), then `_clamped_swiglu(g, u, limit)`: f32 [rows, inter].
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const gg = try self.project(g, x, ids, gate, .gate);
            const uu = try self.project(g, x, ids, up, .up);
            const lim = self.swiglu_limit;
            const uc = try g.clip(uu, try g.scalar(-lim, .float32), try g.scalar(lim, .float32));
            const gc = try g.minimum(gg, try g.scalar(lim, .float32));
            return g.mul(try g.silu(gc), uc);
        }

        /// Down of the SwiGLU rows `h` [rows, inter]: f32 [rows, hidden].
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.project(g, h, ids, d, .down);
        }
    };
}

/// The served routed-expert math: an accepted C2 quant (`quant.checkAccepted`),
/// its decode `gateUp` / `down` (the EXL3 quant: PREP=rin around the decode
/// GEMV) and its prefill waves. The quant reads bf16 routed rows: a call whose
/// rows arrive in another dtype is rounded to bf16 once, here (the RC tier's
/// MoE input is bf16 already).
pub fn QuantMath(comptime G: type, comptime Q: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest decode-lane call (`gateUp` / `down`).
        pub const max_decode_rows: u32 = Q.max_decode_rows;
        q: *Q,

        pub fn init(q: *Q, _: *const v41.Config) Self {
            return .{ .q = q };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            return self.q.gateUp(g, xb, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.q.down(g, h, ids, d);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.q.prefill(g, layer, x, rows, bank);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            return self.q.finishPrefill(g);
        }
    };
}

/// A decode math `D` with one prefill route `P` per layer (`call(g, x, rows,
/// bank)` / `finish(g)`: the quant's prefill shape), for host tests.
pub fn WithPrefillRoutes(comptime G: type, comptime D: type, comptime P: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const max_decode_rows: u32 = if (@hasDecl(D, "max_decode_rows")) D.max_decode_rows else max_route_ids;
        d: D,
        routes: []P,

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            return self.d.gateUp(g, x, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.d.down(g, h, ids, d);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.routes[layer].call(g, x, rows, bank);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            for (self.routes) |*r| try r.finish(g);
        }
    };
}

/// The GEMV's launch contract on the trace backend (the kernel manifest's
/// inputs: xh f32 [rows, in], ids uint32 [rows], code int16 [cap, in/16,
/// out/16, 48] at K 3); the output is a kernel node f32 [rows, out].
pub const TraceGemv = struct {
    pub fn project(_: TraceGemv, g: *ops.TraceOps, _: xq.Proj, xh: u32, ids: u32, code: u32) !u32 {
        const sx = g.shapeOf(xh);
        const si = g.shapeOf(ids);
        const sc = g.shapeOf(code);
        if (g.dtypeOf(xh) != .float32 or g.dtypeOf(ids) != .uint32 or g.dtypeOf(code) != .int16) return error.GemvDtype;
        if (sx.n != 2 or si.n != 1 or sc.n != 4 or si.d[0] != sx.d[0]) return error.GemvShape;
        if (sc.d[1] * 16 != sx.d[1] or sc.d[3] != 48) return error.GemvShape;
        return g.kernel(&.{ sx.d[0], sc.d[2] * 16 }, .float32);
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

/// `argsort(positions)` of unique positions: `inv[pos[j]] = j` (the row of
/// the joined outputs that holds routed position `pos[j]`).
fn invertPositions(pos: []const u32, inv: []u32) void {
    for (pos, 0..) |p, j| inv[p] = @intCast(j);
}

/// Construction-time routes of the executor.
pub const Routes = struct {
    /// Pass `route` the next routed layer's gate scores (the streamer's
    /// lookahead predictor, evaluated with the routing barrier).
    lookahead: bool = false,
    /// Event gates instead of host waits: every wave is built at once over
    /// event-wait aliases of the bank arrays (the typical tier's gate).
    gated: bool = false,
    /// The wide lane (calls of more than `max_route_ids` routed rows) through
    /// the math's `prefill(g, layer, x, rows, bank)` (a KEPT result) and
    /// `finishPrefill(g)`; off refuses a wide call (`PrefillLaneNotPorted`).
    prefill: bool = false,
};

/// The wide lane's read schedule, chosen at construction (`Options.wide`; both
/// exact: the same rows, slots and kernels, only the order and timing of reads
/// and evals change).
pub const Wide = struct {
    /// Each call seeds its layer's residency with its own ids (the most routed
    /// experts protected in the persistent rows) and drains each group once
    /// (its banks' waves queued).
    seed: bool = false,
    /// Each call feeds its groups hottest first (routed rows descending, ties by
    /// id) instead of in first appearance. `seed` + `hot_first` = the wide feed.
    hot_first: bool = false,
    /// Groups in flight: 2 routes (reads) the next group before this
    /// group's waves (the source's `wideDepth` must allow it).
    depth: u8 = 1,
    /// Experts with at most this many rows in the call run the decode lane's
    /// math (the GEMV, in chunks of the math's `max_decode_rows`) after their
    /// bank's wide waves, not the wide route (0: none). ROUNDING-CLASS against
    /// the wide route: its own reference family.
    cold_rows: u8 = 0,
    pub const max_cold_rows = 8;
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
        /// The backend the banks were bound on.
        g: *G,
        wide: WideScratch = .{},
        /// The wide lane's read schedule (`Options.wide`).
        wide_route: Wide = .{},

        pub const Options = struct { gates: []const Gate = &.{}, event: ?expert_event.Event = null, wide: Wide = .{} };

        /// The wide lane's host scratch, reused across calls.
        const WideScratch = struct {
            ids: std.ArrayList(u16) = .empty,
            /// Per expert: its index in `distinct`, or -1.
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            /// Per expert: its rows in the call (the feed's order).
            count: std.ArrayList(u32) = .empty,
            cold_slot: std.ArrayList(u32) = .empty,
            cold_act: std.ArrayList(u32) = .empty,
            cold_pos: std.ArrayList(u32) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act_row: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,
            /// JOINLESS: each assignment's (output, row), int32 pairs.
            loc: std.ArrayList(i32) = .empty,

            fn deinit(w: *WideScratch, a: std.mem.Allocator) void {
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.cold_slot, &w.cold_act, &w.cold_pos, &w.slot, &w.act_row, &w.pos, &w.inv, &w.kept, &w.loc }) |l| l.deinit(a);
            }
        };

        pub fn init(a: std.mem.Allocator, g: *G, source: *S, math: M, c: *const v41.Config) !Self {
            return initWith(a, g, source, math, c, .{});
        }

        pub fn initWith(a: std.mem.Allocator, g: *G, source: *S, math: M, c: *const v41.Config, opt: Options) !Self {
            // A decode-width forward must fit one route: its calls are the decode lane's by arithmetic.
            if (@as(u64, decode_forward_rows) * c.n_experts_per_tok > max_route_ids) return error.DecodeRowsWiderThanRoute;
            if (routes.lookahead and opt.gates.len != c.n_layers) return error.LookaheadNeedsGates;
            if (routes.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            const wr = opt.wide;
            if (wr.depth < 1 or wr.depth > expert_stream.max_wide_depth) return error.InvalidWideRoute;
            if ((wr.seed or wr.hot_first or wr.depth > 1 or wr.cold_rows > 0) and !routes.prefill) return error.InvalidWideRoute;
            if (wr.cold_rows > Wide.max_cold_rows) return error.InvalidWideRoute;
            if (wr.seed and comptime !@hasDecl(S, "seedPrefill")) return error.InvalidWideRoute;
            if (wr.depth > 1) {
                if (comptime @hasDecl(S, "wideDepth")) {
                    if (source.wideDepth() < wr.depth) return error.WideDepthExceedsSource;
                } else return error.WideDepthExceedsSource;
            }
            const banks = try a.alloc([n_banks]?Arrays, c.n_layers);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@backingInt(kind)] = try bind(g, source, @intCast(l), kind);
            }
            var self: Self = .{ .a = a, .source = source, .math = math, .hidden = @intCast(c.hidden_size), .n_experts = c.n_routed_experts, .banks = banks, .gates = opt.gates, .g = g, .wide_route = wr };
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
            self.wide.deinit(self.a);
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
            for (self.banks, 0..) |*b, l| b[@backingInt(BankKind.ext)] = try bind(g, self.source, @intCast(l), .ext);
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

            /// JOINLESS (a wide call only: more than max_route_ids ids): the unjoined outputs and
            /// each assignment's (output, row); `releaseParts` after the combines are evaluated.
            pub fn routedParts(h: Hook, g: *G, xf: T, indices: T) !Parts {
                if (comptime !routes.prefill) return error.PrefillLaneNotPorted;
                const n: u32 = @intCast(g.shapeOf(xf).dim(0));
                const k: u32 = @intCast(g.shapeOf(indices).dim(1));
                if (n * k <= max_route_ids) return error.WideLaneUnderMinIds;
                return h.ex.runWideParts(g, h.layer, xf, indices, n, k);
            }

            pub fn releaseParts(h: Hook, g: *G) void {
                h.ex.releaseParts(g);
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
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
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
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
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
                const b = @backingInt(ref.bank);
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
            // are the wide lane (the DIG kernels), when it is installed.
            if (n_ids > max_route_ids) {
                if (comptime !routes.prefill) {
                    return error.PrefillLaneNotPorted;
                } else {
                    return self.runWide(g, layer, xf, indices, n, k);
                }
            }
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

        /// The wide lane: the routing barrier; the call's distinct experts (first
        /// appearance) routed in groups of at most `max_route_ids`; per group,
        /// every part waited, then per bank the group's rows through the layer's
        /// DIG-X route (slot = the row the source serves the expert in, act row =
        /// its token), drained and evaluated before the group is released (the
        /// next route may refill those slots); the outputs joined in the
        /// router's order, `[n, k, hidden]` f32. The DIG kernels read bf16
        /// activations (the lane of record's MoE input): another dtype is
        /// rounded to bf16 once, here. `Options.wide.seed` seeds the layer from
        /// the call and drains each group once; `.hot_first` orders the groups hottest first;
        /// `Options.wide.depth` 2 routes group g + 1 (its reads) before group g's waves.
        fn runWide(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            try self.runWideCore(g, layer, xf, indices, n, k);
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // `take(concatenate(outputs), argsort(positions))`, the permutation made on the host.
            const joined = try g.concat(w.kept.items, 0);
            for (w.kept.items) |x| g.release(x);
            w.kept.clearRetainingCapacity();
            try w.inv.resize(a, n_ids);
            invertPositions(w.pos.items, w.inv.items);
            const ord = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// JOINLESS: the wide call's outputs unjoined (at most `max_parts`, adjacent ones concatenated
        /// beyond that) and each assignment's (output, row) as int32 [n, k, 2]; the combine reads the
        /// rows in place. The outputs stay kept until `releaseParts`.
        pub const max_parts = 24;
        pub const Parts = struct { outs: []const T, loc: T };

        fn runWideParts(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !Parts {
            try self.runWideCore(g, layer, xf, indices, n, k);
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // Each output's rows, in join order; beyond max_parts, runs of adjacent outputs concatenated.
            const n_out = w.kept.items.len;
            const per = (n_out + max_parts - 1) / max_parts;
            try w.loc.resize(a, 2 * n_ids);
            var merged: std.ArrayList(T) = .empty;
            defer merged.deinit(a);
            var j: usize = 0;
            var src: usize = 0;
            var o: usize = 0;
            while (o < n_out) : (src += 1) {
                const end = @min(o + per, n_out);
                var row: i32 = 0;
                for (w.kept.items[o..end]) |x| {
                    const r: usize = @intCast(g.shapeOf(x).dim(0));
                    for (w.pos.items[j .. j + r]) |p| {
                        w.loc.items[2 * p] = @intCast(src);
                        w.loc.items[2 * p + 1] = row;
                        row += 1;
                    }
                    j += r;
                }
                if (end - o == 1) {
                    try merged.append(a, w.kept.items[o]);
                } else {
                    const cat = g.keep(try g.concat(w.kept.items[o..end], 0));
                    for (w.kept.items[o..end]) |x| g.release(x);
                    try merged.append(a, cat);
                }
                o = end;
            }
            w.kept.clearRetainingCapacity();
            try w.kept.appendSlice(a, merged.items);
            const loc = try g.hostArray(std.mem.sliceAsBytes(w.loc.items), &.{ @intCast(n), @intCast(k), 2 }, .int32);
            return .{ .outs = w.kept.items, .loc = loc };
        }

        /// Releases the outputs `runWideParts` kept (after the combines that read them are evaluated).
        pub fn releaseParts(self: *Self, g: *G) void {
            for (self.wide.kept.items) |x| g.release(x);
            self.wide.kept.clearRetainingCapacity();
        }

        fn runWideCore(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !void {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // The wide lane takes prefill-width calls only (the prefill texts bind small inputs as
            // `constant`; a decode-width call is the decode lane's).
            if (n_ids < wide_min_ids) return error.WideLaneUnderMinIds;
            const feed = self.wide_route.seed;
            const hot_first = self.wide_route.hot_first;
            const depth: usize = self.wide_route.depth;
            const cold: u32 = self.wide_route.cold_rows;
            const cold_chunk: usize = if (@hasDecl(M, "max_decode_rows")) M.max_decode_rows else max_route_ids;
            try w.ids.resize(a, n_ids);
            _ = try g.hostIds(indices, w.ids.items);
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            for (w.ids.items) |e| if (w.first.items[e] < 0) {
                w.first.items[e] = @intCast(w.distinct.items.len);
                try w.distinct.append(a, e);
            };
            if (feed or hot_first or cold > 0) {
                try w.count.resize(a, self.n_experts);
                @memset(w.count.items, 0);
                for (w.ids.items) |e| w.count.items[e] += 1;
            }
            // The call's residency seed (the seed route), then its experts hottest first (the order route).
            if (feed) {
                if (comptime @hasDecl(S, "seedPrefill")) try self.source.seedPrefill(layer, w.ids.items) else unreachable;
            }
            if (hot_first) {
                std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                    fn lt(c: []const u32, x: u16, y: u16) bool {
                        return if (c[x] != c[y]) c[x] > c[y] else x < y;
                    }
                }.lt);
                for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            }
            const act = if (g.dtypeOf(xf) == .bfloat16) xf else try g.astype(xf, .bfloat16);
            w.pos.clearRetainingCapacity();
            w.kept.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            const n_groups = (n_distinct + max_route_ids - 1) / max_route_ids;
            // The live groups' calls, by group index mod `depth` (read ahead up to `depth` groups).
            var calls: [expert_stream.max_wide_depth]?*S.Call = @splat(null);
            errdefer {
                for (&calls) |*c| if (c.*) |cl| {
                    self.source.release(cl);
                    c.* = null;
                };
                for (w.kept.items) |x| g.release(x);
                w.kept.clearRetainingCapacity();
            }
            const groupOf = struct {
                fn f(d: []const u16, gi: usize) []const u16 {
                    return d[gi * max_route_ids .. @min((gi + 1) * max_route_ids, d.len)];
                }
            }.f;
            for (0..@min(depth, n_groups)) |gi| calls[gi % depth] = try self.source.route(layer, groupOf(w.distinct.items, gi), &.{});
            for (0..n_groups) |gi| {
                const start = gi * max_route_ids;
                const group = groupOf(w.distinct.items, gi);
                const call = calls[gi % depth].?;
                const sv = self.source.served(call);
                for (0..sv.n_parts) |p| {
                    try self.source.waitGu(call, @intCast(p));
                    try self.source.waitDown(call, @intCast(p));
                }
                const k0 = w.kept.items.len;
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    const kb = w.kept.items.len;
                    w.slot.clearRetainingCapacity();
                    w.act_row.clearRetainingCapacity();
                    w.cold_slot.clearRetainingCapacity();
                    w.cold_act.clearRetainingCapacity();
                    w.cold_pos.clearRetainingCapacity();
                    for (w.ids.items, 0..) |e, row| {
                        const fi: usize = @intCast(w.first.items[e]);
                        if (fi < start or fi >= start + group.len) continue;
                        const ref = sv.refs[fi - start];
                        if (ref.bank != kind) continue;
                        const act_row: u32 = @intCast(row / @as(usize, k));
                        if (cold > 0 and w.count.items[e] <= cold) {
                            try w.cold_slot.append(a, ref.row);
                            try w.cold_act.append(a, act_row);
                            try w.cold_pos.append(a, @intCast(row));
                            continue;
                        }
                        try w.slot.append(a, ref.row);
                        try w.act_row.append(a, act_row);
                        try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0 and w.cold_slot.items.len == 0) continue;
                    const b = self.banks[layer][@backingInt(kind)].?;
                    const hot = w.slot.items.len > 0;
                    if (hot) {
                        try w.kept.ensureUnusedCapacity(a, 1);
                        const y = try self.math.prefill(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b);
                        w.kept.appendAssumeCapacity(y);
                    }
                    // Cold rows: the decode GEMV over their slots, encoded after the bank's wide waves
                    // (the act-row index is built right before its take).
                    var c0: usize = 0;
                    while (c0 < w.cold_slot.items.len) : (c0 += cold_chunk) {
                        const c1 = @min(c0 + cold_chunk, w.cold_slot.items.len);
                        const m: c_int = @intCast(c1 - c0);
                        const ar = try g.hostArray(std.mem.sliceAsBytes(w.cold_act.items[c0..c1]), &.{m}, .uint32);
                        const xs = try g.take(act, ar, 0);
                        const sid = try g.hostArray(std.mem.sliceAsBytes(w.cold_slot.items[c0..c1]), &.{m}, .uint32);
                        const h = try self.math.gateUp(g, xs, sid, b.gate, b.up);
                        try w.kept.ensureUnusedCapacity(a, 1);
                        w.kept.appendAssumeCapacity(g.keep(try self.math.down(g, h, sid, b.down)));
                        try w.pos.appendSlice(a, w.cold_pos.items[c0..c1]);
                    }
                    if (!feed) {
                        if (hot) try self.math.finishPrefill(g);
                        try g.evalAll(w.kept.items[kb..]);
                    }
                }
                // Feed: the group's banks queued back to back, one drain before its slots go back.
                if (feed) {
                    try self.math.finishPrefill(g);
                    try g.evalAll(w.kept.items[k0..]);
                }
                self.source.release(call);
                calls[gi % depth] = null;
                if (gi + depth < n_groups) calls[gi % depth] = try self.source.route(layer, groupOf(w.distinct.items, gi + depth), &.{});
            }
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
        if (wv != w or seen[@backingInt(ref.bank)]) continue;
        seen[@backingInt(ref.bank)] = true;
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
    c.n_experts_per_tok = 6;
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
    try testing.expect(ex.banks[1][@backingInt(BankKind.ext)] == null);
    try ex.grow(&g, &.{ 8, 8 });
    try testing.expect(ex.banks[1][@backingInt(BankKind.ext)] != null);
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

/// The wide lane's host tests: a registry from the embedded manifest (no MLX).
fn hostRegistry() !xk.Registry {
    var diag: xk.Diag = .{};
    return xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("dsv41 experts: {s}\n", .{diag.message()});
        return e;
    };
}

/// A wide-lane route that records every call: its rows, the bank it was
/// handed and, from the fake source's live route, the served slots of the
/// group (so the test can derive what each call should have been).
const RecRoute = struct {
    a: std.mem.Allocator,
    calls: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    /// The source whose live route each call snapshots.
    var source: ?*FakeSource = null;

    const Rec = struct { slot: []u32, act_row: []u32, bank_code: u32, route: u64, refs: []SlotRef, at: usize, node: usize, act_dtype: ops.Dtype };

    pub fn init(a: std.mem.Allocator) RecRoute {
        return .{ .a = a };
    }

    pub fn deinit(self: *RecRoute, _: *TraceOps) void {
        for (self.calls.items) |r| {
            self.a.free(r.slot);
            self.a.free(r.act_row);
            self.a.free(r.refs);
        }
        self.calls.deinit(self.a);
    }

    pub fn call(self: *RecRoute, g: *TraceOps, act: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const src = source.?;
        const live = for (&src.calls) |*c| {
            if (c.state == .live) break c;
        } else return error.NoLiveRoute;
        try self.calls.append(self.a, .{
            .slot = try self.a.dupe(u32, rows.slot),
            .act_row = try self.a.dupe(u32, rows.act_row.?),
            .bank_code = bank.gate.code,
            .route = src.counters.route_calls,
            .refs = try self.a.dupe(SlotRef, live.refs[0..live.plan.n_ids]),
            .at = src.log.items.len,
            .node = g.nodes.items.len,
            .act_dtype = g.dtypeOf(act),
        });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *RecRoute, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "dsv41 experts: a wide call routes its experts in groups and runs each bank's rows through the wide route" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 64, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    src.trace = &g;
    RecRoute.source = &src;
    defer RecRoute.source = null;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    var rrs = [_]RecRoute{RecRoute.init(a)};
    defer rrs[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c);
    defer ex.deinit();
    // 20 tokens x top-6 over 64 experts: 120 rows, every expert id distinct within a token.
    const n = 20;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((7 * (i / k) + 11 * (i % k)) % 64);
    var script: Script = .{ .calls = &.{&ids} };
    g.host_values = script.values();
    // An f32 stream reaches the wide route rounded to bf16 (the DIG kernels' activations).
    const out = try ex.at(0).routed(&g, try g.input(&.{ n, 256 }, .float32), try g.input(&.{ n, k }, .int32));
    try testing.expect(g.shapeOf(out).eql(ops.Shape.of(&.{ n, k, 256 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(out));

    // The call's distinct experts, in first appearance, routed in groups of max_route_ids.
    var first: [64]i32 = @splat(-1);
    var distinct: std.ArrayList(u16) = .empty;
    defer distinct.deinit(a);
    for (ids) |e| if (first[e] < 0) {
        first[e] = @intCast(distinct.items.len);
        try distinct.append(a, e);
    };
    const n_groups = (distinct.items.len + max_route_ids - 1) / max_route_ids;
    try testing.expectEqual(@as(u64, n_groups), src.counters.route_calls);
    // The source protocol: per group route, every part's waits, release (the next route flushes it).
    var kb: [64]u8 = undefined;
    const kinds = kindsOf(src.log.items, &kb);
    try testing.expectEqual(@as(usize, n_groups), std.mem.count(u8, kinds, "R"));
    try testing.expectEqual(@as(usize, n_groups), std.mem.count(u8, kinds, "r"));
    try testing.expectEqual(std.mem.count(u8, kinds, "g"), std.mem.count(u8, kinds, "d"));
    // The default wide route (no feed, depth 1): one group live at a time, and one finish + eval
    // per bank call (the schedule before the wide routes existed).
    var open = false;
    for (kinds) |kd| switch (kd) {
        'R' => {
            try testing.expect(!open);
            open = true;
        },
        'r' => {
            try testing.expect(open);
            open = false;
        },
        else => {},
    };
    try testing.expect(!open);
    try testing.expectEqual(rrs[0].calls.items.len, g.evals.items.len);

    // Each recorded call is exactly its group's rows in one bank, in routed order, read
    // from the slot the source serves the expert in, after the group's waits and before
    // its release; together they cover every row once.
    const rr = &rrs[0];
    try testing.expect(rr.calls.items.len >= n_groups);
    try testing.expectEqual(@as(u32, @intCast(rr.calls.items.len)), rr.finishes);
    var rows_seen: usize = 0;
    for (rr.calls.items) |rec| {
        try testing.expectEqual(ops.Dtype.bfloat16, rec.act_dtype);
        const gidx: usize = @intCast(rec.route - 1);
        const kind: BankKind = for ([_]BankKind{ .base, .ext, .transient }) |kd| {
            if (ex.banks[0][@backingInt(kd)]) |b| if (b.gate.code == rec.bank_code) break kd;
        } else return error.UnknownBank;
        var want_slot: std.ArrayList(u32) = .empty;
        defer want_slot.deinit(a);
        var want_act: std.ArrayList(u32) = .empty;
        defer want_act.deinit(a);
        for (ids, 0..) |e, row| {
            const gi: usize = @intCast(first[e]);
            if (gi / max_route_ids != gidx) continue;
            const ref = rec.refs[gi - gidx * max_route_ids];
            if (ref.bank != kind) continue;
            try want_slot.append(a, ref.row);
            try want_act.append(a, @intCast(row / k));
        }
        try testing.expectEqualSlices(u32, want_slot.items, rec.slot);
        try testing.expectEqualSlices(u32, want_act.items, rec.act_row);
        // after the group's route and waits, before its release
        try testing.expect(kinds[rec.at - 1] == 'd' or kinds[rec.at - 1] == 'R');
        try testing.expect(rec.at < kinds.len);
        // evaluated before the group is released (the next route may refill its slots)
        const release_at = for (src.log.items[rec.at..]) |e| {
            if (e.kind == .release) break e.at;
        } else return error.NoRelease;
        var evaluated = false;
        for (g.evals.items) |ev| evaluated = evaluated or (ev >= rec.node and ev <= release_at);
        try testing.expect(evaluated);
        rows_seen += rec.slot.len;
    }
    try testing.expectEqual(@as(usize, n * k), rows_seen);
}

/// dump_prefill_waves.route_rows: slot j repeated counts[j] times, then
/// Fisher-Yates from the end with j = splitmix64(seed) output % (i + 1).
fn sampleRows(a: std.mem.Allocator, seed: u64, slots: []const u32, counts: []const u32) ![]u16 {
    var n: usize = 0;
    for (counts) |c| n += c;
    const rows = try a.alloc(u16, n);
    var i: usize = 0;
    for (slots, counts) |s, c| for (0..c) |_| {
        rows[i] = @intCast(s);
        i += 1;
    };
    var st = seed;
    var j = n;
    while (j > 1) {
        j -= 1;
        const r: usize = @intCast(xk.splitmix64(&st) % (j + 1));
        std.mem.swap(u16, &rows[j], &rows[r]);
    }
    return rows;
}

test "dsv41 experts: the kernels' decode GEMV launches configs prepared on the model's backend at construction" {
    var reg = try hostRegistry();
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    {
        var gemv = try xq.Gemv(TraceOps).init(&g, &reg);
        defer gemv.deinit(&g);
        try testing.expect(g.prepared_live > 0);
        // gate / up: xh f32 [rows, 5120] at slot rows ids, code i16 [cap, 320, 144, 48] -> [rows, 2304] f32
        const z = try gemv.project(&g, .gate, try g.input(&.{ 6, 5120 }, .float32), try g.input(&.{6}, .uint32), try g.input(&.{ 64, 320, 144, 48 }, .int16));
        try testing.expect(g.shapeOf(z).eql(ops.Shape.of(&.{ 6, 2304 })));
        try testing.expectEqual(@as(usize, 1), g.prepared_launches);
    }
    try testing.expectEqual(@as(usize, 0), g.prepared_live);
}

test "dsv41 experts: the joined outputs are put back in routed order" {
    // Outputs joined as positions 3, 0, 4, 1, 2: routed position p reads joined row inv[p].
    var inv: [5]u32 = undefined;
    invertPositions(&.{ 3, 0, 4, 1, 2 }, &inv);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 4, 0, 2 }, &inv);
}

test "dsv41 experts: a decode-width call never takes the wide lane: construction proves it fits a route, the wide lane refuses it by name" {
    const a = testing.allocator;
    var c = testConfig(256, 128, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 64, .rows = &.{16} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    src.trace = &g;
    RecRoute.source = &src;
    defer RecRoute.source = null;
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, RecRoute);
    var rrs = [_]RecRoute{RecRoute.init(a)};
    defer rrs[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    // A top-k whose decode block would overflow one route is refused at construction.
    var wide_k = c;
    wide_k.n_experts_per_tok = 7;
    try testing.expectError(error.DecodeRowsWiderThanRoute, Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &wide_k));
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &rrs }, &c);
    defer ex.deinit();
    // 8 rows x top-6 = 48 ids: the wide lane refuses them by name (the dispatch sends them to the decode lane).
    try testing.expectEqual(@as(u32, 48), decode_forward_rows * 6);
    try testing.expectError(error.WideLaneUnderMinIds, ex.runWide(&g, 0, try g.input(&.{ 8, 256 }, .float32), try g.input(&.{ 8, 6 }, .int32), 8, 6));
    try testing.expectEqual(@as(usize, 0), rrs[0].calls.items.len);
}

test "dsv41 experts: JOINLESS hands out the wide call's outputs unjoined, each assignment at one (output, row)" {
    const a = testing.allocator;
    var c = testConfig(5120, 2304, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = 64, .rows = &.{64} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, xq.DigXPrefill(TraceOps));
    var digx = [_]xq.DigXPrefill(TraceOps){try xq.DigXPrefill(TraceOps).init(a, &reg, .tier, null)};
    defer digx[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &digx }, &c);
    defer ex.deinit();
    // 40 tokens x top-6 over 60 experts (two groups of at most 48): a deterministic spread.
    const n: u32 = 40;
    const k: u32 = 6;
    var rows: [40 * 6]u16 = undefined;
    for (&rows, 0..) |*e, i| e.* = @intCast((i * 7 + i / 6) % 60);
    var script: Script = .{ .calls = &.{&rows} };
    g.host_values = script.values();
    const parts = try ex.at(0).routedParts(&g, try g.input(&.{ @intCast(n), 5120 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(k) }, .int32));
    try testing.expect(parts.outs.len >= 2 and parts.outs.len <= Ex.max_parts);
    try testing.expect(g.shapeOf(parts.loc).eql(ops.Shape.of(&.{ @intCast(n), @intCast(k), 2 })));
    // Every (output, row) is some assignment's, once.
    var seen = std.AutoHashMap(u64, void).init(a);
    defer seen.deinit();
    const loc = ex.wide.loc.items;
    for (0..n * k) |q| {
        const s_: usize = @intCast(loc[2 * q]);
        const r_: i32 = loc[2 * q + 1];
        try testing.expect(s_ < parts.outs.len and r_ >= 0 and r_ < g.shapeOf(parts.outs[s_]).dim(0));
        try testing.expect(!(try seen.getOrPut((@as(u64, s_) << 32) | @as(u64, @intCast(r_)))).found_existing);
    }
    var total: c_int = 0;
    for (parts.outs) |o| total += g.shapeOf(o).dim(0);
    try testing.expectEqual(@as(c_int, @intCast(n * k)), total);
    ex.at(0).releaseParts(&g);
}

test "dsv41 experts: a wide call runs the DIG-X prefill route with the lane samples' wave structure" {
    const a = testing.allocator;
    const Sample = struct {
        cases: []const struct {
            case: []const u8,
            calls: []const struct { name: []const u8, a_rows: u32, route: struct { seed: u64, slots: []const u32, counts: []const u32 }, events: []const []const u8 },
        },
    };
    const parsed = try std.json.parseFromSlice(Sample, a, @embedFile("fixtures/dsv41_prefill_wave_samples.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const tier = for (parsed.value.cases) |cs| {
        if (std.mem.eql(u8, cs.case, "tier")) break cs;
    } else return error.NoTierCase;
    var c = testConfig(5120, 2304, 1);
    c.n_routed_experts = 64;
    var reg = try hostRegistry();
    defer reg.deinit();
    var src = try FakeSource.init(a, .{ .hidden = 5120, .inter = 2304, .n_experts = 64, .rows = &.{64} });
    defer src.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const Chain = EagerChain(TraceOps, TraceGemv);
    const Math = WithPrefillRoutes(TraceOps, Chain, xq.DigXPrefill(TraceOps));
    var digx = [_]xq.DigXPrefill(TraceOps){try xq.DigXPrefill(TraceOps).init(a, &reg, .tier, null)};
    defer digx[0].deinit(&g);
    const Ex = ExpertsWith(TraceOps, FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = Chain.init(.{}, &c), .routes = &digx }, &c);
    defer ex.deinit();
    // The sample calls one route serves whole (at most max_route_ids distinct experts),
    // their rows as tokens x k: the lane's slots are this call's experts.
    const Pick = struct { name: []const u8, k: u32 };
    const picks = [_]Pick{ .{ .name = "chunk183", .k = 6 }, .{ .name = "drained", .k = 6 }, .{ .name = "solo_carried", .k = 4 }, .{ .name = "budget_gt", .k = 3 } };
    var checked: usize = 0;
    for (picks) |pk| {
        const cl = for (tier.calls) |x| {
            if (std.mem.eql(u8, x.name, pk.name)) break x;
        } else return error.NoSampleCall;
        try testing.expect(cl.route.slots.len <= max_route_ids);
        const rows = try sampleRows(a, cl.route.seed, cl.route.slots, cl.route.counts);
        defer a.free(rows);
        const n: u32 = @intCast(rows.len / pk.k);
        try testing.expectEqual(rows.len, n * pk.k);
        var script: Script = .{ .calls = &.{rows} };
        g.host_values = script.values();
        const first_node = g.nodes.items.len;
        const resets_before = g.freed.items.len;
        const calls_before = src.counters.route_calls;
        const out = try ex.at(0).routed(&g, try g.input(&.{ @intCast(n), 5120 }, .bfloat16), try g.input(&.{ @intCast(n), @intCast(pk.k) }, .int32));
        try testing.expect(g.shapeOf(out).eql(ops.Shape.of(&.{ @intCast(n), @intCast(pk.k), 5120 })));
        try testing.expectEqual(calls_before + 1, src.counters.route_calls);
        // The lane's waves: 5 launches each (7 outputs: take2 2, gate|up GEMM 2, onepass,
        // down GEMM, widen1), a reset per wave plus the join's.
        var launches: usize = 0;
        for (cl.events) |e| launches += @intFromBool(std.mem.startsWith(u8, e, "launch "));
        const waves = launches / 5;
        var kernels: usize = 0;
        for (g.nodes.items[first_node..]) |nd| kernels += @intFromBool(nd.op == .kernel);
        try testing.expectEqual(7 * waves, kernels);
        try testing.expectEqual(waves + 1, g.freed.items.len - resets_before);
        // A prefill route builds each launch per call (its rows vary up to 2^20).
        try testing.expectEqual(@as(usize, 0), g.prepared_launches);
        checked += 1;
        src.flush() catch {};
    }
    try testing.expectEqual(picks.len, checked);
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
            try testing.expectEqualSlices(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(layer, slot, @fromBackingInt(@intCast(comp)))[0..seg.length]);
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
    const base = ex.banks[0][@backingInt(BankKind.base)].?;
    try testing.expect(g.shapeOf(base.gate.code).eql(ops.Shape.of(&.{ 4, 4, 2, 48 })));
    try testing.expect(g.shapeOf(ex.banks[1][@backingInt(BankKind.transient)].?.down.rin).eql(ops.Shape.of(&.{ 12, 32 })));

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
    try testing.expect(g.shapeOf(ex.banks[1][@backingInt(BankKind.ext)].?.up.rout).eql(ops.Shape.of(&.{ 4, 32 })));
    try testing.expect(ex.banks[0][@backingInt(BankKind.ext)] == null);
    _ = try ex.at(1).routed(&g, xf, idx);
    const r1 = src.calls[
        for (src.calls, 0..) |cl, i| {
            if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1) break i;
        } else unreachable
    ];
    try expectRowsHold(s, &sb, 1, script.calls[1], src.served(&r1));
    try testing.expectEqual(@as(u32, 2), src.served(&r1).n_parts); // six misses: 1, 2, 9 | 12, 20, 21
    _ = try ex.at(1).routed(&g, xf, idx);
    const r2 = src.calls[
        for (src.calls, 0..) |cl, i| {
            if (cl.route != null and cl.route.?.state == .released and cl.route.?.layer == 1 and cl.route.?.plan.n_hits > 0) break i;
        } else unreachable
    ];
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
        if (w > 0) kinds[@backingInt(r.bank)] = true;
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

/// A wide-lane route over a real Stream (trace backend) that names, per call row, the expert
/// whose record its slot holds (verified byte for byte) and its act row; and records the live
/// routes and the evals so far at the call.
const StreamRec = struct {
    a: std.mem.Allocator,
    recs: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    var stream: ?*expert_stream.Stream = null;
    var bank: ?*const SynthBank = null;
    /// The hook's bound banks of the layer (to name a call's bank kind).
    var base_code: u32 = 0;
    var transient_code: u32 = 0;

    const Rec = struct { experts: []u16, act_row: []u32, base: bool, live: u32, evals: usize };

    fn init(a: std.mem.Allocator) StreamRec {
        return .{ .a = a };
    }

    fn deinit(self: *StreamRec, _: *TraceOps) void {
        for (self.recs.items) |r| {
            self.a.free(r.experts);
            self.a.free(r.act_row);
        }
        self.recs.deinit(self.a);
    }

    fn expertIn(s: *expert_stream.Stream, sb: *const SynthBank, slot: u32) !u16 {
        const geom = &sb.bank.layers[0];
        for (0..sb.bank.n_experts) |e| {
            const off = sb.bank.recordOffset(0, @intCast(e));
            const held = for (geom.segments, 0..) |seg, c| {
                if (!std.mem.eql(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(0, slot, @fromBackingInt(@intCast(c)))[0..seg.length])) break false;
            } else true;
            if (held) return @intCast(e);
        }
        return error.SlotHoldsNoRecord;
    }

    pub fn call(self: *StreamRec, g: *TraceOps, act: u32, rows: quant.PrefillRows, b: BankArraysOf(u32)) !u32 {
        const s = stream.?;
        const base = b.gate.code == base_code;
        if (!base and b.gate.code != transient_code) return error.UnknownBank;
        const cap = s.layers[0].policy.capacity;
        const experts = try self.a.alloc(u16, rows.slot.len);
        errdefer self.a.free(experts);
        for (rows.slot, experts) |r, *e| e.* = try expertIn(s, bank.?, if (base) r else cap + r);
        var live: u32 = 0;
        for (&s.routes) |*r| live += @intFromBool(r.state == .live);
        try self.recs.append(self.a, .{ .experts = experts, .act_row = try self.a.dupe(u32, rows.act_row.?), .base = base, .live = live, .evals = g.evals.items.len });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *StreamRec, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "dsv41 experts: the wide feed and read-ahead serve every routed row from its record, hottest first, one drain per group, reads ahead" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    // 60 tokens x top-6, expert j of a token in band j (distinct in a row): 64 experts routed 3..15 times.
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| {
        const t = i / k;
        const j = i % k;
        e.* = @intCast(j * 20 + (t * (j + 3)) % 20);
    }
    var count: [128]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    var hot: [128]u16 = undefined;
    var n_distinct: usize = 0;
    for (0..128) |e| if (count[e] > 0) {
        hot[n_distinct] = @intCast(e);
        n_distinct += 1;
    };
    try testing.expectEqual(@as(usize, 64), n_distinct);
    std.sort.pdq(u16, hot[0..n_distinct], @as([]const u32, &count), struct {
        fn lt(cn: []const u32, x: u16, y: u16) bool {
            return if (cn[x] != cn[y]) cn[x] > cn[y] else x < y;
        }
    }.lt);
    var first_seen: [128]bool = @splat(false);
    var appear: [128]u16 = undefined;
    var n_app: usize = 0;
    for (ids) |e| if (!first_seen[e]) {
        first_seen[e] = true;
        appear[n_app] = e;
        n_app += 1;
    };

    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    for ([_]Wide{ .{}, .{ .seed = true, .hot_first = true }, .{ .depth = 2 }, .{ .seed = true, .hot_first = true, .depth = 2 }, .{ .seed = true, .depth = 2 }, .{ .hot_first = true } }) |wide| {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = wide.depth, .transient_rows = @as(u32, wide.depth) * max_route_ids });
        defer s.deinit();
        StreamRec.stream = s;
        defer StreamRec.stream = null;
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        var rrs = [_]StreamRec{StreamRec.init(a)};
        defer rrs[0].deinit(&g);
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = wide });
        defer ex.deinit();
        StreamRec.base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        const evals0 = g.evals.items.len;
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
        try ex.flush();
        const recs = rrs[0].recs.items;

        // Exact: every routed row once, computed from its own expert's record with its token.
        var seen: [n * k]bool = @splat(false);
        for (recs) |r| for (r.experts, r.act_row) |e, t| {
            const row = for (0..k) |j| {
                const i = t * k + j;
                if (ids[i] == e and !seen[i]) break i;
            } else return error.RowNotRouted;
            seen[row] = true;
        };
        for (seen) |x| try testing.expect(x);
        // Every distinct record read once (the call starts from empty rows).
        try testing.expectEqual(@as(u64, n_distinct * sb.bank.layers[0].logical_bytes), s.stats().expert_bytes_read);
        try testing.expectEqual(@as(u64, 2), s.stats().route_calls);

        // The groups: the first 48 of the feed order (hottest first) or of first appearance.
        const order: []const u16 = if (wide.hot_first) hot[0..n_distinct] else appear[0..n_app];
        var in_first: [128]bool = @splat(false);
        for (order[0..max_route_ids]) |e| in_first[e] = true;
        var second_started = false;
        for (recs) |r| {
            const grp0 = in_first[r.experts[0]];
            for (r.experts) |e| try testing.expectEqual(grp0, in_first[e]);
            if (!grp0) second_started = true else try testing.expect(!second_started);
            // Group 0's waves run with group 1 already routed (its reads issued) at depth 2.
            if (grp0) try testing.expectEqual(@as(u32, wide.depth), r.live) else try testing.expectEqual(@as(u32, 1), r.live);
        }
        // Drains: one per group with the feed, else one per bank call.
        const drains = g.evals.items.len - evals0;
        try testing.expectEqual(if (wide.seed) @as(usize, 2) else recs.len, drains);
        try testing.expectEqual(@as(u32, @intCast(if (wide.seed) 2 else recs.len)), rrs[0].finishes);
        // The feed's seed: the persistent rows hold the 16 hottest (protected), the rest transient.
        if (wide.seed and wide.hot_first) {
            var in_top: [128]bool = @splat(false);
            for (hot[0..16]) |e| in_top[e] = true;
            for (recs) |r| for (r.experts) |e| try testing.expectEqual(in_top[e], r.base);
        }
    }
}

test "dsv41 experts: a read-ahead deeper than the source's windows is refused at construction" {
    const a = testing.allocator;
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 4, 4 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
    defer s.deinit();
    var src = StreamSource.init(s);
    var g = TraceOps.init(a);
    defer g.deinit();
    const c = testConfig(64, 32, 2);
    const Math = WithPrefillRoutes(TraceOps, TraceMath, StreamRec);
    var rrs = [_]StreamRec{ StreamRec.init(a), StreamRec.init(a) };
    const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
    try testing.expectError(error.WideDepthExceedsSource, Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .depth = 2 } }));
    try testing.expectError(error.InvalidWideRoute, Ex.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .depth = 3 } }));
    const Plain = ExpertsWith(TraceOps, StreamSource, Math, .{});
    try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .seed = true } }));
    try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .hidden = 64, .inter = 32 }, .routes = &rrs }, &c, .{ .wide = .{ .hot_first = true } }));
}

/// Decode-lane math that records each gateUp call's bank, the expert whose record each slot
/// holds at the call (StreamRec's stream and bank) and the act rows (the act-row index is the
/// host array made right before the call's `take`), in calls of <= 8 rows.
const ColdRec = struct {
    inner: TraceMath,
    pub const max_decode_rows: u32 = 8;
    var log: std.ArrayList(Call) = .empty;
    const Call = struct { code: u32, experts: []u16, act_row: []u32, evals: usize };

    fn reset(a: std.mem.Allocator) void {
        for (log.items) |c| {
            a.free(c.experts);
            a.free(c.act_row);
        }
        log.clearAndFree(a);
    }

    pub fn gateUp(self: *const ColdRec, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        const slots = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(g.hostBytesOf(ids) orelse return error.NoHostBytes)));
        const acts = g.hostBytesOf(x - 1) orelse return error.NoHostBytes;
        const s = StreamRec.stream.?;
        const base = gate.code == StreamRec.base_code;
        const experts = try g.gpa.alloc(u16, slots.len);
        errdefer g.gpa.free(experts);
        for (slots, experts) |slot, *e| e.* = try StreamRec.expertIn(s, StreamRec.bank.?, if (base) slot else s.layers[0].policy.capacity + slot);
        try log.append(g.gpa, .{
            .code = gate.code,
            .experts = experts,
            .act_row = try g.gpa.dupe(u32, std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(acts)))),
            .evals = g.evals.items.len,
        });
        return self.inner.gateUp(g, x, ids, gate, up);
    }

    pub fn down(self: *const ColdRec, g: *TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, d);
    }
};

test "dsv41 experts: cold rows run the decode lane over their own records after their bank's wide waves, every row once" {
    const a = testing.allocator;
    var sb = try SynthBank.open(128);
    defer sb.close();
    StreamRec.bank = &sb;
    defer StreamRec.bank = null;
    var c = testConfig(64, 32, 1);
    c.n_routed_experts = 128;
    const n = 60;
    const k = 6;
    var ids: [n * k]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast((i % k) * 20 + ((i / k) * ((i % k) + 3)) % 20);
    var count: [128]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    const Math = WithPrefillRoutes(TraceOps, ColdRec, StreamRec);
    try testing.expectEqual(@as(u32, 8), Math.max_decode_rows);
    for ([_]Wide{ .{ .cold_rows = 3 }, .{ .cold_rows = 3, .seed = true, .hot_first = true, .depth = 2 } }) |wide| {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 }, .wide_depth = wide.depth, .transient_rows = @as(u32, wide.depth) * max_route_ids });
        defer s.deinit();
        StreamRec.stream = s;
        defer StreamRec.stream = null;
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        g.record_host = true;
        defer ColdRec.reset(a);
        var rrs = [_]StreamRec{StreamRec.init(a)};
        defer rrs[0].deinit(&g);
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        var ex = try Ex.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = wide });
        defer ex.deinit();
        const base_code = ex.banks[0][@backingInt(BankKind.base)].?.gate.code;
        StreamRec.base_code = base_code;
        StreamRec.transient_code = ex.banks[0][@backingInt(BankKind.transient)].?.gate.code;
        var script: Script = .{ .calls = &.{&ids} };
        g.host_values = script.values();
        _ = try ex.at(0).routed(&g, try g.input(&.{ n, 64 }, .bfloat16), try g.input(&.{ n, k }, .int32));
        try ex.flush();

        // Every routed row once: the wide route's rows are the hot experts', the decode lane's the cold ones'.
        var seen: [n * k]bool = @splat(false);
        const mark = struct {
            fn f(sn: *[n * k]bool, idv: []const u16, e: u16, t: u32) !void {
                for (0..k) |j| {
                    const i = t * k + j;
                    if (idv[i] == e and !sn[i]) {
                        sn[i] = true;
                        return;
                    }
                }
                return error.RowNotRouted;
            }
        }.f;
        for (rrs[0].recs.items) |r| for (r.experts, r.act_row) |e, t| {
            try testing.expect(count[e] > 3);
            try mark(&seen, &ids, e, t);
        };
        var cold_calls: usize = 0;
        for (ColdRec.log.items) |cl| {
            try testing.expect(cl.experts.len >= 1 and cl.experts.len <= 8);
            for (cl.experts, cl.act_row) |e, t| {
                try testing.expectEqual(@as(u32, 3), count[e]);
                try mark(&seen, &ids, e, t);
            }
            cold_calls += 1;
        }
        for (seen) |x| try testing.expect(x);
        // 40 experts routed 3 times: 120 cold rows in calls of at most 8.
        var cold_rows: usize = 0;
        for (ColdRec.log.items) |cl| cold_rows += cl.experts.len;
        try testing.expectEqual(@as(usize, 120), cold_rows);
        try testing.expect(cold_calls >= 15);
        try testing.expectEqual(@as(u64, 64 * sb.bank.layers[0].logical_bytes), s.stats().expert_bytes_read);
    }
    // A cold threshold past the bound, or without the wide lane, is refused at construction.
    {
        const s = try expert_stream.Stream.init(a, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 } });
        defer s.deinit();
        var src = StreamSource.init(s);
        var g = TraceOps.init(a);
        defer g.deinit();
        var rrs = [_]StreamRec{StreamRec.init(a)};
        const Ex = ExpertsWith(TraceOps, StreamSource, Math, .{ .prefill = true });
        try testing.expectError(error.InvalidWideRoute, Ex.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = .{ .cold_rows = Wide.max_cold_rows + 1 } }));
        const Plain = ExpertsWith(TraceOps, StreamSource, Math, .{});
        try testing.expectError(error.InvalidWideRoute, Plain.initWith(a, &g, &src, .{ .d = .{ .inner = .{ .hidden = 64, .inter = 32 } }, .routes = &rrs }, &c, .{ .wide = .{ .cold_rows = 2 } }));
    }
}
