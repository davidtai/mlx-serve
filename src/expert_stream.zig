//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages). `Stream`: per-layer slot pools, routes, deferred release, growth.

const std = @import("std");
const mlx = @import("mlx.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("expert_io.zig");
const expert_policy = @import("expert_policy.zig");

const n_components = expert_bank.n_components;
const Component = expert_bank.Component;
const Layer = expert_bank.Layer;
const LayerPolicy = expert_policy.LayerPolicy;
const Phase = expert_policy.Phase;
const Plan = expert_policy.Plan;
const max_route_ids = expert_policy.max_route_ids;

pub const HostSlotRows = struct {
    rows: u32,
    row_bytes: [n_components]u64,
    banks: [n_components][]u8,

    /// Rows sized for `layer`'s segments; one page-aligned bank per component.
    pub fn init(layer: *const Layer, rows: u32) !HostSlotRows {
        var s: HostSlotRows = .{ .rows = rows, .row_bytes = undefined, .banks = undefined };
        var n: usize = 0;
        errdefer for (s.banks[0..n]) |b| std.heap.page_allocator.free(b);
        for (layer.segments, 0..) |seg, c| {
            s.row_bytes[c] = seg.length;
            const bytes = std.math.mul(u64, seg.length, rows) catch return error.OutOfMemory;
            s.banks[c] = try std.heap.page_allocator.alloc(u8, @intCast(bytes));
            n += 1;
        }
        return s;
    }

    /// After the pool that wrote into the rows has stopped.
    pub fn deinit(self: *HostSlotRows) void {
        for (self.banks) |b| std.heap.page_allocator.free(b);
        self.* = undefined;
    }

    pub fn row(self: *const HostSlotRows, c: Component, r: u32) []u8 {
        const n = self.row_bytes[@intFromEnum(c)];
        return self.banks[@intFromEnum(c)][r * n ..][0..n];
    }

    pub fn rowDest(self: *const HostSlotRows, r: u32) [n_components]u64 {
        var d: [n_components]u64 = undefined;
        for (&d, 0..) |*a, c| a.* = @intFromPtr(self.row(@enumFromInt(c), r).ptr);
        return d;
    }
};

pub const LayerSlotBank = struct {
    arrays: [n_components]mlx.mlx_array,
    base: [n_components]u64,
    row_bytes: [n_components]u64,
    rows: u32,

    /// Nine arrays in the Python bank's dtypes (code int16 [rows, in/16, out/16,
    /// 16K], rout / rin float16 [rows, out] / [rows, in]), zero-filled and
    /// evaluated once on `stream`. The data pointers are taken here; the arrays
    /// stay held (never donated or recycled) until `deinit`, after the pool stops.
    /// MLX allocates through Metal even on the CPU stream: callers hold the GPU lock.
    pub fn init(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
        var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = undefined, .row_bytes = undefined, .rows = rows };
        errdefer b.deinit();
        for (layer.segments, 0..) |seg, c| {
            var shape: [4]c_int = undefined;
            shape[0] = @intCast(rows);
            for (seg.shape[0..seg.rank], 1..) |d, k| shape[k] = @intCast(d);
            const dtype: mlx.mlx_dtype = switch (seg.dtype) {
                .I16 => .int16,
                .F16 => .float16,
            };
            b.arrays[c] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&b.arrays[c], &shape, seg.rank + 1, dtype, stream));
            try mlx.check(mlx.mlx_array_eval(b.arrays[c]));
            const p = mlx.mlx_array_data_uint8(b.arrays[c]) orelse return error.MlxNoData;
            b.base[c] = @intFromPtr(p);
            b.row_bytes[c] = seg.length;
        }
        return b;
    }

    pub fn deinit(self: *LayerSlotBank) void {
        for (self.arrays) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
        self.* = undefined;
    }

    pub fn row(self: *const LayerSlotBank, c: Component, r: u32) []u8 {
        const n = self.row_bytes[@intFromEnum(c)];
        const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
        return p[0..n];
    }

    pub fn rowDest(self: *const LayerSlotBank, r: u32) [n_components]u64 {
        var d: [n_components]u64 = undefined;
        for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
        return d;
    }
};

// ── Stream: per-layer slot pools, routes, deferred release, growth ──

pub const Options = struct {
    /// Persistent rows per routed layer for the prefill phase: the caller's
    /// admission result (nothing here sizes memory).
    rows: []const u32,
    /// Widest route the caller passes (verify rows x top-k).
    max_route_ids: u32 = max_route_ids,
    /// Transient rows shared by every layer; at least `max_route_ids`, so no
    /// route's misses can overflow them.
    transient_rows: u32 = max_route_ids,
    /// Decode misses per completion part (one pool job).
    records_per_part: u32 = 3,
    pool: expert_io.Options = .{ .tickets = 1024 },
};

pub const Stats = struct {
    route_calls: u64 = 0,
    /// Unique experts per route that were resident / had to be loaded.
    expert_cache_hits: u64 = 0,
    expert_cache_misses: u64 = 0,
    expert_cache_evictions: u64 = 0,
    persistent_loads: u64 = 0,
    transient_loads: u64 = 0,
    /// Loads whose slot still held the record: no read.
    loads_skipped: u64 = 0,
    expert_bytes_read: u64 = 0,
    preadv_calls: u64 = 0,
    /// Sum over read ranges of first syscall to publication.
    expert_read_seconds: f64 = 0,
    /// Wall time with a read in flight (the pool's gauge).
    read_wall_ns: u64 = 0,
    /// Speculative reads, from the lookahead pools (phase 2).
    claimed: u64 = 0,
    spec_bytes: u64 = 0,
};

pub const Error = error{
    StreamFailed,
    RoutesExhausted,
    SlotStillPinned,
    ReadFailed,
    Timeout,
    TicketsBusy,
    QueueFull,
    SubmitRefused,
    InvalidJob,
};

const SlotState = enum(u8) { empty, loading, ready, failed };

/// What a physical row holds and how many live routes serve from it.
const SlotMeta = struct { pins: u16 = 0, state: SlotState = .empty, layer: u16 = 0, expert: u16 = 0 };

pub const Part = struct {
    /// Its loads: `plan.loads[order[first + i]]`, i < n.
    first: u32,
    n: u32,
    /// Pool tickets of the loads that read: gate/up of read i = ticket + i,
    /// down = ticket + n_reads + i.
    ticket: u32 = 0,
    n_reads: u32 = 0,
    settled: bool = false,
};

pub const Route = struct {
    layer: u32 = 0,
    plan: Plan = .{},
    /// The slot of each hit (plan.hits order).
    hit_slots: [max_route_ids]u32 = undefined,
    /// Load indices in placement (file offset) order; parts are runs of it.
    order: [max_route_ids]u8 = undefined,
    /// Per load (plan order): false when its slot still held the record.
    reads: [max_route_ids]bool = undefined,
    n_parts: u32 = 0,
    parts: [max_route_ids]Part = undefined,
    state: enum { free, live, released } = .free,

    pub fn partsOf(r: *const Route) []const Part {
        return r.parts[0..r.n_parts];
    }
};

/// The route being served plus released ones awaiting the next flush.
const route_capacity = 4;
const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;

/// Expert residency for one model: per-layer persistent slot pools at the
/// caller's row bound, a transient scratch shared by all layers, and the read
/// pool. One inference thread calls it; pool workers only write slot bytes.
pub const Stream = struct {
    allocator: std.mem.Allocator,
    bank: *const expert_bank.Bank,
    pool: *expert_io.Pool,
    layers: []LayerSlots,
    transient: HostSlotRows,
    transient_meta: []SlotMeta,
    max_route_ids: u32,
    records_per_part: u32,
    phase: Phase = .prefill,
    failed: bool = false,
    routes: [route_capacity]Route = @splat(.{}),
    counters: Stats = .{},
    read_ns: u64 = 0,

    const LayerSlots = struct {
        policy: LayerPolicy,
        /// The prefill rows, then the rows `grow` added.
        base: HostSlotRows,
        ext: ?HostSlotRows = null,
        /// [n_experts]: one entry per persistent slot.
        meta: []SlotMeta,
        lens: [n_components]u64,
    };

    const Location = struct { rows: *const HostSlotRows, row: u32, meta: *SlotMeta };

    pub fn init(a: std.mem.Allocator, bank: *const expert_bank.Bank, opt: Options) !*Stream {
        const n_layers = bank.layers.len;
        if (opt.rows.len != n_layers) return error.InvalidRows;
        for (opt.rows) |r| if (r > bank.n_experts) return error.InvalidRows;
        if (opt.max_route_ids == 0 or opt.max_route_ids > max_route_ids or opt.transient_rows < opt.max_route_ids or
            opt.records_per_part == 0 or opt.records_per_part > expert_io.max_items) return error.InvalidOptions;
        // One transient row must hold any layer's record.
        var widest: usize = 0;
        for (bank.layers, 0..) |l, i| if (l.logical_bytes > bank.layers[widest].logical_bytes) {
            widest = i;
        };
        for (bank.layers) |l| for (l.segments, bank.layers[widest].segments) |s, w| {
            if (s.length > w.length) return error.MixedGeometry;
        };

        const self = try a.create(Stream);
        errdefer a.destroy(self);
        const layers = try a.alloc(LayerSlots, n_layers);
        errdefer a.free(layers);
        var n_init: usize = 0;
        errdefer for (layers[0..n_init]) |*ls| {
            ls.policy.deinit(a);
            ls.base.deinit();
            a.free(ls.meta);
        };
        for (layers, opt.rows, bank.layers) |*ls, rows, *geom| {
            var policy = try LayerPolicy.init(a, bank.n_experts, rows);
            errdefer policy.deinit(a);
            var base = try HostSlotRows.init(geom, rows);
            errdefer base.deinit();
            const meta = try a.alloc(SlotMeta, bank.n_experts);
            @memset(meta, .{});
            var lens: [n_components]u64 = undefined;
            for (&lens, geom.segments) |*l, s| l.* = s.length;
            ls.* = .{ .policy = policy, .base = base, .meta = meta, .lens = lens };
            n_init += 1;
        }
        var transient = try HostSlotRows.init(&bank.layers[widest], opt.transient_rows);
        errdefer transient.deinit();
        const transient_meta = try a.alloc(SlotMeta, opt.transient_rows);
        errdefer a.free(transient_meta);
        @memset(transient_meta, .{});
        const pool = try expert_io.Pool.start(a, opt.pool);
        self.* = .{
            .allocator = a,
            .bank = bank,
            .pool = pool,
            .layers = layers,
            .transient = transient,
            .transient_meta = transient_meta,
            .max_route_ids = opt.max_route_ids,
            .records_per_part = opt.records_per_part,
        };
        return self;
    }

    /// Stops the pool (draining its reads) before freeing the rows it writes.
    pub fn deinit(self: *Stream) void {
        const a = self.allocator;
        self.pool.stop();
        for (self.layers) |*ls| {
            ls.policy.deinit(a);
            ls.base.deinit();
            if (ls.ext) |*e| e.deinit();
            a.free(ls.meta);
        }
        a.free(self.layers);
        self.transient.deinit();
        a.free(self.transient_meta);
        a.destroy(self);
    }

    fn fail(self: *Stream, err: Error) Error {
        self.failed = true;
        return err;
    }

    fn locate(self: *Stream, layer: u32, slot: u32) Location {
        const ls = &self.layers[layer];
        if (slot < ls.base.rows) return .{ .rows = &ls.base, .row = slot, .meta = &ls.meta[slot] };
        if (slot < ls.policy.capacity) return .{ .rows = &ls.ext.?, .row = slot - ls.base.rows, .meta = &ls.meta[slot] };
        const t = slot - ls.policy.capacity;
        return .{ .rows = &self.transient, .row = t, .meta = &self.transient_meta[t] };
    }

    /// One component row of a layer's slot (for the kernels' binding and tests).
    pub fn slotRow(self: *Stream, layer: u32, slot: u32, c: Component) []u8 {
        const loc = self.locate(layer, slot);
        return loc.rows.row(c, loc.row);
    }

    /// prepare_prefill_seed: the prompt's routed ids of `layer`, before its
    /// prefill routes.
    pub fn seedPrefill(self: *Stream, layer: u32, ids: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        try self.layers[layer].policy.prepareSeed(self.allocator, ids);
    }

    /// Resolves `ids` (the router's top-k of one layer call, host values read
    /// from an evaluated array) to slots: pins every slot it serves and submits
    /// the misses' reads in parts. The eval that produced `ids` also finished
    /// every released route's consumers, so their slots are recycled first.
    pub fn route(self: *Stream, layer: u32, ids: []const u16) Error!*Route {
        if (self.failed) return error.StreamFailed;
        std.debug.assert(ids.len > 0 and ids.len <= self.max_route_ids);
        try self.flush();
        const r = for (&self.routes) |*cand| {
            if (cand.state == .free) break cand;
        } else return self.fail(error.RoutesExhausted);
        r.* = .{ .layer = layer };
        const ls = &self.layers[layer];
        ls.policy.plan(ids, self.phase, &r.plan);
        const plan = &r.plan;
        for (plan.hitsOf(), r.hit_slots[0..plan.n_hits]) |e, *s| {
            s.* = ls.policy.slotOf(e).?;
            self.locate(layer, s.*).meta.pins += 1;
        }
        for (plan.loadsOf(), 0..) |l, i| {
            const m = self.locate(layer, l.slot).meta;
            // A row a live route still serves from is never refilled.
            if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
            const held = m.state == .ready and m.layer == layer and m.expert == l.expert;
            r.reads[i] = !held;
            if (!held) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
            m.pins = 1;
        }
        try self.submitParts(r);
        const c = &self.counters;
        c.route_calls += 1;
        c.expert_cache_hits += plan.n_hits;
        c.expert_cache_misses += plan.n_misses;
        c.expert_cache_evictions += plan.n_evictions;
        for (plan.loadsOf(), r.reads[0..plan.n_loads]) |l, reads| {
            if (l.persistent) c.persistent_loads += 1 else c.transient_loads += 1;
            if (!reads) c.loads_skipped += 1;
        }
        r.state = .live;
        return r;
    }

    /// Loads in file order, cut into parts (decode: the bounded parts of
    /// `records_per_part`; prefill: pool-job chunks), one pool job per part.
    fn submitParts(self: *Stream, r: *Route) Error!void {
        const plan = &r.plan;
        const n = plan.n_loads;
        if (n == 0) return;
        const bank = self.bank;
        for (r.order[0..n], 0..) |*o, i| o.* = @intCast(i);
        std.sort.insertion(u8, r.order[0..n], @as(*const Route, r), struct {
            fn less(rr: *const Route, a: u8, b: u8) bool {
                return rr.plan.loads[a].expert < rr.plan.loads[b].expert;
            }
        }.less);
        var offsets: [max_route_ids]u64 = undefined;
        var lengths: [max_route_ids]u64 = undefined;
        const logical = bank.layers[r.layer].logical_bytes;
        for (r.order[0..n], 0..) |li, k| {
            offsets[k] = bank.recordOffset(r.layer, plan.loads[li].expert);
            lengths[k] = logical;
        }
        var ends_buf: [max_route_ids]u32 = undefined;
        const ends = if (plan.phase == .decode)
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
        const ls = &self.layers[r.layer];
        var start: u32 = 0;
        for (ends) |end| {
            var part: Part = .{ .first = start, .n = end - start };
            var rows: [expert_io.max_items][n_components]u64 = undefined;
            var gu: [expert_io.max_items]u64 = undefined;
            var down: [expert_io.max_items]u64 = undefined;
            var nr: u32 = 0;
            for (r.order[start..end]) |li| {
                if (!r.reads[li]) continue;
                const l = plan.loads[li];
                const loc = self.locate(r.layer, l.slot);
                rows[nr] = loc.rows.rowDest(loc.row);
                const sp = bank.spans(r.layer, l.expert);
                gu[nr] = sp.gu_offset;
                down[nr] = sp.down_offset;
                nr += 1;
            }
            if (nr > 0) {
                part.ticket = self.pool.submit(bank.sidecar_fd, bank.sidecar_file_size, gu[0..nr], down[0..nr], rows[0..nr], &ls.lens) catch |e| return self.fail(e);
                part.n_reads = nr;
            } else part.settled = true;
            r.parts[r.n_parts] = part;
            r.n_parts += 1;
            start = end;
        }
    }

    /// Blocks until the part's gate/up segments landed (its down segments may
    /// still be reading): the gate/up kernels of its records can run.
    pub fn waitGu(self: *Stream, r: *Route, part: u32) Error!void {
        const p = &r.parts[part];
        if (p.settled) return;
        self.pool.wait(p.ticket, p.n_reads, wait_timeout_ns) catch |e| return self.fail(e);
        for (0..p.n_reads) |i| {
            if (self.pool.result(p.ticket + @as(u32, @intCast(i))).status != .ok) return self.fail(error.ReadFailed);
        }
    }

    /// Blocks until every segment of the part landed; its rows are then ready.
    pub fn waitDown(self: *Stream, r: *Route, part: u32) Error!void {
        return self.settle(r, &r.parts[part]);
    }

    fn settle(self: *Stream, r: *Route, p: *Part) Error!void {
        if (p.settled) return;
        const count = 2 * p.n_reads;
        const waited = self.pool.wait(p.ticket, count, wait_timeout_ns);
        var ok = if (waited) |_| true else |_| false;
        if (ok) {
            for (0..count) |k| {
                const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                if (res.status != .ok) ok = false;
                self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                self.read_ns += @intCast(@max(res.t_end_ns - res.t_start_ns, 0));
            }
        }
        p.settled = true;
        const ls = &self.layers[r.layer];
        for (r.order[p.first..][0..p.n]) |li| {
            if (!r.reads[li]) continue;
            const l = r.plan.loads[li];
            self.locate(r.layer, l.slot).meta.state = if (ok) .ready else .failed;
            if (!ok and l.persistent) ls.policy.invalidate(l.expert);
        }
        if (!ok) return self.fail(if (waited) |_| error.ReadFailed else |e| e);
    }

    /// Hands a route back. Its slots stay pinned until the next flush: the
    /// kernels that read them finish only with a later eval.
    pub fn release(self: *Stream, r: *Route) void {
        _ = self;
        std.debug.assert(r.state == .live);
        r.state = .released;
    }

    /// Unpins every released route; call only after an eval that consumed
    /// them (`route` does, since its ids come from such an eval). A release
    /// whose reads are still landing waits for them first.
    pub fn flush(self: *Stream) Error!void {
        var first_error: ?Error = null;
        for (&self.routes) |*r| {
            if (r.state != .released) continue;
            for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                if (first_error == null) first_error = e;
            };
            for (r.hit_slots[0..r.plan.n_hits]) |s| self.locate(r.layer, s).meta.pins -= 1;
            for (r.plan.loadsOf()) |l| self.locate(r.layer, l.slot).meta.pins -= 1;
            r.state = .free;
        }
        if (first_error) |e| return e;
    }

    /// The one phase change: each layer's persistent rows become
    /// `decode_rows` (the added rows empty, residents unmoved); routes are
    /// decode routes from here on. Needs every route released.
    pub fn grow(self: *Stream, decode_rows: []const u32) !void {
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (self.failed) return error.StreamFailed;
        if (decode_rows.len != self.layers.len) return error.InvalidRows;
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
        for (self.layers, decode_rows) |*ls, rows| {
            if (rows < ls.policy.capacity or rows > ls.policy.n_experts) return error.InvalidRows;
        }
        const a = self.allocator;
        const exts = try a.alloc(?HostSlotRows, self.layers.len);
        defer a.free(exts);
        @memset(exts, null);
        errdefer for (exts) |*e| if (e.*) |*rows| rows.deinit();
        for (self.layers, decode_rows, exts, self.bank.layers) |*ls, rows, *e, *geom| {
            if (rows > ls.policy.capacity) e.* = try HostSlotRows.init(geom, rows - ls.policy.capacity);
        }
        for (self.layers, decode_rows, exts) |*ls, rows, e| {
            ls.ext = e;
            ls.policy.grow(rows) catch unreachable;
        }
        self.phase = .decode;
    }

    pub fn stats(self: *Stream) Stats {
        var s = self.counters;
        s.expert_read_seconds = @as(f64, @floatFromInt(self.read_ns)) / 1e9;
        s.read_wall_ns = @intCast(@max(self.pool.readGauge()[4], 0));
        return s;
    }

    /// Live pins on a layer's slot (tests).
    fn pinsOf(self: *Stream, layer: u32, slot: u32) u16 {
        return self.locate(layer, slot).meta.pins;
    }
};

// ── Tests ──

const testing = std.testing;

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

/// Only creating a Metal device maps a GPU driver bundle (AGXMetal*), so its
/// absence proves this process did no Metal work.
fn metalDriverLoaded() bool {
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

const FixSeg = struct { component: []const u8, offset: u64, length: u64, sha256: []const u8, head16: []const u8 };
const FixRec = struct {
    layer: u32,
    expert: u32,
    sidecar_offset: u64,
    record_bytes: u64,
    logical_bytes: u64,
    v2_sha256: []const u8,
    v1_sha256: []const u8,
    segments: []const FixSeg,
};
const Fixture = struct { layer_set: []const FixRec, pick_set: []const FixRec };

const Env = struct {
    bank: expert_bank.Bank,
    text: []u8,
    parsed: std.json.Parsed(Fixture),

    /// DSV41_BANK + DSV41_PHASE0_FIXTURE, else the test is skipped.
    fn open() !Env {
        const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
        const fixture = std.mem.span(std.c.getenv("DSV41_PHASE0_FIXTURE") orelse return error.SkipZigTest);
        var diag: expert_bank.Diag = .{};
        var bank = expert_bank.Bank.open(testing.allocator, std.testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        errdefer bank.deinit();
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, fixture, testing.allocator, .limited(4 << 20));
        errdefer testing.allocator.free(text);
        const parsed = try std.json.parseFromSlice(Fixture, testing.allocator, text, .{ .ignore_unknown_fields = true });
        return .{ .bank = bank, .text = text, .parsed = parsed };
    }

    fn close(self: *Env) void {
        self.parsed.deinit();
        testing.allocator.free(self.text);
        self.bank.deinit();
    }
};

fn hexEq(hex: []const u8, bytes: []const u8) bool {
    var buf: [64]u8 = undefined;
    if (hex.len != 2 * bytes.len or hex.len > 2 * buf.len) return false;
    const got = std.fmt.hexToBytes(&buf, hex) catch return false;
    return std.mem.eql(u8, got, bytes);
}

/// Reads `set` (one job, <= 8 records of one geometry) through the pool into
/// `rows`, then checks every record against a direct pread of the full record,
/// both manifest digests and the Python reader's per-component sha256.
fn checkSet(name: []const u8, bank: *const expert_bank.Bank, set: []const FixRec, rows: anytype) !u64 {
    const n = set.len;
    var refs: [expert_io.max_items]expert_io.RecordRef = undefined;
    var dests: [expert_io.max_items][n_components]u64 = undefined;
    for (set, 0..) |r, i| {
        try testing.expectEqual(bank.recordOffset(r.layer, r.expert), r.sidecar_offset);
        refs[i] = .{ .layer = r.layer, .expert = r.expert };
        dests[i] = rows.rowDest(@intCast(i));
    }
    // Stops (drains + joins) before the caller frees `rows`.
    var pool = try expert_io.Pool.start(testing.allocator, .{});
    defer pool.stop();
    const first = try expert_io.submitRecords(pool, bank, refs[0..n], dests[0..n]);
    try pool.wait(first, @intCast(2 * n), 60 * std.time.ns_per_s);
    var calls: i64 = 0;
    var payload: i64 = 0;
    var returned: i64 = 0;
    var t0: i64 = std.math.maxInt(i64);
    var t1: i64 = 0;
    for (0..2 * n) |t| {
        const res = pool.result(first + @as(u32, @intCast(t)));
        try testing.expectEqual(expert_io.Status.ok, res.status);
        calls += res.preadv_calls;
        payload += res.payload;
        returned += res.bytes_returned;
        t0 = @min(t0, res.t_start_ns);
        t1 = @max(t1, res.t_end_ns);
    }
    std.debug.print("{s}: {d} records, pool payload {d} B in {d} preadv ({d} B returned) over {d} us; verify pread {d} B\n", .{ name, n, payload, calls, returned, @divTrunc(t1 - t0, 1000), n * set[0].record_bytes });

    const Sha256 = std.crypto.hash.sha2.Sha256;
    const whole = try testing.allocator.alloc(u8, @intCast(set[0].record_bytes));
    defer testing.allocator.free(whole);
    for (set, 0..) |r, i| {
        const layer = &bank.layers[r.layer];
        try testing.expectEqual(layer.record_bytes, r.record_bytes);
        var got: usize = 0;
        while (got < whole.len) {
            const k = std.c.pread(bank.sidecar_fd, whole[got..].ptr, whole.len - got, @intCast(r.sidecar_offset + got));
            if (k <= 0) return error.ShortRead;
            got += @intCast(k);
        }
        var d: [32]u8 = undefined;
        Sha256.hash(whole, &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).padded, &d);
        try testing.expect(hexEq(r.v2_sha256, &d));
        Sha256.hash(whole[0..@intCast(layer.logical_bytes)], &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).logical, &d);
        try testing.expect(hexEq(r.v1_sha256, &d));
        try testing.expectEqual(@as(usize, n_components), r.segments.len);
        for (layer.segments, r.segments, 0..) |seg, fs, c| {
            const comp: Component = @enumFromInt(c);
            try testing.expectEqualStrings(comp.name(), fs.component);
            try testing.expectEqual(r.sidecar_offset + seg.offset, fs.offset);
            try testing.expectEqual(seg.length, fs.length);
            const slot = rows.row(comp, @intCast(i))[0..@intCast(seg.length)];
            try testing.expectEqualSlices(u8, whole[@intCast(seg.offset)..][0..@intCast(seg.length)], slot);
            Sha256.hash(slot, &d, .{});
            try testing.expect(hexEq(fs.sha256, &d));
            try testing.expect(hexEq(fs.head16, slot[0..16]));
        }
    }
    return @intCast(calls);
}

test "dsv41 slots: layer 13, 8 rows through the pool == pread == v2 sha == Python fixture" {
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.layer_set;
    try testing.expectEqual(@as(usize, 8), set.len);
    var rows = try HostSlotRows.init(&env.bank.layers[set[0].layer], @intCast(set.len));
    defer rows.deinit();
    const calls = try checkSet("layer_set", &env.bank, set, &rows);
    try testing.expect(calls >= 2 * set.len);
    try testing.expect(!metalDriverLoaded());
}

test "dsv41 slots: cross-layer PICK set" {
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.pick_set;
    try testing.expectEqual(@as(usize, 8), set.len);
    var rows = try HostSlotRows.init(&env.bank.layers[set[0].layer], @intCast(set.len));
    defer rows.deinit();
    _ = try checkSet("pick_set", &env.bank, set, &rows);
    try testing.expect(!metalDriverLoaded());
}

// Phase 0b, inside a guarded window (GPU lock held): DSV41_PHASE0B_MLX=1.
test "dsv41 slots 0b: an MLX LayerSlotBank on the CPU stream fills like the host rows" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.layer_set;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var slots = try LayerSlotBank.init(&env.bank.layers[set[0].layer], @intCast(set.len), stream);
    defer slots.deinit();
    try testing.expectEqual(mlx.mlx_dtype.int16, mlx.mlx_array_dtype(slots.arrays[0]));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(slots.arrays[1]));
    try testing.expectEqual(@as(usize, 4), mlx.mlx_array_ndim(slots.arrays[0]));
    try testing.expectEqualSlices(c_int, &.{ @intCast(set.len), 320, 144, 48 }, mlx.mlx_array_shape(slots.arrays[0])[0..4]);
    _ = try checkSet("layer_set (MLX LayerSlotBank)", &env.bank, set, &slots);
}

test "dsv41 slots: no Metal device in this process" {
    if (std.c.getenv("DSV41_PHASE0B_MLX") != null) return error.SkipZigTest;
    try testing.expect(!metalDriverLoaded());
}

/// A 2-layer synthetic bank on disk (hidden 64, inter 32: 2,880-byte records
/// in 4 KiB slots), opened, with its experts.bin image.
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

const test_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 };

/// Routes `ids` and waits for every part (gate/up first, as the kernels do).
fn serve(s: *Stream, layer: u32, ids: []const u16) !*Route {
    const r = try s.route(layer, ids);
    for (0..r.n_parts) |p| {
        try s.waitGu(r, @intCast(p));
        try s.waitDown(r, @intCast(p));
    }
    return r;
}

/// Every routed id's slot holds its record's bytes.
fn expectServed(s: *Stream, sb: *const SynthBank, r: *const Route, ids: []const u16) !void {
    const geom = &sb.bank.layers[r.layer];
    for (ids, r.plan.slotsOf()) |e, slot| {
        const off = sb.bank.recordOffset(r.layer, e);
        for (geom.segments, 0..) |seg, c| {
            const want = sb.image[off + seg.offset ..][0..seg.length];
            try testing.expectEqualSlices(u8, want, s.slotRow(r.layer, slot, @enumFromInt(c))[0..seg.length]);
        }
    }
}

fn readsOf(r: *const Route) u64 {
    var n: u64 = 0;
    for (r.partsOf()) |p| n += p.n_reads;
    return n;
}

test "dsv41 stream: every routed id is served from a slot holding its record" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var unique: u64 = 0;
    var reads: u64 = 0;
    // Prefill: layer 0 seeded, one wave per layer.
    try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
    const waves = [_]struct { layer: u32, ids: []const u16 }{
        .{ .layer = 0, .ids = &.{ 1, 2, 3, 5, 9 } },
        .{ .layer = 1, .ids = &.{ 7, 8, 9 } },
    };
    for (waves) |w| {
        const r = try serve(s, w.layer, w.ids);
        try expectServed(s, &sb, r, w.ids);
        unique += r.plan.n_hits + r.plan.n_misses;
        reads += readsOf(r);
        s.release(r);
    }
    try s.grow(&.{ 6, 4 });
    // Decode: a deterministic pseudo-random trace over both layers.
    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();
    var ids: [12]u16 = undefined;
    for (0..80) |step| {
        const layer: u32 = @intCast(step % 2);
        const n = rand.intRangeAtMost(usize, 1, 12);
        const span: u16 = if (step % 5 == 0) 32 else 10;
        for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, span);
        const r = try serve(s, layer, ids[0..n]);
        try expectServed(s, &sb, r, ids[0..n]);
        // Parts: at most three records each, in file order.
        var prev: ?u16 = null;
        for (r.partsOf()) |p| {
            try testing.expect(p.n >= 1 and p.n <= 3);
            for (r.order[p.first..][0..p.n]) |li| {
                const e = r.plan.loads[li].expert;
                if (prev) |pe| try testing.expect(pe < e);
                prev = e;
            }
        }
        unique += r.plan.n_hits + r.plan.n_misses;
        reads += readsOf(r);
        s.release(r);
    }
    try s.flush();
    for (0..2) |l| for (0..s.layers[l].policy.capacity + 12) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(@intCast(l), @intCast(slot)));
    };
    const st = s.stats();
    try testing.expectEqual(@as(u64, 82), st.route_calls);
    try testing.expectEqual(unique, st.expert_cache_hits + st.expert_cache_misses);
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual(st.expert_cache_misses, reads + st.loads_skipped);
    try testing.expectEqual(reads * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    try testing.expect(st.expert_cache_hits > 0 and st.expert_cache_evictions > 0 and st.transient_loads > 0);
    try testing.expect(st.preadv_calls >= 2 * reads);
}

test "dsv41 stream: a released route keeps its slots pinned until the next route" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    // 1, 2 fill layer 0's two slots; 3 is served from transient row 0 (slot 2).
    const r = try serve(s, 0, &.{ 1, 2, 3 });
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, r.plan.slotsOf());
    s.release(r);
    try testing.expectEqual(@as(u16, 1), s.pinsOf(0, 0));
    try testing.expectEqual(@as(u16, 1), s.pinsOf(0, 2));
    // The next route flushes first: layer 1 then reuses transient row 0.
    const r2 = try serve(s, 1, &.{4});
    try testing.expectEqual(@as(u16, 0), s.pinsOf(0, 0));
    try testing.expectEqual(@as(u16, 0), s.pinsOf(0, 1));
    try testing.expectEqualSlices(u32, &.{0}, r2.plan.slotsOf());
    try testing.expectEqual(@as(u16, 1), s.pinsOf(1, 0));
    try expectServed(s, &sb, r2, &.{4});
    s.release(r2);
}

test "dsv41 stream: a row an unreleased route serves from is never refilled" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 0, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    _ = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.SlotStillPinned, s.route(1, &.{3}));
    try testing.expectError(error.StreamFailed, s.route(1, &.{3}));
}

test "dsv41 stream: routes the caller never releases run out, by name" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    for (0..4) |_| _ = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesExhausted, s.route(0, &.{ 1, 2 }));
}

test "dsv41 stream: a transient row still holding the record is not read again" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 0, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const steps = [_]struct { layer: u32, ids: []const u16, reads: u64 }{
        .{ .layer = 0, .ids = &.{ 5, 6 }, .reads = 2 },
        .{ .layer = 0, .ids = &.{ 5, 6 }, .reads = 0 },
        .{ .layer = 1, .ids = &.{5}, .reads = 1 },
        .{ .layer = 0, .ids = &.{5}, .reads = 1 },
    };
    for (steps) |st| {
        const r = try serve(s, st.layer, st.ids);
        try testing.expectEqual(st.reads, readsOf(r));
        try expectServed(s, &sb, r, st.ids);
        s.release(r);
    }
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2), st.loads_skipped);
    try testing.expectEqual(4 * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
}

test "dsv41 stream: growth is the one phase change" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var r = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesLive, s.grow(&.{ 4, 4 }));
    s.release(r);
    try testing.expectError(error.InvalidRows, s.grow(&.{ 1, 4 }));
    try testing.expectError(error.InvalidRows, s.grow(&.{4}));
    try s.grow(&.{ 4, 3 });
    try testing.expectError(error.AlreadyGrown, s.grow(&.{ 4, 4 }));
    try testing.expectError(error.NotPrefill, s.seedPrefill(0, &.{1}));
    // Residents keep their slots and bytes; the added rows fill before any eviction.
    r = try serve(s, 0, &.{ 1, 2, 3, 4 });
    try testing.expectEqual(@as(u32, 2), r.plan.n_hits);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, r.plan.slotsOf());
    try testing.expectEqual(@as(u32, 0), r.plan.n_evictions);
    try expectServed(s, &sb, r, &.{ 1, 2, 3, 4 });
    s.release(r);
}

test "dsv41 stream: a failed read fails the route and every later one" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    defer expert_io.clearFaults();
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 1).gu_offset / page * page, 2, 0);
    const r = try s.route(0, &.{1});
    try testing.expectError(error.ReadFailed, s.waitDown(r, 0));
    try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(1));
    try testing.expectError(error.StreamFailed, s.route(0, &.{1}));
}

/// sha256 of a served slot's logical record (its nine component rows).
fn slotDigest(s: *Stream, layer: u32, slot: u32, geom: *const Layer) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (geom.segments, 0..) |seg, c| h.update(s.slotRow(layer, slot, @enumFromInt(c))[0..seg.length]);
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

// DSV41_BANK=<bank dir> DSV41_PHASE1_ROUTE_FIXTURE=<json from R/exl3/runtime/dump_phase1_route_fixture.py>
test "dsv41 stream: a recorded trace on the real bank serves every slot's bytes" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE1_ROUTE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(struct { bank_trace: expert_policy.BankTrace }, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const bt = parsed.value.bank_trace;
    const L = bt.layer;
    var rows: [40]u32 = @splat(0);
    rows[L] = bt.prefill_rows;
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = bt.transient, .transient_rows = bt.transient });
    defer s.deinit();
    const geom = &bank.layers[L];
    var served: u64 = 0;
    try s.seedPrefill(L, bt.seed);
    for ([_][]const expert_policy.FixPlan{ bt.prefill, bt.routes }, 0..) |plans, phase| {
        if (phase == 1) {
            rows[L] = bt.decode_rows;
            try s.grow(&rows);
        }
        for (plans) |want| {
            const r = try serve(s, L, want.ids);
            try expert_policy.expectPlan(&r.plan, want);
            // Every served expert's slot holds its record: sha256 == the runtime manifest's.
            for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |e, slot| {
                const d = slotDigest(s, L, slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, e).logical, &d);
                served += 1;
            }
            for (r.plan.loadsOf()) |l| {
                const d = slotDigest(s, L, l.slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, l.expert).logical, &d);
                served += 1;
            }
            s.release(r);
        }
    }
    try s.flush();
    const st = s.stats();
    const ru = std.posix.getrusage(std.c.rusage.SELF);
    std.debug.print(
        "real bank layer {d}: {d} routes, {d} served slots sha256-checked; hits {d} misses {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv ({d:.3} s read, {d} ms wall); {d} ms total; peak RSS {d} B\n",
        .{ L, st.route_calls, served, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions, st.persistent_loads, st.transient_loads, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.expert_read_seconds, @divTrunc(st.read_wall_ns, std.time.ns_per_ms), @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
}
