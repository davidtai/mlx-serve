//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
const mlx = @import("mlx.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("expert_io.zig");
const expert_policy = @import("expert_policy.zig");
const expert_lookahead = @import("expert_lookahead.zig");

const n_components = expert_bank.n_components;
const gu_components = expert_bank.gu_components;
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

/// Where the stream keeps its slot rows: host pages (the default; hermetic
/// tests and CPU checks) or MLX arrays the kernels bind, created and evaluated
/// on `mlx` (creating any MLX array creates the Metal device: callers hold the
/// GPU lock). Chosen once, at Stream.init.
pub const SlotMemory = union(enum) { host, mlx: mlx.mlx_stream };

/// One bank of slot rows. Both memories are addressed by the same row
/// arithmetic, so the read path never asks which one it has.
const Rows = struct {
    rows: u32 = 0,
    base: [n_components]u64 = @splat(0),
    row_bytes: [n_components]u64 = @splat(0),
    backing: union(enum) { none, host: HostSlotRows, mlx: LayerSlotBank } = .none,

    fn init(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
        if (rows == 0) return .{};
        switch (memory) {
            .host => {
                const h = try HostSlotRows.init(layer, rows);
                var r: Rows = .{ .rows = rows, .row_bytes = h.row_bytes, .backing = .{ .host = h } };
                for (&r.base, h.banks) |*b, bank| b.* = @intFromPtr(bank.ptr);
                return r;
            },
            .mlx => |stream| {
                const m = try LayerSlotBank.init(layer, rows, stream);
                return .{ .rows = rows, .base = m.base, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
            },
        }
    }

    /// After the pool that wrote into the rows has stopped.
    fn deinit(self: *Rows) void {
        switch (self.backing) {
            .none => {},
            .host => |*h| h.deinit(),
            .mlx => |*m| m.deinit(),
        }
        self.* = .{};
    }

    fn row(self: *const Rows, c: Component, r: u32) []u8 {
        const n = self.row_bytes[@intFromEnum(c)];
        const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
        return p[0..n];
    }

    fn rowDest(self: *const Rows, r: u32) [n_components]u64 {
        var d: [n_components]u64 = undefined;
        for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
        return d;
    }
};

/// Which of a layer's banks holds a slot: its prefill rows, the rows `grow`
/// added, or the transient scratch every layer shares.
pub const BankKind = enum(u8) { base, ext, transient };
/// A slot's bank and its row in that bank (the index the kernels gather).
pub const SlotRef = struct { bank: BankKind, row: u32 };
/// One projection's arrays in a bank: code int16 [rows, in/16, out/16, 16K],
/// rout f16 [rows, out], rin f16 [rows, in].
pub const ProjArrays = struct { code: mlx.mlx_array, rout: mlx.mlx_array, rin: mlx.mlx_array };
/// A bank's nine arrays by projection; the stream owns them for its life.
pub const BankArrays = struct { gate: ProjArrays, up: ProjArrays, down: ProjArrays };

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
    /// Decode routes read the next layer's predicted records ahead and
    /// pre-read their own certain misses (the lookahead4 lane).
    lookahead: ?Lookahead = null,
    /// Gate each call's reads on an event the GPU waits for (needs `lookahead`).
    event: ?Event = null,
    /// Host pages, or MLX arrays on the given stream (the serving form).
    slot_memory: SlotMemory = .host,
    /// Prefill routes live at once in one layer (the wide lane's read-ahead:
    /// 2 = the next group's reads issued before this group's waves). Each
    /// live route owns a window of `max_route_ids` transient rows, so the
    /// transient rows must hold `wide_depth` windows.
    wide_depth: u8 = 1,
};

/// DSV41_LOOKAHEAD4=<k>:<tau>:<budget>:<chunks> at horizon 1.
pub const Lookahead = struct {
    k: u32 = 8,
    /// inf keeps each row's plain top-K.
    tau: f32 = std.math.inf(f32),
    /// Records read ahead per layer call; 2 x budget staging slots.
    budget: u32 = 2,
    /// Chunks per speculative record: 1, 2, 4 or 8.
    chunks: u32 = 4,
    /// Speculative threads >= 1 start a chunk only while at most this many
    /// demand jobs run (0 = demand idle).
    idle_busy: u32 = 0,
    preread: bool = true,
};

pub const Event = struct {
    /// .host: an int64 word the stream owns (CPU backend, checks); .metal: an
    /// id<MTLSharedEvent> on MLX's device whose signaled value is 0.
    backend: union(enum) { host, metal: u64 } = .host,
    /// A gate whose bytes have not landed by then is forced (the stream fails).
    watchdog_ms: u32 = 2000,
};

/// The event values a gated call's waves wait for: its gate/up wave `gu`, the
/// down wave of part p `down_first + p`.
pub const Gates = struct { gu: u64, down_first: u64, n_parts: u32 };

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
    /// The lookahead class: records claimed by a demand read, physical bytes
    /// of speculative reads, records issued / fully landed, demand ranges
    /// copied out of a speculative record (no preadv) and their bytes.
    claimed: u64 = 0,
    spec_bytes: u64 = 0,
    spec_issued: u64 = 0,
    spec_landed: u64 = 0,
    adopt_ranges: u64 = 0,
    adopt_bytes: u64 = 0,
    /// Pre-read ranges queued, served to a demand read, dropped unbound.
    pre_issued: u64 = 0,
    pre_served: u64 = 0,
    pre_expired: u64 = 0,
    /// Event gates registered and forced by the watchdog.
    gates: u64 = 0,
    gates_forced: u64 = 0,
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
    SpecRefused,
    PreReadRefused,
    GateInvalid,
    GatesFull,
    GateRefused,
    /// The watchdog released a gate before its bytes landed: the GPU may have
    /// read them early, so the outputs since are invalid.
    GateForced,
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
    /// Its transient window (prefill routes beside a live one: `Options.wide_depth`).
    window: u8 = 0,
    /// Decode layer calls with the lookahead class: this call's settle value.
    tag: i64 = 0,
    gates: ?Gates = null,

    pub fn partsOf(r: *const Route) []const Part {
        return r.parts[0..r.n_parts];
    }

    /// The loads (expert, slot) of part `part`, in file order: what its
    /// gate/up and down kernels read once `waitGu` / `waitDown` return.
    pub fn partLoads(r: *const Route, part: u32, out: *[max_route_ids]expert_policy.Load) []expert_policy.Load {
        const p = r.parts[part];
        for (r.order[p.first..][0..p.n], out[0..p.n]) |li, *l| l.* = r.plan.loads[li];
        return out[0..p.n];
    }
};

/// The route being served plus released ones awaiting the next flush.
const route_capacity = 4;
/// Prefill routes one layer may hold live at once (`Options.wide_depth`).
pub const max_wide_depth = 2;
const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;

/// Expert residency for one model: per-layer persistent slot pools at the
/// caller's row bound, a transient scratch shared by all layers, and the read
/// pool. One inference thread calls it; pool workers only write slot bytes.
pub const Stream = struct {
    allocator: std.mem.Allocator,
    bank: *const expert_bank.Bank,
    pool: *expert_io.Pool,
    layers: []LayerSlots,
    transient: Rows,
    transient_meta: []SlotMeta,
    memory: SlotMemory = .host,
    max_route_ids: u32,
    records_per_part: u32,
    wide_depth: u8 = 1,
    phase: Phase = .prefill,
    failed: bool = false,
    routes: [route_capacity]Route = @splat(.{}),
    /// Free routes (a stack) and released ones awaiting the next flush, in
    /// release order: `route` and `flush` touch only these, never the ring.
    free: [route_capacity]u8 = blk: {
        var f: [route_capacity]u8 = undefined;
        for (&f, 0..) |*x, i| x.* = route_capacity - 1 - i;
        break :blk f;
    },
    n_free: u8 = route_capacity,
    released: [route_capacity]u8 = undefined,
    n_released: u8 = 0,
    /// The phase's route: the lookahead class (decode with a selector) and
    /// its pre-reads, set at construction and at the phase change.
    route_lookahead: bool = false,
    route_preread: bool = false,
    counters: Stats = .{},
    read_ns: u64 = 0,
    selector: ?expert_lookahead.Selector = null,
    preread: bool = false,
    /// Decode layer calls so far; a call's pre-reads and speculative records
    /// carry its tag, settled by its own step.
    clock: i64 = 0,
    event_word: ?*i64 = null,
    gated: bool = false,
    /// The last event value handed out.
    gate_value: u64 = 0,
    forced_seen: i64 = 0,
    /// The thread that built the stream (mlx-serve: the inference thread, the
    /// only MLX caller); `grow` allocates slot memory and refuses any other.
    owner: std.Thread.Id,

    const LayerSlots = struct {
        policy: LayerPolicy,
        /// The prefill rows, then the rows `grow` added.
        base: Rows,
        ext: ?Rows = null,
        /// [n_experts]: one entry per persistent slot.
        meta: []SlotMeta,
        lens: [n_components]u64,
    };

    const Location = struct { rows: *const Rows, row: u32, meta: *SlotMeta };

    pub fn init(a: std.mem.Allocator, bank: *const expert_bank.Bank, opt: Options) !*Stream {
        const n_layers = bank.layers.len;
        if (opt.rows.len != n_layers) return error.InvalidRows;
        for (opt.rows) |r| if (r > bank.n_experts) return error.InvalidRows;
        if (opt.wide_depth < 1 or opt.wide_depth > max_wide_depth or opt.transient_rows < @as(u32, opt.wide_depth) * opt.max_route_ids) return error.InvalidOptions;
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
        if (opt.event != null and opt.lookahead == null) return error.InvalidOptions;
        if (opt.event) |ev| if (ev.watchdog_ms < 50 or ev.watchdog_ms > 60_000) return error.InvalidOptions;
        var pool_opt = opt.pool;
        if (opt.lookahead) |la| {
            if (la.chunks == 0 or la.chunks > 8 or !std.math.isPowerOfTwo(la.chunks) or la.idle_busy > 1 or
                la.budget == 0 or la.budget > expert_lookahead.max_budget) return error.InvalidOptions;
            const page = std.heap.pageSize();
            const record = bank.layers[widest].logical_bytes;
            pool_opt.spec = .{ .threads = @min(la.budget, 2), .slots = 2 * la.budget, .record_bytes = record, .chunk_bytes = expert_io.chunkBytes(la.chunks, record, page), .idle_busy = la.idle_busy };
            // A pre-read range is the record's gate/up span or its down span,
            // back to back, each no larger than a staging buffer.
            if (la.preread) for (bank.layers) |l| {
                var gu: u64 = 0;
                for (l.segments[0..gu_components]) |sg| gu += sg.length;
                if (l.segments[0].offset != 0 or l.segments[gu_components].offset != gu) return error.MixedGeometry;
                if (gu > pool_opt.staging_bytes or l.logical_bytes - gu > pool_opt.staging_bytes) return error.InvalidOptions;
            };
        }
        var selector: ?expert_lookahead.Selector = null;
        if (opt.lookahead) |la| selector = try expert_lookahead.Selector.init(a, bank.n_experts, la.k, la.tau, la.budget);
        errdefer if (selector) |*sel| sel.deinit(a);
        // The pool writes the event word until it stops.
        var word: ?*i64 = null;
        if (opt.event) |ev| if (ev.backend == .host) {
            word = try a.create(i64);
            word.?.* = 0;
        };
        errdefer if (word) |w| a.destroy(w);

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
            var base = try Rows.init(geom, rows, opt.slot_memory);
            errdefer base.deinit();
            const meta = try a.alloc(SlotMeta, bank.n_experts);
            @memset(meta, .{});
            var lens: [n_components]u64 = undefined;
            for (&lens, geom.segments) |*l, s| l.* = s.length;
            ls.* = .{ .policy = policy, .base = base, .meta = meta, .lens = lens };
            n_init += 1;
        }
        var transient = try Rows.init(&bank.layers[widest], opt.transient_rows, opt.slot_memory);
        errdefer transient.deinit();
        const transient_meta = try a.alloc(SlotMeta, opt.transient_rows);
        errdefer a.free(transient_meta);
        @memset(transient_meta, .{});
        const pool = try expert_io.Pool.start(a, pool_opt);
        errdefer pool.stop();
        if (opt.lookahead) |la| if (la.preread) try pool.armPreRead(&layers[widest].lens);
        if (opt.event) |ev| {
            const object: u64 = switch (ev.backend) {
                .host => @intFromPtr(word.?),
                .metal => |ptr| ptr,
            };
            try pool.armEvent(if (ev.backend == .host) .host else .metal, object, @as(i64, ev.watchdog_ms) * std.time.ns_per_ms, 0);
        }
        self.* = .{
            .allocator = a,
            .bank = bank,
            .pool = pool,
            .layers = layers,
            .transient = transient,
            .transient_meta = transient_meta,
            .memory = opt.slot_memory,
            .max_route_ids = opt.max_route_ids,
            .records_per_part = opt.records_per_part,
            .wide_depth = opt.wide_depth,
            .selector = selector,
            .preread = if (opt.lookahead) |la| la.preread else false,
            .event_word = word,
            .gated = opt.event != null,
            .owner = std.Thread.getCurrentId(),
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
        if (self.selector) |*sel| sel.deinit(a);
        if (self.event_word) |w| a.destroy(w);
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

    /// The bank holding a layer's slot and the slot's row in it.
    pub fn slotRef(self: *const Stream, layer: u32, slot: u32) SlotRef {
        const ls = &self.layers[layer];
        if (slot < ls.base.rows) return .{ .bank = .base, .row = slot };
        if (slot < ls.policy.capacity) return .{ .bank = .ext, .row = slot - ls.base.rows };
        return .{ .bank = .transient, .row = slot - ls.policy.capacity };
    }

    /// Per routed id of `r` (plan order), its slot's bank and row.
    pub fn refsOf(self: *const Stream, r: *const Route, out: *[max_route_ids]SlotRef) []SlotRef {
        for (r.plan.slotsOf(), out[0..r.plan.n_ids]) |slot, *ref| ref.* = self.slotRef(r.layer, slot);
        return out[0..r.plan.n_ids];
    }

    /// A layer's bank as the kernels bind it (MLX slot memory; null for host
    /// rows or a bank without rows). The stream owns the arrays: never free them.
    pub fn bankArrays(self: *const Stream, layer: u32, kind: BankKind) ?BankArrays {
        const rows: *const Rows = switch (kind) {
            .base => &self.layers[layer].base,
            .ext => if (self.layers[layer].ext) |*e| e else return null,
            .transient => &self.transient,
        };
        const m = switch (rows.backing) {
            .mlx => |*m| m,
            else => return null,
        };
        const x = m.arrays;
        return .{ .gate = .{ .code = x[0], .rout = x[1], .rin = x[2] }, .up = .{ .code = x[3], .rout = x[4], .rin = x[5] }, .down = .{ .code = x[6], .rout = x[7], .rin = x[8] } };
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
    /// With the lookahead class a decode call first pre-reads its certain
    /// misses, and after its submits (which claim the records read ahead for
    /// it) settles what it did not claim and reads ahead the next layer's
    /// predicted records: `scores` = that layer's gate on this call's rows
    /// (rows x n_experts f32, evaluated with `ids`); empty = settle only. The
    /// read-ahead is the decode phase's: before the phase change (and on a
    /// stream without the lookahead class) the scores are not read.
    pub fn route(self: *Stream, layer: u32, ids: []const u16, scores: []const f32) Error!*Route {
        if (self.failed) return error.StreamFailed;
        std.debug.assert(ids.len > 0 and ids.len <= self.max_route_ids);
        const lookahead = self.route_lookahead;
        const tag = self.clock + 1;
        if (self.route_preread) try self.preRead(layer, ids, tag);
        try self.flush();
        if (self.n_free == 0) return self.fail(error.RoutesExhausted);
        self.n_free -= 1;
        const r = &self.routes[self.free[self.n_free]];
        // A prefill route beside live ones of its layer: their slots are held,
        // its transient loads take the first window no route holds.
        var held_buf: [route_capacity * 2 * max_route_ids]u32 = undefined;
        var n_held: usize = 0;
        var used_windows: u8 = 0;
        if (self.wide_depth > 1) for (&self.routes) |*o| {
            if (o.state == .free) continue;
            used_windows |= @as(u8, 1) << @intCast(o.window);
            if (o.layer != layer) continue;
            for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < self.layers[layer].policy.capacity) {
                held_buf[n_held] = hs;
                n_held += 1;
            };
            for (o.plan.loadsOf()) |l| if (l.persistent) {
                held_buf[n_held] = l.slot;
                n_held += 1;
            };
        };
        const window: u8 = @intCast(@ctz(~used_windows));
        if (window >= self.wide_depth) return self.fail(error.RoutesExhausted);
        r.* = .{ .layer = layer, .window = window };
        const ls = &self.layers[layer];
        ls.policy.planWith(ids, self.phase, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held_buf[0..n_held] });
        const plan = &r.plan;
        for (plan.hitsOf(), r.hit_slots[0..plan.n_hits]) |e, *s| {
            s.* = ls.policy.slotOf(e).?;
            self.locate(layer, s.*).meta.pins += 1;
        }
        var skipped: u64 = 0;
        for (plan.loadsOf(), 0..) |l, i| {
            const m = self.locate(layer, l.slot).meta;
            // A row a live route still serves from is never refilled.
            if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
            const held = m.state == .ready and m.layer == layer and m.expert == l.expert;
            r.reads[i] = !held;
            skipped += @intFromBool(held);
            if (!held) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
            m.pins = 1;
        }
        try self.submitParts(r);
        if (lookahead) {
            r.tag = tag;
            self.clock = tag;
            try self.speculate(r, scores);
        }
        const c = &self.counters;
        c.route_calls += 1;
        c.expert_cache_hits += plan.n_hits;
        c.expert_cache_misses += plan.n_misses;
        c.expert_cache_evictions += plan.n_evictions;
        c.persistent_loads += plan.n_persistent;
        c.transient_loads += plan.n_loads - plan.n_persistent;
        c.loads_skipped += skipped;
        r.state = .live;
        return r;
    }

    /// The call's certain misses (its plan's misses) as pre-read ranges,
    /// before the plan; the submit binds them.
    fn preRead(self: *Stream, layer: u32, ids: []const u16, tag: i64) Error!void {
        var experts: [expert_lookahead.max_candidates]u16 = undefined;
        const misses = self.selector.?.certainMisses(ids, &self.layers[layer].policy, &experts);
        if (misses.len == 0) return;
        var bases: [expert_lookahead.max_candidates]i64 = undefined;
        for (misses, bases[0..misses.len]) |e, *b| b.* = @intCast(self.bank.recordOffset(layer, e));
        _ = self.pool.preRead(self.bank.sidecar_fd, self.bank.sidecar_file_size, tag, bases[0..misses.len], &self.layers[layer].lens) catch
            return self.fail(error.PreReadRefused);
    }

    /// Settles every record read ahead for this call that it did not claim,
    /// then reads ahead the next layer's predicted records (keyed by offset).
    fn speculate(self: *Stream, r: *const Route, scores: []const f32) Error!void {
        const next = r.layer + 1;
        var bases: [expert_lookahead.max_budget]i64 = undefined;
        var n: usize = 0;
        var len: u64 = 0;
        if (scores.len > 0 and next < self.layers.len) {
            const sel = &self.selector.?;
            var chosen: [expert_lookahead.max_budget]u16 = undefined;
            for (sel.select(scores, &self.layers[next].policy, chosen[0..sel.budget])) |e| {
                bases[n] = @intCast(self.bank.recordOffset(next, e));
                n += 1;
            }
            len = self.bank.layers[next].logical_bytes;
        }
        _ = self.pool.specStep(self.bank.sidecar_fd, self.bank.sidecar_file_size, r.tag, bases[0..n], len) catch
            return self.fail(error.SpecRefused);
    }

    /// Event gates for a live route's reads, registered before the GPU commits
    /// its waves: the gate/up wave waits for `gu` (every gate/up ticket of the
    /// call), part p's down wave for `down_first + p` (that part's down
    /// tickets). Null when nothing is read.
    pub fn gate(self: *Stream, r: *Route) Error!?Gates {
        std.debug.assert(self.gated and r.state == .live and r.gates == null);
        const n = r.n_parts;
        if (n == 0) return null;
        const lo = self.gate_value;
        const hi = lo + 1 + n;
        self.gate_value = hi;
        var values: [max_route_ids + 1]u64 = undefined;
        var counts: [max_route_ids + 1]i32 = undefined;
        var tickets: [2 * max_route_ids]i64 = undefined;
        var k: usize = 0;
        for (r.partsOf()) |p| for (0..p.n_reads) |i| {
            tickets[k] = @intCast(p.ticket + i);
            k += 1;
        };
        values[0] = lo + 1;
        counts[0] = @intCast(k);
        for (r.partsOf(), 0..) |p, pi| {
            values[1 + pi] = lo + 2 + pi;
            counts[1 + pi] = @intCast(p.n_reads);
            for (0..p.n_reads) |i| {
                tickets[k] = @intCast(p.ticket + p.n_reads + i);
                k += 1;
            }
        }
        self.pool.registerGates(values[0 .. n + 1], counts[0 .. n + 1], tickets[0..k]) catch |e| {
            self.pool.releaseGates(hi);
            return self.fail(switch (e) {
                error.GateInvalid => error.GateInvalid,
                error.GatesFull => error.GatesFull,
                else => error.GateRefused,
            });
        };
        r.gates = .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = n };
        return r.gates;
    }

    /// The host event word (Event.backend = .host), for a CPU-stream wait.
    pub fn eventWord(self: *const Stream) ?*const i64 {
        return self.event_word;
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
        std.debug.assert(r.state == .live);
        r.state = .released;
        self.released[self.n_released] = @intCast((@intFromPtr(r) - @intFromPtr(&self.routes[0])) / @sizeOf(Route));
        self.n_released += 1;
    }

    /// Unpins every released route; call only after an eval that consumed
    /// them (`route` does, since its ids come from such an eval). A release
    /// whose reads are still landing waits for them first. A gate the
    /// watchdog forced since the last flush fails the stream here.
    pub fn flush(self: *Stream) Error!void {
        var first_error: ?Error = null;
        for (self.released[0..self.n_released]) |ri| {
            const r = &self.routes[ri];
            for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                if (first_error == null) first_error = e;
            };
            for (r.hit_slots[0..r.plan.n_hits]) |s| self.locate(r.layer, s).meta.pins -= 1;
            for (r.plan.loadsOf()) |l| self.locate(r.layer, l.slot).meta.pins -= 1;
            r.state = .free;
            self.free[self.n_free] = ri;
            self.n_free += 1;
        }
        self.n_released = 0;
        if (first_error) |e| return e;
        // The watchdog's forced gates (a count that moves only on a gated stream).
        const forced = self.pool.counter(.ev_wd_forced);
        if (forced != self.forced_seen) {
            self.forced_seen = forced;
            return self.fail(error.GateForced);
        }
    }

    /// The one phase change: each layer's persistent rows become
    /// `decode_rows` (the added rows empty, residents unmoved); routes are
    /// decode routes from here on. Needs every route released.
    pub fn grow(self: *Stream, decode_rows: []const u32) !void {
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (self.failed) return error.StreamFailed;
        if (decode_rows.len != self.layers.len) return error.InvalidRows;
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
        for (self.layers, decode_rows) |*ls, rows| {
            if (rows < ls.policy.capacity or rows > ls.policy.n_experts) return error.InvalidRows;
        }
        const a = self.allocator;
        const exts = try a.alloc(?Rows, self.layers.len);
        defer a.free(exts);
        @memset(exts, null);
        errdefer for (exts) |*e| if (e.*) |*rows| rows.deinit();
        for (self.layers, decode_rows, exts, self.bank.layers) |*ls, rows, *e, *geom| {
            if (rows > ls.policy.capacity) e.* = try Rows.init(geom, rows - ls.policy.capacity, self.memory);
        }
        for (self.layers, decode_rows, exts) |*ls, rows, e| {
            ls.ext = e;
            ls.policy.grow(rows) catch unreachable;
        }
        self.phase = .decode;
        self.route_lookahead = self.selector != null;
        self.route_preread = self.route_lookahead and self.preread;
    }

    pub fn stats(self: *Stream) Stats {
        var s = self.counters;
        s.expert_read_seconds = @as(f64, @floatFromInt(self.read_ns)) / 1e9;
        s.read_wall_ns = @intCast(@max(self.pool.readGauge()[4], 0));
        const p = self.pool;
        const pairs = .{
            .{ "claimed", .claimed },           .{ "spec_bytes", .spec_bytes },   .{ "spec_issued", .submitted },
            .{ "spec_landed", .landed },        .{ "adopt_ranges", .adopt_ranges }, .{ "adopt_bytes", .adopt_bytes },
            .{ "pre_issued", .pre_issued },     .{ "pre_served", .pre_served },   .{ "pre_expired", .pre_expired },
            .{ "gates", .ev_gates },            .{ "gates_forced", .ev_wd_forced },
        };
        inline for (pairs) |pr| @field(s, pr[0]) = @intCast(@max(p.counter(pr[1]), 0));
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
    // With DSV41_PHASE0B_MLX the 0b tests run earlier in this process and made the Metal device.
    if (std.c.getenv("DSV41_PHASE0B_MLX") == null) try testing.expect(!metalDriverLoaded());
}

test "dsv41 slots: cross-layer PICK set" {
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.pick_set;
    try testing.expectEqual(@as(usize, 8), set.len);
    var rows = try HostSlotRows.init(&env.bank.layers[set[0].layer], @intCast(set.len));
    defer rows.deinit();
    _ = try checkSet("pick_set", &env.bank, set, &rows);
    // With DSV41_PHASE0B_MLX the 0b tests run earlier in this process and made the Metal device.
    if (std.c.getenv("DSV41_PHASE0B_MLX") == null) try testing.expect(!metalDriverLoaded());
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
    const r = try s.route(layer, ids, &.{});
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
    try testing.expectError(error.SlotStillPinned, s.route(1, &.{3}, &.{}));
    try testing.expectError(error.StreamFailed, s.route(1, &.{3}, &.{}));
}

test "dsv41 stream: routes the caller never releases run out, by name" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    for (0..4) |_| _ = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesExhausted, s.route(0, &.{ 1, 2 }, &.{}));
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

test "dsv41 stream: growth from any thread but the one that built the stream is refused" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const Helper = struct {
        fn run(st: *Stream, out: *?anyerror) void {
            st.grow(&.{ 4, 4 }) catch |e| {
                out.* = e;
                return;
            };
            out.* = null;
        }
    };
    var got: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, Helper.run, .{ s, &got });
    t.join();
    try testing.expectEqual(@as(?anyerror, error.NotInferenceThread), got);
    try s.grow(&.{ 4, 4 });
}

test "dsv41 stream: a failed read fails the route and every later one" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    defer expert_io.clearFaults();
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 1).gu_offset / page * page, 2, 0);
    const r = try s.route(0, &.{1}, &.{});
    try testing.expectError(error.ReadFailed, s.waitDown(r, 0));
    try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(1));
    try testing.expectError(error.StreamFailed, s.route(0, &.{1}, &.{}));
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
    try realBankTrace(.host);
}

/// The phase-1 one-layer recorded trace on the real bank, the slot rows in `memory`.
fn realBankTrace(memory: SlotMemory) !void {
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
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = bt.transient, .transient_rows = bt.transient, .slot_memory = memory });
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
        "real bank layer {d} ({s} rows): {d} routes, {d} served slots sha256-checked; hits {d} misses {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv ({d:.3} s read, {d} ms wall); {d} ms total; peak RSS {d} B\n",
        .{ L, @tagName(memory), st.route_calls, served, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions, st.persistent_loads, st.transient_loads, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.expert_read_seconds, @divTrunc(st.read_wall_ns, std.time.ns_per_ms), @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
}

// Inside a guarded window: DSV41_PHASE0B_MLX=1 + the bank env above.
test "dsv41 stream 0b: the recorded trace on the real bank fills MLX slot banks" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    try realBankTrace(.{ .mlx = stream });
}

test "dsv41 stream: slot refs name each served slot's bank and row" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var refs: [max_route_ids]SlotRef = undefined;
    // Prefill: 1 and 2 fill the two rows, 3 lands in the shared transient scratch.
    var r = try serve(s, 0, &.{ 1, 2, 3 });
    try testing.expectEqualSlices(SlotRef, &.{ .{ .bank = .base, .row = 0 }, .{ .bank = .base, .row = 1 }, .{ .bank = .transient, .row = 0 } }, s.refsOf(r, &refs));
    s.release(r);
    try s.grow(&.{ 4, 2 });
    // Decode: the grown rows of layer 0 are its ext bank.
    r = try serve(s, 0, &.{ 1, 4, 5 });
    try testing.expectEqualSlices(SlotRef, &.{ .{ .bank = .base, .row = 0 }, .{ .bank = .ext, .row = 0 }, .{ .bank = .ext, .row = 1 } }, s.refsOf(r, &refs));
    try expectServed(s, &sb, r, &.{ 1, 4, 5 });
    // A part's loads are the rows its waves read, in file order.
    var loads: [max_route_ids]expert_policy.Load = undefined;
    try testing.expectEqual(@as(u32, 1), r.n_parts);
    const pl = r.partLoads(0, &loads);
    try testing.expectEqual(@as(usize, 2), pl.len);
    try testing.expectEqual(@as(u16, 4), pl[0].expert);
    try testing.expectEqual(@as(u16, 5), pl[1].expert);
    // Host rows have no MLX arrays to bind.
    try testing.expectEqual(@as(?BankArrays, null), s.bankArrays(0, .base));
    s.release(r);
}

fn expectShape(arr: mlx.mlx_array, dtype: mlx.mlx_dtype, want: []const c_int) !void {
    try testing.expectEqual(dtype, mlx.mlx_array_dtype(arr));
    try testing.expectEqual(want.len, mlx.mlx_array_ndim(arr));
    try testing.expectEqualSlices(c_int, want, mlx.mlx_array_shape(arr)[0..want.len]);
}

// DSV41_PHASE0B_MLX=1, inside a guarded window.
test "dsv41 stream 0b: MLX slot memory is filled by the pool like host rows" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .slot_memory = .{ .mlx = stream } });
    defer s.deinit();
    // Hidden 64 / inter 32 synthetic geometry: code [rows, 4, 2, 48] (gate/up), [rows, 2, 4, 48] (down).
    const base = s.bankArrays(0, .base).?;
    try expectShape(base.gate.code, .int16, &.{ 4, 4, 2, 48 });
    try expectShape(base.gate.rout, .float16, &.{ 4, 32 });
    try expectShape(base.down.code, .int16, &.{ 4, 2, 4, 48 });
    try expectShape(base.down.rin, .float16, &.{ 4, 32 });
    try testing.expectEqual(s.bankArrays(0, .transient).?.up.code.ctx, s.bankArrays(1, .transient).?.up.code.ctx);
    try testing.expectEqual(@as(?BankArrays, null), s.bankArrays(0, .ext));
    s.release(try serve(s, 0, &.{ 1, 2, 3, 5, 9 }));
    s.release(try serve(s, 1, &.{ 7, 8, 9 }));
    try s.grow(&.{ 6, 4 });
    try expectShape(s.bankArrays(0, .ext).?.up.rin, .float16, &.{ 2, 64 });
    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();
    var ids: [12]u16 = undefined;
    for (0..40) |step| {
        const layer: u32 = @intCast(step % 2);
        const n = rand.intRangeAtMost(usize, 1, 12);
        for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, if (step % 5 == 0) 32 else 10);
        const r = try serve(s, layer, ids[0..n]);
        try expectServed(s, &sb, r, ids[0..n]);
        s.release(r);
    }
    try s.flush();
}

// DSV41_PHASE0B_MLX=1, inside a guarded window: the GPU reads the slot arrays behind the event gate.
test "dsv41 stream 0b: gated waves over the MLX slot arrays read the landed bytes on the GPU" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const expert_event = @import("expert_event.zig");
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var sb = try SynthBank.open(32);
    defer sb.close();
    const ev = try expert_event.createMetal();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .slot_memory = .{ .mlx = stream }, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false }, .event = .{ .backend = .{ .metal = ev.object }, .watchdog_ms = 10_000 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    try s.grow(&.{ 8, 8 });
    // Expert 4's read is held 300 ms: the GPU, not the host, waits for it.
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 4).gu_offset / page * page, 5, 300 * std.time.ns_per_ms);
    const ids = [_]u16{ 3, 1, 4, 5, 9 };
    const r = try s.route(0, &ids, &.{});
    const g = (try s.gate(r)).?;
    var refs: [max_route_ids]SlotRef = undefined;
    const rf = s.refsOf(r, &refs);
    var rows_i: [ids.len]i32 = undefined;
    for (rf, &rows_i) |ref, *ri| {
        try testing.expectEqual(BankKind.base, ref.bank);
        ri.* = @intCast(ref.row);
    }
    const bank_arrays = s.bankArrays(0, .base).?;
    const src = [n_components]mlx.mlx_array{ bank_arrays.gate.code, bank_arrays.gate.rout, bank_arrays.gate.rin, bank_arrays.up.code, bank_arrays.up.rout, bank_arrays.up.rin, bank_arrays.down.code, bank_arrays.down.rout, bank_arrays.down.rin };
    var gated: [n_components]mlx.mlx_array = @splat(.{});
    defer for (gated) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // Every wave of the call has landed at the last down value.
    try expert_event.wait(&src, ev, g.down_first + g.n_parts - 1, &.{}, false, stream, &gated);
    const idx = mlx.mlx_array_new_data(&rows_i, &[_]c_int{ids.len}, 1, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var taken: [n_components]mlx.mlx_array = undefined;
    for (&taken, gated) |*t, x| {
        t.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(t, x, idx, 0, stream));
    }
    defer for (taken) |t| {
        _ = mlx.mlx_array_free(t);
    };
    const t0 = std.Io.Timestamp.now(std.testing.io, .boot);
    const vec = mlx.mlx_vector_array_new_data(&taken, taken.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    const ms = @divTrunc(t0.untilNow(std.testing.io, .boot).nanoseconds, std.time.ns_per_ms);
    const geom = &sb.bank.layers[0];
    for (taken, geom.segments) |t, seg| {
        const got = (mlx.mlx_array_data_uint8(t) orelse return error.MlxNoData)[0 .. ids.len * seg.length];
        for (ids, 0..) |e, i| {
            const off = sb.bank.recordOffset(0, e) + seg.offset;
            try testing.expectEqualSlices(u8, sb.image[off..][0..seg.length], got[i * seg.length ..][0..seg.length]);
        }
    }
    std.debug.print("gated MLX slot arrays: GPU gather evaluated after {d} ms, {d} rows x 9 components equal the records\n", .{ ms, ids.len });
    try testing.expect(ms >= 250);
    s.release(r);
    try s.flush();
    try testing.expectEqual(@as(u64, 0), s.stats().gates_forced);
}

// ── Lookahead, pre-read and event gates ──

const la_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 512 };

/// One score row over 32 experts: `top` in descending order, the rest 0.
fn scoresFor(top: []const u16) [32]f32 {
    var s: [32]f32 = @splat(0);
    for (top, 0..) |e, i| s[e] = @floatFromInt(top.len - i);
    return s;
}

fn waitCounter(s: *Stream, which: expert_io.Counter, at_least: i64) !void {
    var t: u32 = 0;
    while (s.pool.counter(which) < at_least) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn waitWord(s: *const Stream, value: u64) !void {
    const w = s.eventWord().?;
    var t: u32 = 0;
    while (@as(u64, @intCast(@atomicLoad(i64, w, .acquire))) < value) : (t += 1) {
        if (t > 20_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
    }
}

/// Routes a call and, as the GPU would, waits on its gates instead of the pool.
fn serveGated(s: *Stream, layer: u32, ids: []const u16, scores: []const f32) !*Route {
    const r = try s.route(layer, ids, scores);
    if (try s.gate(r)) |g| try waitWord(s, g.down_first + g.n_parts - 1);
    return r;
}

test "dsv41 stream: the next layer's predicted records are read ahead and claimed by its route" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false } });
    defer s.deinit();
    try s.grow(&.{ 4, 4 });
    // Layer 0's call predicts layer 1's experts 20 and 21.
    const pred = scoresFor(&.{ 20, 21, 3, 4, 5, 6 });
    const r0 = try s.route(0, &.{ 1, 2 }, &pred);
    try s.waitDown(r0, 0);
    s.release(r0);
    try waitCounter(s, .landed, 2);
    const r1 = try serve(s, 1, &.{ 20, 9, 21 });
    try expectServed(s, &sb, r1, &.{ 20, 9, 21 });
    s.release(r1);
    try s.flush();
    const st = s.stats();
    const rec = sb.bank.layers[1].logical_bytes;
    try testing.expectEqual(@as(u64, 2), st.spec_issued);
    try testing.expectEqual(@as(u64, 2), st.claimed);
    try testing.expectEqual(@as(u64, 4), st.adopt_ranges);
    try testing.expectEqual(2 * rec, st.adopt_bytes);
    // Every record still lands in its slot: read, or copied out of the speculative staging.
    try testing.expectEqual(5 * rec, st.expert_bytes_read);
}

test "dsv41 stream: pre-read ranges carry a decode call's certain misses to its reads" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 1, .chunks = 1 } });
    defer s.deinit();
    // Prefill routes never pre-read.
    s.release(try serve(s, 0, &.{ 7, 8 }));
    try testing.expectEqual(@as(i64, 0), s.pool.counter(.pre_calls));
    try s.grow(&.{ 4, 4 });
    const ids = [_]u16{ 1, 7, 2, 3, 1, 8 };
    const r = try serve(s, 0, &ids);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
    // Certain misses 1, 2, 3: one gate/up and one down range each, bound by the submit or read by it.
    try testing.expectEqual(@as(i64, 1), s.pool.counter(.pre_calls));
    try testing.expectEqual(@as(i64, 6), s.pool.counter(.pre_issued));
    try testing.expectEqual(@as(i64, 6), s.pool.counter(.pre_bound) + s.pool.counter(.pre_cancelled));
    try testing.expectEqual(s.pool.counter(.pre_bound), s.pool.counter(.pre_served));
    // A call whose experts are all resident pre-reads nothing.
    s.release(try serve(s, 0, &.{ 7, 8 }));
    try testing.expectEqual(@as(i64, 1), s.pool.counter(.pre_calls));
    try s.flush();
    try testing.expectEqual(5 * sb.bank.layers[0].logical_bytes, s.stats().expert_bytes_read);
}

test "dsv41 stream: gated routes register the gate/up wave, then each part's down wave" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1 }, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    try s.grow(&.{ 8, 8 });
    // Five misses in file order: parts of three and two.
    const ids = [_]u16{ 3, 1, 4, 5, 9 };
    const r = try s.route(0, &ids, &.{});
    try testing.expectEqual(@as(u32, 2), r.n_parts);
    const g = (try s.gate(r)).?;
    try testing.expectEqual(Gates{ .gu = 1, .down_first = 2, .n_parts = 2 }, g);
    try waitWord(s, g.gu);
    for (r.partsOf()) |p| for (0..p.n_reads) |i| try testing.expect(s.pool.result(p.ticket + @as(u32, @intCast(i))).status != .pending);
    try waitWord(s, 3);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    // Values continue across calls; a call that reads nothing is not gated.
    const r2 = try s.route(1, &.{ 2, 6 }, &.{});
    try testing.expectEqual(Gates{ .gu = 4, .down_first = 5, .n_parts = 1 }, (try s.gate(r2)).?);
    try waitWord(s, 5);
    s.release(r2);
    const r3 = try s.route(0, &.{ 3, 9 }, &.{});
    try testing.expectEqual(@as(?Gates, null), try s.gate(r3));
    s.release(r3);
    try s.flush();
    const st = s.stats();
    try testing.expectEqual(@as(u64, 5), st.gates);
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
}

test "dsv41 stream: a gate the watchdog forces fails the stream at the next flush" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false }, .event = .{ .watchdog_ms = 50 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    try s.grow(&.{ 4, 4 });
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 11).gu_offset / page * page, 5, 400 * std.time.ns_per_ms);
    const r = try s.route(0, &.{11}, &.{});
    const g = (try s.gate(r)).?;
    try waitWord(s, g.down_first);
    try testing.expect(s.stats().gates_forced >= 1);
    s.release(r);
    try testing.expectError(error.GateForced, s.route(1, &.{2}, &.{}));
    try testing.expectError(error.StreamFailed, s.route(1, &.{2}, &.{}));
}

test "dsv41 stream: lookahead options outside the lane's ranges are refused at construction" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const a = testing.allocator;
    const base: Options = .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool };
    var o = base;
    o.lookahead = .{ .k = 5 };
    try testing.expectError(error.InvalidSelector, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .budget = 5 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .chunks = 3 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .idle_busy = 2 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .tau = std.math.nan(f32) };
    try testing.expectError(error.InvalidSelector, Stream.init(a, &sb.bank, o));
    o = base;
    o.event = .{};
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{};
    o.event = .{ .watchdog_ms = 10 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    // The tier's values construct.
    o.event = .{};
    const s = try Stream.init(a, &sb.bank, o);
    s.deinit();
}

test "dsv41 stream: a verify trace of 1-8 rows with lookahead, pre-read and gates serves every slot's bytes" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // Verify widths up to 8 rows x top-6 = 48 ids, the transient scratch sized for them.
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 2 }, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
    s.release(try serve(s, 0, &.{ 1, 2, 3, 5, 9 }));
    s.release(try serve(s, 1, &.{ 7, 8, 9 }));
    try s.grow(&.{ 6, 4 });
    // Verify forwards of 1..8 rows x top-6 over both layers; layer 0 predicts layer 1.
    var rng = std.Random.DefaultPrng.init(21);
    const rand = rng.random();
    var ids: [48]u16 = undefined;
    var scores: [8 * 32]f32 = undefined;
    var gates: u64 = 0;
    var reads: u64 = 0;
    for (0..60) |step| {
        const layer: u32 = @intCast(step % 2);
        const m = rand.intRangeAtMost(usize, 1, 8);
        const span: u16 = if (step % 7 == 0) 32 else 12;
        for (ids[0 .. 6 * m]) |*e| e.* = rand.intRangeLessThan(u16, 0, span);
        for (scores[0 .. 32 * m]) |*v| v.* = rand.float(f32);
        const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m] else &.{};
        const r = try serveGated(s, layer, ids[0 .. 6 * m], pred);
        try expectServed(s, &sb, r, ids[0 .. 6 * m]);
        if (r.n_parts > 0) gates += r.n_parts + 1;
        reads += readsOf(r);
        s.release(r);
    }
    try s.flush();
    for (0..2) |l| for (0..s.layers[l].policy.capacity + 48) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(@intCast(l), @intCast(slot)));
    };
    const st = s.stats();
    try testing.expectEqual(gates, st.gates);
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expectEqual(reads * sb.bank.layers[0].logical_bytes + 8 * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0);
}

/// The phase-2 fixture's calls for one layer as M = 1 routes, and the P1 scores of each row.
const TraceCall = struct { ids: []const u16, pre: []const u16, sel: []const []const u16, cand: []const []const u16 };

// DSV41_BANK=<bank dir> DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py>
test "dsv41 stream: a two-layer recorded trace with lookahead and gates on the real bank serves every slot's bytes" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(16 << 20));
    defer a.free(text);
    const Fix = struct { layers: u32, experts: u32, rows: []const u32, resident0: []const []const u16, scores_file: []const u8, calls: []const TraceCall };
    const parsed = try std.json.parseFromSlice(Fix, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var pbuf: [1024]u8 = undefined;
    const spath = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}", .{ std.fs.path.dirname(fixture) orelse ".", f.scores_file }, 0);
    const sfd = std.c.open(spath.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (sfd < 0) return error.OpenFailed;
    defer _ = std.c.close(sfd);

    // Layers 13 and 14 at 3 -> 4 rows, 6 transient rows, the tier's lookahead (8:inf:2, 4 chunks) + event gates.
    const L: u32 = 13;
    var rows: [40]u32 = @splat(0);
    rows[L] = 3;
    rows[L + 1] = 3;
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = 6, .transient_rows = 6, .lookahead = .{}, .event = .{} });
    defer s.deinit();
    const geom = &bank.layers[L];
    for ([_]u32{ L, L + 1 }) |l| {
        var seed: [3]u16 = undefined;
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        @memcpy(&seed, sorted[0..3]);
        try s.seedPrefill(l, &seed);
        s.release(try serve(s, l, &seed));
    }
    rows[L] = 4;
    rows[L + 1] = 4;
    try s.grow(&rows);

    var served: u64 = 0;
    var routes: u64 = 0;
    var score_row: [384]f32 = undefined;
    var raw: [384 * 4]u8 = undefined;
    var at: u64 = 0; // scored rows before this cycle
    var cycle: usize = 0;
    outer: while (cycle < f.rows.len) : (cycle += 1) {
        const m = f.rows[cycle];
        const c13 = f.calls[cycle * f.layers + L];
        const c14 = f.calls[cycle * f.layers + L + 1];
        for (0..m) |r| {
            if (routes >= 40) break :outer;
            // Row r of layer 13's call predicts layer 14 (P1 scores of that row).
            const off = (at + L * m + r) * 384 * 4;
            if (std.c.pread(sfd, &raw, raw.len, @intCast(off)) != raw.len) return error.ShortRead;
            for (&score_row, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
            for ([_]struct { l: u32, ids: []const u16, scores: []const f32 }{
                .{ .l = L, .ids = c13.ids[6 * r ..][0..6], .scores = &score_row },
                .{ .l = L + 1, .ids = c14.ids[6 * r ..][0..6], .scores = &.{} },
            }) |call| {
                const rt = try serveGated(s, call.l, call.ids, call.scores);
                for (rt.plan.slotsOf(), call.ids) |slot, e| {
                    const d = slotDigest(s, call.l, slot, geom);
                    try testing.expectEqualSlices(u8, &bank.digest(call.l, e).logical, &d);
                    served += 1;
                }
                routes += 1;
                s.release(rt);
            }
        }
        at += @as(u64, m) * (f.layers - 1);
    }
    try s.flush();
    const st = s.stats();
    const ru = std.posix.getrusage(std.c.rusage.SELF);
    std.debug.print(
        "real bank layers {d}+{d}: {d} gated routes, {d} served slots sha256-checked; misses {d} skipped {d}; {d} B landed ({d} preadv); lookahead: issued {d} landed {d} claimed {d} adopted {d} ranges / {d} B, spec {d} B; pre-read issued {d} served {d} expired {d}; gates {d} forced {d}; {d} ms total; peak RSS {d} B\n",
        .{ L, L + 1, st.route_calls, served, st.expert_cache_misses, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.spec_issued, st.spec_landed, st.claimed, st.adopt_ranges, st.adopt_bytes, st.spec_bytes, st.pre_issued, st.pre_served, st.pre_expired, st.gates, st.gates_forced, @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0 and st.gates > 0);
}

test "dsv41 stream: wide depth 2 holds two prefill routes of a layer in disjoint slots and transient windows" {
    var sb = try SynthBank.open(128);
    defer sb.close();
    // The transient rows must hold both windows.
    try testing.expectError(error.InvalidOptions, Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool, .wide_depth = 2 }));
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool, .wide_depth = 2, .transient_rows = 2 * max_route_ids });
    defer s.deinit();
    var a_ids: [max_route_ids]u16 = undefined;
    var b_ids: [max_route_ids]u16 = undefined;
    for (&a_ids, &b_ids, 0..) |*x, *y, i| {
        x.* = @intCast(i);
        y.* = @intCast(max_route_ids + i);
    }
    // Group A fills the 16 persistent rows and 32 transient rows of window 0.
    const ra = try serve(s, 0, &a_ids);
    try testing.expectEqual(@as(u8, 0), ra.window);
    // Group B while A is live: no victim among A's slots (they are held), window 1's rows.
    const rb = try serve(s, 0, &b_ids);
    try testing.expectEqual(@as(u8, 1), rb.window);
    try testing.expectEqual(@as(u32, 0), rb.plan.n_persistent);
    for (rb.plan.loadsOf()) |l| {
        try testing.expect(!l.persistent);
        try testing.expect(l.slot >= 16 + max_route_ids and l.slot < 16 + 2 * max_route_ids);
    }
    try expectServed(s, &sb, ra, &a_ids);
    try expectServed(s, &sb, rb, &b_ids);
    s.release(ra);
    s.release(rb);
    try s.flush();
    // After both went back, a route takes window 0 again and may evict their persistent rows.
    var c_ids: [8]u16 = .{ 100, 101, 102, 103, 104, 105, 106, 107 };
    const rc = try serve(s, 0, &c_ids);
    try testing.expectEqual(@as(u8, 0), rc.window);
    try testing.expectEqual(@as(u32, 8), rc.plan.n_persistent);
    try expectServed(s, &sb, rc, &c_ids);
    const st = s.stats();
    try testing.expectEqual(@as(u64, (2 * max_route_ids + 8) * sb.bank.layers[0].logical_bytes), st.expert_bytes_read);
    s.release(rc);
    try s.flush();
}

test "dsv41 stream: wide depth 1 plans as before (one window, a live route's slots are not held)" {
    var sb = try SynthBank.open(128);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool });
    defer s.deinit();
    var ids: [8]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const r = try serve(s, 0, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    for (r.plan.loadsOf()) |l| try testing.expect(l.persistent and l.slot < 16);
    s.release(r);
    try s.flush();
}
