//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
/// PROFILE builds only (`-Ddsv41-prefill-timers=true`): P1's read-ahead record; every call compiles to nothing otherwise.
const prof = @import("dsv41_prefill_timers.zig");
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
    /// evaluated on `stream` in one eval. The data pointers are taken here; the
    /// arrays stay held (never donated or recycled) until `deinit`, after the pool
    /// stops. MLX allocates through Metal even on the CPU stream: callers hold the GPU lock.
    pub fn init(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
        var b = try initLazy(layer, rows, stream);
        errdefer b.deinit();
        try evalArrays(&b.arrays);
        try b.bind();
        return b;
    }

    /// `init`'s zero arrays, not yet evaluated: a grow builds every layer's, evaluates them all in one eval
    /// (`Stream.grow`: one GPU round trip, not nine per layer), then `bind`s each.
    fn initLazy(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
        var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = @splat(0), .row_bytes = undefined, .rows = rows };
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
            b.row_bytes[c] = seg.length;
        }
        return b;
    }

    /// The evaluated arrays' data pointers (after `initLazy` and an eval that covered them).
    fn bind(b: *LayerSlotBank) !void {
        for (b.arrays, &b.base) |arr, *base| base.* = @intFromPtr(mlx.mlx_array_data_uint8(arr) orelse return error.MlxNoData);
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

/// MLX's allocated bytes (MLX slot memory only: the first read creates the Metal device).
fn mlxActive() u64 {
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

/// One eval over `arrays` (a single GPU round trip).
fn evalArrays(arrays: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Where the stream keeps its slot rows: host pages (the default; hermetic
/// tests and CPU checks) or MLX arrays the kernels bind, created and evaluated
/// on `mlx` (creating any MLX array creates the Metal device: callers hold the
/// GPU lock). Chosen once, at Stream.init.
pub const SlotMemory = union(enum) { host, mlx: mlx.mlx_stream };

/// One bank of slot rows. Both memories are addressed by the same row
/// arithmetic, so the read path never asks which one it has.
/// One eval over every MLX bank among `rows` (none for host rows).
fn evalRows(a: std.mem.Allocator, rows: []const ?Rows) !void {
    var arrays: std.ArrayList(mlx.mlx_array) = .empty;
    defer arrays.deinit(a);
    for (rows) |r| if (r) |x| switch (x.backing) {
        .mlx => |m| try arrays.appendSlice(a, &m.arrays),
        .none, .host => {},
    };
    if (arrays.items.len > 0) try evalArrays(arrays.items);
}

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

    /// `init` for a grow: an MLX bank's arrays built without their eval (`bind` after one eval of every layer's,
    /// `evalRows`); host rows are complete at once.
    fn initLazy(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
        if (rows == 0) return .{};
        switch (memory) {
            .host => return init(layer, rows, memory),
            .mlx => |stream| {
                const m = try LayerSlotBank.initLazy(layer, rows, stream);
                return .{ .rows = rows, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
            },
        }
    }

    /// After the eval that covered an MLX bank's arrays: its data pointers (host rows have theirs).
    fn bind(self: *Rows) !void {
        switch (self.backing) {
            .mlx => |*m| {
                try m.bind();
                self.base = m.base;
            },
            .none, .host => {},
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
    /// The phase change's transient release installed (`releaseTransient`, then the grow's window 0); off: the whole
    /// scratch stays through decode.
    transient_release: bool = false,
    /// A0 (a): the first verify's warm reads (`warmIssue` after the grow, settled at each layer's first decode route)
    /// on the read pool's warm class; null: no warm tickets, no warm state.
    first_verify_warm: ?FirstVerifyWarm = null,
};

/// A0 (a)'s budget: at most `max_records` warm records (layer-major: the earliest layers first) and the jobs the
/// reader may run at once while demand is idle (`expert_io.Warm.busy_max`; below the worker count, so one stays free).
pub const FirstVerifyWarm = struct { max_records: u32 = 320, busy_max: u32 = 3 };

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
    /// P1's read-ahead: records posted; at each barrier, the records read ahead that the call routes (its hits)
    /// and its seed's records not read ahead (its demand loads); the bytes read ahead (in `expert_bytes_read` too).
    ahead_posted: u64 = 0,
    ahead_hits: u64 = 0,
    ahead_demand: u64 = 0,
    ahead_bytes: u64 = 0,
    /// A0 (a)'s warm reads: records issued at the grow, landed / cancelled by their layer's first decode route, and
    /// that route's hits on landed ones (once per layer; warm_landed + warm_cancelled == warm_issued).
    warm_issued: u64 = 0,
    warm_landed: u64 = 0,
    warm_cancelled: u64 = 0,
    warm_hits: u64 = 0,
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

/// The route being served plus released ones awaiting the next flush: every wide window live, and one more.
const route_capacity = max_wide_depth + 1;
/// Prefill routes one layer may hold live at once (`Options.wide_depth`; P1c: 5, the served default).
pub const max_wide_depth = 5;
/// SERVED16: the phase change frees the transient scratch (`Stream.releaseTransient`) and the grow allocates decode's
/// window 0 plus `decode_staging_rows`; the bill (deepseek_v41_bill.zig) reads this declaration.
pub const phase_change_releases_wide_windows = true;
/// The release as a construction-time route (on by default since SERVED19E; off: the control arm).
pub const transient_release_default = true;
/// Slot rows decode reserves beside window 0 for staged reads: none (the lookahead and A1 stage in the read pool).
pub const decode_staging_rows: u32 = 0;
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
    /// The widest layer: the transient rows' geometry (decode's window 0 is allocated in it at the grow).
    transient_layer: u32,
    /// `releaseTransient` ran: the scratch is freed until the grow allocates window 0.
    transient_released: bool = false,
    /// The release route, installed at construction (`Options.transient_release`).
    release_installed: bool = false,
    /// The prompt phase's scratch rows and windows (`Options`): `regrowTransient` re-creates them for a later prompt.
    prompt_transient_rows: u32 = 0,
    prompt_wide_depth: u8 = 1,
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
    /// Persistent slots of released prefill routes still pinned for a deferred call's waves
    /// (`holdBase`), held in every later route of `held_layer` until `releaseHeld`.
    held_base: std.ArrayList(u32) = .empty,
    held_layer: u32 = 0,
    /// A route's held-slot scratch (live routes' slots and `held_base`).
    held_scratch: std.ArrayList(u32) = .empty,
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
    /// P1: the prompt pass's read-ahead in flight (one layer's predicted seed).
    ahead: Ahead,
    /// A0 (a): the warm reads (`Options.first_verify_warm`); `warm_live` while a layer's are unsettled.
    warm: ?Warm = null,
    warm_live: bool = false,

    /// A0 (a)'s records (layer-major, as issued) and each layer's span of them.
    const Warm = struct {
        loads: []WarmLoad,
        admitted: []LayerPolicy.ReadAhead,
        layers: []WarmLayer,
        n: u32 = 0,
        pending_layers: u32 = 0,
    };
    /// A warm record: its slot, its job's first ticket (`no_ticket`: the slot still held it, nothing read).
    const WarmLoad = struct { expert: u16, slot: u32, ticket: u32, landed: bool = false };
    const WarmLayer = struct { lo: u32 = 0, n: u32 = 0, issued: bool = false, pending: bool = false, wait_ns: u64 = 0 };
    const no_ticket = std.math.maxInt(u32);

    /// P1's read-ahead of one layer: `loads[0..n]` (expert, slot) in file order, `reads[i]` false when the
    /// slot still held the record; its pool jobs `parts[0..n_parts]` over them (a Route's tickets).
    const Ahead = struct {
        live: bool = false,
        /// The layer's barrier has counted it (`seedPrefill`: hits and demand, once).
        tallied: bool = false,
        layer: u32 = 0,
        n: u32 = 0,
        n_parts: u32 = 0,
        loads: []LayerPolicy.ReadAhead,
        reads: []bool,
        parts: []Part,
    };

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
        if (opt.first_verify_warm) |w| {
            if (w.max_records == 0 or w.max_records > n_layers * bank.n_experts or w.busy_max == 0 or w.busy_max >= pool_opt.workers)
                return error.InvalidOptions;
            pool_opt.tickets += 2 * w.max_records;
            pool_opt.warm = .{ .tickets = 2 * w.max_records, .busy_max = w.busy_max };
        }
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
        const ahead_loads = try a.alloc(LayerPolicy.ReadAhead, bank.n_experts);
        errdefer a.free(ahead_loads);
        const ahead_reads = try a.alloc(bool, bank.n_experts);
        errdefer a.free(ahead_reads);
        const ahead_parts = try a.alloc(Part, std.math.divCeil(u32, bank.n_experts, expert_io.max_items) catch unreachable);
        errdefer a.free(ahead_parts);
        var warm: ?Warm = null;
        if (opt.first_verify_warm) |w| {
            const loads = try a.alloc(WarmLoad, w.max_records);
            errdefer a.free(loads);
            const admitted = try a.alloc(LayerPolicy.ReadAhead, bank.n_experts);
            errdefer a.free(admitted);
            const wl = try a.alloc(WarmLayer, n_layers);
            @memset(wl, .{});
            warm = .{ .loads = loads, .admitted = admitted, .layers = wl };
        }
        errdefer if (warm) |w| {
            a.free(w.loads);
            a.free(w.admitted);
            a.free(w.layers);
        };
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
            .transient_layer = @intCast(widest),
            .release_installed = opt.transient_release,
            .prompt_transient_rows = opt.transient_rows,
            .prompt_wide_depth = opt.wide_depth,
            .memory = opt.slot_memory,
            .max_route_ids = opt.max_route_ids,
            .records_per_part = opt.records_per_part,
            .wide_depth = opt.wide_depth,
            .selector = selector,
            .preread = if (opt.lookahead) |la| la.preread else false,
            .event_word = word,
            .gated = opt.event != null,
            .owner = std.Thread.getCurrentId(),
            .ahead = .{ .loads = ahead_loads, .reads = ahead_reads, .parts = ahead_parts },
            .warm = warm,
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
        self.held_base.deinit(a);
        self.held_scratch.deinit(a);
        self.transient.deinit();
        a.free(self.transient_meta);
        if (self.selector) |*sel| sel.deinit(a);
        if (self.event_word) |w| a.destroy(w);
        a.free(self.ahead.loads);
        a.free(self.ahead.reads);
        a.free(self.ahead.parts);
        if (self.warm) |w| {
            a.free(w.loads);
            a.free(w.admitted);
            a.free(w.layers);
        }
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

    /// Keep a live prefill route's persistent slots (its hits and persistent loads) pinned and held past
    /// its release, for a later call over them (the wide lane's deferred base-bank waves), until
    /// `releaseHeld`. One layer at a time.
    pub fn holdBase(self: *Stream, r: *const Route) !void {
        std.debug.assert(r.state == .live and self.phase == .prefill);
        if (self.held_base.items.len > 0 and self.held_layer != r.layer) return error.HeldOtherLayer;
        self.held_layer = r.layer;
        const cap = self.layers[r.layer].policy.capacity;
        for (r.hit_slots[0..r.plan.n_hits]) |s| if (s < cap) {
            try self.held_base.append(self.allocator, s);
            self.locate(r.layer, s).meta.pins += 1;
        };
        for (r.plan.loadsOf()) |l| if (l.persistent) {
            try self.held_base.append(self.allocator, l.slot);
            self.locate(r.layer, l.slot).meta.pins += 1;
        };
    }

    /// Unpin and stop holding what `holdBase` kept (after the deferred call's waves are evaluated).
    pub fn releaseHeld(self: *Stream) void {
        for (self.held_base.items) |s| self.locate(self.held_layer, s).meta.pins -= 1;
        self.held_base.clearRetainingCapacity();
    }

    /// Construction's end: every layer's residents and prompt state forgotten (`LayerPolicy.forgetAll`: the warm-up's),
    /// so a first prompt's seed and read-ahead start from empty rows. Slot bytes stay: a later load reuses them only on
    /// an exact (layer, expert, slot) match. Refused while anything is live; returns the residents forgotten.
    pub fn forgetResidents(self: *Stream) !u32 {
        if (self.phase != .prefill) return error.NotPrefill;
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
        if (self.ahead.live or self.held_base.items.len > 0) return error.RoutesLive;
        var n: u32 = 0;
        for (self.layers) |*ls| n += ls.policy.forgetAll();
        return n;
    }

    /// The last `seedPrefill` of `layer`: its seed's ranks (the call's hottest experts, hottest first).
    pub fn seedRanks(self: *const Stream, layer: u32) u32 {
        return self.layers[layer].policy.seed_ranks;
    }

    /// prepare_prefill_seed: the prompt's routed ids of `layer`, before its
    /// prefill routes.
    pub fn seedPrefill(self: *Stream, layer: u32, ids: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        const ah = &self.ahead;
        if (ah.live and ah.layer == layer) try self.awaitReadAhead(layer);
        const policy = &self.layers[layer].policy;
        policy.prepareSeed(ids);
        // P1's engagement, once at the layer's barrier: the records read ahead that its call routes, and its seed's
        // records not read ahead (the seed's misses, loaded on demand).
        if (ah.layer != layer or ah.n == 0 or ah.tallied) return;
        ah.tallied = true;
        for (ah.loads[0..ah.n]) |l| self.counters.ahead_hits += @intFromBool(policy.call_counts[l.expert] > 0);
        self.counters.ahead_demand += policy.seed.count();
        if (comptime prof.enabled) {
            // The layer's barrier record: hits, and each demand record's predicted count against the cut.
            var hits: u32 = 0;
            for (ah.loads[0..ah.n]) |l| hits += @intFromBool(policy.call_counts[l.expert] > 0);
            var near: u32 = 0;
            var far: u32 = 0;
            if (layer < prof.max_layers) {
                const cut = prof.ra[layer].cut;
                var it = policy.seed.iterator(.{});
                while (it.next()) |e| {
                    const pc = if (e < prof.max_experts) prof.ra_counts[layer][e] else 0;
                    if (prof.nearCut(pc, cut)) near += 1 else far += 1;
                }
            }
            prof.recordBarrier(layer, hits, @intCast(policy.seed.count()), near, far);
        }
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
        // A layer's read-ahead lands before a route plans over its rows (a hit must never read a loading row).
        if (self.ahead.live and self.ahead.layer == layer) try self.awaitReadAhead(layer);
        // A0 (a): a layer's warm reads settle at its first decode route (the cancelled ones become misses).
        const settled_warm = self.warm_live and self.warm.?.layers[layer].pending;
        if (settled_warm) try self.settleWarm(layer);
        const lookahead = self.route_lookahead;
        const tag = self.clock + 1;
        if (self.route_preread) try self.preRead(layer, ids, tag);
        try self.flush();
        if (self.n_free == 0) return self.fail(error.RoutesExhausted);
        self.n_free -= 1;
        const r = &self.routes[self.free[self.n_free]];
        // A prefill route beside live ones of its layer: their slots are held,
        // its transient loads take the first window no route holds; so are the
        // slots a deferred call still reads (`holdBase`).
        const held_set = &self.held_scratch;
        held_set.clearRetainingCapacity();
        var used_windows: u8 = 0;
        if (self.wide_depth > 1) for (&self.routes) |*o| {
            if (o.state == .free) continue;
            used_windows |= @as(u8, 1) << @intCast(o.window);
            if (o.layer != layer) continue;
            for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < self.layers[layer].policy.capacity) {
                held_set.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
            };
            for (o.plan.loadsOf()) |l| if (l.persistent) {
                held_set.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
            };
        };
        if (self.held_base.items.len > 0 and self.held_layer == layer)
            held_set.appendSlice(self.allocator, self.held_base.items) catch return self.fail(error.RoutesExhausted);
        const window: u8 = @intCast(@ctz(~used_windows));
        if (window >= self.wide_depth) return self.fail(error.RoutesExhausted);
        r.* = .{ .layer = layer, .window = window };
        const ls = &self.layers[layer];
        ls.policy.planWith(ids, self.phase, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held_set.items });
        const plan = &r.plan;
        for (plan.hitsOf(), r.hit_slots[0..plan.n_hits]) |e, *s| {
            s.* = ls.policy.slotOf(e).?;
            self.locate(layer, s.*).meta.pins += 1;
        }
        if (settled_warm) self.countWarmHits(layer, plan.hitsOf());
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

    /// P1: layer `layer`'s predicted seed (`experts`, hottest first) read into its empty persistent rows, no route, while
    /// its attention runs: `LayerPolicy.admitReadAhead` (unprotected; the seed re-protects its choices), pool jobs of
    /// `max_items` within the ticket ring. `awaitReadAhead` or a route of the layer lands it; a live one lands first.
    pub fn readAheadSeed(self: *Stream, layer: u32, experts: []const u16) !void {
        if (self.failed) return error.StreamFailed;
        if (self.phase != .prefill) return error.NotPrefill;
        if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
        const ah = &self.ahead;
        const fit = @min(ah.loads.len, (self.pool.published.len - 2 * expert_io.max_items) / 2);
        const admitted = self.layers[layer].policy.admitReadAhead(experts, ah.loads[0..fit]);
        if (comptime prof.enabled) {
            // The predicted seed: the ranking's top as many as the layer's unprotected rows (the seed's rule); blocked:
            // those neither resident nor admitted (no empty row: every row held by a resident, the read-ahead never evicts).
            const pol = &self.layers[layer].policy;
            const room: usize = pol.capacity -| @as(u32, @intCast(pol.protected.count()));
            const top = experts[0..@min(experts.len, room)];
            var blocked: u32 = 0;
            for (top) |e| blocked += @intFromBool(pol.slotOf(e) == null);
            const cut: u32 = if (top.len > 0 and layer < prof.max_layers and top[top.len - 1] < prof.max_experts) prof.ra_counts[layer][top[top.len - 1]] else 0;
            prof.recordAdmission(layer, @intCast(top.len), @intCast(admitted.len), blocked, cut);
        }
        const n: u32 = @intCast(admitted.len);
        ah.* = .{ .layer = layer, .n = n, .loads = ah.loads, .reads = ah.reads, .parts = ah.parts };
        if (n == 0) return;
        std.sort.pdq(LayerPolicy.ReadAhead, admitted, {}, struct {
            fn less(_: void, x: LayerPolicy.ReadAhead, y: LayerPolicy.ReadAhead) bool {
                return x.expert < y.expert;
            }
        }.less);
        for (admitted, ah.reads[0..n]) |l, *rd| {
            const m = self.locate(layer, l.slot).meta;
            if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
            rd.* = !(m.state == .ready and m.layer == layer and m.expert == l.expert);
            if (rd.*) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
            m.pins = 1;
        }
        ah.live = true;
        const lens = &self.layers[layer].lens;
        var start: u32 = 0;
        while (start < n) {
            const end = @min(start + expert_io.max_items, n);
            var part: Part = .{ .first = start, .n = end - start };
            var rows: [expert_io.max_items][n_components]u64 = undefined;
            var gu: [expert_io.max_items]u64 = undefined;
            var down: [expert_io.max_items]u64 = undefined;
            var nr: u32 = 0;
            for (admitted[start..end], ah.reads[start..end]) |l, rd| if (rd) {
                const loc = self.locate(layer, l.slot);
                rows[nr] = loc.rows.rowDest(loc.row);
                const sp = self.bank.spans(layer, l.expert);
                gu[nr] = sp.gu_offset;
                down[nr] = sp.down_offset;
                nr += 1;
            };
            if (nr > 0) {
                part.ticket = self.pool.submit(self.bank.sidecar_fd, self.bank.sidecar_file_size, gu[0..nr], down[0..nr], rows[0..nr], lens) catch |e| return self.fail(e);
                part.n_reads = nr;
                self.counters.ahead_posted += nr;
                prof.addPosted(layer, nr);
            } else part.settled = true;
            ah.parts[ah.n_parts] = part;
            ah.n_parts += 1;
            start = end;
        }
    }

    /// P1: lands layer `layer`'s read-ahead (none live for it: nothing to do). Every job waited and its
    /// rows ready, the rows' pins dropped; a record that failed is forgotten by the policy and fails the
    /// stream, as a route's failed load does.
    pub fn awaitReadAhead(self: *Stream, layer: u32) Error!void {
        const ah = &self.ahead;
        if (!ah.live or ah.layer != layer) return;
        ah.live = false;
        const policy = &self.layers[layer].policy;
        var first_error: ?Error = null;
        for (ah.parts[0..ah.n_parts]) |*p| {
            if (p.settled) continue;
            p.settled = true;
            const count = 2 * p.n_reads;
            const waited = self.pool.wait(p.ticket, count, wait_timeout_ns);
            var ok = if (waited) |_| true else |_| false;
            if (ok) for (0..count) |k| {
                const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                if (res.status != .ok) ok = false;
                self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                self.counters.ahead_bytes += @intCast(@max(res.payload, 0));
                self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                self.read_ns += @intCast(@max(res.t_end_ns - res.t_start_ns, 0));
            };
            for (ah.loads[p.first..][0..p.n], ah.reads[p.first..][0..p.n]) |l, rd| if (rd) {
                self.locate(layer, l.slot).meta.state = if (ok) .ready else .failed;
                if (!ok) policy.invalidate(l.expert);
            };
            if (!ok and first_error == null) first_error = if (waited) |_| error.ReadFailed else |e| e;
        }
        for (ah.loads[0..ah.n]) |l| self.locate(layer, l.slot).meta.pins -= 1;
        if (first_error) |e| return self.fail(e);
    }

    /// P1's construction self-check: `experts` of `layer` (none resident, one route wide) read ahead, hashed, zeroed and
    /// forgotten, then read again by a demand route; each record's bytes must equal both ways, bit for bit. The layer
    /// is left as found (the experts not resident, their rows empty).
    pub fn checkReadAhead(self: *Stream, layer: u32, experts: []const u16) !void {
        const ls = &self.layers[layer];
        if (experts.len == 0 or experts.len > self.max_route_ids or experts.len > ls.policy.capacity - ls.policy.occupancy) return error.ReadAheadCheckShape;
        for (experts) |e| if (ls.policy.slotOf(e) != null) return error.ReadAheadCheckShape;
        var ahead_sums: [max_route_ids][32]u8 = undefined;
        try self.readAheadSeed(layer, experts);
        if (self.ahead.n != experts.len) return error.ReadAheadCheckNotAdmitted;
        try self.awaitReadAhead(layer);
        for (experts, ahead_sums[0..experts.len]) |e, *sum| {
            const slot = ls.policy.slotOf(e).?;
            sum.* = self.recordDigest(layer, slot);
            for (ls.lens, 0..) |len, c| @memset(self.slotRow(layer, slot, @enumFromInt(c))[0..len], 0);
            self.locate(layer, slot).meta.state = .empty;
            ls.policy.invalidate(e);
        }
        const r = try self.route(layer, experts, &.{});
        for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
            self.release(r);
            return e;
        };
        // Every expert a load that read (the zeroed rows refilled), its bytes the read-ahead's.
        var mismatch = r.plan.n_loads != experts.len;
        for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, rd| {
            const i = std.mem.indexOfScalar(u16, experts, l.expert) orelse {
                mismatch = true;
                continue;
            };
            const d = self.recordDigest(layer, l.slot);
            if (!rd or !std.mem.eql(u8, &ahead_sums[i], &d)) mismatch = true;
        }
        self.release(r);
        try self.flush();
        for (experts) |e| if (ls.policy.slotOf(e)) |slot| {
            self.locate(layer, slot).meta.state = .empty;
            ls.policy.invalidate(e);
        };
        self.ahead.n = 0;
        if (mismatch) return error.ReadAheadCheckMismatch;
    }

    /// sha256 of the record a layer's slot holds (its component rows at their logical lengths).
    fn recordDigest(self: *Stream, layer: u32, slot: u32) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (self.layers[layer].lens, 0..) |len, c| h.update(self.slotRow(layer, slot, @enumFromInt(c))[0..len]);
        var d: [32]u8 = undefined;
        h.final(&d);
        return d;
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

    /// The phase change's first free (the Module's frees stage, before its cache clear and boundary check): the
    /// whole transient scratch, with grow's preconditions; `grow` allocates decode's window 0. MLX rows: the
    /// allocator's active bytes must drop by the scratch's, else a holder survived. Returns the bytes freed.
    pub fn releaseTransient(self: *Stream) !u64 {
        if (!self.release_installed) return error.TransientReleaseNotInstalled;
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (self.failed) return error.StreamFailed;
        if (self.transient_released) return error.TransientAlreadyReleased;
        if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
        if (self.held_base.items.len > 0) return error.RoutesLive;
        var bytes: u64 = 0;
        for (self.transient.row_bytes) |n| bytes += n * self.transient.rows;
        const before = if (self.memory == .mlx) mlxActive() else 0;
        self.transient.deinit();
        self.allocator.free(self.transient_meta);
        self.transient_meta = self.transient_meta[0..0];
        self.transient_released = true;
        self.wide_depth = 1;
        if (self.memory == .mlx and before -| mlxActive() < bytes) {
            self.failed = true;
            return error.TransientStillReferenced;
        }
        return bytes;
    }

    /// The one phase change: each layer's persistent rows become
    /// `decode_rows` (the added rows empty, residents unmoved); routes are
    /// decode routes from here on. Needs every route released and the scratch released first: decode's window 0
    /// (plus `decode_staging_rows`) is allocated here, before the added rows.
    pub fn grow(self: *Stream, decode_rows: []const u32) !void {
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (self.failed) return error.StreamFailed;
        if (self.release_installed and !self.transient_released) return error.TransientNotReleased;
        if (decode_rows.len != self.layers.len) return error.InvalidRows;
        if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
        for (self.layers, decode_rows) |*ls, rows| {
            if (rows < ls.policy.capacity or rows > ls.policy.n_experts) return error.InvalidRows;
        }
        const a = self.allocator;
        var window0: ?Rows = null;
        var meta0: []SlotMeta = &.{};
        errdefer if (window0) |*w| {
            w.deinit();
            a.free(meta0);
        };
        if (self.transient_released) {
            window0 = try Rows.init(&self.bank.layers[self.transient_layer], self.max_route_ids + decode_staging_rows, self.memory);
            meta0 = a.alloc(SlotMeta, window0.?.rows) catch |e| {
                window0.?.deinit();
                window0 = null;
                return e;
            };
            @memset(meta0, .{});
        }
        const exts = try a.alloc(?Rows, self.layers.len);
        defer a.free(exts);
        @memset(exts, null);
        errdefer for (exts) |*e| if (e.*) |*rows| rows.deinit();
        for (self.layers, decode_rows, exts, self.bank.layers) |*ls, rows, *e, *geom| {
            if (rows > ls.policy.capacity) e.* = try Rows.initLazy(geom, rows - ls.policy.capacity, self.memory);
        }
        // Every layer's new MLX arrays in one eval (growth-overlap step 1: not nine evals per layer, 360 at 40).
        try evalRows(a, exts);
        for (exts) |*e| if (e.*) |*r| try r.bind();
        if (window0) |w| {
            self.transient = w;
            self.transient_meta = meta0;
            self.transient_released = false;
        }
        for (self.layers, decode_rows, exts) |*ls, rows, e| {
            ls.ext = e;
            ls.policy.grow(rows) catch unreachable;
        }
        self.phase = .decode;
        self.route_lookahead = self.selector != null;
        self.route_preread = self.route_lookahead and self.preread;
    }

    /// The reverse phase change's free (the return to the prompt phase before a later prompt; the caller synchronized
    /// first): every route settled and unpinned (a cancelled request's included), each layer's grown rows freed and the residents in them
    /// forgotten (`LayerPolicy.shrink`), decode's window 0 freed under the release route; the stream is in its prompt
    /// phase with the scratch absent until `regrowTransient`, which the caller runs only after these frees landed.
    /// Returns the bytes freed. MLX rows: the allocator's active bytes must drop by them, else a holder survived.
    pub fn shrink(self: *Stream, prompt_rows: []const u32) !u64 {
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .decode) return error.NotGrown;
        if (self.failed) return error.StreamFailed;
        if (prompt_rows.len != self.layers.len) return error.InvalidRows;
        for (self.layers, prompt_rows) |*ls, rows| if (rows != ls.base.rows) return error.InvalidRows;
        if (self.warm) |*w| if (w.pending_layers > 0) for (w.layers, 0..) |wl, l| {
            if (wl.pending) try self.settleWarm(@intCast(l));
        };
        try self.settleRoutes();
        var bytes: u64 = 0;
        for (self.layers) |*ls| if (ls.ext) |e| {
            for (e.row_bytes) |n| bytes += n * e.rows;
        };
        if (self.release_installed) for (self.transient.row_bytes) |n| {
            bytes += n * self.transient.rows;
        };
        const before = if (self.memory == .mlx) mlxActive() else 0;
        for (self.layers, prompt_rows) |*ls, rows| {
            for (ls.meta[rows..ls.policy.capacity]) |*m| {
                if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                m.* = .{};
            }
            ls.policy.shrink(rows) catch unreachable;
            // The one place that decides which residents a later prompt finds: none, as at construction (its
            // schedule then equals the first prompt's; slot bytes stay, a load of the same record skips its read).
            _ = ls.policy.forgetAll();
            if (ls.ext) |*e| e.deinit();
            ls.ext = null;
        }
        if (self.release_installed) {
            self.transient.deinit();
            self.allocator.free(self.transient_meta);
            self.transient_meta = self.transient_meta[0..0];
            self.transient_released = true;
        }
        if (self.warm) |*w| {
            @memset(w.layers, .{});
            w.n = 0;
            w.pending_layers = 0;
        }
        self.warm_live = false;
        self.phase = .prefill;
        self.route_lookahead = false;
        self.route_preread = false;
        if (self.memory == .mlx and before -| mlxActive() < bytes) {
            self.failed = true;
            return error.GrownRowsStillReferenced;
        }
        return bytes;
    }

    /// A request's end (the caller synchronized: no command still reads a slot): a cancelled request's live and held
    /// routes released, the prompt's read-ahead awaited, and everything settled and unpinned by the flush (reads still
    /// landing are waited for, never cancelled). Nothing live after it.
    pub fn settleRoutes(self: *Stream) !void {
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.failed) return error.StreamFailed;
        if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
        for (&self.routes) |*r| if (r.state == .live) self.release(r);
        self.releaseHeld();
        try self.flush();
        for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
    }

    /// The prompt scratch's bytes (`regrowTransient` allocates them).
    pub fn promptTransientBytes(self: *const Stream) u64 {
        var n: u64 = 0;
        for (self.bank.layers[self.transient_layer].segments) |seg| n += seg.length;
        return n * self.prompt_transient_rows;
    }

    /// The reverse phase change's allocation, after its frees landed: the prompt's scratch (`Options.transient_rows`
    /// rows, `wide_depth` windows) re-created, so the next prompt routes as the first did. Returns its bytes.
    pub fn regrowTransient(self: *Stream) !u64 {
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (self.failed) return error.StreamFailed;
        if (!self.transient_released) return error.TransientNotReleased;
        var t = try Rows.init(&self.bank.layers[self.transient_layer], self.prompt_transient_rows, self.memory);
        errdefer t.deinit();
        const meta = try self.allocator.alloc(SlotMeta, t.rows);
        @memset(meta, .{});
        self.transient = t;
        self.transient_meta = meta;
        self.transient_released = false;
        self.wide_depth = self.prompt_wide_depth;
        var bytes: u64 = 0;
        for (t.row_bytes) |n| bytes += n * t.rows;
        return bytes;
    }

    /// A0 (a): after the grow, `layer`'s warm set (its prompt tail's experts, ascending) read below demand: each
    /// expert not resident takes an empty persistent slot (no eviction, the policy's read-ahead admission) and one
    /// warm job reads it; a slot that still holds the record is ready at once. Layers in order, once each, up to the
    /// budget's records. Returns the records issued.
    pub fn warmIssue(self: *Stream, layer: u32, experts: []const u16) !u32 {
        if (self.warm == null) return error.WarmNotInstalled;
        const w = &self.warm.?;
        if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
        if (self.phase != .decode) return error.WarmBeforeGrow;
        if (self.failed) return error.StreamFailed;
        if (layer >= self.layers.len or w.layers[layer].issued) return error.InvalidWarm;
        const wl = &w.layers[layer];
        wl.* = .{ .lo = w.n, .issued = true };
        const ls = &self.layers[layer];
        const room = @min(w.loads.len - w.n, w.admitted.len);
        const admitted = ls.policy.admitReadAhead(experts, w.admitted[0..room]);
        for (admitted) |ad| {
            const loc = self.locate(layer, ad.slot);
            const m = loc.meta;
            if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
            var ticket: u32 = no_ticket;
            if (m.state == .ready and m.layer == layer and m.expert == ad.expert) {
                self.counters.loads_skipped += 1;
            } else {
                m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = ad.expert };
                const sp = self.bank.spans(layer, ad.expert);
                const rows = [1][n_components]u64{loc.rows.rowDest(loc.row)};
                ticket = self.pool.submitWarm(self.bank.sidecar_fd, self.bank.sidecar_file_size, &.{sp.gu_offset}, &.{sp.down_offset}, &rows, &ls.lens) catch |e| return self.fail(e);
            }
            w.loads[w.n] = .{ .expert = ad.expert, .slot = ad.slot, .ticket = ticket };
            w.n += 1;
        }
        wl.n = @intCast(admitted.len);
        if (wl.n > 0) {
            wl.pending = true;
            w.pending_layers += 1;
            self.warm_live = true;
        }
        self.counters.warm_issued += wl.n;
        return wl.n;
    }

    /// A0 (a): `layer`'s warm reads at its first decode route, before its pre-read and plan. The queued jobs are
    /// cancelled (their experts forgotten, their slots empty again), the started ones waited for (`warmWaitNs`), the
    /// landed ones ready. A failed read fails the stream, as a demand read does.
    fn settleWarm(self: *Stream, layer: u32) Error!void {
        const w = &self.warm.?;
        const wl = &w.layers[layer];
        wl.pending = false;
        w.pending_layers -= 1;
        if (w.pending_layers == 0) self.warm_live = false;
        const loads = w.loads[wl.lo..][0..wl.n];
        // The layer's jobs hold consecutive tickets (one record each, issued in order; the warm ring holds the budget).
        var first: u32 = no_ticket;
        var end: u32 = 0;
        for (loads) |l| if (l.ticket != no_ticket) {
            if (first == no_ticket) first = l.ticket;
            end = l.ticket + 2;
        };
        if (first != no_ticket) {
            _ = self.pool.cancelWarm(first, end - first);
            const t0 = expert_io.monotonicNs();
            self.pool.wait(first, end - first, wait_timeout_ns) catch |e| return self.fail(e);
            wl.wait_ns = @intCast(@max(expert_io.monotonicNs() - t0, 0));
        }
        const ls = &self.layers[layer];
        for (loads) |*l| {
            const m = self.locate(layer, l.slot).meta;
            if (l.ticket == no_ticket) {
                l.landed = true;
                self.counters.warm_landed += 1;
                continue;
            }
            const gu = self.pool.result(l.ticket).status;
            const down = self.pool.result(l.ticket + 1).status;
            if (gu == .ok and down == .ok) {
                m.state = .ready;
                l.landed = true;
                self.counters.warm_landed += 1;
                for ([2]u32{ l.ticket, l.ticket + 1 }) |t| {
                    const res = self.pool.result(t);
                    self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                    self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                    self.read_ns += @intCast(@max(res.t_end_ns - res.t_start_ns, 0));
                }
            } else if (gu == .skipped and down == .skipped) {
                ls.policy.invalidate(l.expert);
                m.* = .{};
                self.counters.warm_cancelled += 1;
            } else return self.fail(error.ReadFailed);
        }
    }

    /// A0 (a): the settling route's hits on `layer`'s landed warm records (once per layer).
    fn countWarmHits(self: *Stream, layer: u32, hits: []const u16) void {
        const w = &self.warm.?;
        const wl = w.layers[layer];
        for (hits) |e| for (w.loads[wl.lo..][0..wl.n]) |l| {
            if (l.landed and l.expert == e) {
                self.counters.warm_hits += 1;
                break;
            }
        };
    }

    /// A0 (a) (profile builds read it): the ns `layer`'s first decode route waited for its started warm jobs; 0
    /// without the route or with none in flight.
    pub fn warmWaitNs(self: *const Stream, layer: u32) u64 {
        const w = self.warm orelse return 0;
        return if (layer < w.layers.len) w.layers[layer].wait_ns else 0;
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
    for (0..route_capacity) |_| _ = try serve(s, 0, &.{ 1, 2 });
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
    // Off its route (the default) the release is refused by name and the scratch stays whole through decode.
    try testing.expectError(error.TransientReleaseNotInstalled, s.releaseTransient());
    try testing.expectError(error.InvalidRows, s.grow(&.{ 1, 4 }));
    try testing.expectError(error.InvalidRows, s.grow(&.{4}));
    try s.grow(&.{ 4, 3 });
    try testing.expectEqual(@as(u32, 12), s.transient.rows);
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

test "dsv41 stream: A0 (a): the grow's warm reads fill empty rows below demand; a layer's first decode route lands the started, cancels the queued (served on demand) and counts its hits" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const page = std.heap.pageSize();
    // Off its route the class is refused by name and counts nothing.
    {
        const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
        defer s.deinit();
        try s.grow(&.{ 4, 4 });
        try testing.expectError(error.WarmNotInstalled, s.warmIssue(0, &.{3}));
        try testing.expectEqual(@as(u64, 0), s.stats().warm_issued);
    }
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .first_verify_warm = .{ .max_records = 8, .busy_max = 1 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    var r = try serve(s, 0, &.{ 1, 2 });
    s.release(r);
    r = try serve(s, 1, &.{ 1, 2 });
    s.release(r);
    try testing.expectError(error.WarmBeforeGrow, s.warmIssue(0, &.{3}));
    try s.grow(&.{ 6, 6 });
    // Layer 0's first warm record (expert 3) holds the reader 80 ms, so the rest wait in the warm ring (busy limit 1).
    const off3 = sb.bank.recordOffset(0, 3);
    expert_io.injectFault(off3 - off3 % page, 5, 80 * std.time.ns_per_ms);
    // Layer 0: 1 is resident (skipped), 3, 4 and 5 issued; layer 1: 7; a layer issues once.
    try testing.expectEqual(@as(u32, 3), try s.warmIssue(0, &.{ 1, 3, 4, 5 }));
    try testing.expectEqual(@as(u32, 1), try s.warmIssue(1, &.{7}));
    try testing.expectError(error.InvalidWarm, s.warmIssue(0, &.{6}));
    while (s.pool.counter(.warm_started) == 0) std.Thread.yield() catch {};
    // Layer 0's first decode route: 3 lands (waited for), 4 and 5 are cancelled; 3 is a hit, 4 and 6 are read on demand.
    r = try serve(s, 0, &.{ 3, 4, 6 });
    try testing.expectEqual(@as(u32, 1), r.plan.n_hits);
    try testing.expectEqualSlices(u16, &.{3}, r.plan.hitsOf());
    try expectServed(s, &sb, r, &.{ 3, 4, 6 });
    try testing.expect(s.warmWaitNs(0) > 0);
    s.release(r);
    // Layer 1's warm record lands below demand; its first route serves it as a hit.
    const w = &s.warm.?;
    const t7 = w.loads[w.layers[1].lo].ticket;
    try s.pool.wait(t7, 2, 10 * std.time.ns_per_s);
    r = try serve(s, 1, &.{ 7, 8 });
    try testing.expectEqualSlices(u16, &.{7}, r.plan.hitsOf());
    try expectServed(s, &sb, r, &.{ 7, 8 });
    s.release(r);
    // Settled once: a later route of layer 0 counts no warm hit.
    r = try serve(s, 0, &.{3});
    s.release(r);
    const st = s.stats();
    try testing.expectEqual(@as(u64, 4), st.warm_issued);
    try testing.expectEqual(@as(u64, 2), st.warm_landed);
    try testing.expectEqual(@as(u64, 2), st.warm_cancelled);
    try testing.expectEqual(@as(u64, 2), st.warm_hits);
    try testing.expectEqual(st.warm_issued, st.warm_landed + st.warm_cancelled);
    try testing.expect(!s.warm_live);
}

test "dsv41 stream: slots held for a deferred call are never refilled until released" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var r = try serve(s, 0, &.{ 1, 2 });
    const kept = [2]u32{ s.layers[0].policy.expert_to_slot[1], s.layers[0].policy.expert_to_slot[2] };
    try testing.expect(kept[0] < 2 and kept[1] < 2);
    try s.holdBase(r);
    s.release(r);
    // One layer at a time.
    const other = try serve(s, 1, &.{7});
    try testing.expectError(error.HeldOtherLayer, s.holdBase(other));
    s.release(other);
    // A later route of the layer: the held slots are neither evicted nor refilled; it is served right.
    r = try serve(s, 0, &.{ 3, 4, 5 });
    for (r.plan.slotsOf()) |sl| try testing.expect(sl != kept[0] and sl != kept[1]);
    try expectServed(s, &sb, r, &.{ 3, 4, 5 });
    try testing.expectEqual(kept[0], s.layers[0].policy.expert_to_slot[1]);
    try testing.expectEqual(kept[1], s.layers[0].policy.expert_to_slot[2]);
    s.release(r);
    // Released: the slots take loads again.
    s.releaseHeld();
    r = try serve(s, 0, &.{ 6, 7, 8, 9 });
    try expectServed(s, &sb, r, &.{ 6, 7, 8, 9 });
    s.release(r);
    try s.flush();
    for (0..2) |sl| try testing.expectEqual(@as(u16, 0), s.layers[0].meta[sl].pins);
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

/// One layer's prompt call on a fresh stream (the pool is the process's: one stream at a time), its predicted
/// seed read ahead first when given: every routed id served from its record; the call's hits and stats.
fn p1Call(sb: *const SynthBank, predicted: ?[]const u16) !struct { hits: u64, st: Stats } {
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 6, 6 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    if (predicted) |p| {
        try s.readAheadSeed(0, p);
        try testing.expect(s.ahead.live and s.ahead.n == p.len);
        try s.awaitReadAhead(0);
        for (p) |e| {
            const slot = s.layers[0].policy.slotOf(e).?;
            try testing.expectEqual(SlotState.ready, s.locate(0, slot).meta.state);
            try testing.expectEqual(@as(u16, 0), s.pinsOf(0, slot));
        }
    }
    // The layer's routed ids at its barrier, then its distinct experts in two groups.
    try s.seedPrefill(0, &.{ 1, 2, 3, 1, 2, 1, 4, 5, 6, 9, 10, 4, 1, 2 });
    var hits: u64 = 0;
    for ([_][]const u16{ &.{ 1, 2, 4, 3, 5 }, &.{ 6, 9, 10 } }) |grp| {
        const r = try serve(s, 0, grp);
        try expectServed(s, sb, r, grp);
        hits += r.plan.n_hits;
        s.release(r);
    }
    try s.flush();
    return .{ .hits = hits, .st = s.stats() };
}

test "dsv41 stream: P1: a read-ahead's records land before the layer's routes, which serve them as hits; every served byte is a stream's without it" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // The predicted seed: three of the call's four hottest, and 7 (mispredicted).
    const on = try p1Call(&sb, &.{ 1, 2, 7, 4 });
    const off = try p1Call(&sb, null);
    // 1, 2 and 4: hits, read during the attention; 7: read and never served (the waste); the rest as without.
    try testing.expectEqual(off.hits + 3, on.hits);
    try testing.expectEqual(off.st.expert_cache_misses - 3, on.st.expert_cache_misses);
    const rec = sb.bank.layers[0].logical_bytes;
    try testing.expectEqual(off.st.expert_bytes_read + (4 - 3) * rec, on.st.expert_bytes_read);
    // Its engagement at the barrier: 4 posted, 3 routed by the call (7 not), the seed's 3, 5 and 6 on demand.
    try testing.expectEqual(@as(u64, 4), on.st.ahead_posted);
    try testing.expectEqual(@as(u64, 3), on.st.ahead_hits);
    try testing.expectEqual(@as(u64, 3), on.st.ahead_demand);
    try testing.expectEqual(4 * rec, on.st.ahead_bytes);
    try testing.expectEqual(@as(u64, 0), off.st.ahead_posted + off.st.ahead_hits + off.st.ahead_demand + off.st.ahead_bytes);
}

test "dsv41 stream: P1: a route lands its layer's read-ahead first, another layer's lands the live one, full rows admit none, the phase change lands one" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 3, 3 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    // Three rows: 4, 5 and 6 admitted, 7 does not fit.
    try s.readAheadSeed(0, &.{ 4, 5, 6, 7 });
    try testing.expectEqual(@as(u32, 3), s.ahead.n);
    try s.readAheadSeed(1, &.{8});
    try testing.expect(s.ahead.live and s.ahead.layer == 1);
    for ([_]u16{ 4, 5, 6 }) |e| try testing.expectEqual(SlotState.ready, s.locate(0, s.layers[0].policy.slotOf(e).?).meta.state);
    try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(7));
    // A route of layer 1 before its barrier's await: the read-ahead lands first, 8 is a hit.
    const r = try serve(s, 1, &.{ 8, 9 });
    try testing.expect(!s.ahead.live);
    try testing.expectEqual(@as(u32, 1), r.plan.n_hits);
    try expectServed(s, &sb, r, &.{ 8, 9 });
    s.release(r);
    try s.readAheadSeed(0, &.{ 10, 11 });
    try testing.expect(!s.ahead.live and s.ahead.n == 0);
    try s.readAheadSeed(1, &.{12});
    try testing.expect(s.ahead.live);
    try s.grow(&.{ 4, 4 });
    try testing.expect(!s.ahead.live);
    try testing.expectEqual(SlotState.ready, s.locate(1, s.layers[1].policy.slotOf(12).?).meta.state);
    try testing.expectError(error.NotPrefill, s.readAheadSeed(1, &.{13}));
}

test "dsv41 stream: P1: a read-ahead whose record fails to land forgets its job's records and fails the stream" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    defer expert_io.clearFaults();
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 2).gu_offset / page * page, 2, 0);
    try s.readAheadSeed(0, &.{ 1, 2, 3 });
    try testing.expectError(error.ReadFailed, s.awaitReadAhead(0));
    for ([_]u16{ 1, 2, 3 }) |e| try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(e));
    try testing.expectError(error.StreamFailed, s.route(0, &.{1}, &.{}));
}

test "dsv41 stream: P1's construction self-check: records read ahead equal their demand reads over two jobs; the layer is left as found" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 12, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const experts = [_]u16{ 31, 30, 29, 28, 27, 26, 25, 24, 23 };
    try s.checkReadAhead(0, &experts);
    try testing.expectEqual(@as(u32, 0), s.layers[0].policy.occupancy);
    for (0..12) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(0, @intCast(slot)));
        try testing.expectEqual(SlotState.empty, s.locate(0, @intCast(slot)).meta.state);
    }
    try testing.expect(!s.ahead.live and !s.failed);
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2 * experts.len) * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    // Refused shapes: a resident expert; more experts than free rows.
    const r = try serve(s, 0, &.{5});
    s.release(r);
    try testing.expectError(error.ReadAheadCheckShape, s.checkReadAhead(0, &.{5}));
    try testing.expectError(error.ReadAheadCheckShape, s.checkReadAhead(0, &(experts ++ [_]u16{ 22, 21, 20 })));
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

// DSV41_PHASE0B_MLX=1 only (lock-held; 2 GB at a time, freed between). Growth-overlap step 2's premise, measured
// before it is built: the inference thread's cost of 2 GB of new slot memory under the server's wired policy, (a) as
// zeros + one eval (step 1), (b) as a page-aligned mapping wrapped no-copy while untouched, then its first GPU use,
// (c) the same after a helper thread touched every page (the helper's time printed apart). Asserts only no-copy.
test "dsv41 growth 0b: new slot memory's cost on the inference thread, zeros vs a no-copy wrap before and after a helper's touch" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const io = testing.io;
    _ = mlx.applyWiredPolicy();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bytes: usize = 2 << 30;
    const elems: c_int = @intCast(bytes / 2);
    const page = std.heap.pageSize();
    const Probe = struct {
        fn ms(t: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(t.untilNow(testing.io, .boot).nanoseconds)) / 1e6;
        }
        const Payload = struct { m: []align(std.heap.page_size_min) u8 };
        fn dtor(ctx: ?*anyopaque) callconv(.c) void {
            const pl: *Payload = @ptrCast(@alignCast(ctx.?));
            std.posix.munmap(pl.m);
            std.heap.c_allocator.destroy(pl);
        }
        fn map(len: usize) ![]align(std.heap.page_size_min) u8 {
            return std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        }
        /// The mapping as an int16 array, no copy (MLX calls `dtor` at once when it had to copy).
        fn wrap(m: []align(std.heap.page_size_min) u8, n: c_int) !struct { arr: mlx.mlx_array, no_copy: bool } {
            const pl = try std.heap.c_allocator.create(Payload);
            pl.* = .{ .m = m };
            const base = m.ptr;
            const arr = mlx.mlx_array_new_data_managed_payload(@ptrCast(base), &[_]c_int{n}, 1, .int16, pl, dtor);
            const d = mlx.mlx_array_data_uint8(arr);
            return .{ .arr = arr, .no_copy = if (d) |p| @intFromPtr(p) == @intFromPtr(base) else false };
        }
        /// One GPU command that reads the array (a sum over its first 1024 elements).
        fn use(arr: mlx.mlx_array, st: mlx.mlx_stream) !void {
            var sl = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sl);
            try mlx.check(mlx.mlx_slice(&sl, arr, &[_]c_int{0}, 1, &[_]c_int{1024}, 1, &[_]c_int{1}, 1, st));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_sum(&r, sl, false, st));
            try mlx.check(mlx.mlx_array_eval(r));
        }
        fn touch(m: []u8, step: usize) void {
            var i: usize = 0;
            while (i < m.len) : (i += step) m[i] = 0;
        }
    };
    // (a) step 1: zeros and one eval.
    var t = std.Io.Timestamp.now(io, .boot);
    var z = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{elems}, 1, .int16, s));
    try evalArrays(&.{z});
    const zeros_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(z);
    _ = mlx.mlx_clear_cache();
    // (b) an untouched mapping, wrapped, then its first use.
    const mb = try Probe.map(bytes);
    t = std.Io.Timestamp.now(io, .boot);
    const wb = try Probe.wrap(mb, elems);
    const wrap_untouched_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    try Probe.use(wb.arr, s);
    const use_untouched_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(wb.arr);
    // (c) a helper touches every page first (off the timed thread), then the wrap and its first use.
    const mc = try Probe.map(bytes);
    t = std.Io.Timestamp.now(io, .boot);
    const helper = try std.Thread.spawn(.{}, Probe.touch, .{ @as([]u8, mc), page });
    helper.join();
    const helper_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    const wc = try Probe.wrap(mc, elems);
    const wrap_touched_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    try Probe.use(wc.arr, s);
    const use_touched_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(wc.arr);
    _ = mlx.mlx_clear_cache();
    std.debug.print("\nGROWTH_OVERLAP_PROBE {{\"bytes\": {d}, \"zeros_eval_ms\": {d:.2}, \"wrap_untouched_ms\": {d:.2}, \"first_use_untouched_ms\": {d:.2}, \"helper_touch_ms\": {d:.2}, \"wrap_touched_ms\": {d:.2}, \"first_use_touched_ms\": {d:.2}, \"no_copy\": [{}, {}]}}\n", .{
        bytes, zeros_ms, wrap_untouched_ms, use_untouched_ms, helper_ms, wrap_touched_ms, use_touched_ms, wb.no_copy, wc.no_copy,
    });
    try testing.expect(wb.no_copy and wc.no_copy);
}

test "dsv41 stream: the construction's forget: the warm-up's seeded residents cleared, the first prompt's seed takes every row and its read-ahead fills them" {
    var sb = try SynthBank.open(64);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .pool = test_pool });
    defer s.deinit();
    // The warm-up's wide call on layer 0: its 6 experts seeded (protected) and read into the rows.
    try s.seedPrefill(0, &.{ 40, 41, 42, 43, 44, 45, 40, 41 });
    s.release(try serve(s, 0, &.{ 40, 41, 42, 43, 44, 45 }));
    try s.flush();
    try testing.expectEqual(@as(u32, 6), s.layers[0].policy.occupancy);
    try testing.expectEqual(@as(usize, 6), s.layers[0].policy.protected.count());
    // Kept, they would hold 6 of layer 0's 8 rows through a prompt (a seed of 8 - 6 = 2). Forgotten once:
    try testing.expectEqual(@as(u32, 6), try s.forgetResidents());
    for (s.layers) |*ls| {
        try testing.expectEqual(@as(u32, 0), ls.policy.occupancy);
        try testing.expectEqual(@as(usize, 0), ls.policy.protected.count());
    }
    // The first prompt: each layer's read-ahead posts every row and each seed takes every row (1..8 twice, 9 and 10 once).
    const prompt = [_]u16{ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10 };
    for (0..2) |l| {
        try s.readAheadSeed(@intCast(l), &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        try s.seedPrefill(@intCast(l), &prompt);
        try testing.expectEqual(@as(u32, 8), s.seedRanks(@intCast(l)));
    }
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2 * 8), st.ahead_posted);
    try testing.expectEqual(@as(u64, 0), st.ahead_demand);
    // Refused while anything is live.
    const r = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesLive, s.forgetResidents());
    s.release(r);
}

/// The 0b box probes' readings (the growth probe's and the transient release's): the footprint on either side of a
/// posix_spawn'd vm_stat, the settles, the report lines.
const ProbeBox = struct {
    const ar = @import("deepseek_v41_ar.zig");
    const status = @import("status.zig");

    pages: ar.VmStatPages,
    fp: u64,
    pm: status.ProcessMemory,

    const Self = @This();
    const Settled = struct { b: Self, ms: ?u64 };

    /// One moment's reading: the footprint on either side of a posix_spawn'd vm_stat, within the harnesses' bound.
    fn mark(buf: []u8) !@This() {
        var n: u32 = 0;
        while (n < ar.box_mark_attempts) : (n += 1) {
            if (n > 0) std.Io.sleep(testing.io, .fromMilliseconds(ar.box_mark_retry_ms), .awake) catch {};
            const f0 = status.footprint().now;
            const pages = try ar.vmStatPages(try ar.readVmStat(buf));
            const f1 = status.footprint().now;
            if (@max(f0, f1) - @min(f0, f1) <= ar.box_mark_stable_bytes) return .{ .pages = pages, .fp = @max(f0, f1), .pm = status.processMemory() };
        }
        return error.BoxMarkUnstable;
    }
    fn d(x: u64, y: u64) i64 {
        return @as(i64, @intCast(y)) - @as(i64, @intCast(x));
    }
    /// Physical growth outside the footprint since `b0`, the file-backed pages excluded (the guard credits the cache).
    fn outside(b0: @This(), b: @This()) i64 {
        return d(b0.pages.physical(), b.pages.physical()) - d(b0.pages.file_backed, b.pages.file_backed) - d(b0.fp, b.fp);
    }
    /// Marks every 50 ms until the growth outside the footprint since `b0` is within `limit`, at most `bound_ms`
    /// (ms: null when it never is).
    fn settle(b0: Self, buf: []u8, bound_ms: u64, limit: i64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        while (true) {
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (outside(b0, b) <= limit) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
        }
    }
    /// Marks every 50 ms until the footprint is at most `limit` above `ref`'s (a release measured by the process's own
    /// ledger, not only by the growth outside it), at most `bound_ms` (ms: null when it never is).
    fn settleFootprint(ref: Self, buf: []u8, bound_ms: u64, limit: i64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        while (true) {
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (d(ref.fp, b.fp) <= limit) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
        }
    }
    /// A baseline once wired and physical hold still across two marks 100 ms apart (an earlier step's pages still being
    /// retired would land inside the measured steps), at most `bound_ms` (ms: null when they never did).
    fn settled(buf: []u8, bound_ms: u64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        var prev = try mark(buf);
        const stable: u64 = ar.box_mark_stable_bytes;
        while (true) {
            std.Io.sleep(testing.io, .fromMilliseconds(100), .awake) catch {};
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (@abs(d(prev.pages.wired, b.pages.wired)) <= stable and @abs(d(prev.pages.physical(), b.pages.physical())) <= stable) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            prev = b;
        }
    }
    /// One more GPU command (a release the driver retires only at a later submission).
    fn nextCommand(s: mlx.mlx_stream) !void {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{1}, 1, .float32, s));
        try evalArrays(&.{z});
        _ = mlx.mlx_array_free(z);
    }
    fn msOf(x: ?u64, out: []u8) []const u8 {
        return if (x) |v| std.fmt.bufPrint(out, "{d}", .{v}) catch out[0..0] else "null";
    }
    fn line(b0: @This(), b: @This(), out: []u8) []const u8 {
        return std.fmt.bufPrint(out, "{{\"d_footprint\": {d}, \"d_physical\": {d}, \"d_wired\": {d}, \"d_file_backed\": {d}, \"d_graphics_nofootprint\": {d}, \"d_internal\": {d}, \"outside\": {d}}}", .{
            d(b0.fp, b.fp), d(b0.pages.physical(), b.pages.physical()), d(b0.pages.wired, b.pages.wired), d(b0.pages.file_backed, b.pages.file_backed),
            d(b0.pm.graphics_nofootprint, b.pm.graphics_nofootprint), d(b0.pm.internal, b.pm.internal), outside(b0, b),
        }) catch out[0..0];
    }
    const Payload = struct { m: []align(std.heap.page_size_min) u8 };
    fn dtor(ctx: ?*anyopaque) callconv(.c) void {
        const pl: *Payload = @ptrCast(@alignCast(ctx.?));
        std.posix.munmap(pl.m);
        std.heap.c_allocator.destroy(pl);
    }
    fn touch(m: []u8, step: usize) void {
        var i: usize = 0;
        while (i < m.len) : (i += step) @as(*volatile u8, &m[i]).* = 0;
    }
};

// DSV41_PHASE0B_MLX=1, inside a guarded window: SERVED13's kill (12 GB outside the footprint late in decode; the grow's
// wrapped rows its one new mechanism) at 2 GB: the box's pages around each step of the overlapped grow's mechanics,
// the release included (SERVED14's probe: the release left 2.15 GB wired outside the footprint).
test "dsv41 growth 0b: box probe: a no-copy wrap of 2 GB of touched anonymous pages, its first GPU read of every page and its release stay inside the footprint" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    _ = mlx.applyWiredPolicy();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bytes: usize = 2 << 30;
    const elems: c_int = @intCast(bytes / 2);
    const page = std.heap.pageSize();
    const Box = ProbeBox;
    var buf: [1 << 16]u8 = undefined;
    const base = try Box.settled(&buf, 3000);
    const b0 = base.b;
    // The overlapped grow's mechanics: an untouched private anonymous mapping, a helper's touch of every page.
    const m = try std.posix.mmap(null, bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    const helper = try std.Thread.spawn(.{}, Box.touch, .{ @as([]u8, m), page });
    helper.join();
    const b1 = try Box.mark(&buf);
    // The no-copy wrap (its deleter unmaps), a view of it and one eval.
    const pl = try std.heap.c_allocator.create(Box.Payload);
    pl.* = .{ .m = m };
    const arr = mlx.mlx_array_new_data_managed_payload(@ptrCast(m.ptr), &[_]c_int{elems}, 1, .int16, pl, Box.dtor);
    const no_copy = if (mlx.mlx_array_data_uint8(arr)) |p| @intFromPtr(p) == @intFromPtr(m.ptr) else false;
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, arr, &[_]c_int{ 1024, @divExact(elems, 1024) }, 2, s));
    try evalArrays(&.{view});
    const b2 = try Box.mark(&buf);
    // The first GPU read of every page: a sum over the whole array.
    var sum = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&sum, view, false, s));
    try evalArrays(&.{sum});
    const b3 = try Box.mark(&buf);
    // Released: the arrays freed (the wrap's deleter unmaps), synchronize, MLX's cache cleared, the box settled from the
    // post-touch state (2 s bound). Else one more GPU command and a second settle (1 s): a release the driver retires only
    // at a later submission. The first GPU read is judged from the post-touch state too.
    _ = mlx.mlx_array_free(sum);
    _ = mlx.mlx_array_free(view);
    _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    const limit: i64 = @intCast(bytes / 10);
    const r1 = try Box.settle(b1, &buf, 2000, limit);
    var r2: ?Box.Settled = null;
    if (r1.ms == null) {
        try Box.nextCommand(s);
        r2 = try Box.settle(b1, &buf, 1000, limit);
    }
    const released = r1.ms != null or (r2 != null and r2.?.ms != null);
    const out3 = Box.outside(b1, b3);
    const verdict = if (out3 > limit) "GrowWrapOutsideFootprint" else if (!released) "GrowWrapReleaseOutsideFootprint" else "inside";
    var l: [6][320]u8 = undefined;
    var ms: [3][24]u8 = undefined;
    std.debug.print("\nGROWTH_BOX_PROBE {{\"bytes\": {d}, \"no_copy\": {}, \"baseline_settle_ms\": {s}, \"touch\": {s}, \"wrap_eval\": {s}, \"first_gpu_read\": {s}, \"first_gpu_read_from_touch\": {s}, \"release_from_touch\": {s}, \"release_settle_ms\": {s}, \"after_next_command_from_touch\": {s}, \"after_next_command_settle_ms\": {s}, \"outside_limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        bytes, no_copy, Box.msOf(base.ms, &ms[2]), Box.line(b0, b1, &l[0]), Box.line(b0, b2, &l[1]), Box.line(b0, b3, &l[2]), Box.line(b1, b3, &l[5]), Box.line(b1, r1.b, &l[3]), Box.msOf(r1.ms, &ms[0]),
        if (r2) |x| Box.line(b1, x.b, &l[4]) else "null", if (r2) |x| Box.msOf(x.ms, &ms[1]) else "null", limit, verdict,
    });
    // Control (the growth's way back): an MLX-allocated array of the same bytes, written and read in full on the GPU,
    // released through the allocator (synchronize, cache cleared) from its own settled baseline; its release is the
    // footprint back within the limit of that baseline (reported beside the wrap's, not judged).
    const cbase = try Box.settled(&buf, 3000);
    const c0 = cbase.b;
    var za = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&za, &[_]c_int{elems}, 1, .int16, s));
    var zs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&zs, za, false, s));
    try evalArrays(&.{ za, zs });
    const c1 = try Box.mark(&buf);
    _ = mlx.mlx_array_free(zs);
    _ = mlx.mlx_array_free(za);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    const cr = try Box.settleFootprint(c0, &buf, 2000, limit);
    var cr2: ?Box.Settled = null;
    if (cr.ms == null) {
        try Box.nextCommand(s);
        cr2 = try Box.settleFootprint(c0, &buf, 1000, limit);
    }
    const c_released = cr.ms != null or (cr2 != null and cr2.?.ms != null);
    var cl: [3][320]u8 = undefined;
    var cms: [3][24]u8 = undefined;
    std.debug.print("\nGROWTH_BOX_PROBE_CONTROL {{\"bytes\": {d}, \"baseline_settle_ms\": {s}, \"written_read\": {s}, \"release\": {s}, \"release_footprint_settle_ms\": {s}, \"after_next_command\": {s}, \"after_next_command_footprint_settle_ms\": {s}, \"footprint_limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        bytes, Box.msOf(cbase.ms, &cms[2]), Box.line(c0, c1, &cl[0]), Box.line(c0, cr.b, &cl[1]), Box.msOf(cr.ms, &cms[0]),
        if (cr2) |x| Box.line(c0, x.b, &cl[2]) else "null", if (cr2) |x| Box.msOf(x.ms, &cms[1]) else "null", limit, if (c_released) "released" else "ControlReleaseKeptFootprint",
    });
    try testing.expect(no_copy);
    if (out3 > limit) return error.GrowWrapOutsideFootprint;
    if (!released) return error.GrowWrapReleaseOutsideFootprint;
}

// DSV41_PHASE0B_MLX=1 and DSV41_BANK=<bank dir>, inside a guarded window (SERVED16): the 240-row MLX scratch filled with
// records and read on the GPU, released (the allocator check) and cache-cleared back inside the footprint (the box
// probe's after-release rule); then decode's window 0 serves a route's records.
test "dsv41 stream 0b: the transient release frees the 240-row MLX scratch back inside the footprint, and window 0 serves decode" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    _ = mlx.applyWiredPolicy();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const depth = max_wide_depth;
    if (bank.n_experts < depth * max_route_ids) return error.TooFewExperts;
    const L: u32 = 0;
    const rows = try a.alloc(u32, bank.layers.len);
    defer a.free(rows);
    @memset(rows, 0);
    var buf: [1 << 16]u8 = undefined;
    const base = try ProbeBox.settled(&buf, 3000);
    const b0 = base.b;
    const s = try Stream.init(a, &bank, .{ .rows = rows, .transient_rows = depth * max_route_ids, .wide_depth = depth, .slot_memory = .{ .mlx = stream }, .transient_release = true });
    defer s.deinit();
    // The prompt's wide reads: five live routes of layer L fill every window with records (no persistent rows).
    var live: [depth]*Route = undefined;
    for (&live, 0..) |*r, w| {
        var ids: [max_route_ids]u16 = undefined;
        for (&ids, 0..) |*e, i| e.* = @intCast(w * max_route_ids + i);
        r.* = try serve(s, L, &ids);
        try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
    }
    // The GPU reads every row, as the prompt's waves do.
    var sums: [n_components]mlx.mlx_array = @splat(.{});
    for (&sums, s.transient.backing.mlx.arrays) |*x, arr| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum(x, arr, false, stream));
    }
    try evalArrays(&sums);
    for (sums) |x| _ = mlx.mlx_array_free(x);
    for (live) |r| s.release(r);
    _ = mlx.mlx_synchronize(stream);
    const b1 = try ProbeBox.mark(&buf);
    // The release (its allocator check), synchronize, MLX's cache cleared; from the post-fill state (2 s bounds; else one
    // more GPU command and 1 s): the footprint falls by the freed scratch (to within 10 %), and the growth outside the
    // footprint does not rise past 10 % of it.
    var active: [2]usize = .{ 0, 0 };
    _ = mlx.mlx_get_active_memory(&active[0]);
    const freed = try s.releaseTransient();
    _ = mlx.mlx_get_active_memory(&active[1]);
    try testing.expectEqual(transientBytes(s, depth * max_route_ids), freed);
    _ = mlx.mlx_synchronize(stream);
    _ = mlx.mlx_clear_cache();
    const limit: i64 = @intCast(freed / 10);
    const fp_limit: i64 = limit - @as(i64, @intCast(freed));
    const f1 = try ProbeBox.settleFootprint(b1, &buf, 2000, fp_limit);
    const r1 = try ProbeBox.settle(b1, &buf, 2000, limit);
    var r2: ?ProbeBox.Settled = null;
    var f2: ?ProbeBox.Settled = null;
    if (f1.ms == null or r1.ms == null) {
        try ProbeBox.nextCommand(stream);
        f2 = try ProbeBox.settleFootprint(b1, &buf, 1000, fp_limit);
        r2 = try ProbeBox.settle(b1, &buf, 1000, limit);
    }
    const released = r1.ms != null or (r2 != null and r2.?.ms != null);
    const kept = !(f1.ms != null or (f2 != null and f2.?.ms != null));
    // Decode: window 0 and layer L's four rows; the route's misses past them land in window 0 and hold their records.
    rows[L] = 4;
    try s.grow(rows);
    const b2 = try ProbeBox.mark(&buf);
    var ids: [max_route_ids]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(bank.n_experts - 1 - i);
    const r = try serve(s, L, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    const geom = &bank.layers[L];
    var in_window0 = true;
    for (r.plan.loadsOf()) |ld| {
        in_window0 = in_window0 and ld.slot < s.layers[L].policy.capacity + max_route_ids;
        const dg = slotDigest(s, L, ld.slot, geom);
        try testing.expectEqualSlices(u8, &bank.digest(L, ld.expert).logical, &dg);
    }
    s.release(r);
    try s.flush();
    const verdict = if (kept) "TransientReleaseKeptFootprint" else if (!released) "TransientReleaseOutsideFootprint" else "inside";
    var l: [5][320]u8 = undefined;
    var ms: [5][24]u8 = undefined;
    std.debug.print("\nTRANSIENT_RELEASE_PROBE {{\"transient_rows\": {d}, \"freed_bytes\": {d}, \"d_active\": {d}, \"window0_rows\": {d}, \"baseline_settle_ms\": {s}, \"filled\": {s}, \"release_from_fill\": {s}, \"footprint_settle_ms\": {s}, \"outside_settle_ms\": {s}, \"after_next_command_from_fill\": {s}, \"after_next_command_footprint_settle_ms\": {s}, \"after_next_command_outside_settle_ms\": {s}, \"grown\": {s}, \"limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        depth * max_route_ids, freed, active[0] -| active[1], s.transient.rows, ProbeBox.msOf(base.ms, &ms[4]), ProbeBox.line(b0, b1, &l[0]), ProbeBox.line(b1, r1.b, &l[1]), ProbeBox.msOf(f1.ms, &ms[0]), ProbeBox.msOf(r1.ms, &ms[1]),
        if (r2) |x| ProbeBox.line(b1, x.b, &l[2]) else "null", if (f2) |x| ProbeBox.msOf(x.ms, &ms[2]) else "null", if (r2) |x| ProbeBox.msOf(x.ms, &ms[3]) else "null", ProbeBox.line(b0, b2, &l[3]), limit, verdict,
    });
    try testing.expect(in_window0);
    if (kept) return error.TransientReleaseKeptFootprint;
    if (!released) return error.TransientReleaseOutsideFootprint;
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

/// The bytes of `rows` rows of the widest layer's record (the transient scratch's row).
fn transientBytes(s: *const Stream, rows: u64) u64 {
    var n: u64 = 0;
    for (s.bank.layers[s.transient_layer].segments) |seg| n += seg.length;
    return rows * n;
}

test "dsv41 stream: the transient release frees the whole scratch with nothing live or held, and the grow allocates decode's window 0" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 5 * 12, .wide_depth = 5, .pool = test_pool, .transient_release = true });
    defer s.deinit();
    // The prompt: two live routes of layer 0 (windows 0 and 1), the first one's persistent slots held for a deferred call.
    const r0 = try serve(s, 0, &.{ 1, 2, 3, 4, 5, 6 });
    const r1 = try serve(s, 0, &.{ 7, 8, 9, 10, 11, 12 });
    try testing.expectEqual(@as(u8, 1), r1.window);
    try testing.expectError(error.RoutesLive, s.releaseTransient());
    try s.holdBase(r0);
    s.release(r0);
    s.release(r1);
    try testing.expectError(error.RoutesLive, s.releaseTransient());
    s.releaseHeld();
    // The grow needs the release first (the bill counts it).
    try testing.expectError(error.TransientNotReleased, s.grow(&.{ 6, 6 }));
    try testing.expectEqual(transientBytes(s, 5 * 12), try s.releaseTransient());
    try testing.expectEqual(@as(u32, 0), s.transient.rows);
    try testing.expectEqual(@as(usize, 0), s.transient_meta.len);
    try testing.expectEqual(@as(u8, 1), s.wide_depth);
    try testing.expectError(error.TransientAlreadyReleased, s.releaseTransient());
    const Off = struct {
        fn release(st: *Stream, out: *?anyerror) void {
            _ = st.releaseTransient() catch |e| {
                out.* = e;
                return;
            };
            out.* = null;
        }
    };
    var got: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, Off.release, .{ s, &got });
    t.join();
    try testing.expectEqual(@as(?anyerror, error.NotInferenceThread), got);
    try s.grow(&.{ 6, 6 });
    try testing.expect(!s.transient_released);
    try testing.expectEqual(@as(u32, 12 + decode_staging_rows), s.transient.rows);
    try testing.expectEqual(@as(usize, 12 + decode_staging_rows), s.transient_meta.len);
    for (s.transient_meta) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
    try testing.expectError(error.AlreadyGrown, s.releaseTransient());
    // Decode: layer 1's misses past its six rows land in window 0 and serve their records.
    const ids = [_]u16{ 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };
    const r = try serve(s, 1, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    for (r.plan.loadsOf()) |l| try testing.expect(l.slot < s.layers[1].policy.capacity + 12);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
}

test "dsv41 stream: after the transient release, decode with the served lookahead, pre-reads and gates loads only window 0" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // The served decode configuration (Lookahead{}: k 8, tau inf, budget 2, 4 chunks, pre-read; event gates) at depth 5.
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = la_pool, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = true });
    defer s.deinit();
    // The prompt fills all five windows of layer 0 (five live routes), then releases them.
    var live: [5]*Route = undefined;
    for (&live, 0..) |*r, w| {
        var pids: [6]u16 = undefined;
        for (&pids, 0..) |*e, i| e.* = @intCast(w * 6 + i);
        r.* = try serve(s, 0, &pids);
        try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
    }
    for (live) |r| s.release(r);
    _ = try s.releaseTransient();
    try s.grow(&.{ 6, 4 });
    try testing.expectEqual(@as(u32, max_route_ids + decode_staging_rows), s.transient.rows);
    // Verify forwards of 1..8 rows x top-6; layer 0's scores favour layer 1's next ids, so its reads are claimed.
    var rng = std.Random.DefaultPrng.init(1616);
    const rand = rng.random();
    var ids: [2][48]u16 = undefined;
    var scores: [8 * 32]f32 = undefined;
    for (0..40) |_| {
        const m = [2]usize{ rand.intRangeAtMost(usize, 1, 8), rand.intRangeAtMost(usize, 1, 8) };
        for (0..2) |layer| for (ids[layer][0 .. 6 * m[layer]]) |*e| {
            e.* = rand.intRangeLessThan(u16, 0, 32);
        };
        for (0..m[0]) |row| for (scores[row * 32 ..][0..32], 0..) |*v, e| {
            v.* = if (std.mem.indexOfScalar(u16, ids[1][0 .. 6 * m[1]], @intCast(e)) != null) 1 + rand.float(f32) else rand.float(f32) / 2;
        };
        for (0..2) |layer| {
            const n = 6 * m[layer];
            const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m[0]] else &.{};
            const r = try serveGated(s, @intCast(layer), ids[layer][0..n], pred);
            // No other route is live or awaiting its flush, so every load (demand, pre-read bound at submit, or
            // adopted from the speculative staging) lands in a persistent row or window 0.
            try testing.expectEqual(@as(u8, 0), r.window);
            try testing.expectEqual(@as(u8, 0), s.n_released);
            var n_live: u32 = 0;
            for (&s.routes) |*o| n_live += @intFromBool(o.state != .free);
            try testing.expectEqual(@as(u32, 1), n_live);
            const cap = s.layers[layer].policy.capacity;
            for (r.plan.loadsOf()) |l| try testing.expect(l.slot < cap + max_route_ids);
            try expectServed(s, &sb, r, ids[layer][0..n]);
            s.release(r);
        }
    }
    try s.flush();
    const st = s.stats();
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0 and (st.claimed > 0 or st.adopt_ranges > 0));
}

test "dsv41 stream: three requests: each prompt, phase change, decode and reverse change serve every slot's bytes; the reverse change restores the prompt configuration (a cancelled request's routes included)" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = la_pool, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = true });
    defer s.deinit();
    const prompt_bytes = s.promptTransientBytes();
    try testing.expectEqual(transientBytes(s, 5 * max_route_ids), prompt_bytes);
    var rng = std.Random.DefaultPrng.init(777);
    const rand = rng.random();
    for (0..3) |req| {
        // The prompt: five live routes of layer 0 in their windows (the same prompt every request), layer 1 after.
        try testing.expectEqual(Phase.prefill, s.phase);
        try testing.expectEqual(@as(u8, 5), s.wide_depth);
        var live: [5]*Route = undefined;
        for (&live, 0..) |*r, w| {
            var pids: [6]u16 = undefined;
            for (&pids, 0..) |*e, i| e.* = @intCast(w * 6 + i);
            r.* = try serve(s, 0, &pids);
            try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
            try expectServed(s, &sb, r.*, &pids);
        }
        for (live) |r| s.release(r);
        _ = try s.releaseTransient();
        try s.grow(&.{ 10, 9 });
        var ids: [48]u16 = undefined;
        for (0..12) |_| for (0..2) |layer| {
            const n = 6 * rand.intRangeAtMost(usize, 1, 8);
            for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, 32);
            const r = try serveGated(s, @intCast(layer), ids[0..n], &.{});
            try expectServed(s, &sb, r, ids[0..n]);
            s.release(r);
        };
        // Request 1 is cancelled mid-forward: a route stays live (never released) into the reverse change.
        if (req == 1) _ = try serveGated(s, 1, &.{ 3, 4, 5, 6, 7, 8 }, &.{});
        // Freed: the grown rows (6 + 5) and decode's window 0.
        const record = prompt_bytes / (5 * max_route_ids);
        try testing.expectEqual((6 + 5 + max_route_ids + decode_staging_rows) * record, try s.shrink(&.{ 4, 4 }));
        try testing.expectEqual(Phase.prefill, s.phase);
        try testing.expect(s.transient_released);
        for (s.layers) |*ls| {
            try testing.expectEqual(@as(u32, 4), ls.policy.capacity);
            // Residents forgotten (the next prompt's schedule equals the first's).
            try testing.expectEqual(@as(u32, 0), ls.policy.occupancy);
            try testing.expect(ls.ext == null);
            for (ls.meta[4..]) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
            for (ls.meta[0..4]) |m| try testing.expectEqual(@as(u16, 0), m.pins);
        }
        for (&s.routes) |*r| try testing.expect(r.state == .free);
        try testing.expectError(error.NotGrown, s.shrink(&.{ 4, 4 }));
        try testing.expectEqual(prompt_bytes, try s.regrowTransient());
        try testing.expectEqual(@as(u32, 5 * max_route_ids), s.transient.rows);
        for (s.transient_meta) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
        try testing.expectError(error.TransientNotReleased, s.regrowTransient());
    }
    try testing.expectEqual(@as(u64, 0), s.stats().gates_forced);
}

test "dsv41 stream: a request cancelled in its prompt phase: its live and held routes are settled at its end, and the next prompt routes" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 2 * 12, .wide_depth = 2, .pool = test_pool, .transient_release = true });
    defer s.deinit();
    const r0 = try serve(s, 0, &.{ 1, 2, 3, 4, 5, 6 });
    try s.holdBase(r0);
    s.release(r0);
    _ = try serve(s, 0, &.{ 7, 8, 9, 10, 11, 12 }); // never released: the cancel
    try s.settleRoutes();
    for (&s.routes) |*r| try testing.expect(r.state == .free);
    try testing.expectEqual(@as(usize, 0), s.held_base.items.len);
    for (s.layers[0].meta) |m| try testing.expectEqual(@as(u16, 0), m.pins);
    const ids = [_]u16{ 1, 7, 13, 14, 15, 16 };
    const r = try serve(s, 0, &ids);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
}

/// One replay of the phase-2 fixture's layers 13 and 14 (served lookahead 8:inf:2, 4 chunks, pre-read, gates) at the
/// served depth 5, with the transient release on its route or not: each decode route's signature (layer, window, hits,
/// loads with their slots and reads) into `sigs`, every served slot sha256-checked; returns the stream's Stats.
fn replayLookahead(a: std.mem.Allocator, bank: *const expert_bank.Bank, f: anytype, sfd: std.c.fd_t, release: bool, sigs: *std.ArrayList(u64)) !Stats {
    const L: u32 = 13;
    var rows: [40]u32 = @splat(0);
    rows[L] = 3;
    rows[L + 1] = 3;
    const s = try Stream.init(a, bank, .{ .rows = &rows, .max_route_ids = 6, .transient_rows = 5 * 6, .wide_depth = 5, .lookahead = .{}, .event = .{}, .transient_release = release });
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
    if (release) _ = try s.releaseTransient();
    try s.grow(&rows);
    var routes: u64 = 0;
    var score_row: [384]f32 = undefined;
    var raw: [384 * 4]u8 = undefined;
    var at: u64 = 0;
    var cycle: usize = 0;
    outer: while (cycle < f.rows.len) : (cycle += 1) {
        const m = f.rows[cycle];
        const c13 = f.calls[cycle * f.layers + L];
        const c14 = f.calls[cycle * f.layers + L + 1];
        for (0..m) |r| {
            if (routes >= 120) break :outer;
            const off = (at + L * m + r) * 384 * 4;
            if (std.c.pread(sfd, &raw, raw.len, @intCast(off)) != raw.len) return error.ShortRead;
            for (&score_row, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
            for ([_]struct { l: u32, ids: []const u16, scores: []const f32 }{
                .{ .l = L, .ids = c13.ids[6 * r ..][0..6], .scores = &score_row },
                .{ .l = L + 1, .ids = c14.ids[6 * r ..][0..6], .scores = &.{} },
            }) |call| {
                const rt = try serveGated(s, call.l, call.ids, call.scores);
                var h = std.hash.Wyhash.init(call.l);
                h.update(std.mem.asBytes(&rt.window));
                h.update(std.mem.asBytes(&rt.plan.n_hits));
                for (rt.plan.loadsOf(), rt.reads[0..rt.plan.n_loads]) |ld, rd| {
                    h.update(std.mem.asBytes(&ld.expert));
                    h.update(std.mem.asBytes(&ld.slot));
                    h.update(std.mem.asBytes(&ld.persistent));
                    h.update(std.mem.asBytes(&rd));
                }
                try sigs.append(a, h.final());
                for (rt.plan.slotsOf(), call.ids) |slot, e| {
                    const d = slotDigest(s, call.l, slot, geom);
                    try testing.expectEqualSlices(u8, &bank.digest(call.l, e).logical, &d);
                }
                routes += 1;
                s.release(rt);
            }
        }
        at += @as(u64, m) * (f.layers - 1);
    }
    try s.flush();
    return s.stats();
}

// DSV41_BANK=<bank dir> DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py>: SERVED16's decode
// read regression against the release. The release frees only the transient scratch, so the decode routes and reads
// must be the same with and without it: every route's plan and reads, and the read pool's lookahead and pre-read counts.
test "dsv41 stream: the recorded lookahead trace on the real bank routes and reads the same with and without the transient release" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
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
    var pbuf: [1024]u8 = undefined;
    const spath = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}", .{ std.fs.path.dirname(fixture) orelse ".", parsed.value.scores_file }, 0);
    const sfd = std.c.open(spath.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (sfd < 0) return error.OpenFailed;
    defer _ = std.c.close(sfd);
    var sig: [2]std.ArrayList(u64) = .{ .empty, .empty };
    defer for (&sig) |*x| x.deinit(a);
    var st: [2]Stats = undefined;
    var ms: [2]i64 = undefined;
    for (0..2) |i| {
        const t0 = std.Io.Timestamp.now(io, .boot);
        st[i] = try replayLookahead(a, &bank, parsed.value, sfd, i == 1, &sig[i]);
        ms[i] = @intCast(@divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms));
    }
    for (st, ms, [_][]const u8{ "release off", "release on" }) |x, t, name| std.debug.print(
        "replay ({s}): {d} routes; misses {d} hits {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv; lookahead: issued {d} landed {d} claimed {d} adopted {d} ranges / {d} B, spec {d} B; pre-read issued {d} served {d} expired {d}; gates {d} forced {d}; {d} ms\n",
        .{ name, x.route_calls, x.expert_cache_misses, x.expert_cache_hits, x.expert_cache_evictions, x.persistent_loads, x.transient_loads, x.loads_skipped, x.expert_bytes_read, x.preadv_calls, x.spec_issued, x.spec_landed, x.claimed, x.adopt_ranges, x.adopt_bytes, x.spec_bytes, x.pre_issued, x.pre_served, x.pre_expired, x.gates, x.gates_forced, t },
    );
    // The same routes, plans and reads (window 0 throughout), and the same reads issued, claimed and adopted.
    try testing.expectEqualSlices(u64, sig[0].items, sig[1].items);
    inline for (.{ "route_calls", "expert_cache_misses", "expert_cache_hits", "expert_cache_evictions", "persistent_loads", "transient_loads", "loads_skipped", "expert_bytes_read", "preadv_calls", "spec_issued", "claimed", "pre_issued", "gates" }) |k|
        try testing.expectEqual(@field(st[0], k), @field(st[1], k));
    try testing.expectEqual(@as(u64, 0), st[0].gates_forced + st[1].gates_forced);
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
