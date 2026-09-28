//! Slot rows the read pool fills: one bank per record component, `rows` rows
//! of that component's segment length, handed to the pool as nine destination
//! addresses per row. `LayerSlotBank` is the MLX-owned form the kernels bind;
//! `HostSlotRows` is the same layout in host pages (no MLX, no Metal).

const std = @import("std");
const mlx = @import("mlx.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("expert_io.zig");

const n_components = expert_bank.n_components;
const Component = expert_bank.Component;
const Layer = expert_bank.Layer;

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
