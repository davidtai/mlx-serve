//! An MLX IO reader whose reads bypass the page cache. MLX's own safetensors
//! reader is a plain open + pread, so a load leaves the file's pages cached
//! next to the array buffers. This one opens the file with F_NOCACHE and
//! read-ahead off; `mlx_load_safetensors_reader` reads the header through
//! `read` and each tensor through `read_at_offset` into the array's buffer.
//! A failed read panics naming the file, as MLX's own reader throws.

const std = @import("std");
const status = @import("status.zig");
const mlx = @import("mlx");
const io_util = @import("io_util");

/// One read's staging buffer: page-aligned (the page allocator), a multiple of every page size.
const stage_bytes: usize = 8 << 20;

/// The reader's state (MLX owns it once handed over; `free` releases it).
/// Allocated with the C allocator: MLX may free it from an IO thread.
pub const Desc = struct {
    fd: std.c.fd_t,
    size: u64,
    /// The sequential position (`read`, `seek`, `tell`: the header parse).
    pos: u64 = 0,
    label: [:0]u8,

    /// `path` read-only (symlinked blobs allowed), F_NOCACHE, read-ahead off.
    pub fn open(path: [:0]const u8) !*Desc {
        const fd = io_util.openNoCache(path.ptr, .{}) catch |e| return if (e == error.NoCacheFcntl or e == error.NoCacheUnsupported) e else error.NoCacheOpen;
        errdefer _ = std.c.close(fd);
        // The size by lseek, as qwen4_exp's table does (std.c.Stat is void on Linux).
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 0) return error.NoCacheStat;
        const a = std.heap.c_allocator;
        const d = try a.create(Desc);
        errdefer a.destroy(d);
        d.* = .{ .fd = fd, .size = @intCast(size), .label = try a.dupeSentinel(u8, path, 0) };
        return d;
    }

    pub fn close(d: *Desc) void {
        _ = std.c.close(d.fd);
        std.heap.c_allocator.free(d.label);
        std.heap.c_allocator.destroy(d);
    }

    /// `buf.len` bytes at `off`, through a page-aligned staging buffer. macOS honours F_NOCACHE only
    /// for page-aligned reads (the file offset, the length and the destination): an unaligned read goes
    /// through the unified buffer cache and leaves its pages cached (speculative pages, which the
    /// process's footprint does not count until the kernel ages them under pressure). Measured on resident
    /// shards: 0.19 GB cached per 0.54 GB read unaligned, none aligned; MLX's tensor loads arrive
    /// unaligned (safetensors offsets, MLX buffers). pread is safe from MLX's IO threads: each call
    /// owns its stage.
    pub fn readAt(d: *const Desc, buf: []u8, off: u64) void {
        if (buf.len == 0) return;
        const page: u64 = std.heap.pageSize();
        const stage = std.heap.page_allocator.alloc(u8, stage_bytes) catch
            std.debug.panic("nocache reader: {s}: no {d} B staging buffer", .{ d.label, stage_bytes });
        defer std.heap.page_allocator.free(stage);
        var pos = off;
        const end = off + buf.len;
        var out: usize = 0;
        while (pos < end) {
            const a0 = pos - pos % page;
            const want_end = @min(end, a0 + stage_bytes);
            const need: usize = @intCast(want_end - a0);
            const len = std.mem.alignForward(usize, need, @intCast(page));
            var got: usize = 0;
            while (got < need) {
                const n = std.c.pread(d.fd, stage[got..].ptr, len - got, @intCast(a0 + got));
                if (n < 0) {
                    if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                    std.debug.panic("nocache reader: {s}: pread of {d} B at {d} failed, errno {d}", .{ d.label, len - got, a0 + got, std.c._errno().* });
                }
                if (n == 0) std.debug.panic("nocache reader: {s}: short read at {d} ({d} B file)", .{ d.label, a0 + got, d.size });
                got += @intCast(n);
            }
            const s0: usize = @intCast(pos - a0);
            @memcpy(buf[out..][0 .. need - s0], stage[s0..need]);
            out += need - s0;
            pos = want_end;
        }
    }
};

fn of(ctx: ?*anyopaque) *Desc {
    return @ptrCast(@alignCast(ctx.?));
}

fn isOpen(ctx: ?*anyopaque) callconv(.c) bool {
    return of(ctx).fd >= 0;
}

fn good(ctx: ?*anyopaque) callconv(.c) bool {
    return of(ctx).fd >= 0;
}

fn tell(ctx: ?*anyopaque) callconv(.c) usize {
    return @intCast(of(ctx).pos);
}

fn seek(ctx: ?*anyopaque, off: i64, whence: c_int) callconv(.c) void {
    const d = of(ctx);
    const base: i64 = switch (whence) {
        0 => 0, // SEEK_SET (std::ios_base::beg)
        1 => @intCast(d.pos), // SEEK_CUR
        2 => @intCast(d.size), // SEEK_END
        else => std.debug.panic("nocache reader: {s}: seek whence {d}", .{ d.label, whence }),
    };
    d.pos = @intCast(base + off);
}

fn read(ctx: ?*anyopaque, data: [*]u8, n: usize) callconv(.c) void {
    const d = of(ctx);
    d.readAt(data[0..n], d.pos);
    d.pos += n;
}

fn readAtOffset(ctx: ?*anyopaque, data: [*]u8, n: usize, off: usize) callconv(.c) void {
    of(ctx).readAt(data[0..n], off);
}

fn write(ctx: ?*anyopaque, _: [*]const u8, _: usize) callconv(.c) void {
    std.debug.panic("nocache reader: {s}: write on a reader", .{of(ctx).label});
}

fn label(ctx: ?*anyopaque) callconv(.c) [*:0]const u8 {
    return of(ctx).label.ptr;
}

fn free(ctx: ?*anyopaque) callconv(.c) void {
    of(ctx).close();
}

pub const vtable: mlx.mlx_io_vtable = .{
    .is_open = isOpen,
    .good = good,
    .tell = tell,
    .seek = seek,
    .read = read,
    .read_at_offset = readAtOffset,
    .write = write,
    .label = label,
    .free = free,
};

/// An MLX reader over `path` past the page cache; MLX frees it (`free`) once
/// the last array it loaded no longer needs it (`mlx_io_reader_free` drops the
/// caller's reference).
pub fn reader(path: [:0]const u8) !mlx.mlx_io_reader {
    const d = try Desc.open(path);
    return mlx.mlx_io_reader_new(d, vtable);
}

/// The aligned reads, the row gather and the residency probe are shared I/O helpers (io_util).
pub const readAligned = io_util.readAligned;
pub const RowGather = io_util.RowGather;
pub const residentBytes = io_util.residentBytes;

// ── Tests (host: no MLX array; the reader's callbacks driven as MLX drives them) ──

const testing = std.testing;

/// A safetensors image: 8-byte LE header length, the JSON header, the data.
fn writeSafetensors(a: std.mem.Allocator, tmp: *std.testing.TmpDir, name: []const u8, tensors: []const struct { name: []const u8, bytes: []const u8 }) !void {
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    try header.writer.writeAll("{");
    var off: usize = 0;
    for (tensors, 0..) |t, i| {
        if (i > 0) try header.writer.writeAll(",");
        try header.writer.print("\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ t.name, t.bytes.len, off, off + t.bytes.len });
        off += t.bytes.len;
    }
    try header.writer.writeAll("}");
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, header.written().len, .little);
    try image.appendSlice(a, &len);
    try image.appendSlice(a, header.written());
    for (tensors) |t| try image.appendSlice(a, t.bytes);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = image.items });
}

/// A safetensors file's tensor spans, parsed through the reader's `read` / `seek` / `tell` as MLX does.
const Span = struct { off: u64, len: u64 };
fn spansThroughReader(a: std.mem.Allocator, ctx: ?*anyopaque) ![]Span {
    seek(ctx, 0, 0);
    var len_bytes: [8]u8 = undefined;
    read(ctx, &len_bytes, 8);
    const hlen = std.mem.readInt(u64, &len_bytes, .little);
    const text = try a.alloc(u8, @intCast(hlen));
    defer a.free(text);
    read(ctx, text.ptr, text.len);
    if (tell(ctx) != 8 + hlen) return error.TellMismatch;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(a);
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const offs = kv.value_ptr.object.get("data_offsets").?.array.items;
        const lo: u64 = @intCast(offs[0].integer);
        const hi: u64 = @intCast(offs[1].integer);
        try out.append(a, .{ .off = 8 + hlen + lo, .len = hi - lo });
    }
    return out.toOwnedSlice(a);
}

fn plainPread(fd: std.c.fd_t, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

/// Every tensor read through the reader equals the plain pread of its span (what MLX's own reader copies).
fn compareTensors(a: std.mem.Allocator, path: [:0]const u8) !struct { n: usize, bytes: u64 } {
    const d = try Desc.open(path);
    defer d.close();
    const spans = try spansThroughReader(a, d);
    defer a.free(spans);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var bytes: u64 = 0;
    for (spans) |sp| {
        const x = try a.alloc(u8, @intCast(sp.len));
        defer a.free(x);
        const y = try a.alloc(u8, @intCast(sp.len));
        defer a.free(y);
        readAtOffset(d, x.ptr, x.len, @intCast(sp.off));
        try plainPread(fd, y, sp.off);
        try testing.expectEqualSlices(u8, y, x);
        bytes += sp.len;
    }
    return .{ .n = spans.len, .bytes = bytes };
}

test "dsv41 nocache reader: a safetensors file's header and tensors read through the reader equal the plain reads" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var big: [70_001]u8 = undefined;
    for (&big, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
    try writeSafetensors(a, &tmp, "w.safetensors", &.{
        .{ .name = "a", .bytes = "abc" },
        .{ .name = "big", .bytes = &big },
        .{ .name = "z", .bytes = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 } },
    });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/w.safetensors", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    const r = try compareTensors(a, path);
    try testing.expectEqual(@as(usize, 3), r.n);
    try testing.expectEqual(@as(u64, 3 + big.len + 9), r.bytes);
    // The reader's other callbacks, as MLX calls them.
    const d = try Desc.open(path);
    defer d.close();
    try testing.expect(isOpen(d) and good(d));
    try testing.expectEqualStrings(path, std.mem.span(label(d)));
    seek(d, -9, 2);
    var tail: [9]u8 = undefined;
    read(d, &tail, 9);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &tail);
    try testing.expectEqual(@as(usize, @intCast(d.size)), tell(d));
    try testing.expectError(error.NoCacheOpen, Desc.open("/nonexistent/w.safetensors"));
}

// DSV41_BANK=<bank>: a small real shard's tensors through the reader equal the plain reads.
test "dsv41 nocache reader: a real resident shard's tensors through the reader equal the plain reads" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var pbuf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/model-00003.safetensors", .{bank}, 0);
    const r = try compareTensors(a, path);
    std.debug.print("nocache reader: {s}: {d} tensors, {d} B byte-identical to the plain reads\n", .{ path, r.n, r.bytes });
}

test "dsv41 nocache reader: the row gather reads whole aligned pages, each row once, and scatters them in the caller's order" {
    const a = testing.allocator;
    // 300 rows of 10,240 B (not a page multiple) after a 1,000 B prefix (unaligned), the embedding's row shape.
    const base: u64 = 1000;
    const rb: usize = 10240;
    const n_rows: usize = 300;
    const image = try a.alloc(u8, base + n_rows * rb);
    defer a.free(image);
    for (image[0..base], 0..) |*b, i| b.* = @truncate(i *% 5);
    for (0..n_rows) |r| for (0..rb) |k| {
        image[base + r * rb + k] = @truncate(r *% 131 +% k *% 7 +% 3);
    };
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(testing.io, .{ .sub_path = "rows.bin", .data = image });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/rows.bin", .{root[0..try td.dir.realPath(testing.io, &root)]}, 0);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    // The serial path's bytes: one plain pread per row (what gatherRaw read before), on its own descriptor.
    const plain = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    defer _ = std.c.close(plain);
    // Unsorted ids with repeats, the first and the last row (the file ends inside the last row's page).
    var rng = std.Random.DefaultPrng.init(0x5eed_e3b);
    var ids: [200]u32 = undefined;
    for (&ids) |*d| d.* = rng.random().uintLessThan(u32, n_rows);
    ids[3] = 0;
    ids[7] = n_rows - 1;
    ids[8] = ids[2];
    ids[150] = ids[2];
    for ([_]usize{ 0, 7 }) |helpers| {
        // Pieces of 64 ids (a 200-id call runs 4); the caller alone below 8.
        const rg = try RowGather.init(fd, base, rb, n_rows, helpers, 64, 8);
        defer rg.deinit();
        const out = try a.alloc(u8, ids.len * rb);
        defer a.free(out);
        @memset(out, 0xAA);
        try rg.gather(&ids, out);
        const want = try a.alloc(u8, rb);
        defer a.free(want);
        for (ids, 0..) |r, i| {
            try testing.expectEqual(@as(isize, @intCast(rb)), std.c.pread(plain, want.ptr, rb, @intCast(base + r * rb)));
            try testing.expectEqualSlices(u8, want, out[i * rb ..][0..rb]);
        }
        // The last piece's runs: whole aligned pages, each distinct row in exactly one.
        var rows_in_runs: usize = 0;
        for (rg.runs[0..rg.n_runs]) |run| {
            try testing.expect(rg.aligned(run, rg.stages[0]));
            var k = run.first;
            while (k < run.end) : (k += 1) rows_in_runs += @intFromBool(k == run.first or rg.items[k].id != rg.items[k - 1].id);
        }
        var distinct: usize = 0;
        for (rg.items[0 .. ids.len - 3 * 64], 0..) |it, k| distinct += @intFromBool(k == 0 or it.id != rg.items[k - 1].id);
        try testing.expectEqual(distinct, rows_in_runs);
        // One row (the caller alone), and the refusals by name.
        var one: [10240]u8 = undefined;
        try rg.gather(&.{n_rows - 1}, &one);
        try testing.expectEqualSlices(u8, image[base + (n_rows - 1) * rb ..][0..rb], &one);
        try testing.expectError(error.RowOutOfRange, rg.gather(&.{@intCast(n_rows)}, &one));
        try testing.expectError(error.GatherShape, rg.gather(&.{ 0, 1 }, &one));
        // An unaligned run is refused by the reader's check (never planned: `plan` aligns by construction).
        try testing.expect(!rg.aligned(.{ .off = 1000, .len = 16384, .need = 10240, .first = 0, .end = 1 }, rg.stages[0]));
        try testing.expect(!rg.aligned(.{ .off = 0, .len = 12288, .need = 10240, .first = 0, .end = 1 }, rg.stages[0]) or std.heap.pageSize() == 4096);
    }
    try testing.expectEqual(@as(u64, 16 * RowGather.stage_len + 1024 * 40), RowGather.persistentBytes(15, 1024));
}

test "dsv41 nocache reader: MLX's reader handle owns the descriptor and frees it; reads past one staging buffer equal the plain reads" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // Past one 8 MiB stage: a read crossing it, both ends unaligned.
    const image = try a.alloc(u8, stage_bytes + 40_000);
    defer a.free(image);
    for (image, 0..) |*b, i| b.* = @truncate(i *% 40503 >> 5);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "big.bin", .data = image });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/big.bin", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    const d = try Desc.open(path);
    try testing.expectEqual(@as(u64, image.len), d.size);
    const out = try a.alloc(u8, stage_bytes + 20_001);
    defer a.free(out);
    readAtOffset(d, out.ptr, out.len, 12_345);
    try testing.expectEqualSlices(u8, image[12_345..][0..out.len], out);
    // Zero bytes is a no-op; SEEK_CUR moves from the position.
    d.readAt(out[0..0], 0);
    seek(d, 100, 0);
    seek(d, 23, 1);
    try testing.expectEqual(@as(usize, 123), tell(d));
    // MLX frees the reader through the vtable (`free` closes and releases it).
    free(d);
    const r = try reader(path);
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_io_reader_free(r));
    try testing.expectError(error.NoCacheOpen, reader("/nonexistent/cov-d/w.safetensors"));
}
