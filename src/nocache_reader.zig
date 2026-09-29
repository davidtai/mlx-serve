//! An MLX IO reader whose reads bypass the page cache. MLX's own safetensors
//! reader is a plain open + pread, so a load leaves the file's pages cached
//! next to the array buffers. This one opens the file with F_NOCACHE and
//! read-ahead off; `mlx_load_safetensors_reader` reads the header through
//! `read` and each tensor through `read_at_offset` into the array's buffer.
//! A failed read panics naming the file, as MLX's own reader throws.

const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");

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

    /// `buf.len` bytes at `off` (pread is safe from MLX's IO threads).
    pub fn readAt(d: *const Desc, buf: []u8, off: u64) void {
        var done: usize = 0;
        while (done < buf.len) {
            const n = std.c.pread(d.fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
            if (n < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                std.debug.panic("nocache reader: {s}: pread of {d} B at {d} failed, errno {d}", .{ d.label, buf.len - done, off + done, std.c._errno().* });
            }
            if (n == 0) std.debug.panic("nocache reader: {s}: short read at {d} ({d} B file)", .{ d.label, off + done, d.size });
            done += @intCast(n);
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

// ── Residency probes (proof tests) ──

/// Bytes of `path` resident in the page cache (mincore over a read-only map:
/// the map faults nothing in).
pub fn residentBytes(path: [:0]const u8) !u64 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    const size: usize = @intCast(st.size);
    if (size == 0) return 0;
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    const page = std.heap.pageSize();
    const n_pages = (size + page - 1) / page;
    const vec = try std.heap.c_allocator.alloc(u8, n_pages);
    defer std.heap.c_allocator.free(vec);
    if (std.c.mincore(@ptrCast(map.ptr), size, vec.ptr) != 0) return error.MincoreFailed;
    var resident: u64 = 0;
    for (vec) |v| resident += @intFromBool(v & 1 != 0);
    return resident * page;
}

/// `vm_stat`'s file-backed pages, in bytes (the whole box's; other processes move it too).
pub fn fileBackedBytes(a: std.mem.Allocator, io: std.Io) !u64 {
    const r = try std.process.run(a, io, .{ .argv = &.{"/usr/bin/vm_stat"}, .stdout_limit = .limited(1 << 16) });
    defer a.free(r.stdout);
    defer a.free(r.stderr);
    const page_key = "page size of ";
    const pi = std.mem.indexOf(u8, r.stdout, page_key) orelse return error.VmStatFormat;
    const page_end = std.mem.indexOfScalarPos(u8, r.stdout, pi + page_key.len, ' ') orelse return error.VmStatFormat;
    const page = try std.fmt.parseInt(u64, r.stdout[pi + page_key.len .. page_end], 10);
    const key = "File-backed pages:";
    const ki = std.mem.indexOf(u8, r.stdout, key) orelse return error.VmStatFormat;
    const line_end = std.mem.indexOfScalarPos(u8, r.stdout, ki, '\n') orelse r.stdout.len;
    const v = std.mem.trim(u8, r.stdout[ki + key.len .. line_end], " .\t");
    return page * try std.fmt.parseInt(u64, v, 10);
}

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

// DSV41_BANK=<bank> DSV41_NOCACHE_PROOF=1 [DSV41_QUIET_HOLD=<marker>]: the reads leave no page cached (mincore, vm_stat).
test "dsv41 nocache reader: the resident shards and the Engram rows read past the page cache" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    if (std.c.getenv("DSV41_NOCACHE_PROOF") == null) return error.SkipZigTest;
    const hold: ?[:0]const u8 = if (std.c.getenv("DSV41_QUIET_HOLD")) |v| std.mem.span(v) else null;
    const a = testing.allocator;
    const io = testing.io;
    const held = struct {
        fn f(p: ?[:0]const u8) bool {
            const q = p orelse return false;
            return std.c.access(q.ptr, 0) == 0;
        }
    }.f;

    // The shards the index names (the loader's own rule).
    var ipath_buf: [1024]u8 = undefined;
    const ipath = try std.fmt.bufPrint(&ipath_buf, "{s}/model.safetensors.index.json", .{bank});
    const itext = try std.Io.Dir.cwd().readFileAlloc(io, ipath, a, .limited(16 << 20));
    defer a.free(itext);
    const index = try std.json.parseFromSlice(struct { weight_map: std.json.ArrayHashMap([]const u8) }, a, itext, .{ .ignore_unknown_fields = true });
    defer index.deinit();
    var shards: std.ArrayList([:0]u8) = .empty;
    defer {
        for (shards.items) |s| a.free(s);
        shards.deinit(a);
    }
    for (index.value.weight_map.map.values()) |shard| {
        const full = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ bank, shard }, 0);
        var dup = false;
        for (shards.items) |s| dup = dup or std.mem.eql(u8, s, full);
        if (dup) a.free(full) else try shards.append(a, full);
    }

    var cached_before: u64 = 0;
    for (shards.items) |s| cached_before += try residentBytes(s);
    const fb0 = try fileBackedBytes(a, io);
    const chunk = try a.alloc(u8, 64 << 20);
    defer a.free(chunk);
    var total: u64 = 0;
    const t0 = std.Io.Timestamp.now(io, .boot);
    for (shards.items) |s| {
        const d = try Desc.open(s);
        defer d.close();
        var off: u64 = 0;
        while (off < d.size) : (off += chunk.len) {
            if (held(hold)) return error.QuietHoldAppeared;
            const n: usize = @intCast(@min(chunk.len, d.size - off));
            d.readAt(chunk[0..n], off);
            total += n;
        }
    }
    const read_s = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e9;
    const fb1 = try fileBackedBytes(a, io);
    var cached_after: u64 = 0;
    for (shards.items) |s| cached_after += try residentBytes(s);
    std.debug.print("nocache proof: {d} resident shards, {d} B read through the reader in {d:.2} s ({d:.2} GB/s); their cached bytes {d} -> {d} (mincore); box file-backed {d} -> {d} B (vm_stat, delta {d})\n", .{
        shards.items.len,                                           total,        read_s, @as(f64, @floatFromInt(total)) / 1e9 / read_s, cached_before, cached_after, fb0, fb1,
        @as(i64, @intCast(fb1)) - @as(i64, @intCast(fb0)),
    });
    try testing.expect(cached_after <= cached_before);

    // The Engram rows through the row source's own descriptors (its open sets F_NOCACHE).
    const v41 = @import("deepseek_v41.zig");
    const eng = @import("deepseek_v41_engram.zig");
    var cdiag: v41.Diag = .{};
    const c = try v41.Config.load(a, io, bank, &cdiag);
    var mbuf: [1024]u8 = undefined;
    const map = try std.fmt.bufPrint(&mbuf, "{s}/engram-token-map.u32", .{bank});
    var src = try eng.RowSource.open(a, io, bank, map, &c, &cdiag);
    defer src.deinit();
    const n_files = src.hashing.n_layers;
    const files = try a.alloc([:0]u8, n_files);
    defer a.free(files);
    for (files, 0..) |*f, i| f.* = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank, src.bank.files[i] }, 0);
    defer for (files) |f| a.free(f);
    var eng_before: u64 = 0;
    for (files) |f| eng_before += try residentBytes(f);
    const fb2 = try fileBackedBytes(a, io);
    const n_records: usize = 100_000;
    const rows = try a.alloc(i64, 1);
    defer a.free(rows);
    const codes = try a.alloc(u8, src.bank.head_dim);
    defer a.free(codes);
    const scales = try a.alloc(u8, src.bank.head_dim / 32);
    defer a.free(scales);
    var rng = std.Random.DefaultPrng.init(20260928);
    const t1 = std.Io.Timestamp.now(io, .boot);
    for (0..n_records) |i| {
        if (i % 4096 == 0 and held(hold)) return error.QuietHoldAppeared;
        const li = i % n_files;
        rows[0] = @intCast(rng.random().uintLessThan(u64, src.bank.rows[li]));
        try eng.readRows(src.fds[li], &src.bank, rows, codes, scales);
    }
    const rec_s = @as(f64, @floatFromInt(t1.untilNow(io, .boot).nanoseconds)) / 1e9;
    const fb3 = try fileBackedBytes(a, io);
    var eng_after: u64 = 0;
    for (files) |f| eng_after += try residentBytes(f);
    std.debug.print("nocache proof: {d} random Engram records ({d} B each) in {d:.3} s ({d:.1} us each, one thread); the Engram files' cached bytes {d} -> {d} (mincore); box file-backed {d} -> {d} B (vm_stat, delta {d})\n", .{
        n_records,                                      src.bank.record_bytes, rec_s, rec_s * 1e6 / @as(f64, @floatFromInt(n_records)), eng_before, eng_after, fb2, fb3,
        @as(i64, @intCast(fb3)) - @as(i64, @intCast(fb2)),
    });
    try testing.expect(eng_after <= eng_before + 64 * std.heap.pageSize());
}
