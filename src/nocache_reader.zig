//! An MLX IO reader whose reads bypass the page cache. MLX's own safetensors
//! reader is a plain open + pread, so a load leaves the file's pages cached
//! next to the array buffers. This one opens the file with F_NOCACHE and
//! read-ahead off; `mlx_load_safetensors_reader` reads the header through
//! `read` and each tensor through `read_at_offset` into the array's buffer.
//! A failed read panics naming the file, as MLX's own reader throws.

const std = @import("std");
const status = @import("status.zig");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");

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
    /// guard's metric does not count until the kernel ages them under pressure). Measured on the bank's
    /// resident shards: 0.19 GB cached per 0.54 GB read unaligned, none aligned; MLX's tensor loads
    /// arrive unaligned (safetensors offsets, MLX buffers), and a served cell's construction left
    /// 15.1 GB of page cache (pass3aj 20260930-060358). pread is safe from MLX's IO threads: each call
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

/// `buf.len` bytes at `off` of an F_NOCACHE descriptor, through whole aligned pages into a page-aligned stage (the
/// construction's header reads of a table read past the page cache, `NgramTable.openTensor`); errors by name.
pub fn readAligned(fd: std.c.fd_t, buf: []u8, off: u64) !void {
    if (buf.len == 0) return;
    const page: u64 = std.heap.pageSize();
    const span = std.mem.alignForward(u64, off % page + buf.len, page);
    const stage = try std.heap.page_allocator.alloc(u8, @intCast(@min(span, stage_bytes)));
    defer std.heap.page_allocator.free(stage);
    var pos = off;
    const end = off + buf.len;
    var out: usize = 0;
    while (pos < end) {
        const a0 = pos - pos % page;
        const want_end = @min(end, a0 + stage.len);
        const need = want_end - a0;
        const len = std.mem.alignForward(u64, need, page);
        var got: u64 = 0;
        while (got < need) {
            if (got % page != 0) return error.ReadShort;
            const n = std.c.pread(fd, stage[@intCast(got)..].ptr, @intCast(len - got), @intCast(a0 + got));
            if (n < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.ReadShort;
            got += @intCast(n);
        }
        const s0: usize = @intCast(pos - a0);
        const take: usize = @intCast(need - (pos - a0));
        @memcpy(buf[out..][0..take], stage[s0..][0..take]);
        out += take;
        pos = want_end;
    }
}

// ── Row gather: a table's rows past the page cache ──

/// A table's rows past the page cache (the input embedding's host rows): `row_bytes` at `base + id * row_bytes` of an
/// F_NOCACHE descriptor, gathered into the caller's id order. Every read is page-aligned in offset, length and
/// destination, and checked, since macOS keeps only aligned reads out of the page cache (`Desc.readAt`). Each distinct
/// row is read once; rows whose aligned pages touch are read as one run; `helpers` threads and the caller take the runs
/// in parallel. `init` allocates everything once (the page-aligned stages, the sort and run scratch, the threads): a
/// gather allocates nothing. One gather at a time (the model's thread); the descriptor stays its owner's.
pub const RowGather = struct {
    /// One reader's stage: the widest run it reads (whole pages).
    pub const stage_len: usize = 128 << 10;
    pub const Item = struct { id: u32, at: u32 };
    /// `len` aligned bytes at `off` (aligned) hold `items[first..end]`'s rows; the last of them ends `need` bytes in.
    pub const Run = struct { off: u64, len: u64, need: u64, first: u32, end: u32 };

    fd: std.c.fd_t,
    base: u64,
    row_bytes: usize,
    rows: u64,
    page: usize,
    /// Ids per piece (a longer call runs in pieces of this many).
    max_ids: usize,
    /// Below this many ids the caller reads alone (no helper woken).
    parallel_min: usize,
    /// `helpers + 1` stages; `[0]` is the caller's.
    stages: [][]u8,
    items: []Item,
    runs: []Run,
    threads: []std.Thread,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    /// The piece in flight (set before `gen` moves, read after it).
    out: []u8 = &.{},
    n_runs: usize = 0,
    next: std.atomic.Value(usize) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    unaligned: std.atomic.Value(u32) = .init(0),

    /// Host bytes a gather keeps for the table's life (the bill's term): the stages and the scratch.
    pub fn persistentBytes(helpers: usize, max_ids: usize) u64 {
        return @as(u64, helpers + 1) * stage_len + @as(u64, max_ids) * (@sizeOf(Item) + @sizeOf(Run));
    }

    pub fn init(fd: std.c.fd_t, base: u64, row_bytes: usize, rows: u64, helpers: usize, max_ids: usize, parallel_min: usize) !*RowGather {
        const page = std.heap.pageSize();
        if (row_bytes == 0 or max_ids == 0 or row_bytes + 2 * page > stage_len or stage_len % page != 0) return error.GatherGeometry;
        const a = std.heap.c_allocator;
        const self = try a.create(RowGather);
        errdefer a.destroy(self);
        self.* = .{ .fd = fd, .base = base, .row_bytes = row_bytes, .rows = rows, .page = page, .max_ids = max_ids, .parallel_min = parallel_min, .stages = &.{}, .items = &.{}, .runs = &.{}, .threads = &.{} };
        self.stages = try a.alloc([]u8, helpers + 1);
        errdefer a.free(self.stages);
        var made: usize = 0;
        errdefer for (self.stages[0..made]) |st| std.heap.page_allocator.free(st);
        for (self.stages) |*st| {
            st.* = try std.heap.page_allocator.alloc(u8, stage_len);
            made += 1;
            if (@intFromPtr(st.*.ptr) % page != 0) return error.GatherGeometry;
        }
        self.items = try a.alloc(Item, max_ids);
        errdefer a.free(self.items);
        self.runs = try a.alloc(Run, max_ids);
        errdefer a.free(self.runs);
        self.threads = try a.alloc(std.Thread, helpers);
        errdefer a.free(self.threads);
        var started: usize = 0;
        errdefer self.stop(started);
        for (self.threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, helper, .{ self, i });
            started += 1;
        }
        return self;
    }

    pub fn deinit(self: *RowGather) void {
        self.stop(self.threads.len);
        const a = std.heap.c_allocator;
        for (self.stages) |st| std.heap.page_allocator.free(st);
        a.free(self.stages);
        a.free(self.items);
        a.free(self.runs);
        a.free(self.threads);
        a.destroy(self);
    }

    fn stop(self: *RowGather, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    /// The rows of `ids` (any order, repeats allowed) into `out` (`ids.len * row_bytes`), in `ids` order.
    pub fn gather(self: *RowGather, ids: []const u32, out: []u8) !void {
        if (out.len != ids.len * self.row_bytes) return error.GatherShape;
        for (ids) |id| if (id >= self.rows) return error.RowOutOfRange;
        var start: usize = 0;
        while (start < ids.len) : (start += self.max_ids) {
            const end = @min(start + self.max_ids, ids.len);
            try self.piece(ids[start..end], out[start * self.row_bytes .. end * self.row_bytes]);
        }
    }

    fn lessId(_: void, x: Item, y: Item) bool {
        return x.id < y.id;
    }

    fn alignUp(self: *const RowGather, x: u64) u64 {
        return std.mem.alignForward(u64, x, self.page);
    }

    /// The sorted items' runs: each distinct row once, rows whose aligned pages touch in one run up to a stage.
    fn plan(self: *RowGather, n: usize) usize {
        const p: u64 = self.page;
        const rb: u64 = self.row_bytes;
        const items = self.items[0..n];
        var nr: usize = 0;
        var i: usize = 0;
        while (i < n) {
            const o0 = self.base + @as(u64, items[i].id) * rb;
            const start = o0 - o0 % p;
            var end_row = o0 + rb;
            var j = i + 1;
            while (j < n) : (j += 1) {
                if (items[j].id == items[j - 1].id) continue;
                const o = self.base + @as(u64, items[j].id) * rb;
                if (o - o % p > self.alignUp(end_row)) break;
                if (self.alignUp(o + rb) - start > stage_len) break;
                end_row = o + rb;
            }
            self.runs[nr] = .{ .off = start, .len = self.alignUp(end_row) - start, .need = end_row - start, .first = @intCast(i), .end = @intCast(j) };
            nr += 1;
            i = j;
        }
        return nr;
    }

    fn piece(self: *RowGather, ids: []const u32, out: []u8) !void {
        for (self.items[0..ids.len], ids, 0..) |*it, id, i| it.* = .{ .id = id, .at = @intCast(i) };
        std.mem.sort(Item, self.items[0..ids.len], {}, lessId);
        self.n_runs = self.plan(ids.len);
        self.out = out;
        self.next.store(0, .release);
        self.failed.store(0, .release);
        self.unaligned.store(0, .release);
        if (self.threads.len == 0 or ids.len < self.parallel_min or self.n_runs == 1) {
            self.work(self.stages[0]);
        } else {
            const io = std.Io.Threaded.global_single_threaded.io();
            self.mu.lockUncancelable(io);
            self.pending.store(@intCast(self.threads.len), .release);
            self.gen += 1;
            self.cv.broadcast(io);
            self.mu.unlock(io);
            self.work(self.stages[0]);
            while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
        }
        if (self.unaligned.load(.acquire) != 0) return error.GatherUnaligned;
        if (self.failed.load(.acquire) != 0) return error.GatherRead;
    }

    fn helper(self: *RowGather, idx: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        var seen: u64 = 0;
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
            if (self.quit) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            self.mu.unlock(io);
            self.work(self.stages[idx + 1]);
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }

    /// Runs taken in turn: each read into `stage`, its rows copied to their places in the piece's `out`.
    fn work(self: *RowGather, stage: []u8) void {
        while (true) {
            const r = self.next.fetchAdd(1, .acq_rel);
            if (r >= self.n_runs) return;
            const run = self.runs[r];
            if (!self.aligned(run, stage)) {
                _ = self.unaligned.fetchAdd(1, .acq_rel);
                continue;
            }
            if (!self.readRun(run, stage)) {
                _ = self.failed.fetchAdd(1, .acq_rel);
                continue;
            }
            for (self.items[run.first..run.end]) |it| {
                const off: usize = @intCast(self.base + @as(u64, it.id) * self.row_bytes - run.off);
                @memcpy(self.out[@as(usize, it.at) * self.row_bytes ..][0..self.row_bytes], stage[off..][0..self.row_bytes]);
            }
        }
    }

    /// A run's read is page-aligned in offset, length and destination, and fits its stage.
    pub fn aligned(self: *const RowGather, run: Run, stage: []const u8) bool {
        return run.off % self.page == 0 and run.len % self.page == 0 and @intFromPtr(stage.ptr) % self.page == 0 and run.len <= stage.len and run.need <= run.len;
    }

    /// `run.len` bytes at `run.off` into `stage`, until its rows are in (the file may end inside the last page). Each
    /// pread stays aligned: a short read that is not whole pages is the file's end, never continued unaligned.
    fn readRun(self: *const RowGather, run: Run, stage: []u8) bool {
        var got: u64 = 0;
        while (got < run.need) {
            if (got % self.page != 0) return false;
            const k = std.c.pread(self.fd, stage[@intCast(got)..].ptr, @intCast(run.len - got), @intCast(run.off + got));
            if (k < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return false;
            }
            if (k == 0) return false;
            got += @intCast(k);
        }
        return true;
    }
};

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
    const fb0 = status.vmBytes().external;
    const chunk = try a.alloc(u8, 64 << 20);
    defer a.free(chunk);
    var total: u64 = 0;
    const t0 = std.Io.Timestamp.now(io, .boot);
    for (shards.items) |s| {
        const d = try Desc.open(s);
        defer d.close();
        // Unaligned, as MLX's tensor loads arrive (a safetensors offset, a destination off the page).
        const step = chunk.len - 12_347;
        var off: u64 = 8 + 4_321;
        while (off < d.size) : (off += step) {
            if (held(hold)) return error.QuietHoldAppeared;
            const n: usize = @intCast(@min(step, d.size - off));
            d.readAt(chunk[1..][0..n], off);
            total += n;
        }
    }
    const read_s = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e9;
    const fb1 = status.vmBytes().external;
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
    const fb2 = status.vmBytes().external;
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
    const fb3 = status.vmBytes().external;
    var eng_after: u64 = 0;
    for (files) |f| eng_after += try residentBytes(f);
    std.debug.print("nocache proof: {d} random Engram records ({d} B each) in {d:.3} s ({d:.1} us each, one thread); the Engram files' cached bytes {d} -> {d} (mincore); box file-backed {d} -> {d} B (vm_stat, delta {d})\n", .{
        n_records,                                      src.bank.record_bytes, rec_s, rec_s * 1e6 / @as(f64, @floatFromInt(n_records)), eng_before, eng_after, fb2, fb3,
        @as(i64, @intCast(fb3)) - @as(i64, @intCast(fb2)),
    });
    try testing.expect(eng_after <= eng_before + 64 * std.heap.pageSize());
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
