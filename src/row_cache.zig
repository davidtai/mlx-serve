//! A byte-budgeted LRU of fixed-width records read from one file (any model's on-disk row
//! table): Python's `NGramRowCache` + `FileRowReader` rule for rule, with the lane's parallel
//! miss reads (tcq_runner/packed/native_engram.py, engramfetch/q3_engramfetch.c).

const std = @import("std");

pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, evictions: u64 = 0, reads: u64 = 0, rows_read: u64 = 0, gathers: u64 = 0 };

/// `rows` records of `record_bytes` at `data_offset` of `fd` (the caller owns `fd`).
pub const Table = struct { fd: std.c.fd_t, record_bytes: u32, rows: u64, data_offset: u64 = 0 };

/// Python's slot rule: `max(1, max(record_bytes, budget) // record_bytes)`.
pub fn slotCount(record_bytes: u32, budget_bytes: u64) u32 {
    return @intCast(@max(1, @max(record_bytes, budget_bytes) / record_bytes));
}

/// The row index's capacity for `slots` rows (the std map's 80 % load rule).
pub fn indexCapacity(slots: u32) u32 {
    return @max(8, std.math.ceilPowerOfTwo(u32, @intCast(@as(u64, slots) * 100 / 80 + 1)) catch unreachable);
}

/// Host bytes a cache holds for its whole life: the slot arena, the per-slot links and
/// the row index. A gather's scratch grows with the gather, apart from this.
pub fn hostBytes(record_bytes: u32, budget_bytes: u64) u64 {
    const slots = slotCount(record_bytes, budget_bytes);
    const per_slot = record_bytes + @sizeOf(u64) + 3 * @sizeOf(u32);
    return @as(u64, slots) * per_slot + @as(u64, indexCapacity(slots)) * (@sizeOf(u64) + @sizeOf(u32) + 1);
}

/// One positional read: `len` bytes at `off` into `dst`.
pub const Request = struct { off: u64, dst: [*]u8, len: usize };

/// How a batch spreads over a pool: `slices` = Python's binding pool (`ceil(n / workers)`
/// consecutive requests per task, each task in order); `cursor` = the native prefill pool
/// (`depth` takers of one ascending cursor, at most `depth` reads in flight).
pub const Spread = union(enum) { slices: u32, cursor: u32 };

/// Where a gather's miss reads run: inline on the calling thread (the stock class route,
/// sub-runs of up to the slot count), or on `pool` in requests of up to `read_limit` rows.
pub const Route = struct { pool: ?*ReadPool = null, spread: Spread = .{ .slices = 1 }, read_limit: u32 = std.math.maxInt(u32) };

/// Positional reads of one file, issued together and waited for together.
pub const Batch = struct {
    fd: std.c.fd_t = -1,
    reqs: []const Request = &.{},
    tasks: [ReadPool.max_tasks]Task = undefined,
    /// Tasks not finished (guarded by the pool's mutex).
    unfinished: u32 = 0,
    cursor: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    const Task = struct { batch: *Batch, lo: usize, hi: usize, cursor: bool };

    fn readOne(b: *Batch, i: usize) void {
        const r = b.reqs[i];
        preadAll(b.fd, r.dst[0..r.len], r.off) catch b.failed.store(true, .release);
    }

    fn run(t: *const Task) void {
        const b = t.batch;
        if (!t.cursor) {
            for (t.lo..t.hi) |i| b.readOne(i);
            return;
        }
        while (true) {
            const i = b.cursor.fetchAdd(1, .monotonic);
            if (i >= b.reqs.len) break;
            b.readOne(i);
        }
    }
};

/// A fixed set of reader threads over a FIFO of batch tasks (Python's
/// `ThreadPoolExecutor`, the engramfetch pthread pool). Built once; never grows.
pub const ReadPool = struct {
    pub const max_tasks = 256;
    pub const stack_bytes = 256 << 10;

    gpa: std.mem.Allocator,
    threads: []std.Thread = &.{},
    mu: std.c.pthread_mutex_t = .{},
    work: std.c.pthread_cond_t = .{},
    done: std.c.pthread_cond_t = .{},
    ring: [max_tasks]*const Batch.Task = undefined,
    head: usize = 0,
    len: usize = 0,
    stopping: bool = false,

    pub fn create(gpa: std.mem.Allocator, n_threads: u32) !*ReadPool {
        const self = try gpa.create(ReadPool);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa };
        const ts = try gpa.alloc(std.Thread, n_threads);
        errdefer gpa.free(ts);
        var started: usize = 0;
        errdefer self.stop(ts[0..started]);
        for (ts) |*t| {
            t.* = try std.Thread.spawn(.{ .stack_size = stack_bytes }, worker, .{self});
            started += 1;
        }
        self.threads = ts;
        return self;
    }

    pub fn destroy(self: *ReadPool) void {
        self.stop(self.threads);
        self.gpa.free(self.threads);
        self.gpa.destroy(self);
    }

    fn stop(self: *ReadPool, ts: []std.Thread) void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        self.stopping = true;
        _ = std.c.pthread_cond_broadcast(&self.work);
        _ = std.c.pthread_mutex_unlock(&self.mu);
        for (ts) |t| t.join();
    }

    fn worker(self: *ReadPool) void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        while (true) {
            while (self.len == 0 and !self.stopping) _ = std.c.pthread_cond_wait(&self.work, &self.mu);
            if (self.len == 0) break;
            const t = self.ring[self.head];
            self.head = (self.head + 1) % max_tasks;
            self.len -= 1;
            _ = std.c.pthread_mutex_unlock(&self.mu);
            Batch.run(t);
            _ = std.c.pthread_mutex_lock(&self.mu);
            t.batch.unfinished -= 1;
            if (t.batch.unfinished == 0) _ = std.c.pthread_cond_broadcast(&self.done);
        }
        _ = std.c.pthread_mutex_unlock(&self.mu);
    }

    /// Queue `b`'s requests (read from `fd`) spread as `spread`; `wait` collects them.
    pub fn submit(self: *ReadPool, b: *Batch, fd: std.c.fd_t, reqs: []const Request, spread: Spread) void {
        b.fd = fd;
        b.reqs = reqs;
        b.cursor.store(0, .monotonic);
        b.failed.store(false, .monotonic);
        var n: usize = 0;
        switch (spread) {
            .slices => |workers| {
                const width = (reqs.len + workers - 1) / workers;
                var lo: usize = 0;
                while (lo < reqs.len) : (lo += width) {
                    b.tasks[n] = .{ .batch = b, .lo = lo, .hi = @min(lo + width, reqs.len), .cursor = false };
                    n += 1;
                }
            },
            .cursor => |depth| for (0..@min(depth, reqs.len)) |_| {
                b.tasks[n] = .{ .batch = b, .lo = 0, .hi = 0, .cursor = true };
                n += 1;
            },
        }
        _ = std.c.pthread_mutex_lock(&self.mu);
        b.unfinished = @intCast(n);
        for (b.tasks[0..n]) |*t| {
            while (self.len == max_tasks) _ = std.c.pthread_cond_wait(&self.done, &self.mu);
            self.ring[(self.head + self.len) % max_tasks] = t;
            self.len += 1;
        }
        _ = std.c.pthread_cond_broadcast(&self.work);
        _ = std.c.pthread_mutex_unlock(&self.mu);
    }

    pub fn wait(self: *ReadPool, b: *Batch) !void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        while (b.unfinished > 0) _ = std.c.pthread_cond_wait(&self.done, &self.mu);
        _ = std.c.pthread_mutex_unlock(&self.mu);
        if (b.failed.load(.acquire)) return error.ReadFailed;
    }
};

/// `budget / record_bytes` slots keyed by row id: a gather's distinct rows in first-appearance
/// order, a hit made newest, the misses entered sorted in sub-runs capped at the slot count,
/// each taking the highest free slot or the oldest row's. How misses are read changes nothing.
pub const RowCache = struct {
    const nil: u32 = std.math.maxInt(u32);
    const Miss = struct { row: u64, d: u32 };

    gpa: std.mem.Allocator,
    table: Table,
    slot_count: u32,
    arena: []u8,
    slot_row: []u64,
    prev: []u32,
    next: []u32,
    free: []u32,
    n_free: u32,
    oldest: u32 = nil,
    newest: u32 = nil,
    index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// Removals since the index was last rehashed (its tombstones).
    removed: u32 = 0,
    stats: Stats = .{},
    // One gather's scratch, kept across gathers.
    first: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    distinct: std.ArrayList(u64) = .empty,
    chain_head: std.ArrayList(u32) = .empty,
    chain_tail: std.ArrayList(u32) = .empty,
    chain_next: std.ArrayList(u32) = .empty,
    misses: std.ArrayList(Miss) = .empty,
    /// The misses' records, in sorted-miss order.
    scratch: std.ArrayList(u8) = .empty,
    reqs: std.ArrayList(Request) = .empty,
    batch: Batch = .{},
    // The lookahead (Python `before_groups`): rows read ahead, taken by the next gathers.
    ahead_buf: []u8 = &.{},
    ahead_rows: std.ArrayList(u64) = .empty,
    ahead_at: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    ahead_reqs: std.ArrayList(Request) = .empty,
    ahead_batch: Batch = .{},
    ahead_pool: ?*ReadPool = null,

    /// `lookahead_rows`: the most rows one lookahead reads (0: none).
    pub fn init(gpa: std.mem.Allocator, table: Table, budget_bytes: u64, lookahead_rows: u32) !RowCache {
        const slots = slotCount(table.record_bytes, budget_bytes);
        const arena = try gpa.alloc(u8, @as(usize, slots) * table.record_bytes);
        errdefer gpa.free(arena);
        const slot_row = try gpa.alloc(u64, slots);
        errdefer gpa.free(slot_row);
        const prev = try gpa.alloc(u32, slots);
        errdefer gpa.free(prev);
        const next = try gpa.alloc(u32, slots);
        errdefer gpa.free(next);
        const free = try gpa.alloc(u32, slots);
        errdefer gpa.free(free);
        const ahead = try gpa.alloc(u8, @as(usize, lookahead_rows) * table.record_bytes);
        errdefer gpa.free(ahead);
        for (free, 0..) |*f, i| f.* = @intCast(i);
        var self: RowCache = .{ .gpa = gpa, .table = table, .slot_count = slots, .arena = arena, .slot_row = slot_row, .prev = prev, .next = next, .free = free, .n_free = slots, .ahead_buf = ahead };
        errdefer self.index.deinit(gpa);
        try self.index.ensureTotalCapacity(gpa, slots);
        try self.ahead_at.ensureTotalCapacity(gpa, lookahead_rows);
        return self;
    }

    pub fn deinit(self: *RowCache) void {
        self.settle() catch {};
        const a = self.gpa;
        a.free(self.arena);
        a.free(self.slot_row);
        a.free(self.prev);
        a.free(self.next);
        a.free(self.free);
        a.free(self.ahead_buf);
        self.index.deinit(a);
        self.first.deinit(a);
        self.distinct.deinit(a);
        self.chain_head.deinit(a);
        self.chain_tail.deinit(a);
        self.chain_next.deinit(a);
        self.misses.deinit(a);
        self.scratch.deinit(a);
        self.reqs.deinit(a);
        self.ahead_rows.deinit(a);
        self.ahead_at.deinit(a);
        self.ahead_reqs.deinit(a);
    }

    pub fn residentRows(self: *const RowCache) u32 {
        return self.slot_count - self.n_free;
    }

    fn unlink(self: *RowCache, s: u32) void {
        const p = self.prev[s];
        const n = self.next[s];
        if (p != nil) self.next[p] = n else self.oldest = n;
        if (n != nil) self.prev[n] = p else self.newest = p;
    }

    fn linkNewest(self: *RowCache, s: u32) void {
        self.prev[s] = self.newest;
        self.next[s] = nil;
        if (self.newest != nil) self.next[self.newest] = s else self.oldest = s;
        self.newest = s;
    }

    /// Python `_alloc_slot`: `list.pop()` of the free list, else the oldest row is evicted.
    fn allocSlot(self: *RowCache) u32 {
        if (self.n_free > 0) {
            self.n_free -= 1;
            return self.free[self.n_free];
        }
        const s = self.oldest;
        _ = self.index.remove(self.slot_row[s]);
        self.removed += 1;
        self.unlink(s);
        self.stats.evictions += 1;
        return s;
    }

    fn lessRow(_: void, x: Miss, y: Miss) bool {
        return x.row < y.row;
    }

    fn checkRows(self: *const RowCache, rows: []const i64) !void {
        for (rows) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.table.rows) return error.RowOutOfRange;
    }

    /// Requests for `rows` (sorted) landing at `dst` + their index: consecutive rows with
    /// consecutive destinations join, up to `limit` rows (Python `_requests`).
    fn appendRequest(self: *const RowCache, list: *std.ArrayList(Request), row: u64, dst: [*]u8, limit: u32) !void {
        const rb = self.table.record_bytes;
        const off = self.table.data_offset + row * rb;
        if (list.items.len > 0) {
            const last = &list.items[list.items.len - 1];
            if (last.len / rb < limit and off == last.off + last.len and dst == last.dst + last.len) {
                last.len += rb;
                return;
            }
        }
        try list.append(self.gpa, .{ .off = off, .dst = dst, .len = rb });
    }

    fn execute(self: *RowCache, b: *Batch, reqs: []const Request, route: Route) !void {
        if (reqs.len == 0) return;
        if (route.pool) |p| {
            p.submit(b, self.table.fd, reqs, route.spread);
            return p.wait(b);
        }
        for (reqs) |r| try preadAll(self.table.fd, r.dst[0..r.len], r.off);
    }

    /// Python `gather_bytes`: `sink.put(i, record)` for every position `i` of `rows`, the
    /// misses read on `route`. A row out of range refuses before anything changes.
    pub fn gather(self: *RowCache, rows: []const i64, sink: anytype, route: Route) !void {
        const a = self.gpa;
        const rb = self.table.record_bytes;
        try self.checkRows(rows);
        self.first.clearRetainingCapacity();
        self.distinct.clearRetainingCapacity();
        self.chain_head.clearRetainingCapacity();
        self.chain_tail.clearRetainingCapacity();
        try self.chain_next.resize(a, rows.len);
        for (rows, 0..) |r, i| {
            const gop = try self.first.getOrPut(a, @intCast(r));
            self.chain_next.items[i] = nil;
            if (gop.found_existing) {
                const d = gop.value_ptr.*;
                self.chain_next.items[self.chain_tail.items[d]] = @intCast(i);
                self.chain_tail.items[d] = @intCast(i);
            } else {
                gop.value_ptr.* = @intCast(self.distinct.items.len);
                try self.distinct.append(a, @intCast(r));
                try self.chain_head.append(a, @intCast(i));
                try self.chain_tail.append(a, @intCast(i));
            }
        }
        self.misses.clearRetainingCapacity();
        for (self.distinct.items, 0..) |row, d| {
            if (self.index.get(row)) |slot| {
                self.unlink(slot);
                self.linkNewest(slot);
                self.stats.hits += 1;
                self.put(sink, @intCast(d), self.arena[@as(usize, slot) * rb ..][0..rb]);
            } else try self.misses.append(a, .{ .row = row, .d = @intCast(d) });
        }
        const ms = self.misses.items;
        if (ms.len > 0) {
            std.mem.sort(Miss, ms, {}, lessRow);
            // The lookahead's rows first (it settles here), the rest read on the route.
            if (self.ahead_pool) |p| try p.wait(&self.ahead_batch);
            try self.scratch.resize(a, ms.len * rb);
            const limit = if (route.pool == null) self.slot_count else @min(route.read_limit, self.slot_count);
            self.reqs.clearRetainingCapacity();
            for (ms, 0..) |m, k| {
                const dst = self.scratch.items[k * rb ..].ptr;
                if (self.ahead_at.get(m.row)) |at| {
                    @memcpy(dst[0..rb], self.ahead_buf[@as(usize, at) * rb ..][0..rb]);
                } else try self.appendRequest(&self.reqs, m.row, dst, limit);
            }
            try self.execute(&self.batch, self.reqs.items, route);
        }
        // The stock population, in sorted order: sub-runs of contiguous rows capped at the slot count.
        var i: usize = 0;
        while (i < ms.len) {
            var j = i + 1;
            while (j < ms.len and ms[j].row == ms[j - 1].row + 1) j += 1;
            var off = i;
            while (off < j) {
                const n: usize = @min(j - off, self.slot_count);
                self.stats.reads += 1;
                self.stats.rows_read += n;
                self.stats.misses += n;
                for (ms[off..][0..n], off..) |m, k| {
                    const slot = self.allocSlot();
                    const rec = self.scratch.items[k * rb ..][0..rb];
                    @memcpy(self.arena[@as(usize, slot) * rb ..][0..rb], rec);
                    self.slot_row[slot] = m.row;
                    self.index.putAssumeCapacity(m.row, slot);
                    self.linkNewest(slot);
                    self.put(sink, m.d, rec);
                }
                off += n;
            }
            i = j;
        }
        self.stats.gathers += 1;
        // Evictions leave tombstones in the index: clear them once they reach half the slots.
        if (self.removed >= self.slot_count / 2 + 1) {
            self.index.rehash(std.hash_map.AutoContext(u64){});
            self.removed = 0;
        }
    }

    fn put(self: *const RowCache, sink: anytype, d: u32, rec: []const u8) void {
        var i = self.chain_head.items[d];
        while (i != nil) : (i = self.chain_next.items[i]) sink.put(i, rec);
    }

    /// Python `before_groups` for this cache: start reading the rows of `rows` not resident
    /// now (sorted, each once) on `route`'s pool; the next gathers take them. Changes no
    /// cache state; `settle` ends it.
    pub fn lookahead(self: *RowCache, rows: []const i64, route: Route) !void {
        const rb = self.table.record_bytes;
        try self.settle();
        try self.checkRows(rows);
        self.ahead_rows.clearRetainingCapacity();
        for (rows) |r| {
            const row: u64 = @intCast(r);
            if (self.index.contains(row)) continue;
            const gop = try self.ahead_at.getOrPut(self.gpa, row);
            if (!gop.found_existing) try self.ahead_rows.append(self.gpa, row);
        }
        const rs = self.ahead_rows.items;
        if (rs.len * rb > self.ahead_buf.len) {
            self.ahead_at.clearRetainingCapacity();
            return error.LookaheadRows;
        }
        std.mem.sort(u64, rs, {}, std.sort.asc(u64));
        const limit = @min(route.read_limit, self.slot_count);
        self.ahead_reqs.clearRetainingCapacity();
        for (rs, 0..) |row, k| {
            self.ahead_at.putAssumeCapacity(row, @intCast(k));
            try self.appendRequest(&self.ahead_reqs, row, self.ahead_buf[k * rb ..].ptr, limit);
        }
        if (route.pool) |p| {
            p.submit(&self.ahead_batch, self.table.fd, self.ahead_reqs.items, route.spread);
            self.ahead_pool = p;
        } else for (self.ahead_reqs.items) |r| try preadAll(self.table.fd, r.dst[0..r.len], r.off);
    }

    /// Python `after_groups` for this cache: the lookahead's reads done, its rows released.
    pub fn settle(self: *RowCache) !void {
        const p = self.ahead_pool;
        self.ahead_pool = null;
        self.ahead_at.clearRetainingCapacity();
        if (p) |pool| try pool.wait(&self.ahead_batch);
    }
};

fn preadAll(fd: std.c.fd_t, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
        if (n < 0) {
            if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
            return error.ReadFailed;
        }
        if (n == 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const testing = std.testing;

/// Collects a gather's records in order.
const Collect = struct {
    out: []u8,
    rb: usize,
    pub fn put(c: *Collect, i: usize, rec: []const u8) void {
        @memcpy(c.out[i * c.rb ..][0..c.rb], rec);
    }
};

/// A file of `rows` records of `rb` bytes, record r filled with byte r (mod 256).
fn testTable(path: [:0]const u8, rows: usize, rb: usize) !std.c.fd_t {
    {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.TestOpen;
        defer _ = std.c.close(fd);
        const buf = try testing.allocator.alloc(u8, rows * rb);
        defer testing.allocator.free(buf);
        for (0..rows) |r| @memset(buf[r * rb ..][0..rb], @truncate(r));
        if (std.c.write(fd, buf.ptr, buf.len) != @as(isize, @intCast(buf.len))) return error.TestWrite;
    }
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestOpen;
    return fd;
}

fn lruOrder(c: *const RowCache, out: []u64) usize {
    var n: usize = 0;
    var s = c.oldest;
    while (s != RowCache.nil) : (s = c.next[s]) {
        out[n] = c.slot_row[s];
        n += 1;
    }
    return n;
}

test "dsv41 row cache: Python's gather sequence gives Python's stats, residency and order" {
    // The case `engram_replay.py --selfcheck` runs through Python's own class: 32 rows of 33 B
    // (row r filled with byte r), 3 slots; the same on a pool changes nothing.
    const rb = 33;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/row-cache-test-{d}.bin", .{std.c.getpid()}, 0);
    const fd = try testTable(path, 32, rb);
    defer _ = std.c.unlink(path.ptr);
    defer _ = std.c.close(fd);
    const pool = try ReadPool.create(testing.allocator, 4);
    defer pool.destroy();
    const routes = [_]Route{ .{}, .{ .pool = pool, .spread = .{ .slices = 16 }, .read_limit = 2 }, .{ .pool = pool, .spread = .{ .cursor = 64 }, .read_limit = 1 } };
    for (routes) |route| {
        var c = try RowCache.init(testing.allocator, .{ .fd = fd, .record_bytes = rb, .rows = 32 }, 3 * rb, 0);
        defer c.deinit();
        try testing.expectEqual(@as(u32, 3), c.slot_count);
        const seq = [_][]const i64{ &.{ 5, 6, 5, 9 }, &.{ 6, 10 }, &.{ 11, 12, 13, 14 }, &.{13}, &.{15}, &.{12} };
        var out: [4 * rb]u8 = undefined;
        for (seq) |rows| {
            var sink: Collect = .{ .out = &out, .rb = rb };
            try c.gather(rows, &sink, route);
            for (rows, 0..) |r, i| try testing.expect(std.mem.allEqual(u8, out[i * rb ..][0..rb], @intCast(r)));
        }
        try testing.expectEqual(Stats{ .hits = 2, .misses = 10, .evictions = 7, .reads = 7, .rows_read = 10, .gathers = 6 }, c.stats);
        var order: [3]u64 = undefined;
        try testing.expectEqual(@as(usize, 3), lruOrder(&c, &order));
        try testing.expectEqual([3]u64{ 13, 15, 12 }, order);
        var bad: Collect = .{ .out = &out, .rb = rb };
        try testing.expectError(error.RowOutOfRange, c.gather(&.{ 3, 32 }, &bad, route));
        try testing.expectEqual(@as(u64, 6), c.stats.gathers);
        try testing.expectEqual(indexCapacity(3), c.index.capacity());
    }
}

test "dsv41 row cache: a lookahead, pooled reads and a gather past the arena keep the stock state" {
    // Random gathers over 4,096 rows through a 97-slot cache: the stock inline route vs a
    // pooled route with a lookahead before each pair of gathers; same records, stats, LRU.
    const rb = 20;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/row-cache-test-la-{d}.bin", .{std.c.getpid()}, 0);
    const fd = try testTable(path, 4096, rb);
    defer _ = std.c.unlink(path.ptr);
    defer _ = std.c.close(fd);
    const pool = try ReadPool.create(testing.allocator, 16);
    defer pool.destroy();
    const route: Route = .{ .pool = pool, .spread = .{ .slices = 16 }, .read_limit = 7 };
    var stock = try RowCache.init(testing.allocator, .{ .fd = fd, .record_bytes = rb, .rows = 4096 }, 97 * rb, 0);
    defer stock.deinit();
    var lane = try RowCache.init(testing.allocator, .{ .fd = fd, .record_bytes = rb, .rows = 4096 }, 97 * rb, 256);
    defer lane.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var rows: [256]i64 = undefined;
    var out_a: [256 * rb]u8 = undefined;
    var out_b: [256 * rb]u8 = undefined;
    for (0..60) |step| {
        // Mostly a narrow band (hits, runs), sometimes a gather larger than the arena.
        const n: usize = if (step % 13 == 12) 200 else 1 + rnd.uintLessThan(usize, 60);
        const band: u64 = if (step % 5 == 0) 4096 else 300;
        for (rows[0..n]) |*r| r.* = @intCast(rnd.uintLessThan(u64, band));
        const half = (n + 1) / 2;
        try lane.lookahead(rows[0..n], route);
        var sa: Collect = .{ .out = &out_a, .rb = rb };
        var sb: Collect = .{ .out = &out_b, .rb = rb };
        try stock.gather(rows[0..half], &sa, .{});
        try lane.gather(rows[0..half], &sb, route);
        try testing.expectEqualSlices(u8, out_a[0 .. half * rb], out_b[0 .. half * rb]);
        try stock.gather(rows[half..n], &sa, .{});
        try lane.gather(rows[half..n], &sb, route);
        try lane.settle();
        try testing.expectEqualSlices(u8, out_a[0 .. (n - half) * rb], out_b[0 .. (n - half) * rb]);
        for (rows[half..n], 0..) |r, i| try testing.expect(std.mem.allEqual(u8, out_b[i * rb ..][0..rb], @truncate(@as(u64, @intCast(r)))));
        try testing.expectEqual(stock.stats, lane.stats);
    }
    try testing.expect(stock.stats.evictions > 0 and stock.stats.hits > 0);
    var o1: [97]u64 = undefined;
    var o2: [97]u64 = undefined;
    try testing.expectEqual(lruOrder(&stock, &o1), lruOrder(&lane, &o2));
    try testing.expectEqualSlices(u64, &o1, &o2);
}

test "dsv41 row cache: the host charge is the cache's allocation" {
    try testing.expectEqual(@as(u32, 254_200), slotCount(264, 64 << 20));
    try testing.expectEqual(@as(u32, 524_288), indexCapacity(254_200));
    try testing.expectEqual(@as(u64, 254_200 * (264 + 20) + 524_288 * 13), hostBytes(264, 64 << 20));
    var c = try RowCache.init(testing.allocator, .{ .fd = -1, .record_bytes = 264, .rows = 1 << 24 }, 64 << 20, 0);
    defer c.deinit();
    try testing.expectEqual(@as(u64, 254_200 * 264), c.arena.len);
    try testing.expectEqual(indexCapacity(c.slot_count), c.index.capacity());
}
