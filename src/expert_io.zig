//! Native-issue read pool (lib/expert_io/q3_nativeissue.c): worker threads read
//! each record's gate/up span, then its down span (page-aligned F_NOCACHE preadv
//! into a staging buffer, then a scatter copy into slot rows) and publish every
//! range's status words, then its ticket in an ordered log. Workers never call
//! MLX. The C state is static: one pool per process, stopped before the next.

const std = @import("std");
const expert_bank = @import("expert_bank.zig");

const n_components = expert_bank.n_components;
const gu_components = expert_bank.gu_components;

// 1:1 mirror of lib/expert_io/q3_nativeissue.h.
const c = struct {
    extern fn q3ni_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
    extern fn q3ni_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
    extern fn q3ni_seq() i64;
    extern fn q3ni_wait(seen: i64, timeout_ns: i64) i64;
    extern fn q3ni_gauge(out: *[6]i64) void;
    extern fn q3ni_quiesce(timeout_ns: i64) c_int;
    extern fn q3ni_stop() c_int;
    extern fn q3ni_abi() i32;
    /// Q3NI_INJECT builds (the test module) only.
    extern fn q3ni_test_rules(n: i32, off: [*]const i64, code: [*]const i64, arg: [*]const i64) void;
};

pub const abi_version = 20260924;
pub const max_workers = 8;
/// Records per job (one fill unit).
pub const max_items = 8;
const res_w = 8;
/// `q3ni_submit` reads -1 as no deadline; 0 would expire every range at once.
const no_deadline: i64 = -1;

pub const Status = enum(i64) { ok = 0, short = 1, os_error = 2, deadline = 3, skipped = 4, _ };

/// One range's status words, as the worker published them.
pub const Result = struct {
    status: Status,
    preadv_calls: i64,
    bytes_returned: i64,
    payload: i64,
    errno: i64,
    t_start_ns: i64,
    t_end_ns: i64,
    worker: i64,
};

pub const Options = struct {
    workers: u32 = 4,
    /// One page-aligned staging buffer per worker; 9 MiB holds a whole
    /// 8,877,056-byte gate/up span plus its page alignment.
    staging_bytes: u64 = 9 << 20,
    tickets: u32 = 256,
};

pub const Pool = struct {
    allocator: std.mem.Allocator,
    staging: []align(std.heap.page_size_min) u8,
    /// [ticket * res_w + k], written by the workers; zeroed before start.
    res: []i64,
    /// Completion order: log[seq % tickets] = ticket.
    log: []i64,
    gauge: [6]i64 = @splat(0),
    /// Tickets seen in the log since their last submit.
    published: []bool,
    /// Log entries consumed so far.
    seen: i64 = 0,
    next_ticket: u32 = 0,

    /// Starts the process's pool. Staging, status words and the log live here
    /// and stay put until `stop`.
    pub fn start(allocator: std.mem.Allocator, opt: Options) !*Pool {
        const page = std.heap.pageSize();
        if (opt.workers == 0 or opt.workers > max_workers or opt.staging_bytes == 0 or opt.staging_bytes % page != 0 or opt.tickets < 2 * max_items)
            return error.InvalidOptions;
        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        const staging = try std.posix.mmap(null, @intCast(opt.workers * opt.staging_bytes), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(staging);
        const res = try allocator.alloc(i64, @as(usize, opt.tickets) * res_w);
        errdefer allocator.free(res);
        const log_arr = try allocator.alloc(i64, opt.tickets);
        errdefer allocator.free(log_arr);
        const published = try allocator.alloc(bool, opt.tickets);
        errdefer allocator.free(published);
        @memset(res, 0);
        @memset(log_arr, 0);
        @memset(published, false);
        self.* = .{ .allocator = allocator, .staging = staging, .res = res, .log = log_arr, .published = published };
        var ptrs: [max_workers]u64 = undefined;
        for (0..opt.workers) |w| ptrs[w] = @intFromPtr(staging.ptr) + w * opt.staging_bytes;
        const rc = c.q3ni_start(@intCast(opt.workers), &ptrs, @intCast(opt.staging_bytes), @intCast(page), res.ptr, opt.tickets, log_arr.ptr, opt.tickets, &self.gauge);
        if (rc == -2) _ = c.q3ni_stop(); // fewer threads than asked: join the ones that started
        if (rc != 0) return if (rc == -1) error.PoolUnavailable else error.PoolStart;
        return self;
    }

    /// Drains the queue, joins the workers, then frees what they wrote into.
    pub fn stop(self: *Pool) void {
        _ = c.q3ni_quiesce(10 * std.time.ns_per_s);
        _ = c.q3ni_stop();
        const a = self.allocator;
        std.posix.munmap(self.staging);
        a.free(self.res);
        a.free(self.log);
        a.free(self.published);
        a.destroy(self);
    }

    /// One job: `rows.len` records read by one worker, every gate/up span
    /// (`gu_offsets`, six parts) then every down span (three parts), each part
    /// `lens[c]` bytes into `rows[i][c]`. Returns the first of its 2n tickets:
    /// gate/up of record i = first + i, down = first + n + i.
    pub fn submit(self: *Pool, fd: std.c.fd_t, file_size: u64, gu_offsets: []const u64, down_offsets: []const u64, rows: []const [n_components]u64, lens: *const [n_components]u64) !u32 {
        const n = rows.len;
        if (n == 0 or n > max_items or gu_offsets.len != n or down_offsets.len != n) return error.InvalidJob;
        const count: u32 = @intCast(2 * n);
        self.drain();
        if (self.next_ticket + count > self.published.len) self.next_ticket = 0;
        const first = self.next_ticket;
        var offsets: [2 * max_items]i64 = undefined;
        var row_ptrs: [max_items][*]const u64 = undefined;
        for (0..n) |i| {
            offsets[i] = @intCast(gu_offsets[i]);
            offsets[n + i] = @intCast(down_offsets[i]);
            row_ptrs[i] = &rows[i];
        }
        var lens_i: [n_components]i64 = undefined;
        for (lens.*, &lens_i) |l, *li| li.* = @intCast(l);
        switch (c.q3ni_submit(fd, @intCast(file_size), no_deadline, @intCast(n), gu_components, n_components - gu_components, &offsets, &row_ptrs, &lens_i, first)) {
            0 => {},
            -2 => return error.TicketsBusy,
            -3 => return error.QueueFull,
            else => return error.SubmitRefused,
        }
        @memset(self.published[first..][0..count], false);
        self.next_ticket = first + count;
        return first;
    }

    /// Blocks until every ticket in [first, first + count) is in the log;
    /// `timeout_ns` bounds each wait for the next completion.
    pub fn wait(self: *Pool, first: u32, count: u32, timeout_ns: i64) !void {
        while (true) {
            self.drain();
            if (std.mem.allEqual(bool, self.published[first..][0..count], true)) return;
            if (c.q3ni_wait(self.seen, timeout_ns) == self.seen) return error.Timeout;
        }
    }

    pub fn result(self: *const Pool, ticket: u32) Result {
        const w = self.res[@as(usize, ticket) * res_w ..][0..res_w];
        return .{ .status = @enumFromInt(w[0]), .preadv_calls = w[1], .bytes_returned = w[2], .payload = w[3], .errno = w[4], .t_start_ns = w[5], .t_end_ns = w[6], .worker = w[7] };
    }

    /// Tickets in publication order for log positions [from, to).
    pub fn logOrder(self: *const Pool, from: i64, to: i64, out: []u32) []u32 {
        var n: usize = 0;
        var s = from;
        while (s < to and n < out.len) : (s += 1) {
            out[n] = @intCast(self.log[@intCast(@mod(s, @as(i64, @intCast(self.log.len))))]);
            n += 1;
        }
        return out[0..n];
    }

    /// Marks what the log published since the last call (ACQUIRE on the
    /// sequence, so the status words and slot bytes of those tickets are visible).
    fn drain(self: *Pool) void {
        const s = c.q3ni_seq();
        while (self.seen < s) : (self.seen += 1) {
            const t = self.log[@intCast(@mod(self.seen, @as(i64, @intCast(self.log.len))))];
            self.published[@intCast(t)] = true;
        }
    }
};

pub const RecordRef = struct { layer: u32, expert: u32 };

/// `records` of `bank` (one segment geometry) as one job into `rows`.
pub fn submitRecords(pool: *Pool, bank: *const expert_bank.Bank, records: []const RecordRef, rows: []const [n_components]u64) !u32 {
    if (records.len != rows.len or records.len == 0 or records.len > max_items) return error.InvalidJob;
    var lens: [n_components]u64 = undefined;
    for (&lens, bank.layers[records[0].layer].segments) |*l, s| l.* = s.length;
    var gu: [max_items]u64 = undefined;
    var down: [max_items]u64 = undefined;
    for (records, 0..) |r, i| {
        for (lens, bank.layers[r.layer].segments) |l, s| if (l != s.length) return error.MixedGeometry;
        const sp = bank.spans(r.layer, r.expert);
        gu[i] = sp.gu_offset;
        down[i] = sp.down_offset;
    }
    return pool.submit(bank.sidecar_fd, bank.sidecar_file_size, gu[0..records.len], down[0..records.len], rows, &lens);
}

// ── Tests ──

const testing = std.testing;

/// A temp file whose byte at offset o is a function of o, plus its image.
const PatternFile = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    fd: std.c.fd_t,

    fn init(len: usize) !PatternFile {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        expert_bank.fillPattern(image, 11);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pattern.bin", .data = image });
        var root: [512]u8 = undefined;
        var pbuf: [600]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/pattern.bin", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.OpenFailed;
        return .{ .tmp = tmp, .image = image, .fd = fd };
    }

    fn deinit(self: *PatternFile) void {
        _ = std.c.close(self.fd);
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

/// Nine destination rows per record, `lens[c]` bytes each, in one buffer.
const Dests = struct {
    buf: []u8,
    rows: [max_items][n_components]u64,
    at: [max_items][n_components]usize,

    fn init(n: usize, lens: *const [n_components]u64) !Dests {
        var d: Dests = .{ .buf = undefined, .rows = undefined, .at = undefined };
        var total: usize = 0;
        for (lens.*) |l| total += l;
        d.buf = try testing.allocator.alloc(u8, total * n);
        @memset(d.buf, 0xAA);
        var off: usize = 0;
        for (0..n) |i| for (lens.*, 0..) |l, k| {
            d.at[i][k] = off;
            d.rows[i][k] = @intFromPtr(d.buf.ptr) + off;
            off += l;
        };
        return d;
    }

    fn part(self: *const Dests, i: usize, k: usize, lens: *const [n_components]u64) []const u8 {
        return self.buf[self.at[i][k]..][0..lens[k]];
    }
};

test "dsv41 io: pool scatters a synthetic file" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page + 321);
    defer f.deinit();
    // One-page staging: every range needs several aligned reads.
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    // Unequal, page-unaligned parts; gate/up parts contiguous, down parts contiguous.
    const lens = [n_components]u64{ page + 3, 17, 4099, 1, 777, 2 * page + 5, 1000, page - 1, 33 };
    var gu_len: u64 = 0;
    for (lens[0..gu_components]) |l| gu_len += l;
    for ([_]usize{ 1, 3, 8 }, 0..) |n, round| {
        var d = try Dests.init(n, &lens);
        defer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (0..n) |i| {
            gu[i] = 7 + round * 13 + i * (5 * page + 11);
            down[i] = 40 * page + 3 + i * (2 * page + 7);
        }
        const seq0 = c.q3ni_seq();
        const first = try pool.submit(f.fd, f.image.len, gu[0..n], down[0..n], d.rows[0..n], &lens);
        const count: u32 = @intCast(2 * n);
        try pool.wait(first, count, 10 * std.time.ns_per_s);
        for (0..n) |i| {
            var at = gu[i];
            for (0..gu_components) |k| {
                try testing.expectEqualSlices(u8, f.image[at..][0..lens[k]], d.part(i, k, &lens));
                at += lens[k];
            }
            at = down[i];
            for (gu_components..n_components) |k| {
                try testing.expectEqualSlices(u8, f.image[at..][0..lens[k]], d.part(i, k, &lens));
                at += lens[k];
            }
            const r_gu = pool.result(first + @as(u32, @intCast(i)));
            try testing.expectEqual(Status.ok, r_gu.status);
            try testing.expectEqual(@as(i64, @intCast(gu_len)), r_gu.payload);
            try testing.expect(r_gu.preadv_calls >= @as(i64, @intCast(gu_len / page)));
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
        }
        // One worker runs the job in ticket order: every gate/up before any down.
        var order: [2 * max_items]u32 = undefined;
        const got = pool.logOrder(seq0, c.q3ni_seq(), &order);
        try testing.expectEqual(@as(usize, count), got.len);
        for (got, 0..) |t, k| try testing.expectEqual(first + @as(u32, @intCast(k)), t);
    }
}

test "dsv41 io: injected faults map to status words" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(40 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    defer c.q3ni_test_rules(0, &[_]i64{0}, &[_]i64{0}, &[_]i64{0});
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    // Every range starts 500 bytes into its own page, so its first read skips 500.
    const Case = struct { code: i64, arg: i64, hit: enum { gu1, down0 }, want: [4]Status, err: i64 = 0, hit_calls: i64 = 1 };
    const cases = [_]Case{
        // 1 EINTR: retried inside the call, uncounted.
        .{ .code = 1, .arg = 0, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok } },
        // 2 EIO: that range fails with errno, the rest of the job is skipped.
        .{ .code = 2, .arg = 0, .hit = .gu1, .want = .{ .ok, .os_error, .skipped, .skipped }, .err = @intFromEnum(std.posix.E.IO) },
        // 3 zero return: a short read.
        .{ .code = 3, .arg = 0, .hit = .down0, .want = .{ .ok, .ok, .short, .skipped } },
        // 4 truncated past the skip: 200 bytes land, the range continues.
        .{ .code = 4, .arg = 700, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok }, .hit_calls = 2 },
        // 4 truncated inside the skip: no progress, the same request is retried.
        .{ .code = 4, .arg = 300, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok }, .hit_calls = 2 },
    };
    for (cases, 0..) |cs, round| {
        // Each range on its own page, 500 bytes in, so a rule's aligned offset names one range.
        const base = round * 8 * page;
        const gu = [2]u64{ base + 500, base + 2 * page + 500 };
        const down = [2]u64{ base + 4 * page + 500, base + 6 * page + 500 };
        const hit_off: i64 = @intCast(switch (cs.hit) {
            .gu1 => base + 2 * page,
            .down0 => base + 4 * page,
        });
        c.q3ni_test_rules(1, &[_]i64{hit_off}, &[_]i64{cs.code}, &[_]i64{cs.arg});
        var d = try Dests.init(2, &lens);
        defer testing.allocator.free(d.buf);
        const first = try pool.submit(f.fd, f.image.len, &gu, &down, d.rows[0..2], &lens);
        try pool.wait(first, 4, 10 * std.time.ns_per_s);
        for (cs.want, 0..) |w, k| {
            const r = pool.result(first + @as(u32, @intCast(k)));
            testing.expectEqual(w, r.status) catch |e| {
                std.debug.print("case {d} range {d}\n", .{ round, k });
                return e;
            };
            if (w == .os_error) try testing.expectEqual(cs.err, r.errno);
            if (w == .ok) try testing.expectEqual(@as(i64, if (k < 2) 600 else 150), r.payload);
            const hit_k: usize = if (cs.hit == .gu1) 1 else 2;
            if (k == hit_k and w == .ok) try testing.expectEqual(cs.hit_calls, r.preadv_calls);
        }
        if (cs.want[1] == .ok) try testing.expectEqualSlices(u8, f.image[gu[1]..][0..100], d.part(1, 0, &lens));
    }
}

test "dsv41 io: stop joins and restarts" {
    try testing.expectEqual(@as(i32, abi_version), c.q3ni_abi());
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 });
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 }));
    pool.stop();
    pool = try Pool.start(testing.allocator, .{ .workers = 4, .staging_bytes = 9 << 20, .tickets = 16 });
    pool.stop();
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .staging_bytes = std.heap.pageSize() + 1 }));
}
