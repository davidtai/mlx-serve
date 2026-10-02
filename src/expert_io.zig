//! Lookahead read pool (lib/expert_io/q3_lookahead4_exl3.c). Its demand path is
//! the native-issue pool's: worker threads read each record's gate/up span,
//! then its down span (page-aligned F_NOCACHE preadv into a staging buffer, then
//! a scatter copy into slot rows) and publish every range's status words, then
//! its ticket in an ordered log. Three classes ride on it, each chosen at
//! construction: speculative whole-record reads (a demand submit claims the
//! record holding its range and copies it out), pre-read ranges (queued before
//! a plan, bound by the submit that needs them) and event gates (the satisfied
//! prefix goes to an MTLSharedEvent or a host word; a watchdog forces a gate
//! whose bytes never land). Workers never call MLX. The C state is static: one
//! pool per process, stopped before the next.

const std = @import("std");
const expert_bank = @import("expert_bank.zig");

const n_components = expert_bank.n_components;
const gu_components = expert_bank.gu_components;

// 1:1 mirror of lib/expert_io/q3_lookahead4.h (refusing stand-ins where the sources are not built).
const c = if (@import("build_options").macos_engines) struct {
    extern fn q3ld_spec_config(nthreads: i32, bufs: ?[*]const u64, nslots: i32, slot_bytes: i64, rec_len: i64, chunk: i64, counters: ?[*]i64) c_int;
    extern fn q3ld_spec_streams(idle_busy: i32) c_int;
    extern fn q3ld_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
    extern fn q3ld_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
    extern fn q3ld_spec_step_len(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64, len: i64) i32;
    extern fn q3ld_spec_state(out: *[spec_state_w * max_spec]i64) i32;
    extern fn q3ld_seq() i64;
    extern fn q3ld_wait(seen: i64, timeout_ns: i64) i64;
    extern fn q3ld_gauge(out: *[6]i64) void;
    extern fn q3ld_quiesce(timeout_ns: i64) c_int;
    extern fn q3ld_stop() c_int;
    extern fn q3ld_pre_config(ngu: i32, ndown: i32, lens: ?[*]const i64) c_int;
    extern fn q3ld_pre_read_lens(fd: i32, file_size: i64, tag: i64, n: i32, bases: [*]const i64, lens: [*]const i64) i32;
    extern fn q3ld_pre_state(out: *[pre_state_w * max_pre]i64) i32;
    extern fn q3ld_ev_config(kind: i32, obj: u64, timeout_ns: i64, start: u64) c_int;
    extern fn q3ld_ev_gates(n: i32, values: [*]const u64, counts: [*]const i32, tickets: [*]const i64) i32;
    extern fn q3ld_ev_release(value: u64) i32;
    extern fn q3ld_ev_state(out: *[10]i64) i32;
    extern fn q3ld_warm_config(busy_max: i32) c_int;
    extern fn q3ld_submit_warm(fd: i32, file_size: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
    extern fn q3ld_warm_cancel(first: i64, count: i64) i64;
    extern fn q3ld_monotonic_ns() i64;
    extern fn q3ld_abi() i32;
    extern fn q3ld_counters_n() i32;
    extern fn q3ld_max_spec() i32;
    extern fn q3ld_max_pre() i32;
    extern fn q3ld_max_gates() i32;
    extern fn q3ld_max_gate_tickets() i32;
    /// Q3LD_INJECT builds (the test module) only.
    extern fn q3ld_test_rules(n: i32, off: [*]const i64, code: [*]const i64, arg: [*]const i64) void;
    extern fn q3ld_test_delay(seed: u64, max_ns: i64) void;
    extern fn q3ld_test_ev_log(buf: ?[*]i64, cap: i64) i64;
} else @import("expert_io_stub.zig").q3ld;

pub const abi_version = 2026100101;
pub const max_workers = 8;
/// Records per job (one fill unit).
pub const max_items = 8;
pub const max_spec = 16;
pub const max_spec_threads = 3;
pub const max_pre = 32;
pub const max_gates = 256;
pub const max_gate_tickets = 256;
const res_w = 8;
pub const spec_state_w = 11;
pub const pre_state_w = 5;
/// `q3ld_submit` reads -1 as no deadline; 0 would expire every range at once.
const no_deadline: i64 = -1;

pub const Status = enum(i64) { ok = 0, short = 1, os_error = 2, deadline = 3, skipped = 4, pending = -1, _ };

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

/// The pool's counter words (q3_lookahead4_exl3.c `SC_*`), written under its
/// mutex; the h_* entries are the first of four horizon classes.
pub const Counter = enum(u8) {
    submitted = 0,
    refreshed = 1,
    noslot = 2,
    started = 3,
    landed = 4,
    failed = 5,
    expired = 6,
    cancelled_by_demand = 7,
    claimed = 8,
    claimed_inflight = 9,
    abandoned = 10,
    abandoned_bytes = 11,
    discarded = 12,
    adopt_ranges = 13,
    adopt_bytes = 14,
    adopt_waits = 15,
    adopt_wait_ns = 16,
    spec_bytes = 17,
    spec_chunks = 18,
    pauses = 19,
    max_busy_at_start = 20,
    max_qlen_at_start = 21,
    parks = 22,
    max_nearer_at_start = 23,
    h_submitted = 24,
    h_kept = 28,
    h_dropped = 32,
    h_claimed = 36,
    h_adopt_ranges = 40,
    idle_chunks = 44,
    max_busy_at_start_idle = 45,
    pre_calls = 46,
    pre_issued = 47,
    pre_noslot = 48,
    pre_claims = 49,
    pre_started = 50,
    pre_bound = 51,
    pre_cancelled = 52,
    pre_expired = 53,
    pre_served = 54,
    pre_waits = 55,
    pre_wait_ns = 56,
    pre_bind_wait_ns = 57,
    pre_skip_cancels = 58,
    pre_max_inflight = 59,
    ev_calls = 60,
    ev_gates = 61,
    ev_tickets = 62,
    ev_immediate = 63,
    ev_signals = 64,
    ev_signal_ns = 65,
    ev_lag_ns = 66,
    ev_max_live = 67,
    ev_wd_forced = 68,
    ev_wd_last_value = 69,
    ev_host_released = 70,
    ev_stop_released = 71,
    warm_submitted = 72,
    warm_started = 73,
    warm_cancelled = 74,
    warm_max_busy_at_start = 75,
};
pub const counters_n = 76;

/// The speculative class: `slots` staging slots of `slotBytes(record_bytes)`,
/// read by `threads` threads in `chunk_bytes` preadv steps.
pub const Spec = struct {
    threads: u32,
    slots: u32,
    /// The largest record's payload; a speculative read covers one record.
    record_bytes: u64,
    chunk_bytes: u64,
    /// Queue rule of speculative threads >= 1: an unclaimed chunk starts only
    /// while at most this many demand jobs execute (0 = demand idle).
    idle_busy: u32 = 0,
};

/// The warm class (A0 (a)): stock jobs a worker starts only while no demand job or pre-range is queued and fewer than
/// `busy_max` jobs run, on `tickets` tickets of their own at the top of the ring (demand wraps below them).
pub const Warm = struct { tickets: u32, busy_max: u32 };

pub const Options = struct {
    workers: u32 = 4,
    /// One page-aligned staging buffer per worker; 9 MiB holds a whole
    /// 8,877,056-byte gate/up span plus its page alignment.
    staging_bytes: u64 = 9 << 20,
    tickets: u32 = 256,
    spec: ?Spec = null,
    warm: ?Warm = null,
};

/// A speculative staging slot: the page-rounded record plus two pages.
pub fn slotBytes(record_bytes: u64, page: u64) u64 {
    return std.mem.alignForward(u64, record_bytes, page) + 2 * page;
}

/// Page-aligned chunk so `chunks` chunks cover a staging slot.
pub fn chunkBytes(chunks: u32, record_bytes: u64, page: u64) u64 {
    return (std.math.divCeil(u64, slotBytes(record_bytes, page), chunks * page) catch unreachable) * page;
}

pub const EventKind = enum(i32) { metal = 1, host = 2 };

pub const Pool = struct {
    allocator: std.mem.Allocator,
    staging: []align(std.heap.page_size_min) u8,
    spec_staging: ?[]align(std.heap.page_size_min) u8 = null,
    /// [ticket * res_w + k], written by the workers; zeroed before start.
    res: []i64,
    /// Completion order: log[seq % tickets] = ticket.
    log: []i64,
    gauge: [6]i64 = @splat(0),
    /// Written by the pool under its mutex until `stop`.
    counters: [counters_n]i64 = @splat(0),
    /// Tickets seen in the log since their last submit.
    published: []bool,
    /// Log entries consumed so far.
    seen: i64 = 0,
    next_ticket: u32 = 0,
    /// Demand's tickets end here; the warm class's (`Options.warm`) run from here to the end of the ring.
    demand_tickets: u32 = 0,
    next_warm: u32 = 0,
    record_bytes: u64 = 0,

    /// Starts the process's pool (the speculative class, when given, is
    /// configured first). Staging, status words, counters and the log live
    /// here and stay put until `stop`.
    pub fn start(allocator: std.mem.Allocator, opt: Options) !*Pool {
        const page = std.heap.pageSize();
        if (opt.workers == 0 or opt.workers > max_workers or opt.staging_bytes == 0 or opt.staging_bytes % page != 0 or opt.tickets < 2 * max_items)
            return error.InvalidOptions;
        if (opt.spec) |s| if (s.threads == 0 or s.threads > max_spec_threads or s.slots == 0 or s.slots > max_spec or s.record_bytes == 0 or
            s.chunk_bytes == 0 or s.chunk_bytes % page != 0 or s.idle_busy > 1) return error.InvalidOptions;
        if (opt.warm) |w| if (w.tickets == 0 or w.tickets % 2 != 0 or w.tickets > opt.tickets -| 2 * max_items or w.busy_max == 0 or
            w.busy_max > opt.workers) return error.InvalidOptions;
        if (c.q3ld_abi() != abi_version or c.q3ld_counters_n() != counters_n or c.q3ld_max_spec() != max_spec or c.q3ld_max_pre() != max_pre or
            c.q3ld_max_gates() != max_gates or c.q3ld_max_gate_tickets() != max_gate_tickets) return error.PoolAbi;
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
        const demand: u32 = if (opt.warm) |w| opt.tickets - w.tickets else opt.tickets;
        self.* = .{ .allocator = allocator, .staging = staging, .res = res, .log = log_arr, .published = published, .demand_tickets = demand, .next_warm = demand };
        var bufs: [max_spec]u64 = undefined;
        var threads: i32 = 0;
        var slot_bytes: u64 = 0;
        if (opt.spec) |s| {
            slot_bytes = slotBytes(s.record_bytes, page);
            const spec = try std.posix.mmap(null, @intCast(s.slots * slot_bytes), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            self.spec_staging = spec;
            for (0..s.slots) |i| bufs[i] = @intFromPtr(spec.ptr) + i * slot_bytes;
            threads = @intCast(s.threads);
            self.record_bytes = s.record_bytes;
        }
        errdefer if (self.spec_staging) |s| std.posix.munmap(s);
        // Arguments are valid here, so a refusal means a pool is running.
        const nslots: i32 = if (opt.spec) |s| @intCast(s.slots) else 0;
        const rec_len: i64 = @intCast(self.record_bytes);
        if (c.q3ld_spec_config(threads, &bufs, nslots, @intCast(slot_bytes), rec_len, @intCast(if (opt.spec) |s| s.chunk_bytes else 0), &self.counters) != 0)
            return error.PoolUnavailable;
        if (opt.spec) |s| if (c.q3ld_spec_streams(@intCast(s.idle_busy)) != 0) return error.PoolUnavailable;
        var ptrs: [max_workers]u64 = undefined;
        for (0..opt.workers) |w| ptrs[w] = @intFromPtr(staging.ptr) + w * opt.staging_bytes;
        const rc = c.q3ld_start(@intCast(opt.workers), &ptrs, @intCast(opt.staging_bytes), @intCast(page), res.ptr, opt.tickets, log_arr.ptr, opt.tickets, &self.gauge);
        if (rc == -2) _ = c.q3ld_stop(); // fewer threads than asked: join the ones that started
        if (rc != 0) return if (rc == -1) error.PoolUnavailable else error.PoolStart;
        if (opt.warm) |w| if (c.q3ld_warm_config(@intCast(w.busy_max)) != 0) {
            _ = c.q3ld_stop();
            return error.WarmRefused;
        };
        return self;
    }

    /// Drains the queue, joins the workers (releasing every live gate), then
    /// frees what they wrote into.
    pub fn stop(self: *Pool) void {
        _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
        _ = c.q3ld_stop();
        const a = self.allocator;
        std.posix.munmap(self.staging);
        if (self.spec_staging) |s| std.posix.munmap(s);
        a.free(self.res);
        a.free(self.log);
        a.free(self.published);
        a.destroy(self);
    }

    /// Arms the pre-read class with one layer geometry's component lengths
    /// (needs the speculative class).
    pub fn armPreRead(self: *Pool, lens: *const [n_components]u64) !void {
        _ = self;
        var l: [n_components]i64 = undefined;
        for (lens, &l) |x, *y| y.* = @intCast(x);
        if (c.q3ld_pre_config(gu_components, n_components - gu_components, &l) != 0) return error.PreReadRefused;
    }

    /// Arms the event-gate class: the satisfied prefix goes to `object` (an
    /// id<MTLSharedEvent>, or an 8-aligned int64 host word) from the
    /// publishing worker; a gate unsatisfied after `timeout_ns` is forced.
    pub fn armEvent(self: *Pool, kind: EventKind, object: u64, timeout_ns: i64, start_value: u64) !void {
        _ = self;
        if (c.q3ld_ev_config(@intFromEnum(kind), object, timeout_ns, start_value) != 0) return error.EventRefused;
    }

    /// One job: `rows.len` records read by one worker, every gate/up span
    /// (`gu_offsets`, six parts) then every down span (three parts), each part
    /// `lens[c]` bytes into `rows[i][c]`. Returns the first of its 2n tickets:
    /// gate/up of record i = first + i, down = first + n + i.
    pub fn submit(self: *Pool, fd: std.c.fd_t, file_size: u64, gu_offsets: []const u64, down_offsets: []const u64, rows: []const [n_components]u64, lens: *const [n_components]u64) !u32 {
        return self.submitSplit(fd, file_size, gu_offsets, down_offsets, rows, lens, gu_components, n_components - gu_components);
    }

    /// `submit` for records of another part geometry: each gate/up span `ngu` parts, each down span `ndown`
    /// (`lens[0..ngu]`, then `lens[ngu..][0..ndown]`; the rest of `rows[i]` and `lens` unused).
    pub fn submitSplit(self: *Pool, fd: std.c.fd_t, file_size: u64, gu_offsets: []const u64, down_offsets: []const u64, rows: []const [n_components]u64, lens: *const [n_components]u64, ngu: u32, ndown: u32) !u32 {
        const n = rows.len;
        if (n == 0 or n > max_items or gu_offsets.len != n or down_offsets.len != n or ngu == 0 or ndown == 0 or ngu + ndown > n_components) return error.InvalidJob;
        const count: u32 = @intCast(2 * n);
        self.drain();
        if (self.next_ticket + count > self.demand_tickets) self.next_ticket = 0;
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
        switch (c.q3ld_submit(fd, @intCast(file_size), no_deadline, @intCast(n), @intCast(ngu), @intCast(ndown), &offsets, &row_ptrs, &lens_i, first)) {
            0 => {},
            -2 => return error.TicketsBusy,
            -3 => return error.QueueFull,
            else => return error.SubmitRefused,
        }
        @memset(self.published[first..][0..count], false);
        self.next_ticket = first + count;
        return first;
    }

    /// The warm class (`Options.warm`): one job queued below demand on the warm tickets, laid out as `submit`'s
    /// (gate/up of record i = first + i, down = first + n + i). Returns its first ticket.
    pub fn submitWarm(self: *Pool, fd: std.c.fd_t, file_size: u64, gu_offsets: []const u64, down_offsets: []const u64, rows: []const [n_components]u64, lens: *const [n_components]u64) !u32 {
        const n = rows.len;
        if (n == 0 or n > max_items or gu_offsets.len != n or down_offsets.len != n or self.demand_tickets == self.published.len) return error.InvalidJob;
        const count: u32 = @intCast(2 * n);
        self.drain();
        if (self.next_warm + count > self.published.len) self.next_warm = self.demand_tickets;
        const first = self.next_warm;
        var offsets: [2 * max_items]i64 = undefined;
        var row_ptrs: [max_items][*]const u64 = undefined;
        for (0..n) |i| {
            offsets[i] = @intCast(gu_offsets[i]);
            offsets[n + i] = @intCast(down_offsets[i]);
            row_ptrs[i] = &rows[i];
        }
        var lens_i: [n_components]i64 = undefined;
        for (lens.*, &lens_i) |l, *li| li.* = @intCast(l);
        switch (c.q3ld_submit_warm(fd, @intCast(file_size), @intCast(n), gu_components, n_components - gu_components, &offsets, &row_ptrs, &lens_i, first)) {
            0 => {},
            -2 => return error.TicketsBusy,
            -3 => return error.QueueFull,
            else => return error.SubmitRefused,
        }
        @memset(self.published[first..][0..count], false);
        self.next_warm = first + count;
        return first;
    }

    /// The queued warm jobs overlapping tickets [first, first + count), published skipped now (a started job
    /// finishes). Returns the tickets cancelled.
    pub fn cancelWarm(_: *Pool, first: u32, count: u32) u32 {
        const n = c.q3ld_warm_cancel(first, count);
        return if (n < 0) 0 else @intCast(n);
    }

    /// A layer call's speculative step: settles every unclaimed record tagged
    /// <= `cur`, then queues `bases` (whole records of `len` bytes) tagged
    /// `cur + 1`. Returns how many were newly queued.
    pub fn specStep(self: *Pool, fd: std.c.fd_t, file_size: u64, cur: i64, bases: []const i64, len: u64) !u32 {
        const len_i: i64 = @intCast(if (len == 0) self.record_bytes else len);
        const rc = c.q3ld_spec_step_len(fd, @intCast(file_size), cur, @intCast(bases.len), bases.ptr, len_i);
        if (rc < 0) return error.SpecRefused;
        return @intCast(rc);
    }

    /// A layer call's certain misses (record bases, route order) before its
    /// plan, as gate/up and down ranges of `lens`; `tag` = the call's settle
    /// value. Returns how many ranges were queued.
    pub fn preRead(self: *Pool, fd: std.c.fd_t, file_size: u64, tag: i64, bases: []const i64, lens: *const [n_components]u64) !u32 {
        _ = self;
        var l: [n_components]i64 = undefined;
        for (lens, &l) |x, *y| y.* = @intCast(x);
        const rc = c.q3ld_pre_read_lens(fd, @intCast(file_size), tag, @intCast(bases.len), bases.ptr, &l);
        if (rc < 0) return error.PreReadRefused;
        return @intCast(rc);
    }

    /// Registers gates in the GPU's encode order: gate i waits for `counts[i]`
    /// of `tickets` (in order); values strictly increase above every earlier one.
    pub fn registerGates(self: *Pool, values: []const u64, counts: []const i32, tickets: []const i64) !void {
        _ = self;
        std.debug.assert(values.len == counts.len and values.len > 0);
        switch (c.q3ld_ev_gates(@intCast(values.len), values.ptr, counts.ptr, tickets.ptr)) {
            -2 => return error.GateInvalid,
            -3 => return error.GatesFull,
            else => |rc| if (rc != @as(i32, @intCast(values.len))) return error.GateRefused,
        }
    }

    /// Forces every live gate <= `value` (error paths: nothing may wait for it).
    pub fn releaseGates(self: *Pool, value: u64) void {
        _ = self;
        _ = c.q3ld_ev_release(value);
    }

    pub fn counter(self: *const Pool, which: Counter) i64 {
        return @atomicLoad(i64, &self.counters[@intFromEnum(which)], .monotonic);
    }

    /// Blocks until every ticket in [first, first + count) is in the log;
    /// `timeout_ns` bounds each wait for the next completion.
    pub fn wait(self: *Pool, first: u32, count: u32, timeout_ns: i64) !void {
        while (true) {
            self.drain();
            if (std.mem.allEqual(bool, self.published[first..][0..count], true)) return;
            if (c.q3ld_wait(self.seen, timeout_ns) == self.seen) return error.Timeout;
        }
    }

    /// A copy of the read gauge (taken under the pool mutex): 0 in flight, 1 max in flight, 2 depth sum,
    /// 3 samples, 4 wall ns with a read in flight, 5 busy-since ns.
    pub fn readGauge(self: *Pool) [6]i64 {
        _ = self;
        var out: [6]i64 = undefined;
        c.q3ld_gauge(&out);
        return out;
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
        const s = c.q3ld_seq();
        while (self.seen < s) : (self.seen += 1) {
            const t = self.log[@intCast(@mod(self.seen, @as(i64, @intCast(self.log.len))))];
            self.published[@intCast(t)] = true;
        }
    }
};

/// The reader's monotonic clock (ns), the one its result words use.
pub fn monotonicNs() i64 {
    return c.q3ld_monotonic_ns();
}

/// Test builds (-DQ3LD_INJECT): one scripted preadv fault at an aligned file
/// offset (code 1 EINTR, 2 EIO, 3 zero return, 4 truncate to `arg` bytes, 5
/// sleep `arg` ns first).
pub fn injectFault(aligned_offset: u64, code: i64, arg: i64) void {
    c.q3ld_test_rules(1, &[_]i64{@intCast(aligned_offset)}, &[_]i64{code}, &[_]i64{arg});
}

pub fn injectFaults(aligned_offsets: []const i64, codes: []const i64, args: []const i64) void {
    c.q3ld_test_rules(@intCast(aligned_offsets.len), aligned_offsets.ptr, codes.ptr, args.ptr);
}

pub fn clearFaults() void {
    c.q3ld_test_rules(0, &[_]i64{0}, &[_]i64{0}, &[_]i64{0});
}

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

    /// Record i's nine rows hold the file's bytes at `gu` (six parts) and `down` (three).
    fn expectRecord(self: *const Dests, i: usize, image: []const u8, gu: u64, down: u64, lens: *const [n_components]u64) !void {
        var at = gu;
        for (0..n_components) |k| {
            if (k == gu_components) at = down;
            try testing.expectEqualSlices(u8, image[at..][0..lens[k]], self.part(i, k, lens));
            at += lens[k];
        }
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
        const seq0 = c.q3ld_seq();
        const first = try pool.submit(f.fd, f.image.len, gu[0..n], down[0..n], d.rows[0..n], &lens);
        const count: u32 = @intCast(2 * n);
        try pool.wait(first, count, 10 * std.time.ns_per_s);
        for (0..n) |i| {
            try d.expectRecord(i, f.image, gu[i], down[i], &lens);
            const r_gu = pool.result(first + @as(u32, @intCast(i)));
            try testing.expectEqual(Status.ok, r_gu.status);
            try testing.expectEqual(@as(i64, @intCast(gu_len)), r_gu.payload);
            try testing.expect(r_gu.preadv_calls >= @as(i64, @intCast(gu_len / page)));
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
        }
        // One worker runs the job in ticket order: every gate/up before any down.
        var order: [2 * max_items]u32 = undefined;
        const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
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
    defer clearFaults();
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
        const hit_off: u64 = switch (cs.hit) {
            .gu1 => base + 2 * page,
            .down0 => base + 4 * page,
        };
        injectFault(hit_off, cs.code, cs.arg);
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
    try testing.expectEqual(@as(i32, abi_version), c.q3ld_abi());
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 });
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 }));
    pool.stop();
    pool = try Pool.start(testing.allocator, .{ .workers = 4, .staging_bytes = 9 << 20, .tickets = 16 });
    pool.stop();
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .staging_bytes = std.heap.pageSize() + 1 }));
    const bad_spec: Spec = .{ .threads = 1, .slots = max_spec + 1, .record_bytes = 100, .chunk_bytes = std.heap.pageSize() };
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .spec = bad_spec }));
}

test "dsv41 io: A0 (a): warm jobs start only with demand idle, after a demand job submitted later, and land a demand read's bytes" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    // One worker: the order is the queue rule's. Warm tickets 16..31 above demand's 0..15.
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    // The first demand job holds the worker 80 ms (its first range sleeps), so everything below queues behind it.
    injectFault(0, 5, 80 * std.time.ns_per_ms);
    var d = try Dests.init(4, &lens);
    defer testing.allocator.free(d.buf);
    const seq0 = c.q3ld_seq();
    const j0 = try pool.submit(f.fd, f.image.len, &.{500}, &.{700}, d.rows[0..1], &lens);
    const w0 = try pool.submitWarm(f.fd, f.image.len, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
    const w1 = try pool.submitWarm(f.fd, f.image.len, &.{4 * page + 500}, &.{4 * page + 700}, d.rows[2..3], &lens);
    const j1 = try pool.submit(f.fd, f.image.len, &.{6 * page + 500}, &.{6 * page + 700}, d.rows[3..4], &lens);
    try testing.expect(w0 >= 16 and w1 == w0 + 2 and j1 == j0 + 2 and j1 < 16);
    try pool.wait(j0, 2, 10 * std.time.ns_per_s);
    try pool.wait(j1, 2, 10 * std.time.ns_per_s);
    try pool.wait(w0, 4, 10 * std.time.ns_per_s);
    // Completion order: the held job, the demand job submitted after the warm ones, then the warm jobs.
    var order: [8]u32 = undefined;
    const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
    try testing.expectEqualSlices(u32, &.{ j0, j0 + 1, j1, j1 + 1, w0, w0 + 1, w1, w1 + 1 }, got);
    for ([_]u64{ 500, 2 * page + 500, 4 * page + 500, 6 * page + 500 }, 0..) |gu, i| try d.expectRecord(i, f.image, gu, gu + 200, &lens);
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_submitted));
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_started));
    try testing.expectEqual(@as(i64, 0), pool.counter(.warm_cancelled));
    try testing.expectEqual(@as(i64, 0), pool.counter(.warm_max_busy_at_start));
}

test "dsv41 io: A0 (a): a warm cancel publishes the queued jobs skipped at once and lets a started one land" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    var d = try Dests.init(3, &lens);
    defer testing.allocator.free(d.buf);
    // The started one: its first range sleeps 80 ms, so the two after it stay queued (one worker).
    injectFault(0, 5, 80 * std.time.ns_per_ms);
    const w0 = try pool.submitWarm(f.fd, f.image.len, &.{500}, &.{700}, d.rows[0..1], &lens);
    const w1 = try pool.submitWarm(f.fd, f.image.len, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
    const w2 = try pool.submitWarm(f.fd, f.image.len, &.{4 * page + 500}, &.{4 * page + 700}, d.rows[2..3], &lens);
    while (pool.counter(.warm_started) == 0) std.Thread.yield() catch {};
    try testing.expectEqual(@as(u32, 4), pool.cancelWarm(w0, w2 + 2 - w0));
    try pool.wait(w1, 4, std.time.ns_per_s);
    for (0..4) |k| try testing.expectEqual(Status.skipped, pool.result(w1 + @as(u32, @intCast(k))).status);
    // The cancelled rows were never written; the started one lands.
    try testing.expect(std.mem.allEqual(u8, d.part(1, 0, &lens), 0xAA) and std.mem.allEqual(u8, d.part(2, 8, &lens), 0xAA));
    try pool.wait(w0, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(w0).status);
    try testing.expectEqual(Status.ok, pool.result(w0 + 1).status);
    try d.expectRecord(0, f.image, 500, 700, &lens);
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_cancelled));
    try testing.expectEqual(@as(u32, 0), pool.cancelWarm(w0, 2));
}

test "dsv41 io: A0 (a): demand wraps below the warm tickets; a pool stopping with warm jobs queued returns" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    var d = try Dests.init(2, &lens);
    defer testing.allocator.free(d.buf);
    {
        var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
        defer pool.stop();
        // Demand's 16 tickets: eight 1-record jobs, then the ninth wraps to 0, never into the warm tickets.
        for (0..9) |k| {
            const t = try pool.submit(f.fd, f.image.len, &.{500}, &.{700}, d.rows[0..1], &lens);
            try testing.expectEqual(@as(u32, @intCast((2 * k) % 16)), t);
            try pool.wait(t, 2, 10 * std.time.ns_per_s);
        }
        const w = try pool.submitWarm(f.fd, f.image.len, &.{500}, &.{700}, d.rows[1..2], &lens);
        try testing.expectEqual(@as(u32, 16), w);
        try pool.wait(w, 2, 10 * std.time.ns_per_s);
    }
    {
        var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
        defer clearFaults();
        injectFault(0, 5, 50 * std.time.ns_per_ms);
        _ = try pool.submit(f.fd, f.image.len, &.{500}, &.{700}, d.rows[0..1], &lens);
        _ = try pool.submitWarm(f.fd, f.image.len, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
        pool.stop();
    }
    // Refusals at construction: odd or oversized warm tickets, a busy limit of 0 or past the workers.
    for ([_]Warm{ .{ .tickets = 3, .busy_max = 1 }, .{ .tickets = 32, .busy_max = 1 }, .{ .tickets = 8, .busy_max = 0 }, .{ .tickets = 8, .busy_max = 3 } }) |bad|
        try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32, .warm = bad }));
    // Without the class, a warm submit is refused by name.
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32 });
    defer pool.stop();
    try testing.expectError(error.InvalidJob, pool.submitWarm(f.fd, f.image.len, &.{500}, &.{700}, d.rows[0..1], &lens));
}

test "dsv41 io: speculative slots and chunks follow the lane's sizes" {
    // q3_lookahead4_candidate.slot_bytes / chunk_bytes on the 3.0 bank's 13,315,584-byte record, 16 KiB pages.
    try testing.expectEqual(@as(u64, 13_352_960), slotBytes(13_315_584, 16384));
    try testing.expectEqual(@as(u64, 3_342_336), chunkBytes(4, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 13_352_960), chunkBytes(1, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 49_152), slotBytes(2880, 16384));
    try testing.expectEqual(@as(u64, 16_384), chunkBytes(4, 2880, 16384));
}

// Speculative class. One record = a gate/up range then a down range, back to back.
const spec_lens = [n_components]u64{ 3000, 100, 200, 3000, 100, 200, 3000, 200, 100 };
const spec_gu_len = 6600;
const spec_rec_len = 9900;

fn specPool(workers: u32, slots: u32) !*Pool {
    const page = std.heap.pageSize();
    return Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 128, .spec = .{
        .threads = 1,
        .slots = slots,
        .record_bytes = spec_rec_len,
        .chunk_bytes = page,
    } });
}

const SpecSlot = struct { state: i64, tag: i64, base: i64, landed: i64, claimed: i64 };

fn specSlots(out: *[max_spec]SpecSlot) []SpecSlot {
    var raw: [spec_state_w * max_spec]i64 = undefined;
    const n: usize = @intCast(c.q3ld_spec_state(&raw));
    for (out[0..n], 0..) |*s, i| {
        const w = raw[i * spec_state_w ..][0..spec_state_w];
        s.* = .{ .state = w[0], .tag = w[1], .base = w[2], .landed = w[3], .claimed = w[6] };
    }
    return out[0..n];
}

/// Polls until `pred` holds (10 s).
fn waitFor(ctx: anytype, comptime pred: fn (@TypeOf(ctx)) bool) !void {
    var t: u32 = 0;
    while (!pred(ctx)) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn landedAt(base: i64) bool {
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == base and s.state == 3) return true;
    return false;
}

test "dsv41 io: a landed speculative record serves the demand read with no preadv" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(32 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    const base: u64 = 3 * page + 100;
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.fd, f.image.len, 1, &.{@intCast(base)}, spec_rec_len));
    // A second issue of a live record only refreshes it.
    try testing.expectEqual(@as(u32, 0), try pool.specStep(f.fd, f.image.len, 1, &.{@intCast(base)}, spec_rec_len));
    try waitFor(@as(i64, @intCast(base)), landedAt);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try pool.submit(f.fd, f.image.len, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    for (0..2) |k| {
        const r = pool.result(first + @as(u32, @intCast(k)));
        try testing.expectEqual(Status.ok, r.status);
        try testing.expectEqual(@as(i64, 0), r.preadv_calls);
    }
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed));
    try testing.expectEqual(@as(i64, 2), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, spec_rec_len), pool.counter(.adopt_bytes));
    try testing.expectEqual(@as(i64, 1), pool.counter(.refreshed));
    try testing.expect(pool.counter(.spec_bytes) >= spec_rec_len);
}

test "dsv41 io: a queued speculative record is cancelled by the demand read; settle drops the unclaimed" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(2, 4);
    defer pool.stop();
    defer clearFaults();
    // Two slow demand jobs keep both workers busy, so no unclaimed record may start.
    const slow = [2]u64{ 20 * page, 30 * page };
    injectFaults(&.{ @intCast(slow[0]), @intCast(slow[1]) }, &.{ 5, 5 }, &.{ 300 * std.time.ns_per_ms, 300 * std.time.ns_per_ms });
    var busy = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(busy.buf);
    const j0 = try pool.submit(f.fd, f.image.len, slow[0..1], &.{slow[0] + spec_gu_len}, busy.rows[0..1], &spec_lens);
    const j1 = try pool.submit(f.fd, f.image.len, slow[1..2], &.{slow[1] + spec_gu_len}, busy.rows[1..2], &spec_lens);
    const a: u64 = 2 * page + 7;
    const b: u64 = 6 * page + 9;
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.fd, f.image.len, 1, &.{ @intCast(a), @intCast(b) }, spec_rec_len));
    // The demand read of `a` cancels its queued record and reads it itself.
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try pool.submit(f.fd, f.image.len, &.{a}, &.{a + spec_gu_len}, d.rows[0..1], &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.cancelled_by_demand));
    // The next layer call's settle drops `b`, never claimed.
    _ = try pool.specStep(f.fd, f.image.len, 2, &.{}, spec_rec_len);
    try testing.expectEqual(@as(i64, 1), pool.counter(.expired));
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| try testing.expectEqual(@as(i64, 0), s.state);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try pool.wait(j0, 2, 10 * std.time.ns_per_s);
    try pool.wait(j1, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, a, a + spec_gu_len, &spec_lens);
    try testing.expect(pool.result(first).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 0), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, 0), pool.counter(.started));
}

fn preCount(p: *Pool) bool {
    return p.counter(.pre_started) >= 4;
}

fn preTwo(p: *Pool) bool {
    return p.counter(.pre_started) >= 2;
}

fn preIdle(p: *Pool) bool {
    _ = p;
    var raw: [pre_state_w * max_pre]i64 = undefined;
    _ = c.q3ld_pre_state(&raw);
    for (0..max_pre) |i| if (raw[i * pre_state_w] != 0) return false;
    return true;
}

test "dsv41 io: pre-read ranges bind to the demand submit; unbound ones expire at the settle" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(4, 2);
    defer pool.stop();
    try pool.armPreRead(&spec_lens);
    // Bound: two records pre-read (4 ranges, each worker holds one at its first copy), then demanded.
    const recs = [2]u64{ 5 * page + 11, 9 * page + 13 };
    try testing.expectEqual(@as(u32, 4), try pool.preRead(f.fd, f.image.len, 1, &.{ @intCast(recs[0]), @intCast(recs[1]) }, &spec_lens));
    try waitFor(pool, preCount);
    var d = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try pool.submit(f.fd, f.image.len, &recs, &.{ recs[0] + spec_gu_len, recs[1] + spec_gu_len }, d.rows[0..2], &spec_lens);
    try pool.wait(first, 4, 10 * std.time.ns_per_s);
    for (0..2) |i| try d.expectRecord(i, f.image, recs[i], recs[i] + spec_gu_len, &spec_lens);
    for (0..4) |k| try testing.expect(pool.result(first + @as(u32, @intCast(k))).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_bound));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_cancelled));
    _ = try pool.specStep(f.fd, f.image.len, 1, &.{}, spec_rec_len);
    // Unbound: pre-read, never demanded; the call's settle expires both ranges and frees their workers.
    const lone: u64 = 20 * page + 5;
    try testing.expectEqual(@as(u32, 2), try pool.preRead(f.fd, f.image.len, 2, &.{@intCast(lone)}, &spec_lens));
    try waitFor(pool, preCount6);
    _ = try pool.specStep(f.fd, f.image.len, 2, &.{}, spec_rec_len);
    try waitFor(pool, preIdle);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
}

fn preCount6(p: *Pool) bool {
    return p.counter(.pre_started) >= 6;
}

// Event gates on a host word. Gate/up spans cross a page, so a down range's
// first preadv never shares an aligned offset with its record's gate/up read.
const ev_lens = [n_components]u64{ 8000, 100, 200, 8000, 100, 200, 3000, 200, 100 };
const ev_gu_len = 16600;

const EvRig = struct {
    f: PatternFile,
    pool: *Pool,
    word: *i64,
    dests: std.ArrayList(Dests) = .empty,
    value: u64 = 0,

    fn init(timeout_ms: i64, workers: u32) !EvRig {
        const page = std.heap.pageSize();
        var f = try PatternFile.init(96 * page);
        errdefer f.deinit();
        const word = try testing.allocator.create(i64);
        errdefer testing.allocator.destroy(word);
        word.* = 0;
        const pool = try Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 512 });
        errdefer pool.stop();
        try pool.armEvent(.host, @intFromPtr(word), timeout_ms * std.time.ns_per_ms, 0);
        return .{ .f = f, .pool = pool, .word = word };
    }

    fn deinit(self: *EvRig) void {
        // The pool writes the word until it stops.
        self.pool.stop();
        for (self.dests.items) |d| testing.allocator.free(d.buf);
        self.dests.deinit(testing.allocator);
        testing.allocator.destroy(self.word);
        self.f.deinit();
    }

    fn base(r: u64) u64 {
        return 2 * std.heap.pageSize() + r * 3 * std.heap.pageSize() + 17 * r;
    }

    /// One job of records `recs`; returns its first ticket.
    fn job(self: *EvRig, recs: []const u64) !u32 {
        var d = try Dests.init(recs.len, &ev_lens);
        errdefer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (recs, 0..) |r, i| {
            gu[i] = base(r);
            down[i] = base(r) + ev_gu_len;
        }
        const first = try self.pool.submit(self.f.fd, self.f.image.len, gu[0..recs.len], down[0..recs.len], d.rows[0..recs.len], &ev_lens);
        try self.dests.append(testing.allocator, d);
        return first;
    }

    /// Gates over ticket groups, values continuing from the last one.
    fn register(self: *EvRig, groups: []const []const i64) !void {
        var values: [16]u64 = undefined;
        var counts: [16]i32 = undefined;
        var tickets: [256]i64 = undefined;
        var k: usize = 0;
        for (groups, 0..) |g, i| {
            values[i] = self.value + 1 + i;
            counts[i] = @intCast(g.len);
            @memcpy(tickets[k..][0..g.len], g);
            k += g.len;
        }
        try self.pool.registerGates(values[0..groups.len], counts[0..groups.len], tickets[0..k]);
        self.value += groups.len;
    }

    fn waitWord(self: *EvRig, value: u64, timeout_ms: u32) !void {
        var t: u32 = 0;
        while (@as(u64, @intCast(@atomicLoad(i64, self.word, .acquire))) < value) : (t += 1) {
            if (t > timeout_ms * 2) return error.Timeout;
            std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
        }
    }
};

test "dsv41 io: event gates hand the satisfied prefix to the event in order" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    var log: [4 * 64]i64 = undefined;
    _ = c.q3ld_test_ev_log(&log, 64);
    defer _ = c.q3ld_test_ev_log(null, -1);
    c.q3ld_test_delay(11, 2 * std.time.ns_per_ms);
    defer c.q3ld_test_delay(0, 0);
    // Three jobs; gate 1 = every gate/up ticket, then one gate per job's down tickets.
    var firsts: [3]u32 = undefined;
    for (&firsts, 0..) |*fst, j| {
        const r: u64 = 2 * j;
        fst.* = try rig.job(&.{ r, r + 1 });
    }
    var gu: [6]i64 = undefined;
    var downs: [3][2]i64 = undefined;
    for (firsts, 0..) |fst, j| {
        gu[2 * j] = fst;
        gu[2 * j + 1] = fst + 1;
        downs[j] = .{ fst + 2, fst + 3 };
    }
    const groups = [_][]const i64{ &gu, &downs[0], &downs[1], &downs[2] };
    try rig.register(&groups);
    try rig.waitWord(4, 10_000);
    for (firsts) |fst| try rig.pool.wait(fst, 4, 10 * std.time.ns_per_s);
    // Every signal value is new and higher, and every gate at or below it had all its tickets published before the call.
    const n: usize = @intCast(c.q3ld_test_ev_log(null, 0));
    try testing.expect(n >= 1);
    var prev: i64 = 0;
    for (0..n) |i| {
        const e = log[4 * i ..][0..4];
        try testing.expect(e[0] > prev);
        prev = e[0];
        for (groups, 1..) |g, v| {
            if (@as(i64, @intCast(v)) > e[0]) break;
            for (g) |t| {
                const r = rig.pool.result(@intCast(t));
                try testing.expect(r.status != .pending and r.t_end_ns <= e[1]);
            }
        }
    }
    try testing.expectEqual(@as(i64, 4), prev);
    for (rig.dests.items, 0..) |d, j| for (0..2) |i| {
        const r: u64 = 2 * j + i;
        try d.expectRecord(i, rig.f.image, EvRig.base(r), EvRig.base(r) + ev_gu_len, &ev_lens);
    };
    try testing.expectEqual(@as(i64, 4), rig.pool.counter(.ev_gates));
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
    // Gates whose tickets already landed are handed over inside the registration.
    const late = try rig.job(&.{7});
    try rig.pool.wait(late, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{ &.{late}, &.{late + 1} });
    try testing.expectEqual(@as(i64, 6), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_immediate));
}

test "dsv41 io: a failed read is terminal for its gate" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    injectFault(EvRig.base(4) / page * page, 2, 0);
    const f0 = try rig.job(&.{ 4, 5 });
    const f1 = try rig.job(&.{6});
    try rig.register(&.{ &.{ f0, f0 + 1, f1 }, &.{ f0 + 2, f0 + 3 }, &.{f1 + 1} });
    try rig.waitWord(3, 10_000);
    try rig.pool.wait(f0, 4, 10 * std.time.ns_per_s);
    try rig.pool.wait(f1, 2, 10 * std.time.ns_per_s);
    const want = [4]Status{ .os_error, .skipped, .skipped, .skipped };
    for (want, 0..) |w, k| try testing.expectEqual(w, rig.pool.result(f0 + @as(u32, @intCast(k))).status);
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
}

test "dsv41 io: the watchdog forces a gate whose bytes never land; stop and the host release the rest" {
    var rig = try EvRig.init(100, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    // Record 8's down range is held 600 ms: its gate is forced after ~100 ms, record 9's still waits for its bytes.
    injectFault((EvRig.base(8) + ev_gu_len) / page * page, 5, 600 * std.time.ns_per_ms);
    const f0 = try rig.job(&.{8});
    const f1 = try rig.job(&.{9});
    const t_reg = c.q3ld_monotonic_ns();
    try rig.register(&.{ &.{ f0, f1 }, &.{f0 + 1}, &.{f1 + 1} });
    try rig.waitWord(2, 5_000);
    const forced_ms = @divTrunc(c.q3ld_monotonic_ns() - t_reg, std.time.ns_per_ms);
    try testing.expectEqual(Status.pending, rig.pool.result(f0 + 1).status);
    try testing.expect(forced_ms >= 90 and forced_ms < 450);
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_wd_forced));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_wd_last_value));
    try rig.waitWord(3, 5_000);
    try rig.pool.wait(f0, 2, 10 * std.time.ns_per_s);
    // Host release: a gate over a ticket that never publishes is forced by value.
    const phantom: u32 = 500;
    rig.pool.res[phantom * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom}});
    rig.pool.releaseGates(4);
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_host_released));
    // Stop: a live gate is released so no GPU wait outlives the pool.
    const phantom2: u32 = 501;
    rig.pool.res[phantom2 * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom2}});
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
    _ = c.q3ld_stop();
    try testing.expectEqual(@as(i64, 5), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_stop_released));
    rig.pool.res[phantom * res_w] = 0;
    rig.pool.res[phantom2 * res_w] = 0;
}

test "dsv41 io: event-gate refusals" {
    var rig = try EvRig.init(60_000, 1);
    defer rig.deinit();
    try testing.expectError(error.EventRefused, rig.pool.armEvent(.host, @intFromPtr(rig.word), std.time.ns_per_s, 0));
    const f = try rig.job(&.{11});
    try rig.pool.wait(f, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{&.{f}});
    // Not above the last value; not increasing; out-of-range and repeated tickets; too many tickets.
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{1}, &.{0}, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{ 4, 3 }, &.{ 0, 0 }, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{512}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{-1}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{2}, &.{ f + 1, f + 1 }));
    var many: [max_gate_tickets + 1]i64 = undefined;
    for (&many, 0..) |*t, i| t.* = @intCast(i);
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{max_gate_tickets + 1}, &many));
    // Nothing was registered by a refused call.
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_calls));
}

test "dsv41 io: the event class refuses a pool without it and bad arguments" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    try testing.expectError(error.GateRefused, pool.registerGates(&.{1}, &.{0}, &.{}));
    var word: i64 = 0;
    try testing.expectError(error.EventRefused, pool.armEvent(.host, @intFromPtr(&word), 999_999, 0));
    try testing.expectError(error.EventRefused, pool.armEvent(.host, 0, std.time.ns_per_s, 0));
    // The pre-read class needs the speculative class.
    try testing.expectError(error.PreReadRefused, pool.armPreRead(&spec_lens));
}
