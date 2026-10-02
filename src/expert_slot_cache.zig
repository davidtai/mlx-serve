//! An expert cache over per-expert tensors at known file offsets (safetensors shards): per group (a routed
//! layer) a fixed slot bank, persistent rows [0, capacity) and transient rows after them, planned by the streamer's
//! residency policy (`expert_policy.LayerPolicy`, decode phase) and filled by the streamer's read pool
//! (`expert_io.Pool`, F_NOCACHE page-aligned reads, scatter copy into the rows). A record is `components` parts,
//! read in pairs: one pool job per pair, the even part its gate/up span, the odd part its down span. Slot memory:
//! MLX arrays the kernels bind (served), host pages (tests), or none (the trace backend: the policy alone).

const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const expert_io = @import("expert_io.zig");
const expert_policy = @import("expert_policy.zig");
const expert_bank = @import("expert_bank.zig");
const expert_stream = @import("expert_stream.zig");

pub const max_components = 8;
const max_jobs = expert_policy.max_route_ids * max_components / 2;
const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;
/// MLX's Metal allocator rounds each buffer up to the 16 KiB page.
pub const alloc_page_bytes: u64 = 16_384;

/// One part of every record of a group: its bytes, and the per-row shape and dtype of its bank array.
pub const Component = struct { bytes: u64, shape: []const c_int, dtype: mlx.mlx_dtype };

pub const Geometry = struct {
    n_experts: u32,
    components: []const Component,
    /// Persistent rows per group (the hot set's slots).
    capacity: []const u32,
    /// Transient rows per group: the widest route's ids.
    transient: u32,

    pub fn rows(g: Geometry, group: usize) u64 {
        return @as(u64, g.capacity[group]) + g.transient;
    }

    /// The device bytes the slot banks allocate: every group's arrays, each rounded to the allocator's page.
    pub fn billBytes(g: Geometry) u64 {
        var n: u64 = 0;
        for (0..g.capacity.len) |s| for (g.components) |c| {
            n += std.mem.alignForward(u64, g.rows(s) * c.bytes, alloc_page_bytes);
        };
        return n;
    }
};

pub const Memory = union(enum) { none, host, mlx: mlx.mlx_stream };

pub const Error = error{ CacheGeometry, CacheLocation, CacheFailed, ReadFailed, Timeout, TicketsBusy, QueueFull, SubmitRefused, InvalidJob, OutOfMemory, MlxError, MlxNoData };

/// Where a record's part sits: a file the cache opened and the byte offset in it.
pub const Loc = struct { file: u16 = std.math.maxInt(u16), offset: u64 = 0 };

pub const Cache = struct {
    a: std.mem.Allocator,
    geom: Geometry,
    memory: Memory,
    pool: ?*expert_io.Pool,
    policies: []expert_policy.LayerPolicy,
    /// [group][expert][component].
    locs: []Loc,
    files: std.ArrayList(File) = .empty,
    /// [group][component]: row 0's address (0 without memory) and the backing.
    base: [][max_components]u64,
    host: [][max_components][]u8,
    arrays: [][max_components]mlx.mlx_array,
    stats: expert_stream.Stats = .{},
    /// A read the cache could not see land (a ticket's wait timed out, or a submit refused after earlier jobs of the
    /// call were queued): a worker may still write into that call's rows, so no row may ever be planned again. Latched;
    /// every later route and seed refuses by name (CacheFailed).
    failed: bool = false,
    /// One ticket's wait bound (tests shorten it).
    wait_ns: i64 = wait_timeout_ns,

    pub const File = struct { fd: std.c.fd_t, size: u64 };

    /// The policies at their capacities and the slot banks (an MLX bank: zero arrays, one eval, data pointers bound
    /// once; MLX allocates through Metal even on the CPU stream). Reads need `pool` and a memory.
    pub fn init(a: std.mem.Allocator, geom: Geometry, memory: Memory, pool: ?*expert_io.Pool) Error!*Cache {
        const n_groups = geom.capacity.len;
        const n_comp = geom.components.len;
        if (n_groups == 0 or n_comp == 0 or n_comp > max_components or n_comp % 2 != 0) return error.CacheGeometry;
        for (geom.capacity) |cap| if (cap + geom.transient > geom.n_experts or geom.transient == 0 or geom.transient > expert_policy.max_route_ids) return error.CacheGeometry;
        if (memory != .none and pool == null) return error.CacheGeometry;
        const self = try a.create(Cache);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .geom = geom, .memory = memory, .pool = pool, .policies = &.{}, .locs = &.{}, .base = &.{}, .host = &.{}, .arrays = &.{} };
        {
            const pols = try a.alloc(expert_policy.LayerPolicy, n_groups);
            var n_pol: usize = 0;
            errdefer {
                for (pols[0..n_pol]) |*p| p.deinit(a);
                a.free(pols);
            }
            for (pols, geom.capacity) |*p, cap| {
                p.* = expert_policy.LayerPolicy.init(a, geom.n_experts, cap) catch return error.CacheGeometry;
                n_pol += 1;
            }
            self.policies = pols;
        }
        errdefer self.freeAll();
        self.locs = try a.alloc(Loc, n_groups * geom.n_experts * n_comp);
        @memset(self.locs, .{});
        self.base = try a.alloc([max_components]u64, n_groups);
        @memset(self.base, @splat(0));
        self.host = try a.alloc([max_components][]u8, n_groups);
        const no_rows: [max_components][]u8 = @splat(&[_]u8{});
        @memset(self.host, no_rows);
        self.arrays = try a.alloc([max_components]mlx.mlx_array, n_groups);
        @memset(self.arrays, @splat(.{}));
        switch (memory) {
            .none => {},
            .host => for (0..n_groups) |s| for (geom.components, 0..) |c, k| {
                const buf = try std.heap.page_allocator.alloc(u8, @intCast(geom.rows(s) * c.bytes));
                self.host[s][k] = buf;
                self.base[s][k] = @intFromPtr(buf.ptr);
            },
            .mlx => |stream| {
                var all: std.ArrayList(mlx.mlx_array) = .empty;
                defer all.deinit(a);
                for (0..n_groups) |s| for (geom.components, 0..) |c, k| {
                    var shape: [8]c_int = undefined;
                    shape[0] = @intCast(geom.rows(s));
                    @memcpy(shape[1..][0..c.shape.len], c.shape);
                    self.arrays[s][k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&self.arrays[s][k], &shape, c.shape.len + 1, c.dtype, stream));
                    try all.append(a, self.arrays[s][k]);
                };
                const vec = mlx.mlx_vector_array_new_data(all.items.ptr, all.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                for (0..n_groups) |s| for (0..n_comp) |k| {
                    self.base[s][k] = @intFromPtr(mlx.mlx_array_data_uint8(self.arrays[s][k]) orelse return error.MlxNoData);
                };
            },
        }
        return self;
    }

    fn freeAll(self: *Cache) void {
        const a = self.a;
        for (self.policies) |*p| p.deinit(a);
        if (self.policies.len > 0) a.free(self.policies);
        for (self.host) |bufs| for (bufs) |b| if (b.len > 0) std.heap.page_allocator.free(b);
        for (self.arrays) |arrs| for (arrs) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (self.files.items) |f| _ = std.c.close(f.fd);
        self.files.deinit(a);
        if (self.locs.len > 0) a.free(self.locs);
        if (self.base.len > 0) a.free(self.base);
        if (self.host.len > 0) a.free(self.host);
        if (self.arrays.len > 0) a.free(self.arrays);
    }

    /// After the pool that wrote into the rows has stopped, or with no read in flight.
    pub fn deinit(self: *Cache) void {
        self.freeAll();
        self.a.destroy(self);
    }

    /// Opens `path` past the page cache (F_NOCACHE, read-ahead off); its index for `setLoc`.
    pub fn openFile(self: *Cache, path: [:0]const u8) !u16 {
        const fd = try io_util.openNoCache(path.ptr, .{});
        errdefer _ = std.c.close(fd);
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 0) return error.OpenFailed;
        try self.files.append(self.a, .{ .fd = fd, .size = @intCast(size) });
        return @intCast(self.files.items.len - 1);
    }

    fn locIndex(self: *const Cache, group: usize, expert: usize, k: usize) usize {
        return (group * self.geom.n_experts + expert) * self.geom.components.len + k;
    }

    pub fn setLoc(self: *Cache, group: usize, expert: usize, k: usize, loc: Loc) void {
        self.locs[self.locIndex(group, expert, k)] = loc;
    }

    /// Once, after every `setLoc`: every part placed inside its file, each pair's two parts in one file.
    pub fn checkLocs(self: *const Cache) Error!void {
        const n_comp = self.geom.components.len;
        for (0..self.geom.capacity.len) |s| for (0..self.geom.n_experts) |e| for (0..n_comp) |k| {
            const l = self.locs[self.locIndex(s, e, k)];
            if (l.file >= self.files.items.len) return error.CacheLocation;
            if (l.offset + self.geom.components[k].bytes > self.files.items[l.file].size) return error.CacheLocation;
            if (k % 2 == 1 and self.locs[self.locIndex(s, e, k - 1)].file != l.file) return error.CacheLocation;
        };
    }

    pub fn slotOf(self: *const Cache, group: usize, expert: u16) ?u32 {
        return self.policies[group].slotOf(expert);
    }

    /// Row `slot` of component `k` in group `group`'s bank (host or MLX memory).
    pub fn row(self: *const Cache, group: usize, k: usize, slot: u32) []const u8 {
        const n = self.geom.components[k].bytes;
        const p: [*]const u8 = @ptrFromInt(self.base[group][k] + @as(u64, slot) * n);
        return p[0..n];
    }

    /// One route of group `group`: each id's slot into `slots` (`ids.len`), every load read and landed first.
    pub fn route(self: *Cache, group: usize, ids: []const u16, slots: []u32) Error!void {
        if (self.failed) return error.CacheFailed;
        var plan: expert_policy.Plan = undefined;
        self.policies[group].plan(ids, .decode, &plan);
        const st = &self.stats;
        st.route_calls += 1;
        st.expert_cache_hits += plan.n_hits;
        st.expert_cache_misses += plan.n_misses;
        st.expert_cache_evictions += plan.n_evictions;
        st.persistent_loads += plan.n_persistent;
        st.transient_loads += plan.n_loads - plan.n_persistent;
        self.read(group, plan.loadsOf()) catch |e| {
            for (plan.loadsOf()) |l| if (l.persistent) self.policies[group].invalidate(l.expert);
            return e;
        };
        @memcpy(slots, plan.slotsOf());
    }

    /// The hot set's seed (construction): `experts` into empty persistent slots, in order, read and landed.
    /// Returns how many were admitted.
    pub fn seed(self: *Cache, group: usize, experts: []const u16) Error!u32 {
        if (self.failed) return error.CacheFailed;
        var buf: [512]expert_policy.LayerPolicy.ReadAhead = undefined;
        var done: u32 = 0;
        var rest = experts;
        while (rest.len > 0) {
            const chunk = rest[0..@min(rest.len, expert_policy.max_route_ids)];
            rest = rest[chunk.len..];
            const got = self.policies[group].admitReadAhead(chunk, buf[0..chunk.len]);
            var loads: [expert_policy.max_route_ids]expert_policy.Load = undefined;
            for (got, 0..) |r, i| loads[i] = .{ .expert = r.expert, .slot = r.slot, .persistent = true };
            self.read(group, loads[0..got.len]) catch |e| {
                for (got) |r| self.policies[group].invalidate(r.expert);
                return e;
            };
            done += @intCast(got.len);
            if (got.len < chunk.len) break;
        }
        return done;
    }

    /// Every group's policy anew at its capacity (what a warm-up admitted belongs to no request); the rows keep
    /// their bytes but no plan serves from them until read again.
    pub fn forgetAll(self: *Cache) Error!void {
        for (self.policies, self.geom.capacity) |*p, cap| {
            const fresh = expert_policy.LayerPolicy.init(self.a, self.geom.n_experts, cap) catch return error.CacheGeometry;
            p.deinit(self.a);
            p.* = fresh;
        }
    }

    /// One pool job per (load, component pair) on the pool's aux ring when it has one (the cache's own tickets), in
    /// batches the ring holds; every job of a batch waited and its status words checked before the next. A job the
    /// cache cannot see land latches `failed` (its rows may still be written), so a failed call never frees a row
    /// that can be planned again.
    fn read(self: *Cache, group: usize, loads: []const expert_policy.Load) Error!void {
        if (self.memory == .none or loads.len == 0) return;
        const pool = self.pool.?;
        const n_pairs = self.geom.components.len / 2;
        const ring: usize = if (pool.auxTickets() > 0) pool.auxTickets() else pool.demand_tickets;
        const batch_loads = @max(1, @min(ring / 2, max_jobs) / n_pairs);
        var rest = loads;
        while (rest.len > 0) {
            const now = rest[0..@min(rest.len, batch_loads)];
            rest = rest[now.len..];
            try self.readBatch(pool, group, now);
        }
    }

    fn readBatch(self: *Cache, pool: *expert_io.Pool, group: usize, loads: []const expert_policy.Load) Error!void {
        const n_pairs = self.geom.components.len / 2;
        var tickets: [max_jobs]u32 = undefined;
        var n_jobs: usize = 0;
        var submit_err: ?Error = null;
        submit: for (loads) |l| for (0..n_pairs) |p| {
            const k = 2 * p;
            const gu = self.locs[self.locIndex(group, l.expert, k)];
            const down = self.locs[self.locIndex(group, l.expert, k + 1)];
            const f = self.files.items[gu.file];
            var dest: [expert_bank.n_components]u64 = @splat(0);
            var lens: [expert_bank.n_components]u64 = @splat(0);
            dest[0] = self.base[group][k] + @as(u64, l.slot) * self.geom.components[k].bytes;
            dest[1] = self.base[group][k + 1] + @as(u64, l.slot) * self.geom.components[k + 1].bytes;
            lens[0] = self.geom.components[k].bytes;
            lens[1] = self.geom.components[k + 1].bytes;
            const sub = if (pool.auxTickets() > 0) pool.submitAux(f.fd, f.size, &.{gu.offset}, &.{down.offset}, &.{dest}, &lens, 1, 1) else pool.submitSplit(f.fd, f.size, &.{gu.offset}, &.{down.offset}, &.{dest}, &lens, 1, 1);
            tickets[n_jobs] = sub catch |e| {
                submit_err = e;
                break :submit;
            };
            n_jobs += 1;
        };
        var failed = false;
        // Every submitted job is waited, whatever an earlier one did: no row of this call is released while a worker
        // can still write it.
        for (tickets[0..n_jobs]) |t| {
            pool.wait(t, 2, self.wait_ns) catch |e| {
                self.failed = true;
                return e;
            };
            for (0..2) |i| {
                const r = pool.result(t + @as(u32, @intCast(i)));
                if (r.status != .ok) failed = true;
                self.stats.expert_bytes_read += @intCast(@max(r.payload, 0));
                self.stats.preadv_calls += @intCast(@max(r.preadv_calls, 0));
            }
        }
        if (submit_err) |e| return e;
        if (failed) return error.ReadFailed;
    }
};

// ── Tests (host: host rows or no memory; never MLX) ──

const testing = std.testing;

/// A file of `n_groups x n_experts` records, each part at a shuffled offset, bytes a function of the offset.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    path: [:0]u8,
    offsets: []u64,

    fn init(a: std.mem.Allocator, n_groups: usize, n_experts: usize, comps: []const Component, seed_: u64) !Fixture {
        const n_parts = n_groups * n_experts * comps.len;
        const order = try a.alloc(usize, n_parts);
        defer a.free(order);
        for (order, 0..) |*o, i| o.* = i;
        var prng = std.Random.DefaultPrng.init(seed_);
        prng.random().shuffle(usize, order);
        const offsets = try a.alloc(u64, n_parts);
        errdefer a.free(offsets);
        var at: u64 = 123; // an unaligned header
        for (order) |i| {
            offsets[i] = at;
            at += comps[i % comps.len].bytes;
        }
        const image = try a.alloc(u8, @intCast(at + 77));
        errdefer a.free(image);
        for (image, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard.safetensors", .data = image });
        var root: [512]u8 = undefined;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/shard.safetensors", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        return .{ .tmp = tmp, .image = image, .path = path, .offsets = offsets };
    }

    fn deinit(f: *Fixture, a: std.mem.Allocator) void {
        a.free(f.offsets);
        a.free(f.image);
        a.free(f.path);
        f.tmp.cleanup();
    }

    fn part(f: *const Fixture, comps: []const Component, n_experts: usize, group: usize, expert: usize, k: usize) []const u8 {
        const off = f.offsets[(group * n_experts + expert) * comps.len + k];
        return f.image[off..][0..comps[k].bytes];
    }
};

const test_comps = [_]Component{
    .{ .bytes = 3 * 4096 + 17, .shape = &.{ 3, 4113 }, .dtype = .uint8 },
    .{ .bytes = 513, .shape = &.{513}, .dtype = .uint8 },
    .{ .bytes = 4096, .shape = &.{ 2, 512 }, .dtype = .uint32 },
    .{ .bytes = 99, .shape = &.{99}, .dtype = .uint8 },
};

test "dsv41 slot cache: every routed id's slot holds its record's bytes through evictions, the default bank's input row for row" {
    const a = testing.allocator;
    const n_groups = 2;
    const n_experts = 12;
    var fx = try Fixture.init(a, n_groups, n_experts, &test_comps, 7);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 3, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 256 });
    defer pool.stop();
    const geom: Geometry = .{ .n_experts = n_experts, .components = &test_comps, .capacity = &.{ 2, 3 }, .transient = 6 };
    const cache = try Cache.init(a, geom, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..n_groups) |s| for (0..n_experts) |e| for (0..test_comps.len) |k| {
        cache.setLoc(s, e, k, .{ .file = f, .offset = fx.offsets[(s * n_experts + e) * test_comps.len + k] });
    };
    try cache.checkLocs();
    // The construction seed: the first ids into the persistent slots.
    try testing.expectEqual(@as(u32, 2), try cache.seed(0, &.{ 0, 1 }));
    try testing.expectEqual(@as(u32, 3), try cache.seed(1, &.{ 0, 1, 2 }));
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var evictions: u64 = 0;
    for (0..60) |round| {
        const s = round % n_groups;
        var ids: [6]u16 = undefined; // 2 rows x top-3, repeats across rows allowed
        for (&ids, 0..) |*d, i| {
            d.* = rnd.uintLessThan(u16, n_experts);
            if (i % 3 != 0) while (d.* == ids[i - 1] or (i % 3 == 2 and d.* == ids[i - 2])) {
                d.* = rnd.uintLessThan(u16, n_experts);
            };
        }
        var slots: [6]u32 = undefined;
        const ev0 = cache.stats.expert_cache_evictions;
        try cache.route(s, &ids, &slots);
        evictions += cache.stats.expert_cache_evictions - ev0;
        // The default route's kernel reads bank[id]; this route's reads bank[slot]: the same bytes, every part.
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| {
            try testing.expectEqualSlices(u8, fx.part(&test_comps, n_experts, s, e, k), cache.row(s, k, slot));
        };
    }
    try testing.expect(evictions > 0);
    const st = cache.stats;
    try testing.expectEqual(@as(u64, 60), st.route_calls);
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    var per_load: u64 = 0;
    for (test_comps) |c| per_load += c.bytes;
    try testing.expectEqual((5 + st.expert_cache_misses) * per_load, st.expert_bytes_read);
}

test "dsv41 slot cache: geometry and placement refused by name; a failed read forgets its loads" {
    const a = testing.allocator;
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{3}, .transient = 6 }, .none, null));
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = test_comps[0..3], .capacity = &.{1}, .transient = 2 }, .none, null));
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{1}, .transient = 2 }, .host, null));
    var fx = try Fixture.init(a, 1, 4, &test_comps, 3);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 64 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = 4, .components = &test_comps, .capacity = &.{1}, .transient = 3 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    try testing.expectError(error.CacheLocation, cache.checkLocs());
    for (0..4) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    cache.setLoc(0, 3, 1, .{ .file = f, .offset = fx.image.len });
    try testing.expectError(error.CacheLocation, cache.checkLocs());
    // A part past the end of the file reads short: the call fails by name and its persistent load is forgotten.
    var slots: [1]u32 = undefined;
    try testing.expectError(error.ReadFailed, cache.route(0, &.{3}, &slots));
    try testing.expect(cache.slotOf(0, 3) == null);
}

test "dsv41 slot cache: no memory plans only (the trace backend), with the policy's slots and the bill's bytes" {
    const a = testing.allocator;
    const comps = [_]Component{
        .{ .bytes = 5_898_240, .shape = &.{ 2304, 640 }, .dtype = .uint32 },
        .{ .bytes = 368_640, .shape = &.{ 2304, 160 }, .dtype = .uint8 },
    };
    const cache = try Cache.init(a, .{ .n_experts = 128, .components = &comps, .capacity = &.{ 43, 42 }, .transient = 15 }, .none, null);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 3), try cache.seed(0, &.{ 0, 1, 2 }));
    var slots: [3]u32 = undefined;
    try cache.route(0, &.{ 1, 9, 2 }, &slots);
    try testing.expectEqual(@as(u32, 1), slots[0]);
    try testing.expectEqual(@as(u32, 2), slots[2]);
    try testing.expectEqual(@as(u64, 0), cache.stats.expert_bytes_read);
    // 58 + 57 rows: weights a page multiple, scales rounded up to the page per array.
    try testing.expectEqual(@as(u64, 115 * 5_898_240 + std.mem.alignForward(u64, 58 * 368_640, 16384) + std.mem.alignForward(u64, 57 * 368_640, 16384)), cache.geom.billBytes());
}

test "dsv41 slot cache: one group shared by consecutive routes reuses a row the previous route served as early as the policy allows; every route's rows are its records" {
    // Three "stages" of 12 experts in one group (global id = stage x 12 + e), 2 hot + 6 transient rows: each route is a
    // stage's 2 rows x top-3; stage s + 1 routes right after stage s, the next block's stage 0 right after stage 2.
    const a = testing.allocator;
    const n_stages = 3;
    const per = 12;
    var fx = try Fixture.init(a, 1, n_stages * per, &test_comps, 41);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 3, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 256 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = n_stages * per, .components = &test_comps, .capacity = &.{2}, .transient = 6 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..n_stages * per) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    _ = try cache.seed(0, &.{ 0, 12 });
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var prev: [6]u32 = undefined;
    var prev_ids: [6]u16 = undefined;
    var have_prev = false;
    var reuse_in_block: u32 = 0;
    var reuse_across: u32 = 0;
    for (0..20) |blk| for (0..n_stages) |st| {
        var ids: [6]u16 = undefined;
        for (&ids, 0..) |*d, i| {
            while (true) {
                d.* = @intCast(st * per + rnd.uintLessThan(u16, per));
                if (std.mem.indexOfScalar(u16, ids[i - i % 3 .. i], d.*) == null) break;
            }
        }
        var slots: [6]u32 = undefined;
        try cache.route(0, &ids, &slots);
        // The kernel's input at this route's evaluation (before the next route plans): its records' bytes, row for row.
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| {
            try testing.expectEqualSlices(u8, fx.part(&test_comps, n_stages * per, 0, e, k), cache.row(0, k, slot));
        };
        // A row the previous route read now holds another record: the earliest reuse there is.
        if (have_prev) for (slots, ids) |slot, e| for (prev, prev_ids) |ps, pe| if (slot == ps and e != pe) {
            if (st == 0) reuse_across += 1 else reuse_in_block += 1;
        };
        _ = blk;
        prev = slots;
        prev_ids = ids;
        have_prev = true;
    };
    try testing.expect(reuse_in_block > 0 and reuse_across > 0);
}

test "dsv41 slot cache: a read it cannot see land latches the cache: no row of that call is planned again, even after the late write" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 8, &test_comps, 19);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 64, .aux_tickets = 32 });
    defer pool.stop();
    defer expert_io.clearFaults();
    const cache = try Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{2}, .transient = 3 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..8) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    cache.wait_ns = 30 * std.time.ns_per_ms;
    // Expert 5's second pair (its down span) sleeps 300 ms before its read: one ticket of a multi-ticket call is late.
    const page = std.heap.pageSize();
    const late = fx.offsets[5 * test_comps.len + 3];
    expert_io.injectFault(late / page * page, 5, 300 * std.time.ns_per_ms);
    var slots: [3]u32 = undefined;
    try testing.expectError(error.Timeout, cache.route(0, &.{ 4, 5, 6 }, &slots));
    try testing.expect(cache.failed);
    // Refused by name from here on, before and after the late ticket publishes.
    try testing.expectError(error.CacheFailed, cache.route(0, &.{ 4, 5, 6 }, &slots));
    try testing.expectError(error.CacheFailed, cache.seed(0, &.{1}));
    std.Io.sleep(testing.io, .fromMilliseconds(400), .awake) catch {};
    try testing.expectError(error.CacheFailed, cache.route(0, &.{1}, slots[0..1]));
}

test "dsv41 slot cache: on a pool with an aux ring every read takes the aux tickets; the demand ring is untouched" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 8, &test_comps, 23);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 96, .aux_tickets = 32 });
    defer pool.stop();
    try testing.expectEqual(@as(u32, 64), pool.demand_tickets);
    try testing.expectEqual(@as(u32, 64), pool.aux_first);
    const cache = try Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{2}, .transient = 6 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..8) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    var slots: [6]u32 = undefined;
    // 6 loads x 2 pairs = 24 tickets per route, the aux ring 32: the ring wraps inside [64, 96) across routes.
    for (0..5) |r| {
        const ids = [_]u16{ @intCast(r % 8), @intCast((r + 1) % 8), @intCast((r + 2) % 8), @intCast((r + 3) % 8), @intCast((r + 4) % 8), @intCast((r + 5) % 8) };
        try cache.route(0, &ids, &slots);
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| try testing.expectEqualSlices(u8, fx.part(&test_comps, 8, 0, e, k), cache.row(0, k, slot));
        try testing.expect(pool.next_aux >= pool.aux_first and pool.next_aux <= pool.aux_end);
    }
    try testing.expectEqual(@as(u32, 0), pool.next_ticket);
    // The stream's demand submits stay below the aux ring.
    var dests: [expert_bank.n_components]u64 = @splat(0);
    var buf: [8192]u8 = undefined;
    dests[0] = @intFromPtr(&buf);
    dests[1] = @intFromPtr(&buf) + 4096;
    var lens: [expert_bank.n_components]u64 = @splat(0);
    lens[0] = 100;
    lens[1] = 100;
    for (0..40) |_| {
        const t = try pool.submitSplit(cache.files.items[0].fd, cache.files.items[0].size, &.{0}, &.{200}, &.{dests}, &lens, 1, 1);
        try testing.expect(t + 2 <= pool.demand_tickets);
        try pool.wait(t, 2, 5 * std.time.ns_per_s);
    }
}
