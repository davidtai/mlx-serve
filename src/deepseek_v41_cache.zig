//! DeepSeek-V4.1 per-sequence attention state (Python `deepseek_v41_cache.py`):
//! the window, compressed-KV, index-key and compressor-frontier lanes of each
//! layer, their trim / rollback seam, and the prefill chunk geometry. The
//! backing is a construction-time route; every route hands the attention the
//! same rows the full-history store would (the ring keeps a contiguous suffix
//! whose dropped rows no query can reach). Generic over the op backend `G`.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");

/// Which backing the lanes use (Python precedence KV_BOUNDED > WINDOW_RING >
/// KV_CHUNK_GROW > plain), picked once when the sequence state is built.
pub const Route = enum {
    /// `_grow` concatenate stores: the stock path, every lever off.
    full_history,
    /// W73 `_GrowBuffer` lanes (geometric capacity from 256).
    chunk_grow,
    /// W80 `_WindowRing` window; compress / index `_GrowBuffer`s sized to
    /// `max_kv` when set, else geometric; the frontier a plain store.
    window_ring,
    /// W107: the ring plus compress / index / frontier preallocated to
    /// `max_kv` (geometric when unset), appends past the cap refused.
    bounded,
};

pub const Geometry = struct {
    route: Route = .full_history,
    /// The widest verify block one forward appends (DSpark depth + 1 fits).
    max_verify: u32 = 8,
    slack: u32 = 8,
    /// In-place appends between two ring compactions.
    headroom: u32 = 64,
    /// Preallocation capacity (`--max-kv`); null keeps geometric growth.
    max_kv: ?u32 = null,
};

/// `_BOUNDED_COMP_SLACK`, `_BOUNDED_LATENT_SLACK`.
const bounded_comp_slack = 8;
const bounded_latent_slack = 8;

/// `_bounded_comp_cap`: one row per completed group plus a verify margin.
pub fn boundedCompCap(max_kv: ?u32, ratio: u32) ?u32 {
    const m = max_kv orelse return null;
    const r = if (ratio > 0) ratio else 1;
    return (m + r - 1) / r + bounded_comp_slack;
}

/// `_bounded_latent_cap`: one fed row per token plus a verify margin.
pub fn boundedLatentCap(max_kv: ?u32) ?u32 {
    return (max_kv orelse return null) + bounded_latent_slack;
}

pub const Error = error{ BoundedLaneFull, RingRollbackTooDeep, TrimPastStart };

pub fn Lanes(comptime G: type) type {
    return struct {
        pub const T = G.T;

        /// `x[:, lo:hi]` along the sequence axis.
        fn rowsSlice(g: *G, x: T, lo: u32, hi: u32) !T {
            const s = g.shapeOf(x);
            var start: [ops.max_dims]c_int = @splat(0);
            var stop: [ops.max_dims]c_int = undefined;
            const strides: [ops.max_dims]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            start[1] = @intCast(lo);
            stop[1] = @intCast(hi);
            return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        /// `mx.slice_update(buf, new, mx.array([0, row, 0..], int32), axes=all)`.
        fn write(g: *G, buf: T, new: T, row: u32) !T {
            const n = g.shapeOf(new).n;
            var starts: [ops.max_dims]i32 = @splat(0);
            starts[1] = @intCast(row);
            const sa = try g.hostArray(std.mem.sliceAsBytes(starts[0..n]), &.{@intCast(n)}, .int32);
            return g.sliceUpdateDyn(buf, new, sa);
        }

        /// `mx.zeros((b, cap) + tail, new.dtype)`.
        fn alloc(g: *G, like: T, cap: u32) !T {
            var s = g.shapeOf(like);
            s.d[1] = @intCast(cap);
            return g.zeros(s.slice(), g.dtypeOf(like));
        }

        fn replace(g: *G, slot: *?T, next: T) void {
            const kept = g.keep(next);
            if (slot.*) |old| g.release(old);
            slot.* = kept;
        }

        /// `_grow` / `_truncate`: a plain append-only store.
        pub const Concat = struct {
            rows: ?T = null,
            len: u32 = 0,

            pub fn append(self: *Concat, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.rows) |old| {
                    replace(g, &self.rows, try g.concat(&.{ old, new }, 1));
                } else replace(g, &self.rows, new);
                self.len += n;
            }

            pub fn view(self: *const Concat) ?T {
                return self.rows;
            }

            pub fn truncate(self: *Concat, g: *G, n: u32) !void {
                const rows = self.rows orelse return;
                if (n == 0) {
                    g.release(rows);
                    self.rows = null;
                    self.len = 0;
                    return;
                }
                if (n >= self.len) return;
                replace(g, &self.rows, try rowsSlice(g, rows, 0, n));
                self.len = n;
            }

            pub fn deinit(self: *Concat, g: *G) void {
                if (self.rows) |r| g.release(r);
                self.* = .{};
            }
        };

        /// `_GrowBuffer`: a capacity buffer with a logical length, written in
        /// place; `view` is byte-identical to the concatenated store.
        pub const Grow = struct {
            buf: ?T = null,
            len: u32 = 0,
            cap: u32 = 0,
            init_cap: u32 = 256,
            bounded_cap: ?u32 = null,

            pub fn init(init_cap: u32, bounded_cap: ?u32) Grow {
                return .{ .init_cap = if (bounded_cap) |b| b else init_cap, .bounded_cap = bounded_cap };
            }

            pub fn append(self: *Grow, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.bounded_cap) |b| if (self.len + n > b) return error.BoundedLaneFull;
                if (self.buf == null) {
                    const cap = @max(self.init_cap, n);
                    const buf = try alloc(g, new, cap);
                    replace(g, &self.buf, try write(g, buf, new, 0));
                    self.cap = cap;
                    self.len = n;
                    return;
                }
                if (self.len + n <= self.cap) {
                    replace(g, &self.buf, try write(g, self.buf.?, new, self.len));
                    self.len += n;
                    return;
                }
                const new_cap = @max(self.cap * 2, self.len + n);
                const head = try rowsSlice(g, self.buf.?, 0, self.len);
                var buf = try alloc(g, new, new_cap);
                buf = try write(g, buf, head, 0);
                buf = try write(g, buf, new, self.len);
                replace(g, &self.buf, buf);
                self.cap = new_cap;
                self.len += n;
            }

            pub fn view(self: *const Grow, g: *G) !?T {
                const buf = self.buf orelse return null;
                if (self.len == 0) return null;
                if (self.len == self.cap) return buf;
                return try rowsSlice(g, buf, 0, self.len);
            }

            /// Length only: the capacity stays for the next append.
            pub fn truncateTo(self: *Grow, n: u32) void {
                self.len = @min(n, self.len);
            }

            pub fn deinit(self: *Grow, g: *G) void {
                if (self.buf) |b| g.release(b);
                self.buf = null;
                self.len = 0;
                self.cap = 0;
            }
        };

        /// `_WindowRing`: a contiguous suffix of the window history in two
        /// ping-pong buffers; row j of `view` is absolute position `drop + j`.
        pub const Ring = struct {
            window: u32,
            cap_keep: u32,
            phys_cap: u32,
            base_phys_cap: u32,
            bufs: [2]?T = .{ null, null },
            caps: [2]u32 = .{ 0, 0 },
            cur: u1 = 0,
            len: u32 = 0,
            drop: u32 = 0,

            pub fn init(window: u32, geo: Geometry) Ring {
                const keep = window + geo.max_verify + geo.slack;
                return .{ .window = window, .cap_keep = keep, .phys_cap = keep + geo.headroom, .base_phys_cap = keep + geo.headroom };
            }

            pub fn logicalLen(self: *const Ring) u32 {
                return self.drop + self.len;
            }

            pub fn append(self: *Ring, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.bufs[self.cur] == null) {
                    const cap = @max(self.base_phys_cap, n);
                    self.phys_cap = cap;
                    replace(g, &self.bufs[0], try alloc(g, new, cap));
                    replace(g, &self.bufs[1], try alloc(g, new, cap));
                    self.caps = .{ cap, cap };
                    replace(g, &self.bufs[self.cur], try write(g, self.bufs[self.cur].?, new, 0));
                    self.len = n;
                    self.drop = 0;
                    return;
                }
                if (self.len + n <= self.phys_cap) {
                    replace(g, &self.bufs[self.cur], try write(g, self.bufs[self.cur].?, new, self.len));
                    self.len += n;
                    return;
                }
                // Compaction: keep the newest `keep` rows (the oldest new query's
                // whole causal window) in the OTHER buffer, drop the rest.
                const l = self.drop + self.len;
                const l_new = l + n;
                const keep = @min(l_new, @max(self.cap_keep, n + (self.window - 1)));
                const new_drop = l_new - keep;
                const retained = l - new_drop;
                const src = self.bufs[self.cur].?;
                const target = @max(self.base_phys_cap, keep);
                const dst_idx: u1 = 1 - self.cur;
                var dst: T = undefined;
                if (self.bufs[dst_idx] == null or self.caps[dst_idx] != target) {
                    dst = try alloc(g, new, target);
                } else dst = self.bufs[dst_idx].?;
                if (retained > 0) dst = try write(g, dst, try rowsSlice(g, src, self.len - retained, self.len), 0);
                dst = try write(g, dst, new, retained);
                if (target != self.phys_cap) {
                    self.phys_cap = target;
                    // The other slot re-allocates at the next compaction.
                    if (self.bufs[self.cur]) |b| g.release(b);
                    self.bufs[self.cur] = null;
                    self.caps[self.cur] = 0;
                }
                replace(g, &self.bufs[dst_idx], dst);
                self.caps[dst_idx] = target;
                self.cur = dst_idx;
                self.len = retained + n;
                self.drop = new_drop;
            }

            pub fn view(self: *const Ring, g: *G) !?T {
                const buf = self.bufs[self.cur] orelse return null;
                if (self.len == 0) return null;
                if (self.len == self.caps[self.cur]) return buf;
                return try rowsSlice(g, buf, 0, self.len);
            }

            /// A rollback whose next query still has its whole window resident.
            /// Nothing dropped yet means every row is (Python's predicate also
            /// refuses lengths below window - 1 there, which no row requires).
            pub fn canTruncateToLength(self: *const Ring, n: u32) bool {
                return n == 0 or self.drop == 0 or n + 1 >= self.window + self.drop;
            }

            pub fn truncateToLength(self: *Ring, n: u32) !void {
                if (!self.canTruncateToLength(n)) return error.RingRollbackTooDeep;
                if (n <= self.drop) {
                    self.len = 0;
                    self.drop = n;
                    return;
                }
                self.len = @min(self.len, n - self.drop);
            }

            pub fn deinit(self: *Ring, g: *G) void {
                for (&self.bufs) |*b| if (b.*) |x| {
                    g.release(x);
                    b.* = null;
                };
                self.len = 0;
            }
        };

        /// One store lane under its route.
        pub const Store = union(enum) {
            concat: Concat,
            grow: Grow,

            pub fn append(self: *Store, g: *G, new: T) !void {
                return switch (self.*) {
                    inline else => |*l| l.append(g, new),
                };
            }

            pub fn view(self: *const Store, g: *G) !?T {
                return switch (self.*) {
                    .concat => |*l| l.view(),
                    .grow => |*l| try l.view(g),
                };
            }

            pub fn rows(self: *const Store) u32 {
                return switch (self.*) {
                    inline else => |*l| l.len,
                };
            }

            pub fn truncate(self: *Store, g: *G, n: u32) !void {
                switch (self.*) {
                    .concat => |*l| try l.truncate(g, n),
                    .grow => |*l| l.truncateTo(n),
                }
            }

            pub fn deinit(self: *Store, g: *G) void {
                switch (self.*) {
                    inline else => |*l| l.deinit(g),
                }
            }
        };

        pub const Window = union(enum) {
            store: Store,
            ring: Ring,

            pub fn append(self: *Window, g: *G, new: T) !void {
                return switch (self.*) {
                    inline else => |*l| l.append(g, new),
                };
            }

            pub fn view(self: *const Window, g: *G) !?T {
                return switch (self.*) {
                    .store => |*l| l.view(g),
                    .ring => |*l| l.view(g),
                };
            }

            /// Absolute position of view row 0 (0 unless the ring dropped rows).
            pub fn dropOffset(self: *const Window) u32 {
                return switch (self.*) {
                    .store => 0,
                    .ring => |r| r.drop,
                };
            }

            /// Rows appended so far, dropped ones included (the logical length).
            pub fn rows(self: *const Window) u32 {
                return switch (self.*) {
                    .store => |*l| l.rows(),
                    .ring => |r| r.logicalLen(),
                };
            }

            /// Whether `truncateTo(n)` keeps what a later read needs (a ring's rule; a store keeps every row).
            pub fn canTruncateTo(self: *const Window, n: u32) bool {
                return switch (self.*) {
                    .store => true,
                    .ring => |r| r.canTruncateToLength(n),
                };
            }

            /// Back to `n` rows (logical).
            pub fn truncateTo(self: *Window, g: *G, n: u32) !void {
                switch (self.*) {
                    .store => |*l| try l.truncate(g, n),
                    .ring => |*r| try r.truncateToLength(n),
                }
            }

            pub fn deinit(self: *Window, g: *G) void {
                switch (self.*) {
                    inline else => |*l| l.deinit(g),
                }
            }
        };
    };
}

/// One layer's attention state (Python `LayerAttentionCache` + its
/// `CompressorState`), lanes picked at construction.
pub fn LayerState(comptime G: type) type {
    return struct {
        const Self = @This();
        const L = Lanes(G);
        pub const T = G.T;

        offset: u32 = 0,
        window_size: u32,
        ratio: u32,
        kv_source: bool,
        bounded_max_kv: ?u32 = null,
        window: L.Window,
        compress: L.Store,
        index: L.Store,
        /// The compressor frontier (`raw_kv`, `raw_score`) of a ratio > 1 kv source. On the ring routes a `Ring` of
        /// window `ratio`: a push pools only the groups it completes, whose rows start within `ratio - 1` rows of the
        /// fed length, and a verify's rollback re-exposes at most its own rows, so the ring's retained rows (ratio plus
        /// the verify margin and slack) hold every row a later push reads. Elsewhere a plain store of every fed row.
        frontier: ?struct { kv: L.Window, score: L.Window } = null,

        pub const Mark = struct { offset: u32, window: u32, compress: u32, index: u32, frontier: u32 };

        pub fn init(li: v41.LayerInfo, window_size: u32, geo: Geometry) Self {
            const ratio: u32 = li.ratio;
            var self: Self = .{
                .window_size = window_size,
                .ratio = ratio,
                .kv_source = li.kv_source,
                .window = .{ .store = .{ .concat = .{} } },
                .compress = .{ .concat = .{} },
                .index = .{ .concat = .{} },
            };
            var frontier_lane: L.Window = .{ .store = .{ .concat = .{} } };
            switch (geo.route) {
                .full_history => {},
                .chunk_grow => {
                    self.window = .{ .store = .{ .grow = L.Grow.init(256, null) } };
                    self.compress = .{ .grow = L.Grow.init(256, null) };
                    self.index = .{ .grow = L.Grow.init(256, null) };
                },
                .window_ring => {
                    self.window = .{ .ring = L.Ring.init(window_size, geo) };
                    const icap = geo.max_kv orelse 256;
                    self.compress = .{ .grow = L.Grow.init(icap, null) };
                    self.index = .{ .grow = L.Grow.init(icap, null) };
                    frontier_lane = .{ .ring = L.Ring.init(@max(ratio, 1), geo) };
                },
                .bounded => {
                    self.bounded_max_kv = geo.max_kv;
                    self.window = .{ .ring = L.Ring.init(window_size, geo) };
                    const cc = boundedCompCap(geo.max_kv, ratio);
                    self.compress = .{ .grow = L.Grow.init(cc orelse 256, cc) };
                    self.index = .{ .grow = L.Grow.init(cc orelse 256, cc) };
                    frontier_lane = .{ .ring = L.Ring.init(@max(ratio, 1), geo) };
                },
            }
            if (li.kv_source and ratio > 1) self.frontier = .{ .kv = frontier_lane, .score = frontier_lane };
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.window.deinit(g);
            self.compress.deinit(g);
            self.index.deinit(g);
            if (self.frontier) |*f| {
                f.kv.deinit(g);
                f.score.deinit(g);
            }
        }

        /// The longest sequence `canAdmit` lets through (null: unbounded). A
        /// constant of the layer's geometry: a state checks its minimum once.
        pub fn admitLimit(self: *const Self) ?u32 {
            const m = self.bounded_max_kv orelse return null;
            var lim: ?u32 = null;
            if (self.kv_source and self.ratio >= 1) if (boundedCompCap(m, self.ratio)) |cap| {
                const l = (cap + 1) * self.ratio - 1; // new_len / ratio <= cap
                lim = if (lim) |x| @min(x, l) else l;
            };
            return lim;
        }

        /// `assert_can_admit`: an over-cap forward fails before any lane is written.
        pub fn canAdmit(self: *const Self, n: u32) Error!void {
            const m = self.bounded_max_kv orelse return;
            const new_len = self.offset + n;
            if (self.kv_source and self.ratio >= 1) if (boundedCompCap(m, self.ratio)) |cap| if (new_len / self.ratio > cap) return error.BoundedLaneFull;
        }

        pub fn nFed(self: *const Self) u32 {
            return if (self.frontier) |f| f.kv.rows() else 0;
        }

        /// The groups a push completes: `kv` / `score` appended, then the completed groups' rows (fed rows
        /// [g_before * ratio, g_after * ratio)) as views of the lanes (a ring's view starts at its drop offset).
        pub const Groups = struct { kv: T, score: T, groups: u32 };

        pub fn frontierGroups(self: *Self, g: *G, kv: T, score: T) !?Groups {
            const f = &self.frontier.?;
            const r = self.ratio;
            const n_before = f.kv.rows();
            try f.kv.append(g, kv);
            try f.score.append(g, score);
            const g_before = n_before / r;
            const g_after = f.kv.rows() / r;
            if (g_after == g_before) return null;
            const drop = f.kv.dropOffset();
            // The ring keeps the incomplete group's rows by construction (see `frontier`); checked, never taken.
            if (g_before * r < drop) return error.FrontierRowsDropped;
            const lo = g_before * r - drop;
            const hi = g_after * r - drop;
            return .{ .kv = try L.rowsSlice(g, (try f.kv.view(g)).?, lo, hi), .score = try L.rowsSlice(g, (try f.score.view(g)).?, lo, hi), .groups = g_after - g_before };
        }

        /// `CompressorState.push`: append the fp32 projections, return the
        /// softmax-gated pooled latents of the groups this push completed.
        pub fn frontierPush(self: *Self, g: *G, kv: T, score: T) !?T {
            const gr = (try self.frontierGroups(g, kv, score)) orelse return null;
            const sh = g.shapeOf(kv);
            const grp: [4]c_int = .{ sh.d[0], @intCast(gr.groups), @intCast(self.ratio), sh.d[2] };
            const gk = try g.reshape(gr.kv, &grp);
            const gs = try g.reshape(gr.score, &grp);
            return try g.sum(try g.mul(gk, try g.softmax(gs, 2)), 2, false);
        }

        /// `advance`: the entry offset after a forward's appends.
        pub fn advance(self: *Self, n: u32) void {
            self.offset += n;
        }

        /// Whether `trim(n)` can restore every lane (the ring keeps the window of
        /// the next query).
        pub fn canTrim(self: *const Self, n: u32) bool {
            if (n > self.offset) return false;
            if (self.frontier) |f| if (n > f.kv.rows() or !f.kv.canTruncateTo(f.kv.rows() - n)) return false;
            return switch (self.window) {
                .ring => |r| r.canTruncateToLength(self.offset - n),
                .store => true,
            };
        }

        /// `trim(n)`: back to `offset - n` tokens; 0 (no change) when a ring
        /// cannot recover that far (the session-restore miss contract).
        pub fn trim(self: *Self, g: *G, n: u32) !u32 {
            if (n == 0) return 0;
            if (n > self.offset) return error.TrimPastStart;
            const new_len = self.offset - n;
            // The frontier ring's rule with the window's: a rollback either lands whole or is a clean miss.
            if (self.frontier) |f| if (n <= f.kv.rows() and !f.kv.canTruncateTo(f.kv.rows() - n)) return 0;
            switch (self.window) {
                .ring => |*r| {
                    if (!r.canTruncateToLength(new_len)) return 0;
                    try r.truncateToLength(new_len);
                },
                .store => |*s| try s.truncate(g, new_len),
            }
            if (self.kv_source and self.ratio >= 1) {
                const groups = new_len / self.ratio;
                try self.compress.truncate(g, groups);
                try self.index.truncate(g, groups);
                if (self.frontier) |*f| {
                    if (n > f.kv.rows()) return error.TrimPastStart;
                    const keep = f.kv.rows() - n;
                    try f.kv.truncateTo(g, keep);
                    try f.score.truncateTo(g, keep);
                }
            }
            self.offset = new_len;
            return n;
        }

        pub fn mark(self: *const Self) Mark {
            return .{
                .offset = self.offset,
                .window = switch (self.window) {
                    .store => |s| s.rows(),
                    .ring => |r| r.logicalLen(),
                },
                .compress = self.compress.rows(),
                .index = self.index.rows(),
                .frontier = self.nFed(),
            };
        }

        pub fn rollback(self: *Self, g: *G, m: Mark) !void {
            switch (self.window) {
                .ring => |*r| try r.truncateToLength(m.offset),
                .store => |*s| try s.truncate(g, m.window),
            }
            try self.compress.truncate(g, m.compress);
            try self.index.truncate(g, m.index);
            if (self.frontier) |*f| {
                try f.kv.truncateTo(g, m.frontier);
                try f.score.truncateTo(g, m.frontier);
            }
            self.offset = m.offset;
        }
    };
}

// ── prefill chunk geometry (host) ──

pub const default_chunk_target_bytes: f64 = 8e9;

/// `_prefill_score_bytes_per_row`: one `[H, T]` f32 score row, T = s plus the
/// smallest positive ratio's compressed rows.
pub fn prefillScoreBytesPerRow(c: *const v41.Config, s: u64) u64 {
    var min_ratio: u64 = 0;
    for (c.layers[0 .. c.n_layers + c.dspark.n_stages]) |li| {
        if (li.ratio > 0 and (min_ratio == 0 or li.ratio < min_ratio)) min_ratio = li.ratio;
    }
    const n_comp = if (min_ratio > 0) s / min_ratio else 0;
    return @as(u64, c.n_heads) * (s + n_comp) * 4;
}

/// `_resolve_prefill_chunk`: an explicit chunk wins, else the largest query
/// chunk whose score transient stays under `target_bytes`. A result `>= s`
/// (or `<= 0` explicit) is one-shot.
pub fn resolvePrefillChunk(c: *const v41.Config, s: u64, explicit: ?i64, target_bytes: f64) i64 {
    if (explicit) |e| return e;
    const per_row = prefillScoreBytesPerRow(c, s);
    if (per_row == 0) return @intCast(s);
    const chunk: u64 = @intFromFloat(@floor(@max(target_bytes, 1e9) / @as(f64, @floatFromInt(per_row))));
    return @intCast(@max(1, @min(chunk, s)));
}

/// The query spans of a forward over `s` tokens (`[start, end)` pairs).
pub fn prefillSpans(a: std.mem.Allocator, s: u32, chunk: i64) ![][2]u32 {
    if (chunk <= 0 or chunk >= s) {
        const one = try a.alloc([2]u32, 1);
        one[0] = .{ 0, s };
        return one;
    }
    const k: u32 = @intCast(chunk);
    const n = (s + k - 1) / k;
    const out = try a.alloc([2]u32, n);
    for (out, 0..) |*sp, i| sp.* = .{ @intCast(i * k), @min(s, @as(u32, @intCast((i + 1) * k))) };
    return out;
}

// ── tests: a row-id backend checks the lanes' reachable rows on the host ──

const testing = std.testing;

/// Arrays of row ids (`[1, n]` rows, the feature axis elided): enough of the
/// op surface for the lanes, so every append / compaction / view / trim is
/// checked against the absolute positions it must hold.
const RowOps = struct {
    pub const T = u32;
    gpa: std.mem.Allocator,
    arrays: std.ArrayList(std.ArrayList(i64)) = .empty,
    kept: std.ArrayList(u32) = .empty,
    allocs: u32 = 0,

    fn deinit(g: *RowOps) void {
        for (g.arrays.items) |*a| a.deinit(g.gpa);
        g.arrays.deinit(g.gpa);
        g.kept.deinit(g.gpa);
    }

    fn new(g: *RowOps, ids: []const i64) !u32 {
        var a: std.ArrayList(i64) = .empty;
        try a.appendSlice(g.gpa, ids);
        try g.arrays.append(g.gpa, a);
        return @intCast(g.arrays.items.len - 1);
    }

    fn range(g: *RowOps, lo: i64, hi: i64) !u32 {
        var buf: [4096]i64 = undefined;
        for (0..@intCast(hi - lo)) |i| buf[i] = lo + @as(i64, @intCast(i));
        return g.new(buf[0..@intCast(hi - lo)]);
    }

    fn rows(g: *RowOps, x: u32) []const i64 {
        return g.arrays.items[x].items;
    }

    pub fn shapeOf(g: *RowOps, x: u32) ops.Shape {
        return ops.Shape.of(&.{ 1, @intCast(g.arrays.items[x].items.len), 3 });
    }
    pub fn dtypeOf(_: *RowOps, _: u32) ops.Dtype {
        return .float32;
    }
    pub fn keep(g: *RowOps, x: u32) u32 {
        g.kept.append(g.gpa, x) catch unreachable;
        return x;
    }
    pub fn release(g: *RowOps, x: u32) void {
        const i = std.mem.indexOfScalar(u32, g.kept.items, x) orelse unreachable;
        _ = g.kept.swapRemove(i);
    }
    pub fn zeros(g: *RowOps, shape: []const c_int, _: ops.Dtype) !u32 {
        g.allocs += 1;
        var buf: [4096]i64 = @splat(-1);
        return g.new(buf[0..@intCast(shape[1])]);
    }
    pub fn hostArray(g: *RowOps, bytes: []const u8, _: []const c_int, _: ops.Dtype) !u32 {
        const s = std.mem.bytesAsSlice(i32, bytes);
        return g.new(&.{s[1]});
    }
    pub fn sliceUpdateDyn(g: *RowOps, buf: u32, upd: u32, starts: u32) !u32 {
        const row: usize = @intCast(g.rows(starts)[0]);
        var out: [4096]i64 = undefined;
        const b = g.rows(buf);
        @memcpy(out[0..b.len], b);
        const u = g.rows(upd);
        if (row + u.len > b.len) return error.SliceUpdateBounds;
        @memcpy(out[row..][0..u.len], u);
        return g.new(out[0..b.len]);
    }
    pub fn slice(g: *RowOps, x: u32, start: []const c_int, stop: []const c_int, _: []const c_int) !u32 {
        const r = g.rows(x);
        if (stop[1] > r.len) return error.SliceBounds;
        var out: [4096]i64 = undefined;
        const lo: usize = @intCast(start[1]);
        const hi: usize = @intCast(stop[1]);
        @memcpy(out[0 .. hi - lo], r[lo..hi]);
        return g.new(out[0 .. hi - lo]);
    }
    pub fn concat(g: *RowOps, xs: []const u32, _: c_int) !u32 {
        var out: [4096]i64 = undefined;
        var n: usize = 0;
        for (xs) |x| {
            const r = g.rows(x);
            @memcpy(out[n..][0..r.len], r);
            n += r.len;
        }
        return g.new(out[0..n]);
    }
};

const RS = LayerState(RowOps);

/// The window view must be exactly positions [drop, logical) and hold every
/// row the queries of the last append can reach.
fn expectWindow(g: *RowOps, st: *const RS, logical: u32) !void {
    const v = (try st.window.view(g)) orelse return error.TestUnexpectedResult;
    const r = g.rows(v);
    const drop = st.window.dropOffset();
    try testing.expectEqual(logical - drop, @as(u32, @intCast(r.len)));
    for (r, 0..) |id, j| try testing.expectEqual(@as(i64, @intCast(drop + j)), id);
}

test "dsv41 cache: the window ring keeps every reachable row across prefill chunks, decode, verify and trims" {
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    const li: v41.LayerInfo = .{ .ratio = 0 };
    var st = RS.init(li, 128, .{ .route = .window_ring });
    defer st.deinit(&g);
    var full = RS.init(li, 128, .{ .route = .full_history });
    defer full.deinit(&g);
    // prefill in chunks wider and narrower than the ring, then decode / verify with rejections
    const Step = struct { n: u32, trim: u32 = 0 };
    var steps: [36]Step = @splat(.{ .n = 6, .trim = 2 });
    steps[0..6].* = .{ .{ .n = 300 }, .{ .n = 40 }, .{ .n = 1 }, .{ .n = 6, .trim = 4 }, .{ .n = 1 }, .{ .n = 6, .trim = 5 } };
    var pos: u32 = 0;
    for (steps) |sp| {
        const x = try g.range(pos, pos + sp.n);
        try st.window.append(&g, x);
        try full.window.append(&g, x);
        st.advance(sp.n);
        full.advance(sp.n);
        pos += sp.n;
        try expectWindow(&g, &st, pos);
        // Every query of this append reaches back window-1 rows: all resident.
        try testing.expect(pos - sp.n + 1 >= st.window.dropOffset() + st.window_size or st.window.dropOffset() == 0);
        if (sp.trim > 0) {
            try testing.expectEqual(sp.trim, try st.trim(&g, sp.trim));
            _ = try full.trim(&g, sp.trim);
            pos -= sp.trim;
            try expectWindow(&g, &st, pos);
        }
    }
    // The ring stayed bounded: two buffers of window + 8 + 8 + 64 rows after the wide prefill chunk shrank back.
    try testing.expectEqual(@as(u32, 128 + 8 + 8 + 64), st.window.ring.phys_cap);
    // A rollback past the resident window is a clean miss, never a wrong read.
    try testing.expectEqual(@as(u32, 0), try st.trim(&g, 200));
    try testing.expectEqual(pos, st.offset);
    // Before the first compaction every row is resident: a short sequence rolls back freely.
    var short = RS.init(li, 128, .{ .route = .window_ring });
    defer short.deinit(&g);
    try short.window.append(&g, try g.range(0, 40));
    short.advance(40);
    try testing.expectEqual(@as(u32, 6), try short.trim(&g, 6));
    try expectWindow(&g, &short, 34);
}

test "dsv41 cache: grow and bounded lanes read like the concatenated store, trims keep capacity" {
    const L = Lanes(RowOps);
    for ([_]L.Grow{ L.Grow.init(256, null), L.Grow.init(708, 708) }) |lane0| {
        var g: RowOps = .{ .gpa = testing.allocator };
        defer g.deinit();
        var lane: L.Store = .{ .grow = lane0 };
        defer lane.deinit(&g);
        var pos: u32 = 0;
        for ([_]u32{ 333, 1, 1, 6, 17, 300 }) |n| {
            try lane.append(&g, try g.range(pos, pos + n));
            pos += n;
            const v = (try lane.view(&g)).?;
            try testing.expectEqual(@as(usize, pos), g.rows(v).len);
            for (g.rows(v), 0..) |id, j| try testing.expectEqual(@as(i64, @intCast(j)), id);
        }
        const allocs = g.allocs;
        try lane.truncate(&g, 500);
        try lane.append(&g, try g.range(500, 510));
        try testing.expectEqual(allocs, g.allocs); // a trim keeps the capacity
        for (g.rows((try lane.view(&g)).?), 0..) |id, j| try testing.expectEqual(@as(i64, @intCast(j)), id);
        if (lane.grow.bounded_cap != null) try testing.expectError(error.BoundedLaneFull, lane.append(&g, try g.range(0, 200)));
    }
    // W107 caps at max_kv 700, ratio 2: 358 groups (the frontier is a ring, no cap of its own); admission refuses
    // before any write.
    try testing.expectEqual(@as(?u32, 358), boundedCompCap(700, 2));
    const li: v41.LayerInfo = .{ .ratio = 2, .kv_source = true, .index_source = true, .mode = .full };
    var st = RS.init(li, 128, .{ .route = .bounded, .max_kv = 700 });
    st.offset = 700;
    try testing.expectError(error.BoundedLaneFull, st.canAdmit(18));
    try st.canAdmit(17);
    // The per-state limit agrees with the per-forward check at every length around it,
    // for every compression ratio the model has (with and without the frontier).
    for ([_]u8{ 1, 2, 4, 128 }) |ratio| for ([_]u32{ 1, 7, 700, 4096 }) |max_kv| {
        const lr: v41.LayerInfo = .{ .ratio = ratio, .kv_source = true, .index_source = true, .mode = .full };
        var s2 = RS.init(lr, 128, .{ .route = .bounded, .max_kv = max_kv });
        const lim = s2.admitLimit().?;
        for (lim -| 40..lim + 40) |len| {
            s2.offset = 0;
            const ok = if (s2.canAdmit(@intCast(len))) |_| true else |_| false;
            try testing.expectEqual(len <= lim, ok);
        }
    };
    const unbounded = RS.init(li, 128, .{ .route = .full_history });
    try testing.expectEqual(@as(?u32, null), unbounded.admitLimit());
}

test "dsv41 cache: the compressor frontier pools each group once, across chunks and a trim" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const li: v41.LayerInfo = .{ .ratio = 4, .kv_source = true, .index_source = true, .mode = .full };
    var st = LayerState(ops.TraceOps).init(li, 128, .{ .route = .window_ring });
    defer st.deinit(&g);
    const want = [_]?c_int{ 2, null, 1, null };
    for ([_]c_int{ 9, 2, 1, 3 }, want) |n, w| {
        const kv = try g.input(&.{ 1, n, 512 }, .float32);
        const pooled = try st.frontierPush(&g, kv, kv);
        if (w) |groups| {
            try testing.expect(g.shapeOf(pooled.?).eql(ops.Shape.of(&.{ 1, groups, 512 })));
        } else try testing.expect(pooled == null);
    }
    try testing.expectEqual(@as(u32, 15), st.nFed());
    st.offset = 15;
    // Trim 5: the frontier re-exposes the partial group of 10 fed rows.
    try testing.expectEqual(@as(u32, 5), try st.trim(&g, 5));
    try testing.expectEqual(@as(u32, 10), st.nFed());
    const kv = try g.input(&.{ 1, 2, 512 }, .float32);
    try testing.expect(g.shapeOf((try st.frontierPush(&g, kv, kv)).?).eql(ops.Shape.of(&.{ 1, 1, 512 })));
}

test "dsv41 cache: the frontier ring hands every push its completed groups' rows, as the full store does, across prefill chunks, verify blocks and trims" {
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    for ([_]u8{ 2, 4 }) |ratio| {
        const li: v41.LayerInfo = .{ .ratio = ratio, .kv_source = true, .index_source = true, .mode = .full };
        var st = RS.init(li, 128, .{ .route = .bounded, .max_kv = 20000 });
        defer st.deinit(&g);
        var full = RS.init(li, 128, .{ .route = .full_history });
        defer full.deinit(&g);
        try testing.expect(st.frontier.?.kv == .ring and full.frontier.?.kv == .store);
        // K16's chunks (953 and a ragged tail), then decode / verify blocks with rejected rows trimmed.
        const Step = struct { n: u32, trim: u32 = 0 };
        var steps: [40]Step = @splat(.{ .n = 6, .trim = 2 });
        steps[0..8].* = .{ .{ .n = 953 }, .{ .n = 953 }, .{ .n = 953 }, .{ .n = 183 }, .{ .n = 1 }, .{ .n = 6, .trim = 5 }, .{ .n = 8, .trim = 7 }, .{ .n = 3 } };
        var pos: u32 = 0;
        for (steps) |sp| {
            const x = try g.range(pos, pos + sp.n);
            const got = try st.frontierGroups(&g, x, x);
            const want = try full.frontierGroups(&g, x, x);
            try testing.expectEqual(want == null, got == null);
            if (got) |gr| {
                try testing.expectEqual(want.?.groups, gr.groups);
                try testing.expectEqualSlices(i64, g.rows(want.?.kv), g.rows(gr.kv));
                try testing.expectEqualSlices(i64, g.rows(want.?.score), g.rows(gr.score));
                // The rows of groups g_before .. g_after: every one fed, each group's rows once.
                const first = g.rows(gr.kv)[0];
                try testing.expectEqual(@as(i64, 0), @mod(first, ratio));
                for (g.rows(gr.kv), 0..) |id, j| try testing.expectEqual(first + @as(i64, @intCast(j)), id);
            }
            st.advance(sp.n);
            full.advance(sp.n);
            pos += sp.n;
            if (sp.trim > 0) {
                try testing.expect(st.canTrim(sp.trim));
                try testing.expectEqual(sp.trim, try st.trim(&g, sp.trim));
                _ = try full.trim(&g, sp.trim);
                pos -= sp.trim;
                try testing.expectEqual(full.nFed(), st.nFed());
            }
        }
        // The ring stayed small: two buffers of ratio + 8 + 8 + 64 rows once the wide chunks passed.
        try testing.expectEqual(@as(u32, ratio + 8 + 8 + 64), st.frontier.?.kv.ring.phys_cap);
        // A rollback deeper than the ring keeps is a clean miss (the window's rule), never a wrong group.
        try testing.expect(!st.canTrim(200));
        try testing.expectEqual(@as(u32, 0), try st.trim(&g, 200));
        try testing.expectEqual(full.nFed(), st.nFed());
    }
}

test "dsv41 cache: prefill chunks follow the Python shape-aware derivation" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    // Goldens from `_derive_prefill_chunk` (8 GB target; ratio-1 layers double T).
    try testing.expectEqual(@as(i64, 953), resolvePrefillChunk(&c, 16384, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 2048), resolvePrefillChunk(&c, 2048, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 238), resolvePrefillChunk(&c, 65536, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 32), resolvePrefillChunk(&c, 16384, 32, default_chunk_target_bytes));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spans = try prefillSpans(arena.allocator(), 2000, 953);
    try testing.expectEqual(@as(usize, 3), spans.len);
    try testing.expectEqual([2]u32{ 1906, 2000 }, spans[2]);
    try testing.expectEqual(@as(usize, 1), (try prefillSpans(arena.allocator(), 1, 953)).len);
    // The 16K cell's prompt: the lane of record's 17 x 953 + 183 chunks.
    const cell = try prefillSpans(arena.allocator(), 16384, resolvePrefillChunk(&c, 16384, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(usize, 18), cell.len);
    for (cell[0..17]) |sp| try testing.expectEqual(@as(u32, 953), sp[1] - sp[0]);
    try testing.expectEqual([2]u32{ 16201, 16384 }, cell[17]);
}
