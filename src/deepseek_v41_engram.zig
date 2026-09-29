//! Engram row ids and bank rows for DeepSeek-V4.1 (layers 1 and 14), host
//! side: the manifest's hashing recipe (Python `NgramHashState`: rolling XOR
//! of the n-gram's compressed ids times per-layer multipliers, mod per-head
//! primes, plus flat offsets) and the 264-byte mxfp8 records it indexes. The
//! compressed token map is a converter output (tokenizer normalisation stays
//! Python); the rows' dequantize and the gated add are MLX (`engramApply`).

const std = @import("std");
const v41 = @import("deepseek_v41.zig");

pub const max_ngram = 8;
pub const max_heads = 16;
pub const max_cols = (max_ngram - 1) * max_heads;
pub const max_layers = 8;
/// A masked position (an image span): no n-gram may span it.
pub const dead: i64 = -1;

pub const Hashing = struct {
    max_ngram: u32,
    n_heads: u32,
    n_layers: u32,
    layer_ids: [max_layers]u32 = @splat(0),
    multipliers: [max_layers][max_ngram]i64 = @splat(@splat(0)),
    primes: [max_layers][max_ngram - 1][max_heads]i64 = @splat(@splat(@splat(0))),
    flat_offsets: [max_layers][max_cols]i64 = @splat(@splat(0)),
    total_rows: [max_layers]u64 = @splat(0),
    pad_id: u32,
    compressed_vocab: u32,

    pub fn cols(self: *const Hashing) u32 {
        return (self.max_ngram - 1) * self.n_heads;
    }
};

pub const Bank = struct {
    head_dim: u32,
    record_bytes: u32,
    files: [max_layers][]const u8,
    /// Records per layer file (the manifest's `rows`).
    rows: [max_layers]u64 = @splat(0),
    /// The manifest's own identity field; a token map names the manifest it was built for.
    manifest_sha256: []const u8 = "",
};

/// `engram-manifest.json` checked against the config's Engram fields; any
/// codec, geometry or table this build does not implement is refused.
pub fn parseManifest(a: std.mem.Allocator, text: []const u8, c: *const v41.Config, diag: ?*v41.Diag) !struct { hashing: Hashing, bank: Bank } {
    const Layer = struct {
        layer_id: u32,
        file: []const u8,
        rows: u64,
        record_bytes: u32,
        quant: struct { bits: u32, group_size: u32, mode: []const u8, head_dim: u32 },
    };
    const PerLayer = struct { layer_id: u32, primes: []const []const i64, flat_offsets: []const i64, total_rows: u64 };
    const M = struct {
        format: []const u8,
        manifest_sha256: []const u8 = "",
        layers: []const Layer,
        hashing: struct {
            layer_ids: []const u32,
            max_ngram_size: u32,
            n_heads: u32,
            head_dim: u32,
            compressed_vocab_size: u32,
            pad_id: u32,
            n_hash_cols: u32,
            hash_multipliers: []const []const i64,
            per_layer: []const PerLayer,
        },
    };
    const m = std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return fail(diag, "engram manifest: {s}", .{@errorName(e)}),
    };
    if (!std.mem.eql(u8, m.format, "mtplx-engram-manifest-v1")) return fail(diag, "engram manifest: format {s}", .{m.format});
    const h = m.hashing;
    const e = c.engram;
    if (h.max_ngram_size != e.max_ngram_size or h.n_heads != e.n_heads or h.head_dim != e.head_dim or
        h.compressed_vocab_size != e.compressed_vocab_size or h.pad_id != e.pad_token_id or h.n_hash_cols != e.hashCols())
        return fail(diag, "engram manifest: hashing geometry differs from config.json", .{});
    if (h.max_ngram_size < 2 or h.max_ngram_size > max_ngram or h.n_heads > max_heads) return fail(diag, "engram manifest: {d}-grams x {d} heads not implemented", .{ h.max_ngram_size, h.n_heads });
    if (h.layer_ids.len != e.n_layers or h.per_layer.len != e.n_layers or h.hash_multipliers.len != e.n_layers or m.layers.len != e.n_layers)
        return fail(diag, "engram manifest: {d} layers, config has {d}", .{ h.layer_ids.len, e.n_layers });
    var hs: Hashing = .{ .max_ngram = h.max_ngram_size, .n_heads = h.n_heads, .n_layers = e.n_layers, .pad_id = h.pad_id, .compressed_vocab = h.compressed_vocab_size };
    var bank: Bank = .{ .head_dim = h.head_dim, .record_bytes = h.head_dim + h.head_dim / 32, .files = undefined, .manifest_sha256 = m.manifest_sha256 };
    for (0..e.n_layers) |i| {
        const lid = h.layer_ids[i];
        const pl = h.per_layer[i];
        const ly = m.layers[i];
        if (lid != e.layer_ids[i] or pl.layer_id != lid or ly.layer_id != lid) return fail(diag, "engram manifest: layer order differs from config.json at {d}", .{i});
        if (pl.total_rows != e.num_embeddings[i] or ly.rows != pl.total_rows) return fail(diag, "engram layer {d}: {d} rows, config has {d}", .{ lid, pl.total_rows, e.num_embeddings[i] });
        if (!std.mem.eql(u8, ly.quant.mode, "mxfp8") or ly.quant.bits != 8 or ly.quant.group_size != 32 or ly.quant.head_dim != h.head_dim or ly.record_bytes != bank.record_bytes)
            return fail(diag, "engram layer {d}: codec {s} bits {d} group {d} record {d} not implemented", .{ lid, ly.quant.mode, ly.quant.bits, ly.quant.group_size, ly.record_bytes });
        if (h.hash_multipliers[i].len != h.max_ngram_size or pl.primes.len != h.max_ngram_size - 1 or pl.flat_offsets.len != hs.cols())
            return fail(diag, "engram layer {d}: hash tables have the wrong shape", .{lid});
        hs.layer_ids[i] = lid;
        hs.total_rows[i] = pl.total_rows;
        for (h.hash_multipliers[i], 0..) |v, k| hs.multipliers[i][k] = v;
        for (pl.primes, 0..) |row, k| {
            if (row.len != h.n_heads) return fail(diag, "engram layer {d}: primes row {d} has {d} heads", .{ lid, k, row.len });
            for (row, 0..) |p, hd| {
                if (p <= 0) return fail(diag, "engram layer {d}: prime {d} is not positive", .{ lid, p });
                hs.primes[i][k][hd] = p;
            }
        }
        for (pl.flat_offsets, 0..) |o, k| hs.flat_offsets[i][k] = o;
        bank.files[i] = ly.file;
        bank.rows[i] = ly.rows;
    }
    return .{ .hashing = hs, .bank = bank };
}

fn fail(diag: ?*v41.Diag, comptime fmt: []const u8, args: anytype) error{EngramManifest} {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return error.EngramManifest;
}

/// The per-sequence compressed-id history (Python `NgramHashState._buf`):
/// lookbacks cross the prefill / decode boundary; `trim` rewinds a rejected
/// draft so re-fed tokens hash identically.
pub const HashState = struct {
    hist: std.ArrayList(i64) = .empty,

    pub fn deinit(self: *HashState, gpa: std.mem.Allocator) void {
        self.hist.deinit(gpa);
    }

    pub fn trim(self: *HashState, n: usize) void {
        self.hist.shrinkRetainingCapacity(self.hist.items.len - n);
    }

    /// Feed `ids` (compressed through `map`; a masked position is `dead`) and
    /// write their row ids, `[len(ids)][n_layers][cols]`, into `out`.
    pub fn advance(self: *HashState, gpa: std.mem.Allocator, h: *const Hashing, map: []const u32, ids: []const u32, masked: ?[]const bool, out: []i64) !void {
        const cols = h.cols();
        if (out.len != ids.len * h.n_layers * cols) return error.RowsShape;
        const start = self.hist.items.len;
        for (ids, 0..) |id, i| {
            if (id >= map.len) return error.TokenOutOfMap;
            const dead_here = if (masked) |mk| mk[i] else false;
            try self.hist.append(gpa, if (dead_here) dead else map[id]);
        }
        const pad: i64 = map[h.pad_id];
        for (0..ids.len) |i| {
            const pos = start + i;
            var toks: [max_ngram]i64 = undefined;
            var blocked = false;
            for (0..h.max_ngram) |shift| {
                const src = self.hist.items[if (pos >= shift) pos - shift else 0];
                blocked = blocked or pos < shift or src == dead;
                toks[shift] = if (blocked) pad else src;
            }
            for (0..h.n_layers) |l| {
                var rolling: i64 = toks[0] *% h.multipliers[l][0];
                for (1..h.max_ngram) |k| {
                    rolling ^= toks[k] *% h.multipliers[l][k];
                    for (0..h.n_heads) |hd| {
                        const col = (k - 1) * h.n_heads + hd;
                        out[(i * h.n_layers + l) * cols + col] = @mod(rolling, h.primes[l][k - 1][hd]) + h.flat_offsets[l][col];
                    }
                }
            }
        }
    }
};

/// Read the records of `rows` from one layer's bank: `codes` gets the E4M3
/// words (`head_dim` bytes per row), `scales` the E8M0 bytes (`head_dim / 32`).
pub fn readRows(fd: std.c.fd_t, bank: *const Bank, rows: []const i64, codes: []u8, scales: []u8) !void {
    const rb: usize = bank.record_bytes;
    const hd: usize = bank.head_dim;
    var rec: [4096]u8 = undefined;
    if (rb > rec.len) return error.RecordTooLarge;
    for (rows, 0..) |r, i| {
        if (r < 0) return error.RowOutOfRange;
        const off: u64 = @as(u64, @intCast(r)) * rb;
        var done: usize = 0;
        while (done < rb) {
            const n = std.c.pread(fd, rec[done..].ptr, rb - done, @intCast(off + done));
            if (n < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.ShortRead;
            done += @intCast(n);
        }
        @memcpy(codes[i * hd ..][0..hd], rec[0..hd]);
        @memcpy(scales[i * (hd / 32) ..][0 .. hd / 32], rec[hd..rb]);
    }
}

/// Construction-time refusals of the row source (one named error each; the
/// message says which field or file).
pub const Refusal = error{ EngramManifest, EngramTokenMap, EngramBankFile };

fn refuse(diag: ?*v41.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

/// The compressed token map: `R/exl3/runtime/export_dsv41_engram_token_map.py`
/// runs our Python `build_compressed_token_map` (tokenizer normalisation stays
/// Python) and writes one little-endian u32 per vocab id plus a JSON sidecar
/// naming the tokenizer.json and the manifest it was built for. Checked once.
pub const TokenMap = struct {
    ids: []u32,
    pad_compressed: u32,

    pub const format = "mtplx-dsv41-engram-token-map-v1";

    /// `map_path` + `map_path.json`, against the bank's tokenizer.json, the
    /// manifest identity and the config / hashing geometry. `a` owns the result.
    pub fn load(a: std.mem.Allocator, io: std.Io, map_path: []const u8, bank_dir: []const u8, c: *const v41.Config, h: *const Hashing, bank: *const Bank, diag: ?*v41.Diag) !TokenMap {
        const Meta = struct {
            format: []const u8,
            vocab: u64,
            compressed_vocab_size: u64,
            pad_id: u64,
            pad_compressed: u64,
            map_sha256: []const u8,
            tokenizer_sha256: []const u8,
            manifest_sha256: []const u8,
        };
        const meta_path = try std.fmt.allocPrint(a, "{s}.json", .{map_path});
        const meta_text = std.Io.Dir.cwd().readFileAlloc(io, meta_path, a, .limited(1 << 16)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s} (the converter writes the map and its sidecar together)", .{ meta_path, @errorName(e) }),
        };
        const meta = std.json.parseFromSliceLeaky(Meta, a, meta_text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ meta_path, @errorName(e) }),
        };
        if (!std.mem.eql(u8, meta.format, format)) return refuse(diag, error.EngramTokenMap, "{s}: format {s}", .{ meta_path, meta.format });
        if (meta.vocab != c.vocab_size) return refuse(diag, error.EngramTokenMap, "token map covers {d} ids, config vocab_size is {d}", .{ meta.vocab, c.vocab_size });
        if (meta.compressed_vocab_size != h.compressed_vocab) return refuse(diag, error.EngramTokenMap, "token map compresses to {d} ids, the manifest to {d}", .{ meta.compressed_vocab_size, h.compressed_vocab });
        if (meta.pad_id != h.pad_id) return refuse(diag, error.EngramTokenMap, "token map pad id {d}, the manifest's {d}", .{ meta.pad_id, h.pad_id });
        if (!std.mem.eql(u8, meta.manifest_sha256, bank.manifest_sha256)) return refuse(diag, error.EngramTokenMap, "token map built for manifest {s}, this bank's is {s}", .{ meta.manifest_sha256, bank.manifest_sha256 });
        const tok_path = try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{bank_dir});
        const tok = std.Io.Dir.cwd().readFileAlloc(io, tok_path, a, .limited(256 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ tok_path, @errorName(e) }),
        };
        if (!std.mem.eql(u8, &sha256Hex(tok), meta.tokenizer_sha256)) return refuse(diag, error.EngramTokenMap, "token map built from tokenizer {s}, the bank's tokenizer.json is {s}", .{ meta.tokenizer_sha256, &sha256Hex(tok) });
        const raw = std.Io.Dir.cwd().readFileAlloc(io, map_path, a, .limited(64 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ map_path, @errorName(e) }),
        };
        if (raw.len != meta.vocab * 4) return refuse(diag, error.EngramTokenMap, "{s}: {d} bytes, want {d} (u32 per id)", .{ map_path, raw.len, meta.vocab * 4 });
        if (!std.mem.eql(u8, &sha256Hex(raw), meta.map_sha256)) return refuse(diag, error.EngramTokenMap, "{s}: sha256 differs from its sidecar", .{map_path});
        const ids = try a.alloc(u32, raw.len / 4);
        var top: u32 = 0;
        for (ids, 0..) |*v, i| {
            v.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
            top = @max(top, v.*);
        }
        // Python keys the map by distinct normalised forms: ids 0 .. size-1, all used.
        if (@as(u64, top) + 1 != h.compressed_vocab) return refuse(diag, error.EngramTokenMap, "token map ids reach {d}, want 0 .. {d}", .{ top, h.compressed_vocab - 1 });
        if (ids[h.pad_id] != meta.pad_compressed) return refuse(diag, error.EngramTokenMap, "token map pads to {d}, its sidecar says {d}", .{ ids[h.pad_id], meta.pad_compressed });
        return .{ .ids = ids, .pad_compressed = ids[h.pad_id] };
    }
};

fn sha256Hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// The Engram row source (Python `EngramV41`'s row fetch: `NgramHashState` ids,
/// `NGramRowCache` records): the manifest, the token map and one open file per
/// Engram layer, all checked at `open`. Immutable afterwards; each sequence
/// owns a `HashState`. The dequantize of the fetched records is MLX
/// (`Trunk.engramRows`), the gated add `Trunk.engramApply`.
pub const RowSource = struct {
    arena: std.heap.ArenaAllocator,
    hashing: Hashing,
    bank: Bank,
    map: TokenMap,
    fds: [max_layers]std.c.fd_t = @splat(-1),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, bank_dir: []const u8, map_path: []const u8, c: *const v41.Config, diag: ?*v41.Diag) !RowSource {
        var self: RowSource = .{ .arena = std.heap.ArenaAllocator.init(gpa), .hashing = undefined, .bank = undefined, .map = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        const mpath = try std.fmt.allocPrint(a, "{s}/engram/engram-manifest.json", .{bank_dir});
        const mtext = std.Io.Dir.cwd().readFileAlloc(io, mpath, a, .limited(1 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramManifest, "{s}: {s}", .{ mpath, @errorName(e) }),
        };
        const m = try parseManifest(a, mtext, c, diag);
        self.hashing = m.hashing;
        self.bank = m.bank;
        self.map = try TokenMap.load(a, io, map_path, bank_dir, c, &self.hashing, &self.bank, diag);
        for (0..self.hashing.n_layers) |i| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank_dir, self.bank.files[i] }, 0);
            const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (fd < 0) return refuse(diag, error.EngramBankFile, "{s}: cannot open", .{path});
            self.fds[i] = fd;
            // Records are read one by one at random rows: past the page cache, read-ahead off.
            if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0 or std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0)
                return refuse(diag, error.EngramBankFile, "{s}: F_NOCACHE / F_RDAHEAD refused", .{path});
            var st: std.c.Stat = undefined;
            if (std.c.fstat(fd, &st) != 0) return refuse(diag, error.EngramBankFile, "{s}: fstat failed", .{path});
            const want = self.bank.rows[i] * self.bank.record_bytes;
            if (@as(u64, @intCast(st.size)) != want) return refuse(diag, error.EngramBankFile, "{s}: {d} bytes, the manifest's {d} rows x {d} need {d}", .{ path, st.size, self.bank.rows[i], self.bank.record_bytes, want });
        }
        return self;
    }

    pub fn deinit(self: *RowSource) void {
        for (self.fds) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        self.arena.deinit();
    }

    /// Row ids per position: `[n_layers][cols]`.
    pub fn perToken(self: *const RowSource) usize {
        return self.hashing.n_layers * self.hashing.cols();
    }

    /// `NgramHashState.advance` for one sequence: `out` gets `[ids][n_layers][cols]`.
    pub fn advance(self: *const RowSource, gpa: std.mem.Allocator, st: *HashState, ids: []const u32, out: []i64) !void {
        return st.advance(gpa, &self.hashing, self.map.ids, ids, null, out);
    }

    /// Layer slot `li`'s records for the `n` positions of `rows` (`advance`'s
    /// output): `codes` gets `[n * cols][head_dim]` E4M3 bytes, `scales`
    /// `[n * cols][head_dim / 32]` E8M0 bytes, positions then columns.
    pub fn read(self: *const RowSource, li: usize, rows: []const i64, n: usize, ids_buf: []i64, codes: []u8, scales: []u8) !void {
        const cols = self.hashing.cols();
        const per = self.perToken();
        for (0..n) |t| @memcpy(ids_buf[t * cols ..][0..cols], rows[t * per + li * cols ..][0..cols]);
        return self.readIds(li, ids_buf[0 .. n * cols], codes, scales);
    }

    /// The records of row ids `ids` of layer slot `li`.
    pub fn readIds(self: *const RowSource, li: usize, ids: []const i64, codes: []u8, scales: []u8) !void {
        for (ids) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.bank.rows[li]) return error.RowOutOfRange;
        return readRows(self.fds[li], &self.bank, ids, codes, scales);
    }
};

const testing = std.testing;

/// A small table with hand-checkable arithmetic: 3-grams, 2 heads, 1 layer.
fn tinyHashing() Hashing {
    var h: Hashing = .{ .max_ngram = 3, .n_heads = 2, .n_layers = 1, .pad_id = 0, .compressed_vocab = 10 };
    h.multipliers[0][0] = 3;
    h.multipliers[0][1] = 5;
    h.multipliers[0][2] = 7;
    h.primes[0][0][0] = 11;
    h.primes[0][0][1] = 13;
    h.primes[0][1][0] = 17;
    h.primes[0][1][1] = 19;
    h.flat_offsets[0][0] = 0;
    h.flat_offsets[0][1] = 11;
    h.flat_offsets[0][2] = 24;
    h.flat_offsets[0][3] = 41;
    return h;
}

test "dsv41 engram: the n-gram hash follows the manifest recipe" {
    const h = tinyHashing();
    const map = [_]u32{ 9, 2, 4, 6 }; // pad id 0 -> compressed 9
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows: [2 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, &.{ 1, 2 }, null, &rows);
    // pos 0: toks (2, pad 9, pad 9); pos 1: toks (4, 2, pad 9).
    const r0a: i64 = (2 * 3) ^ (9 * 5);
    const r0b: i64 = r0a ^ (9 * 7);
    const r1a: i64 = (4 * 3) ^ (2 * 5);
    const r1b: i64 = r1a ^ (9 * 7);
    const want = [_]i64{
        @mod(r0a, 11), @mod(r0a, 13) + 11, @mod(r0b, 17) + 24, @mod(r0b, 19) + 41,
        @mod(r1a, 11), @mod(r1a, 13) + 11, @mod(r1b, 17) + 24, @mod(r1b, 19) + 41,
    };
    try testing.expectEqualSlices(i64, &want, &rows);
}

test "dsv41 engram: streaming, masking and trim hash like one pass" {
    const h = tinyHashing();
    const map = [_]u32{ 9, 2, 4, 6, 1, 3 };
    const ids = [_]u32{ 1, 2, 3, 4, 5, 1 };
    var one: HashState = .{};
    defer one.deinit(testing.allocator);
    var all: [6 * 4]i64 = undefined;
    try one.advance(testing.allocator, &h, &map, &ids, null, &all);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var part: [6 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, ids[0..4], null, part[0 .. 4 * 4]);
    // a rejected draft: feed two wrong tokens, trim them, re-feed the real ones
    var junk: [2 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, &.{ 3, 3 }, null, &junk);
    st.trim(2);
    try st.advance(testing.allocator, &h, &map, ids[4..6], null, part[4 * 4 ..]);
    try testing.expectEqualSlices(i64, &all, &part);
    // a masked position pads every n-gram that spans it
    var m: HashState = .{};
    defer m.deinit(testing.allocator);
    var masked: [3 * 4]i64 = undefined;
    try m.advance(testing.allocator, &h, &map, &.{ 1, 2, 3 }, &.{ false, true, false }, &masked);
    var p: HashState = .{};
    defer p.deinit(testing.allocator);
    var fresh: [1 * 4]i64 = undefined;
    try p.advance(testing.allocator, &h, &map, &.{3}, null, &fresh);
    try testing.expectEqualSlices(i64, &fresh, masked[2 * 4 ..]); // pos 2 sees (tok, pad, pad)
    try testing.expectError(error.TokenOutOfMap, p.advance(testing.allocator, &h, &map, &.{6}, null, &fresh));
}

// DSV41_BANK=<bank> DSV41_ENGRAM_FIXTURE=<json from R/exl3/runtime/dump_dsv41_engram_fixture.py>
test "dsv41 engram: the real manifest and token map hash the prompt to the Python oracle's rows" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_ENGRAM_FIXTURE") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank_dir, &diag);
    const mtext = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fmt.allocPrint(a, "{s}/engram/engram-manifest.json", .{bank_dir}), a, .limited(1 << 20));
    const parsed = try parseManifest(a, mtext, &c, &diag);
    const Row = struct { layer: u32, row: i64, sha256: []const u8 };
    const Fixture = struct { token_map: []const u8, ids: []const u32, prompt: u32, rows: []const i64, pad_compressed: i64, bytes: []const Row };
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, try std.Io.Dir.cwd().readFileAlloc(testing.io, fixture, a, .limited(64 << 20)), .{ .ignore_unknown_fields = true });
    const map_bytes = try std.Io.Dir.cwd().readFileAlloc(testing.io, fx.token_map, a, .limited(16 << 20));
    const map = try a.alloc(u32, map_bytes.len / 4);
    for (map, 0..) |*v, i| v.* = std.mem.readInt(u32, map_bytes[i * 4 ..][0..4], .little);
    try testing.expectEqual(fx.pad_compressed, @as(i64, map[parsed.hashing.pad_id]));
    const per = parsed.hashing.n_layers * parsed.hashing.cols();
    const got = try a.alloc(i64, fx.ids.len * per);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    // the prompt in one span, then the decode tokens one by one (the oracle fed them the same way)
    try st.advance(testing.allocator, &parsed.hashing, map, fx.ids[0..fx.prompt], null, got[0 .. fx.prompt * per]);
    for (fx.prompt..fx.ids.len) |t| try st.advance(testing.allocator, &parsed.hashing, map, fx.ids[t .. t + 1], null, got[t * per ..][0..per]);
    try testing.expectEqualSlices(i64, fx.rows, got);
    for (fx.bytes) |r| {
        const slot = std.mem.indexOfScalar(u32, parsed.hashing.layer_ids[0..parsed.hashing.n_layers], r.layer) orelse return error.TestUnexpectedResult;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank_dir, parsed.bank.files[slot] }, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        var codes: [256]u8 = undefined;
        var scales: [8]u8 = undefined;
        try readRows(fd, &parsed.bank, &.{r.row}, &codes, &scales);
        var d = std.crypto.hash.sha2.Sha256.init(.{});
        d.update(&codes);
        d.update(&scales);
        try testing.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(d.finalResult(), .lower));
    }
    std.debug.print("dsv41 engram: {d} positions x {d} rows equal the oracle; {d} records byte-equal\n", .{ fx.ids.len, per, fx.bytes.len });
}

// ── the row source on a synthetic mini bank (hermetic) ──

pub const MiniBank = struct {
    vocab: u32 = 64,
    map_mod: u32 = 50,
    sidecar_vocab: ?u64 = null,
    tokenizer_differs: bool = false,
    manifest_sha: []const u8 = "mini-manifest",
    sidecar_manifest_sha: []const u8 = "mini-manifest",
    flip_map_byte: bool = false,
    drop_sidecar: bool = false,
    short_bank: bool = false,
    codec: []const u8 = "mxfp8",
};

/// Record r byte i of the mini bank: `(r * 7 + i) mod 251`.
fn miniRecordByte(r: u64, i: u64) u8 {
    return @intCast((r * 7 + i) % 251);
}

/// A bank dir for the mini config (Engram layer 1: 97 rows, 3-grams x 2 heads,
/// head_dim 32 -> 33-byte records) with its token map + sidecar; returns the map path.
pub fn writeMiniBank(a: std.mem.Allocator, tmp: *std.testing.TmpDir, root: []const u8, f: MiniBank) ![]const u8 {
    const io = testing.io;
    try tmp.dir.createDirPath(io, "engram");
    const tok = if (f.tokenizer_differs) "{\"model\":\"other\"}" else "{\"model\":\"mini\"}";
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tok });
    const manifest = try std.fmt.allocPrint(a,
        \\{{"format":"mtplx-engram-manifest-v1","manifest_sha256":"{s}",
        \\"layers":[{{"layer_id":1,"file":"engram-L1.bin","rows":97,"record_bytes":33,
        \\"quant":{{"bits":8,"group_size":32,"mode":"{s}","head_dim":32}}}}],
        \\"hashing":{{"layer_ids":[1],"max_ngram_size":3,"n_heads":2,"head_dim":32,"compressed_vocab_size":50,
        \\"pad_id":2,"n_hash_cols":4,"hash_multipliers":[[3,5,7]],
        \\"per_layer":[{{"layer_id":1,"primes":[[11,13],[17,19]],"flat_offsets":[0,11,24,41],"total_rows":97}}]}}}}
    , .{ f.manifest_sha, f.codec });
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/engram-manifest.json", .data = manifest });
    const n_rec: u64 = if (f.short_bank) 96 else 97;
    const bank = try a.alloc(u8, @intCast(n_rec * 33));
    for (0..n_rec) |r| for (0..33) |i| {
        bank[r * 33 + i] = miniRecordByte(r, i);
    };
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/engram-L1.bin", .data = bank });
    const map = try a.alloc(u8, f.vocab * 4);
    for (0..f.vocab) |i| std.mem.writeInt(u32, map[i * 4 ..][0..4], @intCast(i % f.map_mod), .little);
    const map_sha = sha256Hex(map);
    if (f.flip_map_byte) map[5] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "map.u32", .data = map });
    if (!f.drop_sidecar) {
        const meta = try std.fmt.allocPrint(a,
            \\{{"format":"{s}","vocab":{d},"compressed_vocab_size":50,"pad_id":2,"pad_compressed":2,
            \\"map_sha256":"{s}","tokenizer_sha256":"{s}","manifest_sha256":"{s}"}}
        , .{ TokenMap.format, f.sidecar_vocab orelse f.vocab, &map_sha, &sha256Hex("{\"model\":\"mini\"}"), f.sidecar_manifest_sha });
        try tmp.dir.writeFile(io, .{ .sub_path = "map.u32.json", .data = meta });
    }
    return std.fmt.allocPrint(a, "{s}/map.u32", .{root});
}

fn miniConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .mini);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

test "dsv41 engram: a synthetic bank opens and its rows come back through the row source" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const map_path = try writeMiniBank(a, &tmp, root, .{});
    const c = try miniConfig();
    var diag: v41.Diag = .{};
    var src = RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer src.deinit();
    try testing.expectEqual(@as(usize, 4), src.perToken());
    try testing.expectEqual(@as(u32, 2), src.map.pad_compressed);
    // The source hashes like a bare HashState over the same table and map.
    const ids = [_]u32{ 5, 7, 9 };
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows: [3 * 4]i64 = undefined;
    try src.advance(testing.allocator, &st, &ids, &rows);
    var ref: HashState = .{};
    defer ref.deinit(testing.allocator);
    var want: [3 * 4]i64 = undefined;
    try ref.advance(testing.allocator, &src.hashing, src.map.ids, &ids, null, &want);
    try testing.expectEqualSlices(i64, &want, &rows);
    // Records land positions-then-columns, code bytes and scale bytes split.
    var idb: [12]i64 = undefined;
    var codes: [12 * 32]u8 = undefined;
    var scales: [12]u8 = undefined;
    try src.read(0, &rows, 3, &idb, &codes, &scales);
    for (0..12) |k| {
        const r: u64 = @intCast(rows[k]);
        for (0..32) |i| try testing.expectEqual(miniRecordByte(r, i), codes[k * 32 + i]);
        try testing.expectEqual(miniRecordByte(r, 32), scales[k]);
    }
    try testing.expectError(error.RowOutOfRange, src.readIds(0, &.{97}, codes[0..32], scales[0..1]));
}

test "dsv41 engram: the row source refuses a map or bank built for something else, by name" {
    const Case = struct { f: MiniBank, err: anyerror };
    const cases = [_]Case{
        .{ .f = .{ .sidecar_vocab = 63 }, .err = error.EngramTokenMap },
        .{ .f = .{ .tokenizer_differs = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .sidecar_manifest_sha = "another-manifest" }, .err = error.EngramTokenMap },
        .{ .f = .{ .flip_map_byte = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .map_mod = 49 }, .err = error.EngramTokenMap },
        .{ .f = .{ .drop_sidecar = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .short_bank = true }, .err = error.EngramBankFile },
        .{ .f = .{ .codec = "affine" }, .err = error.EngramManifest },
    };
    for (cases, 0..) |cs, i| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var rbuf: [512]u8 = undefined;
        const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
        const map_path = try writeMiniBank(a, &tmp, root, cs.f);
        const c = try miniConfig();
        var diag: v41.Diag = .{};
        if (RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag)) |opened| {
            var o = opened;
            o.deinit();
            std.debug.print("case {d}: opened, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {d}: {s}\n", .{ i, diag.message() });
                return x;
            };
            try testing.expect(diag.message().len > 0);
        }
    }
}

// DSV41_BANK=<bank> DSV41_ENGRAM_TOKEN_MAP=<converter output> DSV41_ENGRAM_FIXTURE=<m0 fixture json>
test "dsv41 engram: the real bank's row source hashes and reads like the Python oracle" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_ENGRAM_FIXTURE") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank_dir, &diag);
    var src = try RowSource.open(testing.allocator, testing.io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const Row = struct { layer: u32, row: i64, sha256: []const u8 };
    const Fixture = struct { ids: []const u32, prompt: u32, rows: []const i64, bytes: []const Row };
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, try std.Io.Dir.cwd().readFileAlloc(testing.io, fixture, a, .limited(64 << 20)), .{ .ignore_unknown_fields = true });
    const per = src.perToken();
    const got = try a.alloc(i64, fx.ids.len * per);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    try src.advance(testing.allocator, &st, fx.ids[0..fx.prompt], got[0 .. fx.prompt * per]);
    for (fx.prompt..fx.ids.len) |t| try src.advance(testing.allocator, &st, fx.ids[t .. t + 1], got[t * per ..][0..per]);
    try testing.expectEqualSlices(i64, fx.rows, got);
    for (fx.bytes) |r| {
        const li = std.mem.indexOfScalar(u32, src.hashing.layer_ids[0..src.hashing.n_layers], r.layer) orelse return error.TestUnexpectedResult;
        var codes: [256]u8 = undefined;
        var scales: [8]u8 = undefined;
        try src.readIds(li, &.{r.row}, &codes, &scales);
        var d = std.crypto.hash.sha2.Sha256.init(.{});
        d.update(&codes);
        d.update(&scales);
        try testing.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(d.finalResult(), .lower));
    }
    std.debug.print("dsv41 engram: row source over {d} rows x {d} layers; {d} positions and {d} records equal the oracle\n", .{ src.bank.rows[0], src.hashing.n_layers, fx.ids.len, fx.bytes.len });
}
