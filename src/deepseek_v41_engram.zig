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
    var bank: Bank = .{ .head_dim = h.head_dim, .record_bytes = h.head_dim + h.head_dim / 32, .files = undefined };
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
