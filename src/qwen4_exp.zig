//! Qwen3.8-Flash-Next (`model_type` qwen4_exp) host-side pieces: the hashed
//! n-gram embedding (PLE) id math and the mmapped quantized n-gram table.
//!
//! The 51B-parameter table (320M rows x 160) is never copied to the GPU:
//! a token touches 16 rows, so the rows are dequantized from the mmap on the
//! host and only the [T, 2560] result is sent, or (serial forwards, when the
//! working set fits it) a kernel reads the mapping in place (`ple_gpu.zig`).
//! Format: `ngram_table.bin` is a safetensors-format file holding one merged
//! affine table (`weight` U32 [R, dim*bits/32], `scales`/`biases` BF16
//! [R, dim/gs]) written by `tests/convert_qwen38_flash_next.py`, or one
//! merged RAW BF16 table (`weight` BF16 [R, dim], no scales/biases,
//! `"bits":"16"`) for bit-exact PLE lookups.

const std = @import("std");
const log = @import("log");
const ple_gpu = @import("ple_gpu.zig");

pub const ngram = @import("ngram");
pub const MAX_HEADS = ngram.MAX_HEADS;
pub const MAX_NGRAM_SIZE = ngram.MAX_NGRAM_SIZE;
pub const NgramHash = ngram.NgramHash;
pub const WARM_LOG_BYTES = ngram.WARM_LOG_BYTES;
pub const WARM_LOG_NS = ngram.WARM_LOG_NS;
pub const WarmProgress = ngram.WarmProgress;
pub const NgramTable = ngram.NgramTable;
pub const RowCache = ngram.RowCache;
const PrefetchPool = ngram.PrefetchPool;
pub const PREFILL_SAY_MIN_ROWS = ngram.PREFILL_SAY_MIN_ROWS;
pub const PREFILL_PREFETCH_MIN_KV = ngram.PREFILL_PREFETCH_MIN_KV;
pub const PrefillPrefetchMode = ngram.PrefillPrefetchMode;
pub const plePrefillPrefetchModeFromEnv = ngram.plePrefillPrefetchModeFromEnv;
pub const plePrefillPrefetchMinKvFromEnv = ngram.plePrefillPrefetchMinKvFromEnv;
pub const plePrefillPrefetchWanted = ngram.plePrefillPrefetchWanted;
pub const bf16ToF32 = ngram.bf16ToF32;
pub const bf16Rne = ngram.bf16Rne;

// ── tests ──

const testing = std.testing;

test "ngram hash reproduces the reference multipliers, primes and offsets" {
    const h = try NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
    try testing.expectEqual(@as(i64, 23703573157769), h.multipliers[0]);
    try testing.expectEqual(@as(i64, 20109073645365), h.multipliers[1]);
    try testing.expectEqual(@as(i64, 8052911324071), h.multipliers[2]);
    try testing.expectEqual(@as(i64, 20000003), h.vocab[0]);
    try testing.expectEqual(@as(i64, 20000171), h.vocab[15]);
    try testing.expectEqual(@as(i64, 300001275), h.offsets[15]);
    try testing.expectEqual(@as(u64, 320001536), h.total_rows);
}

test "ngram row ids match the reference on an eos-split history" {
    // Reference (modeling_qwen4_exp.py, run in python): history
    // [eos, eos | 5, 7, eos, 9, 11]; shifts reset across the eos.
    const h = try NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
    const prev = [_]u32{ 248044, 248044 };
    const ids = [_]u32{ 5, 7, 248044, 9, 11 };
    var out: [5 * 16]i64 = undefined;
    h.rowIds(&prev, &ids, &out);
    const want = [5][16]i64{
        .{ 15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159, 179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898 },
        .{ 12441580, 26378836, 53347667, 75104214, 99467174, 114254887, 126436461, 156012011, 169119442, 187827161, 214803956, 239809754, 242938905, 266427765, 294337448, 314484167 },
        .{ 10204458, 27984170, 41283776, 68842151, 85621153, 118821647, 129504214, 158727320, 176298516, 181690702, 206665473, 238343128, 252151767, 267018740, 285543023, 319927855 },
        .{ 18043673, 37626835, 51159316, 78294604, 94015356, 106720349, 136526052, 144330141, 176817901, 186368539, 203707490, 230017629, 247662678, 266533413, 293096193, 307951937 },
        .{ 10041117, 28960672, 48420531, 71664411, 83016360, 106800418, 122476460, 150044571, 163654473, 184259024, 206781966, 224776026, 248853488, 273290488, 294849492, 303242927 },
    };
    for (want, 0..) |row, t| {
        for (row, 0..) |v, i| try testing.expectEqual(v, out[t * 16 + i]);
    }
}

test "ngram table row dequant follows the MLX affine nibble layout" {
    // Two rows, dim 32, 4-bit, one group: word k packs elements 8k..8k+7,
    // element i at nibble i % 8.
    var buf: [8 + 512 + 2 * 16 + 2 * 2 + 2 * 2]u8 = undefined;
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[2,4],\"data_offsets\":[0,32]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[32,36]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[36,40]}}";
    var hdr: [512]u8 = @splat(' ');
    @memcpy(hdr[0..header.len], header);
    std.mem.writeInt(u64, buf[0..8], 512, .little);
    @memcpy(buf[8..520], &hdr);
    const data = buf[520..];
    // row 0: elements 0..31 = i % 16; row 1: all 3
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        var w: u32 = 0;
        var j: u32 = 0;
        while (j < 8) : (j += 1) w |= ((i * 8 + j) % 16) << @intCast(j * 4);
        std.mem.writeInt(u32, data[i * 4 ..][0..4], w, .little);
        std.mem.writeInt(u32, data[16 + i * 4 ..][0..4], 0x33333333, .little);
    }
    // scales: row0 = 0.5 (bf16 0x3F00), row1 = 2.0 (0x4000); biases: row0 = 1.0 (0x3F80), row1 = -1 (0xBF80)
    std.mem.writeInt(u16, data[32..34], 0x3F00, .little);
    std.mem.writeInt(u16, data[34..36], 0x4000, .little);
    std.mem.writeInt(u16, data[36..38], 0x3F80, .little);
    std.mem.writeInt(u16, data[38..40], 0xBF80, .little);
    const aligned = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), buf.len);
    defer std.heap.page_allocator.free(aligned);
    @memcpy(aligned, &buf);
    const t = try NgramTable.parse(aligned, aligned[8..520], 520);
    try testing.expectEqual(@as(u32, 32), t.dim);
    var out: [32]f32 = undefined;
    t.row(0, &out);
    try testing.expectEqual(@as(f32, 1.0), out[0]);
    try testing.expectEqual(@as(f32, 0.5 * 7 + 1.0), out[7]);
    try testing.expectEqual(@as(f32, 0.5 * 15 + 1.0), out[31]);
    t.row(1, &out);
    try testing.expectEqual(@as(f32, 5.0), out[13]);
}

test "ngram table row dequant reads the dense mx.quantize packing at every width" {
    // mx.quantize packs element i at bit offset i*bits of the little-endian
    // u32 stream (wcols = dim*bits/32); at 3/5/6 bits elements straddle words.
    // dim 32, one group, q[i] = i % 4, scale 1, bias 0 ⇒ out[i] == i % 4.
    inline for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| {
        const wcols = 32 * bits / 32;
        var packed_words: [8]u32 = @splat(0);
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            const off = i * bits;
            const q: u64 = i % 4;
            const w = off / 32;
            packed_words[w] |= @truncate(q << @intCast(off % 32));
            if (off % 32 + bits > 32) packed_words[w + 1] |= @truncate(q >> @intCast(32 - off % 32));
        }
        var hbuf: [256]u8 = undefined;
        const header = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"bits\":\"{d}\",\"group_size\":\"32\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[1,{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[{d},{d}]}}}}", .{ bits, wcols, wcols * 4, wcols * 4, wcols * 4 + 2, wcols * 4 + 2, wcols * 4 + 4 });
        const total = 8 + 256 + wcols * 4 + 4;
        const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), total);
        defer std.heap.page_allocator.free(buf);
        @memset(buf, ' ');
        std.mem.writeInt(u64, buf[0..8], 256, .little);
        @memcpy(buf[8 .. 8 + header.len], header);
        const data = buf[264..];
        for (0..wcols) |w| std.mem.writeInt(u32, data[w * 4 ..][0..4], packed_words[w], .little);
        std.mem.writeInt(u16, data[wcols * 4 ..][0..2], 0x3F80, .little);
        std.mem.writeInt(u16, data[wcols * 4 + 2 ..][0..2], 0, .little);
        const t = try NgramTable.parse(buf, buf[8..264], 264);
        var out: [32]f32 = undefined;
        t.row(0, &out);
        for (out, 0..) |v, k| try testing.expectEqual(@as(f32, @floatFromInt(k % 4)), v);
    }
}

test "ngram table raw BF16 rows do not depend on quantization group size" {
    for ([_][]const u8{ "0", "1", "32", "1024" }) |group_size| {
        var header_buf: [512]u8 = undefined;
        const header = try std.fmt.bufPrint(
            &header_buf,
            "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"{s}\"}}," ++
                "\"weight\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}}}",
            .{group_size},
        );
        const buf = try ngramTestImage(header, 8);
        defer std.heap.page_allocator.free(buf);
        const values = [_]u16{ 0x3F80, 0xC000, 0x3F00, 0x4040 };
        for (values, 0..) |value, i| std.mem.writeInt(u16, buf[520 + i * 2 ..][0..2], value, .little);
        const table = try NgramTable.parse(buf, buf[8..520], 520);
        var row: [2]f32 = undefined;
        table.row(0, &row);
        try testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, &row);
        table.row(1, &row);
        try testing.expectEqualSlices(f32, &.{ 0.5, 3.0 }, &row);
    }
}

test "ngram table raw bf16 rows copy out converted without scales" {
    // bits 16: `weight` BF16 [2,4], no scales/biases keys at all.
    const header = "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]}}";
    const total = 8 + 256 + 16;
    const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), total);
    defer std.heap.page_allocator.free(buf);
    @memset(buf, ' ');
    std.mem.writeInt(u64, buf[0..8], 256, .little);
    @memcpy(buf[8 .. 8 + header.len], header);
    const data = buf[264..];
    const vals = [_]u16{ 0x3F80, 0xC000, 0x4000, 0x0000, 0xBF80, 0x3F00, 0xC040, 0x4040 };
    for (vals, 0..) |v, k| std.mem.writeInt(u16, data[k * 2 ..][0..2], v, .little);
    const t = try NgramTable.parse(buf, buf[8..264], 264);
    try testing.expectEqual(@as(u32, 16), t.bits);
    try testing.expectEqual(@as(u64, 2), t.rows);
    try testing.expectEqual(@as(u32, 4), t.dim);
    var out: [4]f32 = undefined;
    t.row(0, &out);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, -2.0, 2.0, 0.0 }, &out);
    t.row(1, &out);
    try testing.expectEqualSlices(f32, &[_]f32{ -1.0, 0.5, -3.0, 3.0 }, &out);
}

test "dsv41 ngram table: a BF16 tensor inside a checkpoint shard gathers its raw rows past the page cache" {
    // Two tensors, the second a BF16 [5, 3] table after a U8 one; no mlx-serve-ngram metadata.
    const header = "{\"other\":{\"dtype\":\"U8\",\"shape\":[1,7],\"data_offsets\":[0,7]},\"embed.weight\":{\"dtype\":\"BF16\",\"shape\":[5,3],\"data_offsets\":[7,37]}}";
    var image: [8 + header.len + 37 + 3]u8 = undefined;
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    const data = image[8 + header.len ..];
    for (data, 0..) |*b, i| b.* = @truncate(i *% 37 +% 11);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(testing.io, .{ .sub_path = "shard.safetensors", .data = &image });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/shard.safetensors", .{root[0..try td.dir.realPath(testing.io, &root)]}, 0);
    var t = try NgramTable.openTensor(path, "embed.weight");
    defer t.close();
    try testing.expect(t.nocache and t.map.len == 0 and t.warm_thread == null);
    try testing.expectEqual(@as(u64, 5), t.rows);
    try testing.expectEqual(@as(u32, 3), t.dim);
    var out: [4 * 6]u8 = undefined;
    const ids = [_]u32{ 4, 0, 4, 2 };
    try t.gatherRaw(&ids, &out);
    for (ids, 0..) |r, i| try testing.expectEqualSlices(u8, data[7 + r * 6 ..][0..6], out[i * 6 ..][0..6]);
    t.startWarm(); // no warm thread on a no-cache table
    try testing.expect(t.warm_thread == null);
    try testing.expectError(error.NgramTableRegion, t.gatherRaw(&.{5}, out[0..6]));
    try testing.expectError(error.NgramTableRegion, t.gatherRaw(&.{0}, out[0..5]));
    // By name only: another dtype or a missing name is refused.
    try testing.expectError(error.NgramTableHeader, NgramTable.openTensor(path, "other"));
    try testing.expectError(error.NgramTableHeader, NgramTable.openTensor(path, "missing"));
    try testing.expectError(error.FileNotFound, NgramTable.openTensor("/nonexistent/shard.safetensors", "embed.weight"));
}

/// Module-owned state for one loaded qwen4_exp model: the n-gram hash and
/// the mmapped table. Non-null on `Transformer.qwen4` ⇒ the arch is served
/// serially with speculation off (`ownsModuleDecodeState`), which is what
/// the per-request PLE/indexer state in `SSMCacheEntry.aux_state` needs
/// until the snapshot machinery carries it.
pub const Qwen4State = struct {
    hash: NgramHash,
    table: NgramTable,
    /// The table as one no-copy GPU buffer; null = the host gather serves every forward.
    gpu: ?ple_gpu.Table = null,

    pub fn deinit(self: *Qwen4State) void {
        self.table.close();
        if (self.gpu) |g| g.release();
        self.gpu = null;
    }
};

test "ngram prefill prefetch is KV-GATED: a short prompt walks, a long one pools" {
    const min = PREFILL_PREFETCH_MIN_KV;
    try testing.expect(!plePrefillPrefetchWanted(.kv_gated, 0, min));
    for ([_]u64{ 4096, 8192, 16_384, 65_536, 131_072, 262_143 }) |kv| {
        try testing.expect(!plePrefillPrefetchWanted(.kv_gated, kv, min));
    }
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, min, min));
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, 355_000, min));
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, 8192, 4096));
    try testing.expect(!plePrefillPrefetchWanted(.kv_gated, 8192, 131_072));
    try testing.expect(!plePrefillPrefetchWanted(.off, 1_000_000, min));
    try testing.expect(plePrefillPrefetchWanted(.on, 0, min));

    try testing.expectEqual(PrefillPrefetchMode.kv_gated, plePrefillPrefetchModeFromEnv(null));
    try testing.expectEqual(PrefillPrefetchMode.kv_gated, plePrefillPrefetchModeFromEnv(""));
    try testing.expectEqual(PrefillPrefetchMode.off, plePrefillPrefetchModeFromEnv("0"));
    try testing.expectEqual(PrefillPrefetchMode.on, plePrefillPrefetchModeFromEnv("1"));

    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv(null));
    try testing.expectEqual(@as(u64, 131_072), plePrefillPrefetchMinKvFromEnv("131072"));
    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv("64k"));
    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv(""));
    try testing.expectEqual(@as(u64, 0), plePrefillPrefetchMinKvFromEnv("0")); // an explicit always-on
}

test "ngram prefill gather: 4096 rows through the pool equal the direct mmap read" {
    // 4-bit, group 32, dim 32 -> wcols 4 u32, scols 1. 4096 rows = 64 pool batches.
    const ROWS: usize = 4096;
    const HDR: usize = 512;
    const W: usize = ROWS * 16;
    const SB: usize = ROWS * 2;
    const buf = try testing.allocator.alloc(u8, 8 + HDR + W + 2 * SB);
    defer testing.allocator.free(buf);
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
        "\"weight\":{\"dtype\":\"U32\",\"shape\":[4096,4],\"data_offsets\":[0,65536]}," ++
        "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[65536,73728]}," ++
        "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[73728,81920]}}";
    std.mem.writeInt(u64, buf[0..8], HDR, .little);
    @memset(buf[8 .. 8 + HDR], ' ');
    @memcpy(buf[8..][0..header.len], header);
    for (buf[8 + HDR ..], 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);

    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});

    ngram.warm_override = false; // the warm thread would race the fd for no benefit
    defer ngram.warm_override = null;
    var t = try NgramTable.open(path);
    defer t.close();
    const pool = t.pool orelse return error.SkipZigTest;
    try testing.expectEqual(@as(u32, 32), t.dim);

    const ids = try testing.allocator.alloc(i64, ROWS);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*r, i| r.* = @intCast((i *% 1237) % ROWS);
    const ref = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(ref);
    const got = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(got);

    ngram.ple_prefill_arm_said[0][0].store(false, .monotonic);
    ngram.ple_prefill_arm_said[0][1].store(false, .monotonic);
    ngram.ple_prefill_arm_said[1][0].store(false, .monotonic);
    ngram.ple_prefill_arm_said[1][1].store(false, .monotonic);
    const said = struct {
        fn f(arm: usize, bucket: usize) bool {
            return ngram.ple_prefill_arm_said[arm][bucket].load(.monotonic);
        }
    }.f;

    // Below the kv gate: serial mmap walk, no pool round.
    ngram.ple_prefill_min_kv_override = 65_536;
    defer ngram.ple_prefill_min_kv_override = null;
    const before = pool.runs.load(.monotonic);
    t.gather(ids, ref, 8192);
    try testing.expectEqual(before, pool.runs.load(.monotonic));
    try testing.expect(said(0, 1));
    try testing.expect(!said(0, 0));
    try testing.expect(!said(1, 1) and !said(1, 0));

    // Past the threshold the same gather rides the pool.
    t.gather(ids, got, 131_072);
    try testing.expectEqual(before + ROWS / PrefetchPool.MAX_ROWS, pool.runs.load(.monotonic));
    try testing.expect(said(1, 1));
    try testing.expect(!said(1, 0));

    try testing.expectEqualSlices(f32, ref, got);

    ngram.ple_prefill_prefetch_override = false;
    defer ngram.ple_prefill_prefetch_override = null;
    const forced_off = pool.runs.load(.monotonic);
    t.gather(ids, got, 1_000_000);
    try testing.expectEqual(forced_off, pool.runs.load(.monotonic));
    try testing.expectEqualSlices(f32, ref, got);
    ngram.ple_prefill_prefetch_override = true;
    t.gather(ids, got, 0);
    try testing.expectEqual(forced_off + ROWS / PrefetchPool.MAX_ROWS, pool.runs.load(.monotonic));
    try testing.expectEqualSlices(f32, ref, got);

    // A wide-but-short gather (> MAX_ROWS, < PREFILL_SAY_MIN_ROWS) reports the warmup bucket.
    const warm = ids[0..128];
    const w_out = try testing.allocator.alloc(f32, warm.len * t.dim);
    defer testing.allocator.free(w_out);
    t.gather(warm, w_out, 0); // forced on: the warmup forward runs at kv 0
    try testing.expect(said(1, 0));
    try testing.expectEqualSlices(f32, got[0 .. warm.len * t.dim], w_out);

    const serial_warm_before = said(0, 0);
    const dec = ids[0..16];
    const d_ref = try testing.allocator.alloc(f32, dec.len * t.dim);
    defer testing.allocator.free(d_ref);
    const d_got = try testing.allocator.alloc(f32, dec.len * t.dim);
    defer testing.allocator.free(d_got);
    ngram.ple_prefill_prefetch_override = false;
    const dec_before = pool.runs.load(.monotonic);
    t.gather(dec, d_ref, 1_000_000);
    try testing.expectEqual(dec_before + 1, pool.runs.load(.monotonic)); // still pooled
    ngram.ple_prefill_prefetch_override = true;
    t.gather(dec, d_got, 0);
    try testing.expectEqualSlices(f32, d_ref, d_got);
    try testing.expectEqual(serial_warm_before, said(0, 0));
}

test "ngram table warm: touches the whole file in the background; close() joins mid-warm" {
    // The 4-bit fixture from the nibble-layout test, written to a real file
    // so open()'s kept fd serves the warm preads.
    var buf: [8 + 512 + 2 * 16 + 2 * 2 + 2 * 2]u8 = undefined;
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[2,4],\"data_offsets\":[0,32]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[32,36]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[36,40]}}";
    var hdr: [512]u8 = @splat(' ');
    @memcpy(hdr[0..header.len], header);
    std.mem.writeInt(u64, buf[0..8], 512, .little);
    @memcpy(buf[8..520], &hdr);
    @memset(buf[520..], 0x33);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = &buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});

    ngram.warm_override = true;
    defer ngram.warm_override = null;
    var t = try NgramTable.open(path);
    t.startWarm();
    try testing.expect(t.warm_thread != null);
    var spins: u32 = 0;
    while (t.warm_bytes.load(.acquire) < buf.len) : (spins += 1) {
        if (spins > 10_000) return error.WarmNeverFinished;
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }
    try testing.expectEqual(@as(u64, buf.len), t.warm_bytes.load(.acquire));
    t.close();

    // close() during the warm joins instead of racing the fd/munmap.
    var t2 = try NgramTable.open(path);
    t2.startWarm();
    t2.close();

    // Kill switch: no thread.
    ngram.warm_override = false;
    var t3 = try NgramTable.open(path);
    t3.startWarm();
    try testing.expect(t3.warm_thread == null);
    t3.close();
}

/// A whole `ngram_table.bin` image in one page-aligned buffer. Caller frees with the page allocator.
fn ngramTestImage(header: []const u8, data_bytes: usize) ![]align(std.heap.page_size_min) u8 {
    const hlen: usize = 512;
    std.debug.assert(header.len <= hlen);
    const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 8 + hlen + data_bytes);
    @memset(buf, ' ');
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + header.len], header);
    @memset(buf[8 + hlen ..], 0);
    return buf;
}

fn ngramTestParse(header: []const u8, data_bytes: usize) !NgramTable {
    const buf = try ngramTestImage(header, data_bytes);
    return NgramTable.parse(buf, buf[8..520], 520);
}

/// rows 4, dim 64, 4-bit, group 32 => wcols 8, scols 2; w 128 B, s/b 16 B each.
const NGRAM_GOOD_HEADER =
    "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"}," ++
    "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
    "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
    "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}";
const NGRAM_GOOD_BYTES = 160;

test "ngram table header: a missing or wrong-typed field is a named error, never a trap" {
    const t = try ngramTestParse(NGRAM_GOOD_HEADER, NGRAM_GOOD_BYTES);
    try testing.expectEqual(@as(u64, 4), t.rows);
    try testing.expectEqual(@as(u32, 64), t.dim);
    try testing.expectEqual(@as(u32, 4), t.bits);
    std.heap.page_allocator.free(@constCast(t.map));

    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":4,\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"F32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"format\":\"pt\",\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
}

test "ngram table header: bits must be a width mx.quantize actually ships" {
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"32\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,64],\"data_offsets\":[0,1024]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[1024,1040]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[1040,1056]}}",
        1056,
    ));
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"7\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,14],\"data_offsets\":[0,224]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[224,240]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[240,256]}}",
        256,
    ));
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"0\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        160,
    ));
}

test "ngram table header: every region is bounded, sized by its own shape and disjoint" {
    // Weight region too small for rows x wcols x 4.
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,64]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    // Scales overlapping the weights.
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[120,136]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[0,8],\"data_offsets\":[0,0]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[0,2],\"data_offsets\":[0,0]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[0,2],\"data_offsets\":[0,0]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[128,136]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableTruncated, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[400,416]}}",
        NGRAM_GOOD_BYTES,
    ));
}

test "NgramHash.init refuses a config past its fixed arrays instead of asserting" {
    try testing.expectError(error.InvalidQwen4NgramSize, NgramHash.init(248320, 9, 8, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramSize, NgramHash.init(248320, 1, 8, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramHeads, NgramHash.init(248320, 5, 16, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramHeads, NgramHash.init(248320, 3, 0, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramVocab, NgramHash.init(248320, 3, 8, 20_000_000, 0, 1234, 0, 248044));
    const wide = try NgramHash.init(248320, 5, 8, 20_000_000, 128, 1234, 0, 248044);
    try testing.expectEqual(@as(u32, 32), wide.n_heads);
}

test "WarmProgress emits on the byte step, on the silence timeout, and never twice for one step" {
    const GB: u64 = 1 << 30;
    const S: u64 = 1_000_000_000;
    var p: WarmProgress = .{};
    // Nothing before the first step, however long it takes... except that a
    // long silence is itself worth a line.
    try testing.expect(!p.should(1 * GB, 1 * S));
    try testing.expect(!p.should(2 * GB, 9 * S));
    try testing.expect(p.should(3 * GB, 10 * S)); // silence timeout
    try testing.expect(!p.should(4 * GB, 11 * S)); // clock restarted by that line
    // Crossing the byte step emits once, and the step advances past it.
    try testing.expect(p.should(WARM_LOG_BYTES, 12 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES, 13 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES + 1, 13 * S));
    // A jump of several steps still emits exactly once and does not backlog.
    try testing.expect(p.should(WARM_LOG_BYTES * 4, 14 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES * 4 + 1, 15 * S));
    try testing.expect(p.should(WARM_LOG_BYTES * 5, 16 * S));
}

/// A file of `rows` records of `rb` bytes at `path`, record r filled with byte r (mod 256).
fn writeRecordFile(path: [:0]const u8, rows: usize, rb: usize) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestOpen;
    defer _ = std.c.close(fd);
    const buf = try std.testing.allocator.alloc(u8, rows * rb);
    defer std.testing.allocator.free(buf);
    for (0..rows) |r| @memset(buf[r * rb ..][0..rb], @truncate(r));
    if (std.c.write(fd, buf.ptr, buf.len) != @as(isize, @intCast(buf.len))) return error.TestWrite;
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

test "ngram record table: Python's gather sequence gives Python's stats, residency and LRU order" {
    // Python's NGramRowCache (mtplx/ngram_row_cache.py) gives these stats and this order for this
    // sequence (32 rows of 33 B, 3 slots).
    const rb = 33;
    var pbuf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "/tmp/ngram-records-{d}.bin", .{std.c.getpid()}, 0);
    try writeRecordFile(path, 32, rb);
    defer _ = std.c.unlink(path.ptr);
    var t = try NgramTable.openRecords(path, rb, 32, 0);
    defer t.close();
    try t.attachCache(std.testing.allocator, 3 * rb);
    const seq = [_][]const i64{ &.{ 5, 6, 5, 9 }, &.{ 6, 10 }, &.{ 11, 12, 13, 14 }, &.{13}, &.{15}, &.{12} };
    var out: [4 * rb]u8 = undefined;
    for (seq) |rows| {
        try t.gatherRecords(rows, out[0 .. rows.len * rb]);
        for (rows, 0..) |r, i| try std.testing.expect(std.mem.allEqual(u8, out[i * rb ..][0..rb], @intCast(r)));
    }
    const c = t.cache.?;
    try std.testing.expectEqual(RowCache.Stats{ .hits = 2, .misses = 10, .evictions = 7, .reads = 7, .rows_read = 10, .gathers = 6 }, c.stats);
    var order: [3]u64 = undefined;
    try std.testing.expectEqual(@as(usize, 3), lruOrder(c, &order));
    try std.testing.expectEqual([3]u64{ 13, 15, 12 }, order);
    try std.testing.expectError(error.RowOutOfRange, t.gatherRecords(&.{ 3, 32 }, out[0 .. 2 * rb]));
    try std.testing.expectEqual(@as(u64, 6), c.stats.gathers);
}

test "ngram record table: pooled and serial reads, and gathers past the arena, give the same records and stats" {
    const rb = 20;
    var pbuf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "/tmp/ngram-records-pool-{d}.bin", .{std.c.getpid()}, 0);
    try writeRecordFile(path, 4096, rb);
    defer _ = std.c.unlink(path.ptr);
    var pooled = try NgramTable.openRecords(path, rb, 4096, 0);
    defer pooled.close();
    try std.testing.expect(pooled.pool != null);
    try pooled.attachCache(std.testing.allocator, 97 * rb);
    var serial = try NgramTable.openRecords(path, rb, 4096, 0);
    defer serial.close();
    serial.pool.?.destroy();
    serial.pool = null;
    try serial.attachCache(std.testing.allocator, 97 * rb);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var rows: [300]i64 = undefined;
    var a: [300 * rb]u8 = undefined;
    var b: [300 * rb]u8 = undefined;
    for (0..60) |step| {
        const n: usize = if (step % 13 == 12) 300 else 1 + rnd.uintLessThan(usize, 60);
        const band: u64 = if (step % 5 == 0) 4096 else 300;
        for (rows[0..n]) |*r| r.* = @intCast(rnd.uintLessThan(u64, band));
        try pooled.gatherRecords(rows[0..n], a[0 .. n * rb]);
        try serial.gatherRecords(rows[0..n], b[0 .. n * rb]);
        try std.testing.expectEqualSlices(u8, b[0 .. n * rb], a[0 .. n * rb]);
        for (rows[0..n], 0..) |r, i| try std.testing.expect(std.mem.allEqual(u8, a[i * rb ..][0..rb], @truncate(@as(u64, @intCast(r)))));
    }
    try std.testing.expectEqual(serial.cache.?.stats, pooled.cache.?.stats);
    try std.testing.expect(pooled.cache.?.stats.evictions > 0 and pooled.cache.?.stats.hits > 0);
    // Without a cache every row is read, in order.
    var plain = try NgramTable.openRecords(path, rb, 4096, 0);
    defer plain.close();
    try plain.gatherRecords(&.{ 4095, 0, 7, 7 }, a[0 .. 4 * rb]);
    for ([_]u8{ 255, 0, 7, 7 }, 0..) |v, i| try std.testing.expect(std.mem.allEqual(u8, a[i * rb ..][0..rb], v));
}

test "ngram record table: the host charge is the cache's allocation; a short file refuses" {
    try std.testing.expectEqual(@as(u32, 254_200), RowCache.slotCount(264, 64 << 20));
    try std.testing.expectEqual(@as(u32, 524_288), RowCache.indexCapacity(254_200));
    try std.testing.expectEqual(@as(u64, 254_200 * (264 + 20) + 524_288 * 13), RowCache.hostBytes(264, 64 << 20));
    var c = try RowCache.init(std.testing.allocator, 264, 64 << 20);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 254_200 * 264), c.arena.len);
    try std.testing.expectEqual(RowCache.indexCapacity(c.slot_count), c.index.capacity());
    var pbuf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "/tmp/ngram-records-short-{d}.bin", .{std.c.getpid()}, 0);
    try writeRecordFile(path, 10, 8);
    defer _ = std.c.unlink(path.ptr);
    try std.testing.expectError(error.NgramTableTruncated, NgramTable.openRecords(path, 8, 11, 0));
    try std.testing.expectError(error.NgramTableTruncated, NgramTable.openRecords(path, 8, 10, 8));
    try std.testing.expectError(error.NgramTableRegion, NgramTable.openRecords(path, 0, 10, 0));
}
