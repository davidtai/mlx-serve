//! The DSpark-direct decode cycle's host half (Python
//! `deepseek_v41_dspark_decode._decode_cycles`): the verify schedule, the draft
//! length after the confidence early stop, greedy acceptance of the target's
//! argmax rows, and what a cycle commits. The target side of the seam is
//! `Model.forward` (verify rows with every logits row and `main_hidden`) and
//! `Model.trim`; the draft head (three stages, markov, confidence, resident
//! mxfp4 experts) is M2 proper.
//!
//! One cycle: draft `k_cap` ids from `main_h` and the primary token; keep
//! `k_eff`; verify `[primary, d1 .. d_keff]` chunk by chunk (stopping once a row
//! rejects); commit = trim the unaccepted verified rows, seed the draft's stage
//! windows with `main_hidden[0 .. accepted + 1]`, next primary = the correction
//! (or bonus), next `main_h` = `main_hidden[accepted]`.

const std = @import("std");

pub const max_block = 16;

pub const Error = error{ VerifySchedule, DraftDepth };

/// `_normalize_verify_chunks`: the construction-time schedule partitions the
/// `k_cap + 1` verify rows exactly (null = one chunk of all of them).
pub fn verifyChunks(k_cap: u32, chunks: ?[]const u32, buf: *[max_block + 1]u32) Error![]const u32 {
    if (k_cap + 1 > buf.len) return error.DraftDepth;
    const cs = chunks orelse {
        buf[0] = k_cap + 1;
        return buf[0..1];
    };
    if (cs.len == 0 or cs.len > buf.len) return error.VerifySchedule;
    var sum: u32 = 0;
    for (cs, 0..) |w, i| {
        if (w == 0) return error.VerifySchedule;
        sum += w;
        buf[i] = w;
    }
    if (sum != k_cap + 1) return error.VerifySchedule;
    return buf[0..cs.len];
}

/// `_effective_draft_len`: the leading run of drafts whose sigmoid confidence
/// clears `threshold`, at least one (a cycle always verifies a draft).
pub fn effectiveDraftLen(conf: []const f32, k: u32, threshold: ?f32) u32 {
    const t = threshold orelse return k;
    if (k == 0) return 0;
    var keep: u32 = 0;
    for (conf[0..k]) |p| {
        if (p >= t) keep += 1 else break;
    }
    return @max(1, keep);
}

/// A verify chunk's rows `[start, end)` of the block `[primary, d1 .. d_keff]`.
pub fn chunkRows(chunks: []const u32, i: usize, k_eff: u32) ?[2]u32 {
    var start: u32 = 0;
    for (chunks[0..i]) |w| start += w;
    if (start >= k_eff + 1) return null;
    return .{ start, @min(start + chunks[i], k_eff + 1) };
}

pub const Outcome = struct {
    accepted: u32 = 0,
    /// The target's token at the first rejecting row (or the bonus row).
    correction: ?u32 = null,
    /// Verify rows forwarded (chunks after the rejecting one never run).
    verified: u32 = 0,

    /// Rows `Model.trim` drops: the verified rows past the accepted run and its correction.
    pub fn trimRows(o: Outcome) u32 {
        return o.verified - (o.accepted + 1);
    }

    /// The `main_hidden` row the next draft starts from.
    pub fn nextMainRow(o: Outcome) u32 {
        return o.accepted;
    }
};

/// Greedy acceptance of one verify chunk: `target[r]` is the argmax of the
/// chunk's row r, `drafts[d]` the draft at depth d. Returns true once the cycle
/// has its correction (later chunks are not run).
pub fn acceptGreedyChunk(o: *Outcome, drafts: []const u32, k_eff: u32, rows: [2]u32, target: []const u32) bool {
    o.verified = rows[1];
    for (target, rows[0]..) |t, depth| {
        if (depth < k_eff and t == drafts[depth]) {
            o.accepted += 1;
            continue;
        }
        o.correction = t;
        return true;
    }
    return false;
}

const testing = std.testing;

test "dsv41 dspark: the verify schedule partitions K + 1 rows, the early stop keeps a leading run" {
    var buf: [max_block + 1]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{6}, try verifyChunks(5, null, &buf));
    try testing.expectEqualSlices(u32, &.{ 2, 4 }, try verifyChunks(5, &.{ 2, 4 }, &buf));
    try testing.expectError(error.VerifySchedule, verifyChunks(5, &.{ 2, 3 }, &buf));
    try testing.expectError(error.VerifySchedule, verifyChunks(5, &.{ 6, 0 }, &buf));
    const conf = [_]f32{ 0.9, 0.7, 0.4, 0.8, 0.9 };
    try testing.expectEqual(@as(u32, 5), effectiveDraftLen(&conf, 5, null));
    try testing.expectEqual(@as(u32, 2), effectiveDraftLen(&conf, 5, 0.5));
    try testing.expectEqual(@as(u32, 1), effectiveDraftLen(&conf, 5, 0.95)); // always one draft
    try testing.expectEqual(@as(u32, 0), effectiveDraftLen(&conf, 0, 0.5));
}

test "dsv41 dspark: greedy acceptance commits the accepted run and its correction, trims the rest" {
    const drafts = [_]u32{ 11, 12, 13, 14, 15 };
    // One 6-row chunk; the target agrees at depths 0 and 1, then says 99.
    var o: Outcome = .{};
    try testing.expect(acceptGreedyChunk(&o, &drafts, 5, .{ 0, 6 }, &.{ 11, 12, 99, 14, 15, 7 }));
    try testing.expectEqual(@as(u32, 2), o.accepted);
    try testing.expectEqual(@as(?u32, 99), o.correction);
    try testing.expectEqual(@as(u32, 3), o.trimRows()); // 6 verified - (2 + 1) kept
    try testing.expectEqual(@as(u32, 2), o.nextMainRow());
    // Everything accepted: the last row is the bonus, nothing to trim.
    var all: Outcome = .{};
    try testing.expect(acceptGreedyChunk(&all, &drafts, 5, .{ 0, 6 }, &.{ 11, 12, 13, 14, 15, 42 }));
    try testing.expectEqual(@as(u32, 5), all.accepted);
    try testing.expectEqual(@as(?u32, 42), all.correction);
    try testing.expectEqual(@as(u32, 0), all.trimRows());
    // A staged [2, 4] schedule stops after a rejection in its first chunk: 2 rows verified.
    var staged: Outcome = .{};
    var buf: [max_block + 1]u32 = undefined;
    const chunks = try verifyChunks(5, &.{ 2, 4 }, &buf);
    const r0 = chunkRows(chunks, 0, 5).?;
    try testing.expect(acceptGreedyChunk(&staged, &drafts, 5, r0, &.{ 50, 60 }));
    try testing.expectEqual(@as(u32, 0), staged.accepted);
    try testing.expectEqual(@as(u32, 2), staged.verified);
    try testing.expectEqual(@as(u32, 1), staged.trimRows());
    // An early stop at k_eff 2 truncates the last chunk: rows [2, 3) of a [2, 4] schedule.
    try testing.expectEqual([2]u32{ 2, 3 }, chunkRows(chunks, 1, 2).?);
    try testing.expect(chunkRows(chunks, 1, 1) == null);
}
