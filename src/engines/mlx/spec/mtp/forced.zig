//! A forced close committed through the MTP head. A thinking budget's close (`logit_features.thinkBoundTick`, the loop
//! guard's close of a looping thought) or a forced tool-call opener (`toolForceTick`), on a solo qwen4 MTP slot at its
//! spec exit, commits like a fully accepted round (`commitForcedTokensMtp`): one trunk forward with every row's stream
//! captured, the rows appended to the head's history, the last forced token decided and not yet in the cache. The head
//! keeps its whole history and the answer keeps speculating. Every other state (`Decline`) commits plainly, and the
//! head re-seeds (`mtpResumeAfterForced`).

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const generate_mod = @import("../../generate.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const ForcedCommit = @import("../../generate/pipeline.zig").ForcedCommit;
const SpecDisableReason = generate_mod.SpecDisableReason;
const specDecodeUnsupported = @import("../verify.zig").specDecodeUnsupported;
const mtpCommittedLen = @import("chain.zig").mtpCommittedLen;
const mtpHeadPositionDrift = @import("adaptive.zig").mtpHeadPositionDrift;

pub const TbMtpState = struct {
    /// Forced commits that went through the head this request.
    commits: u32 = 0,
    /// The request's one decline line is out.
    decline_logged: bool = false,
};

/// Why a forced commit keeps the plain path, in the order `decide` checks.
pub const Decline = enum {
    reason,
    not_solo_mtp,
    no_qwen4_head,
    spec_off,
    serial_block,
    pipelined,
    no_hidden,
    head_cannot_resume,
    grouped,
    planner,
    other_spec,
    unsupported,
    max_ctx,
    head_drift,
};

pub fn declineText(d: Decline) []const u8 {
    return switch (d) {
        .reason => "not a thinking-budget, loop-recover or tool-choice close",
        .not_solo_mtp => "not this slot's solo MTP tick (a batched group, the planner's tick, or a non-MTP decode mode)",
        .no_qwen4_head => "no qwen4 MTP head on this generator",
        .spec_off => "speculation is already off for this request",
        .serial_block => "a serial block (the probe or its exit ramp) is running",
        .pipelined => "the pipeline holds a forwarded token, not the spec exit",
        .no_hidden => "no hidden for the next draft",
        .head_cannot_resume => "the head does not resume on this request (an M-RoPE turn or the sticky serial arm)",
        .grouped => "the slot drafts inside a batched group",
        .planner => "the group planner owns this slot",
        .other_spec => "a PLD, DFlash, drafter or DSpark generator",
        .unsupported => "a grammar or logprobs request",
        .max_ctx => "the context passes --max-mtp-ctx",
        .head_drift => "the head's position bookkeeping is out of sync",
    };
}

/// What `decide` reads, gathered by `facts`: the policy is a pure function of it.
pub const Facts = struct {
    reason: SpecDisableReason,
    /// The scheduler's solo decode tick for this slot, dispatching `.mtp`.
    solo_mtp_tick: bool,
    /// `mtp` and `mtp_cache` are the in-checkpoint qwen4 head.
    qwen4_head: bool,
    spec_off: bool,
    /// `mtp_serial_left > 0` or `mtp_serial_exit != .none`.
    serial_block: bool,
    /// `pending_token` / `pending_logits`: `next_token_id` is already in the cache.
    pipelined: bool,
    has_last_hidden: bool,
    /// `mtpAdaptiveHeadMayResume`.
    head_may_resume: bool,
    /// `mtp_batch_head` or a group cap.
    grouped: bool,
    planner_owned: bool,
    /// PLD, a drafter, DFlash or DSpark is armed.
    other_spec: bool,
    /// `specDecodeUnsupported`.
    unsupported: bool,
    /// `--max-mtp-ctx` still admits the context once the forced tokens are in.
    ctx_within: bool,
    /// The drift the forced rows' head append would hit (`headDrift`); null = no head state.
    head_drift: ?i64,
};

/// The forced close's policy. Null = the forced commit goes through the head.
pub fn decide(f: Facts) ?Decline {
    if (f.reason != .think_bound and f.reason != .tool_choice) return .reason;
    if (!f.solo_mtp_tick) return .not_solo_mtp;
    if (!f.qwen4_head) return .no_qwen4_head;
    if (f.spec_off) return .spec_off;
    if (f.serial_block) return .serial_block;
    if (f.pipelined) return .pipelined;
    if (!f.has_last_hidden) return .no_hidden;
    if (!f.head_may_resume) return .head_cannot_resume;
    if (f.grouped) return .grouped;
    if (f.planner_owned) return .planner;
    if (f.other_spec) return .other_spec;
    if (f.unsupported) return .unsupported;
    if (!f.ctx_within) return .max_ctx;
    const drift = f.head_drift orelse return .no_qwen4_head;
    if (drift != 0) return .head_drift;
    return null;
}

/// The head's committed length once a forced commit settles it: a pending stash applied (`mtpApplyStash` truncates to
/// the stash origin and appends its rows; `stash_end` = `off0 + n`), else a pre-drafted chain's rows past its origin
/// dropped, else the head as it stands. The forced run's first pair `(t1, last_hidden)` takes this row.
pub fn settledLen(step: usize, pre_draft_off0: ?usize, stash_end: ?usize) usize {
    if (stash_end) |e| return e;
    return mtpCommittedLen(step, pre_draft_off0, null);
}

/// Trunk positions past the one the head's next row (`head_len`) pairs with: 0 = that row is `(stream at
/// trunk_len - 1, t1)`, the head in step with the trunk. A serial block leaves a positive gap; a forced commit keeps the
/// head contiguous either way, so the gap it finds is the gap it leaves.
pub fn headGap(position_base: usize, head_len: usize, trunk_len: usize) i64 {
    return @as(i64, @intCast(trunk_len)) - @as(i64, @intCast(position_base + head_len + 1));
}

/// The scheduler's question for one forced commit: true = `commitForcedTokensMtp` may run. A decline keeps the plain
/// path, said once per request.
pub fn tbMtpWants(self: *Generator, reason: SpecDisableReason, solo_mtp_tick: bool, forced_len: usize) bool {
    const why = decide(tbMtpFacts(self, reason, solo_mtp_tick, forced_len)) orelse return true;
    if (!self.tb_mtp.decline_logged) {
        self.tb_mtp.decline_logged = true;
        log.info("[mtp] forced close through the head declined ({s}: {s}); it takes the plain forward\n", .{ @tagName(why), declineText(why) });
    }
    return false;
}

pub fn tbMtpFacts(self: *Generator, reason: SpecDisableReason, solo_mtp_tick: bool, forced_len: usize) Facts {
    const qwen4_head = self.mtp != null and self.mtp.? == .qwen4 and self.mtp_cache != null and self.mtp_cache.? == .qwen4;
    return .{
        .reason = reason,
        .solo_mtp_tick = solo_mtp_tick,
        .qwen4_head = qwen4_head,
        .spec_off = self.spec_disabled_runtime,
        .serial_block = self.mtp_serial_left > 0 or self.mtp_serial_exit != .none,
        .pipelined = self.has_pending_token or self.has_pending_logits,
        .has_last_hidden = self.has_last_hidden,
        .head_may_resume = self.mtpAdaptiveHeadMayResume(),
        .grouped = self.mtp_batch_head or self.mtp_group_cap > 0,
        .planner_owned = group_planner.enabled() and self.mtp_planner_owned,
        .other_spec = self.pld_enabled or self.drafter != null or self.dflash != null or self.dspark_enabled,
        .unsupported = specDecodeUnsupported(self.sampling, self.logprobs_n),
        .ctx_within = mtp_mod.mtpCtxWithinLimit(generate_mod.max_mtp_ctx, @as(usize, self.mtpKvLen()) + forced_len),
        .head_drift = if (qwen4_head) headDrift(self) else null,
    };
}

/// The drift (`mtpHeadPositionDrift`) the forced rows' head append would hit at the settled length. A read: it only
/// installs this request's head state on the module, as every head read does.
fn headDrift(self: *const Generator) ?i64 {
    const r = switch (self.mtp_cache orelse return null) {
        .qwen4 => |q| q,
        else => return null,
    };
    r.t.qwen4MtpActivate(r.st);
    const m = &(r.t.qwen4_mtp orelse return null);
    const at = settledLen(
        m.seq_offset,
        if (self.mtp_pre_draft) |*pd| pd.off0 else null,
        if (self.mtp_hist_stash) |st| st.off0 + st.n else null,
    );
    return mtpHeadPositionDrift(@intCast(m.pos_base), at, @intCast(at + 1));
}

/// Settle the head's committed history for the forced rows: a pre-drafted chain is dropped with every head row past the
/// committed end, and a pending history stash is applied (`mtpApplyStash`). Returns the committed length.
fn settleHead(self: *Generator, allocator: std.mem.Allocator) !usize {
    if (self.mtp_pre_draft) |*pd| {
        try self.mtp_cache.?.truncate(self.mtpCommittedHistoryLen(), self.xfm.s);
        pd.deinit(allocator);
        self.mtp_pre_draft = null;
    }
    try self.mtpApplyStash();
    return self.mtp_cache.?.step();
}

/// Commit `forced` from the spec exit of a solo qwen4 MTP generator (`tbMtpWants` said yes) so that speculation
/// continues past it. The forced run is a verify row whose drafts are `forced[0..n-1]` and whose bonus is
/// `forced[n-1]`, fully accepted, committed as `mtpRoundFinish` commits one: one trunk forward over
/// `[t1, forced[0..n-1]]` with every row's stream captured, the head given the rows a full accept's stash holds (ids
/// `[t1, forced[0..n-1]]`, hiddens `last_hidden ++ stream[0..n-1]`, at the committed end, as a history-only head
/// forward), `last_hidden` = the last row's stream, and `forced[n-1]` decided but not in the cache: `nextMtp`'s entry
/// state, the head one row short of the trunk as after any round. Publishes what `commitForcedTokens` publishes from a
/// spec exit, in its order; `forced[n-1]` is the next round's first token.
pub fn commitForcedTokensMtp(self: *Generator, allocator: std.mem.Allocator, forced: []const u32, reason: SpecDisableReason) !ForcedCommit {
    std.debug.assert(forced.len > 0);
    std.debug.assert(!self.has_pending_token and !self.has_pending_logits);
    const n = forced.len;
    // A spec exit owes no n-gram advance; settle one before the history moves (a no-op otherwise).
    try self.xfm.flushDeferredPle(&self.ctx);
    const want_from = settledLen(
        self.mtp_cache.?.step(),
        if (self.mtp_pre_draft) |*pd| pd.off0 else null,
        if (self.mtp_hist_stash) |st| st.off0 + st.n else null,
    );
    const head_from = try settleHead(self, allocator);
    if (head_from != want_from) return error.TbMtpHeadSettle;
    const trunk_from = self.ctx.moe_seq_offset.*;

    // `[t1, forced[0..n-1]]`: the trunk's input and the head's ids.
    const ids = try allocator.alloc(u32, n);
    defer allocator.free(ids);
    ids[0] = self.next_token_id;
    @memcpy(ids[1..], forced[0 .. n - 1]);
    const ids_i32 = try allocator.alloc(i32, n);
    defer allocator.free(ids_i32);
    for (ids, ids_i32) |t, *v| v.* = @intCast(t);

    // What `commitForcedTokens` publishes from a spec exit, in its order.
    var emitted = std.ArrayList(u32).empty;
    errdefer emitted.deinit(allocator);
    try self.generated_ids.append(allocator, ids[0]);
    try emitted.append(allocator, ids[0]);
    self.advanceStep(1);
    for (forced[0 .. n - 1]) |f| {
        try self.generated_ids.append(allocator, f);
        try emitted.append(allocator, f);
    }
    self.advanceStep(@intCast(n - 1));
    self.consecutive_pad = 0;
    self.loop_guard_start = self.generated_ids.items.len;

    const tok_input = mlx.mlx_array_new_data(ids_i32.ptr, &[_]c_int{ 1, @intCast(n) }, 2, .int32);
    defer _ = mlx.mlx_array_free(tok_input);
    var last = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(last);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    const logits = try self.xfm.forwardWithCaptureAll(&self.ctx, tok_input, &last, &all);
    // `forced[n-1]` is decided: nothing samples these rows.
    _ = mlx.mlx_array_free(logits);

    // The committed pairs `(t1, last_hidden)`, `(forced[i], stream[i])` for i < n-1: a full accept's stash.
    var hist_hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hist_hidden);
    if (n == 1) {
        try mlx.check(mlx.mlx_array_set(&hist_hidden, self.last_hidden));
    } else {
        try self.mtpPaddedHidden(&hist_hidden, n - 1, all);
    }
    const hist_ids = mlx.mlx_array_new_data(ids_i32.ptr, &[_]c_int{@intCast(n)}, 1, .int32);
    defer _ = mlx.mlx_array_free(hist_ids);
    // A committed-history append: the mixer and the vocab projection are dead work.
    const out = try self.mtp.?.forward(self.xfm, &self.mtp_cache.?, hist_ids, hist_hidden, @intCast(head_from), .none, self.mtpMropeContext(), ids);
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
    if (out.hidden_next.ctx != null) _ = mlx.mlx_array_free(out.hidden_next);
    if (out.rerank_x.ctx != null) _ = mlx.mlx_array_free(out.rerank_x);

    // `nextMtp`'s entry state.
    if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
    self.last_hidden = last;
    self.has_last_hidden = true;
    last = .{ .ctx = null };
    self.next_token_id = forced[n - 1];
    self.mtp_hidden_stale = false;
    // A serial-accept round's rule: the next draw is the token after the pending one.
    if (self.mtp_serial_accept) self.sampling.draw = self.generated_ids.items.len + 1;
    // The forced tick is no round: the next round's wall is no width sample.
    self.invalidateRoundClock();
    self.mtp_gap_watch = null;
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, self.last_hidden);
        self.mtp_cache.?.appendEvalArrays(vec);
        _ = mlx.mlx_async_eval(vec);
    }
    self.tb_mtp.commits += 1;
    const gap = headGap(self.mtp_position_base, head_from, trunk_from);
    log.info("[mtp] forced close committed through the head (n={d}, reason={s}, head rows {d}->{d}, trunk {d}->{d}, head_gap={d})\n", .{ n, @tagName(reason), head_from, head_from + n, trunk_from, self.ctx.moe_seq_offset.*, gap });
    return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = false };
}

// ── tests ──

const testing = std.testing;

/// The facts of a solo qwen4 MTP generator at its spec exit, in step with its head.
fn testFacts() Facts {
    return .{
        .reason = .think_bound,
        .solo_mtp_tick = true,
        .qwen4_head = true,
        .spec_off = false,
        .serial_block = false,
        .pipelined = false,
        .has_last_hidden = true,
        .head_may_resume = true,
        .grouped = false,
        .planner_owned = false,
        .other_spec = false,
        .unsupported = false,
        .ctx_within = true,
        .head_drift = 0,
    };
}

test "mtp forced close: it goes through the head only from a solo qwen4 MTP spec exit, every other state keeps the plain path" {
    const ok = testFacts();
    try testing.expectEqual(@as(?Decline, null), decide(ok));
    var f = ok;
    f.reason = .tool_choice;
    try testing.expectEqual(@as(?Decline, null), decide(f));
    for ([_]SpecDisableReason{ .none, .acceptance, .max_ctx, .adaptive, .company }) |r| {
        f = ok;
        f.reason = r;
        try testing.expectEqual(@as(?Decline, .reason), decide(f));
    }
    const Case = struct { field: []const u8, want: Decline };
    inline for ([_]Case{
        .{ .field = "solo_mtp_tick", .want = .not_solo_mtp },
        .{ .field = "qwen4_head", .want = .no_qwen4_head },
        .{ .field = "has_last_hidden", .want = .no_hidden },
        .{ .field = "head_may_resume", .want = .head_cannot_resume },
        .{ .field = "ctx_within", .want = .max_ctx },
    }) |c| {
        f = ok;
        @field(f, c.field) = false;
        try testing.expectEqual(@as(?Decline, c.want), decide(f));
    }
    inline for ([_]Case{
        .{ .field = "spec_off", .want = .spec_off },
        .{ .field = "serial_block", .want = .serial_block },
        .{ .field = "pipelined", .want = .pipelined },
        .{ .field = "grouped", .want = .grouped },
        .{ .field = "planner_owned", .want = .planner },
        .{ .field = "other_spec", .want = .other_spec },
        .{ .field = "unsupported", .want = .unsupported },
    }) |c| {
        f = ok;
        @field(f, c.field) = true;
        try testing.expectEqual(@as(?Decline, c.want), decide(f));
    }
    f = ok;
    f.head_drift = null;
    try testing.expectEqual(@as(?Decline, .no_qwen4_head), decide(f));
    for ([_]i64{ 3, -1 }) |d| {
        f = ok;
        f.head_drift = d;
        try testing.expectEqual(@as(?Decline, .head_drift), decide(f));
    }
    // The first failing check names the decline.
    f = ok;
    f.spec_off = true;
    f.pipelined = true;
    f.head_drift = 7;
    try testing.expectEqual(@as(?Decline, .spec_off), decide(f));
    for (std.enums.values(Decline)) |d| try testing.expect(declineText(d).len > 0);
}

test "mtp forced close: the generator's own state feeds the policy, and a decline is said once" {
    const head_ref = @import("../mtp_head_ref.zig");
    const Qwen4Trunk = std.meta.Child(@FieldType(head_ref.MtpHeadRef, "qwen4"));
    const Qwen4MtpState = std.meta.Child(@FieldType(head_ref.MtpCacheRef.Qwen4Ref, "st"));
    const prev_planner = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = prev_planner;
    const prev_ctx = generate_mod.max_mtp_ctx;
    defer generate_mod.max_mtp_ctx = prev_ctx;
    // A qwen4 head ref with no module head behind it: the drift reads null and nothing reaches a GPU.
    var t: Qwen4Trunk = undefined;
    t.formats = .{};
    t.qwen4_mtp = null;
    var st: Qwen4MtpState = undefined;
    var gen: Generator = undefined;
    gen.mtp = .{ .qwen4 = &t };
    gen.mtp_cache = .{ .qwen4 = .{ .t = &t, .st = &st, .allocator = testing.allocator } };
    gen.mtp_pre_draft = null;
    gen.mtp_hist_stash = null;
    gen.spec_disabled_runtime = false;
    gen.mtp_serial_left = 0;
    gen.mtp_serial_exit = .none;
    gen.has_pending_token = false;
    gen.has_pending_logits = false;
    gen.has_last_hidden = true;
    gen.mtp_adaptive = .{};
    gen.ctx.mrope_pos = null;
    gen.mtp_batch_head = false;
    gen.mtp_group_cap = 0;
    gen.mtp_planner_owned = false;
    gen.pld_enabled = false;
    gen.drafter = null;
    gen.dflash = null;
    gen.dspark_enabled = false;
    gen.sampling = .{};
    gen.logprobs_n = 0;
    gen.prompt_ids_owned = &.{};
    gen.generated_ids = .empty;
    gen.tb_mtp = .{};
    generate_mod.max_mtp_ctx = 0;
    const f = tbMtpFacts(&gen, .think_bound, true, 24);
    try testing.expect(f.solo_mtp_tick and f.qwen4_head and !f.spec_off and !f.serial_block and !f.pipelined);
    try testing.expect(f.has_last_hidden and f.head_may_resume and !f.grouped and !f.planner_owned);
    try testing.expect(!f.other_spec and !f.unsupported and f.ctx_within and f.head_drift == null);
    try testing.expectEqual(@as(?Decline, .no_qwen4_head), decide(f));
    const Probe = struct {
        fn decline(g: *Generator) ?Decline {
            return decide(tbMtpFacts(g, .think_bound, true, 24));
        }
    };
    gen.spec_disabled_runtime = true;
    try testing.expectEqual(@as(?Decline, .spec_off), Probe.decline(&gen));
    gen.spec_disabled_runtime = false;
    gen.mtp_serial_exit = .drain;
    try testing.expectEqual(@as(?Decline, .serial_block), Probe.decline(&gen));
    gen.mtp_serial_exit = .none;
    gen.has_pending_logits = true;
    try testing.expectEqual(@as(?Decline, .pipelined), Probe.decline(&gen));
    gen.has_pending_logits = false;
    gen.has_last_hidden = false;
    try testing.expectEqual(@as(?Decline, .no_hidden), Probe.decline(&gen));
    gen.has_last_hidden = true;
    gen.mtp_adaptive.sticky_serial = true;
    try testing.expectEqual(@as(?Decline, .head_cannot_resume), Probe.decline(&gen));
    gen.mtp_adaptive.sticky_serial = false;
    gen.mtp_group_cap = 2;
    try testing.expectEqual(@as(?Decline, .grouped), Probe.decline(&gen));
    gen.mtp_group_cap = 0;
    gen.mtp_planner_owned = true;
    try testing.expectEqual(@as(?Decline, .planner), Probe.decline(&gen));
    gen.mtp_planner_owned = false;
    gen.pld_enabled = true;
    try testing.expectEqual(@as(?Decline, .other_spec), Probe.decline(&gen));
    gen.pld_enabled = false;
    gen.logprobs_n = 3;
    try testing.expectEqual(@as(?Decline, .unsupported), Probe.decline(&gen));
    gen.logprobs_n = 0;
    generate_mod.max_mtp_ctx = 10;
    try testing.expectEqual(@as(?Decline, .max_ctx), Probe.decline(&gen));
    generate_mod.max_mtp_ctx = 0;
    // A decline keeps the plain path and says why once per request.
    try testing.expect(!tbMtpWants(&gen, .think_bound, false, 24));
    try testing.expect(gen.tb_mtp.decline_logged);
    try testing.expect(!tbMtpWants(&gen, .tool_choice, true, 1));
    try testing.expect(gen.tb_mtp.decline_logged and gen.tb_mtp.commits == 0);
}

test "mtp forced close: the head settles to the stash end, else the pre-draft origin, else where it stands" {
    // A pending stash: its origin plus its rows, whatever stale draft tail the head holds past the origin.
    try testing.expectEqual(@as(usize, 34), settledLen(37, null, 34));
    // A pre-drafted chain: its rows past its origin go; a chain that appended none leaves the head.
    try testing.expectEqual(@as(usize, 34), settledLen(40, 34, null));
    try testing.expectEqual(mtpCommittedLen(40, 34, null), settledLen(40, 34, null));
    try testing.expectEqual(@as(usize, 40), settledLen(40, 40, null));
    // Nothing pending (the first tick after prefill): the head as it stands.
    try testing.expectEqual(@as(usize, 99), settledLen(99, null, null));
    // The head gap: 0 when the next row pairs the trunk's last cached position.
    try testing.expectEqual(@as(i64, 0), headGap(0, 99, 100));
    try testing.expectEqual(@as(i64, 0), headGap(17, 82, 100));
    try testing.expectEqual(@as(i64, 9), headGap(0, 90, 100));
    try testing.expectEqual(@as(i64, -1), headGap(0, 100, 100));
}
