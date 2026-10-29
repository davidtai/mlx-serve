//! A forced span (the reasoning budget's closer) through the served model's own block drafter
//! (`NativeDrafter.commit_forced`), against a test drafter over the llama plugin that never drafts: each round forwards
//! `t1` through the served trunk and hands its argmax back, and a forced span is one forward of `t1` and every forced
//! token but the last. From a round's exit the span is the drafter's, kept whole, and the last forced token is the next
//! round's `t1`; a drafter without the hook keeps the engine's own commit.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const llama = @import("../../../plugins/llama/plugin.zig");
const proof = @import("proof.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Generator = @import("../generate/generator.zig").Generator;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../transformer.zig").Transformer;

const testing = std.testing;

/// The served trunk the test drafter forwards on, the request's context, and what the drafter was handed.
var trunk: struct { x: *Transformer = undefined, ctx: *ForwardCtx = undefined } = .{};
var seen: struct { commits: u32 = 0, span: [8]u32 = undefined, n: usize = 0, at: u64 = 0, last_t1: u32 = 0 } = .{};

fn nStep(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    plainRound(f, out) catch return abi.Verdict.refuse(.failed, "test native round");
    return .ok;
}

fn nCommitForced(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    forcedSpan(f, out) catch return abi.Verdict.refuse(.failed, "test forced span");
    return .ok;
}

/// `t1` alone, verified by the trunk: no drafts, and the trunk's argmax after it is the next round's `t1`.
fn plainRound(f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const t1 = f.tokens.slice()[0];
    seen.last_t1 = t1;
    var g = ops.Graph.init(testing.allocator, trunk.x.s);
    defer g.deinit();
    const ids = [_]i32{@intCast(t1)};
    const logits = try g.adopt(try trunk.x.forwardWith(trunk.ctx, try fixture.tokens(&g, &ids)));
    const id = try g.argmax(logits, -1, false);
    try mlx.check(mlx.mlx_array_eval(id));
    var next: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&next, id));
    out.round = .{ .n = 1, .next_token = @intCast(next), .rows = 1, .position = f.positions.slice()[0] + 1 };
    out.round.tokens[0] = t1;
}

/// The span forwarded through the trunk and kept: the state moves past it.
fn forcedSpan(f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const span = f.tokens.slice();
    seen.commits += 1;
    seen.n = span.len;
    @memcpy(seen.span[0..span.len], span);
    seen.at = f.positions.slice()[0];
    var g = ops.Graph.init(testing.allocator, trunk.x.s);
    defer g.deinit();
    var ids: [8]i32 = undefined;
    for (span, ids[0..span.len]) |t, *v| v.* = @intCast(t);
    _ = try g.adopt(try trunk.x.forwardWith(trunk.ctx, try fixture.tokens(&g, ids[0..span.len])));
    out.round = .{ .position = f.positions.slice()[0] + span.len };
}

const with_hook: abi.NativeDrafter = .{ .step = &nStep, .commit_forced = &nCommitForced };
const without_hook: abi.NativeDrafter = .{ .step = &nStep };
const spec_with: abi.SpecHooks = .{ .native_drafter = &with_hook, .trunk_weight = llama.table.spec.?.trunk_weight };
const spec_without: abi.SpecHooks = .{ .native_drafter = &without_hook, .trunk_weight = llama.table.spec.?.trunk_weight };
const table_with: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &spec_with;
    break :blk t;
};
const table_without: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &spec_without;
    break :blk t;
};

const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
const forced = [_]u32{ 40, 41, 42 };

/// A greedy request on the model's own cache, two of the test drafter's rounds in, the closer committed.
fn closeAfterRounds(x: *Transformer, gen: *Generator, tok: *Tokenizer) !Generator.ForcedCommit {
    const a = testing.allocator;
    try x.resetCache();
    gen.* = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, x, tok, &prompt, 12, .{ .temperature = 0.0 }, &.{}, .{ .mtp_enabled = true });
    errdefer gen.deinit(a);
    try testing.expect(gen.dspark_enabled);
    trunk = .{ .x = x, .ctx = &gen.ctx };
    seen = .{};
    for (0..2) |_| {
        const res = (try gen.nextDspark(a)) orelse return error.TestUnexpectedResult;
        a.free(res.tokens);
    }
    return gen.commitForcedTokens(a, &forced);
}

// ── tests ──

test "plugin native drafter: a forced span commits through commit_forced, its last token the next round's t1" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &table_with;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    var gen: Generator = undefined;
    const r = try closeAfterRounds(&b.xfm, &gen, &tok);
    defer gen.deinit(a);
    defer a.free(r.emitted);
    // The span: the round's t1 and the closer but its last token, at the state's position, handed over once.
    const t1 = r.emitted[0];
    try testing.expectEqual(@as(u32, 1), seen.commits);
    try testing.expectEqualSlices(u32, &.{ t1, 40, 41 }, seen.span[0..seen.n]);
    try testing.expectEqual(@as(u64, prompt.len + 2), seen.at);
    try testing.expectEqualSlices(u32, &.{ t1, 40, 41 }, r.emitted);
    try testing.expect(!r.stopped);
    // A round's exit: the span in the state, nothing pending, the last forced token outside it.
    try testing.expect(!gen.has_pending_logits and !gen.has_pending_token);
    try testing.expectEqual(@as(u32, 42), gen.next_token_id);
    try testing.expectEqual(prompt.len + gen.generated_ids.items.len, gen.ctx.cache.step);
    try testing.expectEqual(gen.ctx.cache.step, gen.ctx.moe_seq_offset.*);
    // The next round drafts from it.
    const res = (try gen.nextDspark(a)) orelse return error.TestUnexpectedResult;
    defer a.free(res.tokens);
    try testing.expectEqual(@as(u32, 42), seen.last_t1);
    try testing.expectEqual(@as(u32, 42), res.tokens[0]);
}

test "plugin native drafter: without commit_forced a forced span is the engine's own commit" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x9a7e);
    defer b.close();
    b.xfm.plugin.?.table = &table_without;
    b.xfm.config.plugin.?.parsed.spec.native_stages = 1;
    const a = testing.allocator;
    var tok: Tokenizer = undefined; // never read by the Generator
    var gen: Generator = undefined;
    const r = try closeAfterRounds(&b.xfm, &gen, &tok);
    defer gen.deinit(a);
    defer a.free(r.emitted);
    try testing.expectEqual(@as(u32, 0), seen.commits);
    // The engine forwarded the whole span: its last token is in the state, its logits pending.
    try testing.expect(gen.has_pending_logits);
    try testing.expectEqual(@as(u32, 42), gen.next_token_id);
    try testing.expectEqual(prompt.len + gen.generated_ids.items.len + 1, gen.ctx.cache.step);
}
