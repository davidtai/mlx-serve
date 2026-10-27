//! The built-in quant formats' tables, in `block.builtins` order: the index `block.claimant` returns names the table
//! here. Discovery reads `block.zig` (MLX-free); the loader reads this list, and the claimant weighs the installed
//! outside plugins' tables (`outside.zig`) against it.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const common = @import("../../abi/common.zig");
const block = @import("block.zig");
const outside = @import("outside.zig");
const native = @import("native.zig");
const hadamard = @import("hadamard/format.zig");
const exl3 = @import("exl3/format.zig");
const gguf = @import("gguf/format.zig");

pub const tables = [_]*const abi.QuantTable{ &native.affine, &native.fp8, &native.mxfp8, &native.nvfp4, &hadamard.table, &exl3.table, &gguf.table };

/// The table serving `block_json` of `config_json`, a built-in or an installed outside plugin's; null when nobody claims
/// it.
pub fn claimant(block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    return claimantAmong(outside.tables.get().list, block_json, config_json);
}

/// `claimant` with the outside tables `extra`, by the loader's rule for packs (`loader.outsideWinner`): an outside table
/// named as a built-in replaces it; otherwise one claiming above every built-in wins, a built-in wins a tie, and outside
/// tables tied at the top leave the block to the built-ins.
pub fn claimantAmong(extra: []const *const abi.QuantTable, block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    var got: [tables.len]abi.Claim = undefined;
    var top: u8 = 0;
    for (block.builtins, &got) |b, *c| {
        c.* = if (replaced(b.name, extra)) .no else b.claims(.of(block_json), .of(config_json));
        top = @max(top, @intFromEnum(c.*));
    }
    if (extra.len > 0 and extra.len <= max_outside) {
        var ranks: [max_outside]abi.Claim = undefined;
        for (extra, ranks[0..extra.len]) |t, *c| c.* = t.claims(.of(block_json), .of(config_json));
        if (common.pick(ranks[0..extra.len]) catch null) |i| {
            if (@intFromEnum(ranks[i]) > top) return extra[i];
        }
    }
    const i = try common.pick(&got) orelse return null;
    return tables[i];
}

/// The outside tables weighed at most (the loader's cap).
const max_outside = 64;

fn replaced(name: []const u8, extra: []const *const abi.QuantTable) bool {
    for (extra) |t| if (std.mem.eql(u8, t.name.slice(), name)) return true;
    return false;
}

// ── tests ──

const testing = std.testing;
const sdk = @import("../root.zig");
const conformance = @import("../testing/conformance.zig");
const fixture = @import("../testing/fixture.zig");

test "quant formats: the tables follow block.builtins, each with its own claims and validation" {
    try testing.expectEqual(block.builtins.len, tables.len);
    for (block.builtins, tables) |b, t| {
        try testing.expectEqualStrings(b.name, t.name.slice());
        try testing.expect(b.claims == t.claims and b.validate == t.validate);
    }
    try testing.expectEqual(&native.affine, (try claimant("{\"bits\":4}", "{}")).?);
    try testing.expectEqual(@as(?*const abi.QuantTable, null), try claimant("{\"quant_method\":\"bitsandbytes\"}", "{}"));
}

fn claimsAtMarker(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
    return if (std.mem.indexOf(u8, block_json.slice(), "\"exl3\"") != null) .by_marker else .no;
}

fn claimsAtType(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
    return if (std.mem.indexOf(u8, block_json.slice(), "\"exl3\"") != null) .by_type else .no;
}

/// An outside plugin's table as the claimant reads it: its name and its claim.
fn outsideTable(comptime name: []const u8, claims: block.ClaimsFn) abi.QuantTable {
    return .{ .name = abi.Str.of(name), .claims = claims, .validate = undefined, .bill = undefined, .open = undefined, .close = undefined, .load_tensor = undefined, .load_bank = undefined, .release = undefined, .matmul = undefined, .moe = undefined };
}

test "quant formats: an outside table takes a block it claims above the built-ins or by a built-in's name; a tie stays built-in" {
    const bank = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul1\"}";
    const marker = outsideTable("fmt_marker", &claimsAtMarker);
    const marker2 = outsideTable("fmt_marker2", &claimsAtMarker);
    const typed = outsideTable("fmt_typed", &claimsAtType);
    const named = outsideTable("exl3", &claimsAtType);
    try testing.expectEqual(&exl3.table, (try claimantAmong(&.{}, bank, "{}")).?);
    try testing.expectEqual(&marker, (try claimantAmong(&.{&marker}, bank, "{}")).?);
    try testing.expectEqual(&exl3.table, (try claimantAmong(&.{&typed}, bank, "{}")).?);
    try testing.expectEqual(&named, (try claimantAmong(&.{&named}, bank, "{}")).?);
    try testing.expectEqual(&exl3.table, (try claimantAmong(&.{ &marker, &marker2 }, bank, "{}")).?);
    try testing.expectEqual(&native.affine, (try claimantAmong(&.{&marker}, "{\"bits\":4}", "{}")).?);
    // A dense trunk block (no bank rate) has no built-in claimant: the outside table's, else nobody's.
    const trunk = "{\"format\":\"exl3\",\"codebook\":\"mul1\",\"window\":16}";
    try testing.expectEqual(@as(?*const abi.QuantTable, null), try claimantAmong(&.{}, trunk, "{}"));
    try testing.expectEqual(&typed, (try claimantAmong(&.{&typed}, trunk, "{}")).?);
    try testing.expectEqual(@as(?*const abi.QuantTable, null), try claimant(trunk, "{}"));
}

/// A built-in format as the conformance suite reads a quant plugin.
fn Builtin(comptime i: usize, comptime block_json: []const u8, comptime config_json: []const u8) type {
    return struct {
        pub const plugin = sdk.Plugin{ .name = block.builtins[i].name, .kind = .quant };
        pub const table = tables[i].*;
        pub fn conformanceFixture(_: std.mem.Allocator) !fixture.Fixture {
            return .{ .files = &.{.{ .name = "config.json", .bytes = config_json }}, .block = block_json };
        }
    };
}

test "quant formats: every built-in format passes the quant plugin conformance" {
    try conformance.expectPlugin(Builtin(0, "{\"bits\":4,\"group_size\":64}", "{}"));
    try conformance.expectPlugin(Builtin(1, "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", "{}"));
    try conformance.expectPlugin(Builtin(2, "{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\"}", "{}"));
    try conformance.expectPlugin(Builtin(3, "{\"bits\":4,\"group_size\":16,\"mode\":\"nvfp4\"}", "{}"));
    try conformance.expectPlugin(Builtin(4, "{\"bits\":2,\"group_size\":128}", "{\"modules\":[{\"path\":\"lm_head\",\"block\":1024}]}"));
    try conformance.expectPlugin(Builtin(5, "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", "{}"));
    try conformance.expectPlugin(Builtin(6, "{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}", "{}"));
}

test {
    _ = block;
    _ = native;
    _ = hadamard;
    _ = exl3;
    _ = gguf;
    _ = @import("../weights/resident.zig");
    _ = @import("../weights/streamed.zig");
    _ = @import("fp8_block128.zig");
}
