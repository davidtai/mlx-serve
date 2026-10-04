//! The registry's compile-time refusals (docs/plugins.md, negotiation): `zig build conformance` compiles this root
//! once per case (`refusal_case.name`), each with one bad plugin line, and expects the compile error that names it.
//! Never imported by the host.

const std = @import("std");
const sdk = @import("sdk");
const plugins = @import("plugins.zig");
const case = @import("refusal_case").name;
const FakeArch = sdk.testing.FakeArch;

fn is(comptime name: []const u8) bool {
    return std.mem.eql(u8, case, name);
}

const Claims = struct {
    pub const name = "fixture-kind";
    pub fn claims(_: *const sdk.ConfigPeek) ?sdk.Priority {
        return null;
    }
};
const NoClaims = struct {
    pub const name = "fixture-kind";
};
const WrongClaims = struct {
    pub const name = "fixture-kind";
    pub fn claims(_: *const sdk.GroupPeek) ?sdk.Priority {
        return null;
    }
};
const ClaimsNotFn = struct {
    pub const name = "fixture-kind";
    pub const claims: u8 = 5;
};
const ClaimsTwoParams = struct {
    pub const name = "fixture-kind";
    pub fn claims(_: *const sdk.ConfigPeek, _: u8) ?sdk.Priority {
        return null;
    }
};
const ClaimsReturnsBool = struct {
    pub const name = "fixture-kind";
    pub fn claims(_: *const sdk.ConfigPeek) bool {
        return false;
    }
};
const Nameless = struct {
    pub const claims = Claims.claims;
};

/// An arch that claims a process resource without the release.
const ClaimOnly = struct {
    const F = FakeArch(.{});
    pub const name = F.name;
    pub const caps = F.caps;
    pub const claims = F.claims;
    pub const Config = F.Config;
    pub const Module = F.Module;
    pub const parse = F.parse;
    pub const freeConfig = F.freeConfig;
    pub const shell = F.shell;
    pub const applySettings = F.applySettings;
    pub const loadBytes = F.loadBytes;
    pub const init = F.init;
    pub const deinit = F.deinit;
    pub const prefill = F.prefill;
    pub const step = F.step;
    pub const position = F.position;
    pub fn claimProcess() !void {}
};

fn line(comptime name: []const u8, comptime provides: sdk.Provides) sdk.Plugin {
    return .{ .name = name, .api = sdk.api, .mlx = sdk.mlx_pin, .provides = provides };
}

const list: []const sdk.Plugin = if (is("api_major"))
    &.{.{ .name = "bad-api", .api = .{ .major = sdk.api.major + 1, .minor = 0 }, .mlx = sdk.mlx_pin, .provides = .{} }}
else if (is("mlx_pin"))
    &.{.{ .name = "bad-mlx", .api = sdk.api, .mlx = "v0.0.1", .provides = .{} }}
else if (is("mlx_pin_macos_only"))
    &.{.{ .name = "mac-pin", .api = sdk.api, .mlx = "v0.0.1", .macos_only = true, .provides = .{} }}
else if (is("duplicate"))
    &.{ line("twin", .{ .source = Claims }), line("other", .{}), line("twin", .{}) }
else if (is("source_no_claims"))
    &.{line("p", .{ .source = NoClaims })}
else if (is("engine_wrong_claims"))
    &.{line("p", .{ .engine = WrongClaims })}
else if (is("arch_batches_owned_state"))
    &.{line("p", .{ .arch = FakeArch(.{ .caps = .{ .owns_decode_state = true, .batches_decode = true } }) })}
else if (is("source_claims_not_fn"))
    &.{line("p", .{ .source = ClaimsNotFn })}
else if (is("engine_claims_param_count"))
    &.{line("p", .{ .engine = ClaimsTwoParams })}
else if (is("source_claims_returns"))
    &.{line("p", .{ .source = ClaimsReturnsBool })}
else if (is("source_no_name"))
    &.{line("p", .{ .source = Nameless })}
else if (is("arch_claim_unpaired"))
    &.{line("p", .{ .arch = ClaimOnly })}
else if (is("name_not_json_safe"))
    &.{line("quo\"te", .{ .arch = FakeArch(.{}) })}
else
    @compileError("unknown refusal case " ++ case);

/// The macOS-only case registers on a graph without the macOS-only sources: negotiation still refuses it there.
const R = plugins.Registry(list, .{ .macos = !is("mlx_pin_macos_only") });

/// Every table and the served fragments, so each kind's `of` runs.
export fn pluginsRefusalProbe() usize {
    return R.sources.len + R.archs.len + R.engines.len + R.servedJson(null).len;
}
