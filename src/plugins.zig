//! The plugin registry (docs/plugins.md): one line per plugin, negotiated at compile time, routed by `claims`.
//! The host asks it once per model (discovery, load) and keeps the tables it resolves: nothing here runs per
//! request, token or layer.

const std = @import("std");
const sdk = @import("sdk");
const build_options = @import("build_options");

/// One line per plugin: `@import("<its root file>").plugin`. A plugin left out (`-Dmlx-stream=false`) leaves none
/// of its files in the build: the host reaches a plugin through this table only.
pub const all = if (registers_mlx_stream) [_]sdk.Plugin{
    @import("mlx_stream.zig").plugin,
} else [_]sdk.Plugin{};
const registers_mlx_stream = if (@hasDecl(build_options, "plugin_mlx_stream")) build_options.plugin_mlx_stream else true;

/// This build's registry. A macOS-only plugin registers nothing on graphs without the macOS-only sources.
pub const registry = Registry(&all, .{ .macos = build_options.macos_engines });

pub const Platform = struct { macos: bool };

pub fn Entry(comptime K: type) type {
    return struct { plugin: []const u8, kind: K };
}

/// The registry over `plugins`: each is negotiated (sdk.negotiate) and each provided kind checked by its `of`, at
/// compile time; a refusal names the plugin. One table per kind, in registry order.
pub fn Registry(comptime plugins: []const sdk.Plugin, comptime platform: Platform) type {
    comptime {
        for (plugins, 0..) |p, i| {
            sdk.negotiate(p, sdk.host) catch |e| @compileError(std.fmt.comptimePrint(
                "plugin {s}: {s} (built against SDK {d}.{d} on MLX {s}; this host is SDK {d}.{d} on MLX {s})",
                .{ p.name, @errorName(e), p.api.major, p.api.minor, p.mlx, sdk.api.major, sdk.api.minor, sdk.mlx_pin },
            ));
            for (plugins[0..i]) |q| {
                if (std.mem.eql(u8, p.name, q.name)) @compileError("plugin " ++ p.name ++ ": registered twice");
            }
        }
    }
    return struct {
        pub const sources = tableOf(sdk.Source, "source", plugins, platform);
        pub const quants = tableOf(sdk.Quant, "quant", plugins, platform);
        pub const expert_sources = tableOf(sdk.ExpertSource, "expert_source", plugins, platform);
        pub const archs = tableOf(sdk.Arch, "arch", plugins, platform);
        pub const engines = tableOf(sdk.Engine, "engine", plugins, platform);

        /// The arch that serves a model (discovery): the highest claim; among the tied, the plugin
        /// model-settings.json names (`prefer`), else the first in registry order. Null: no arch claims it.
        pub fn arch(peek: *const sdk.ConfigPeek, prefer: ?[]const u8) ?*const Entry(sdk.Arch) {
            return route(sdk.Arch, &archs, peek, prefer);
        }
        pub fn source(peek: *const sdk.ConfigPeek, prefer: ?[]const u8) ?*const Entry(sdk.Source) {
            return route(sdk.Source, &sources, peek, prefer);
        }
        pub fn engine(peek: *const sdk.ConfigPeek, prefer: ?[]const u8) ?*const Entry(sdk.Engine) {
            return route(sdk.Engine, &engines, peek, prefer);
        }
        pub fn expertSource(peek: *const sdk.ConfigPeek, prefer: ?[]const u8) ?*const Entry(sdk.ExpertSource) {
            return route(sdk.ExpertSource, &expert_sources, peek, prefer);
        }
        /// The quant that claims a weight group (load, once per group); `why` keeps the last decline.
        pub fn quant(group: *const sdk.GroupPeek, why: ?*sdk.Diag, prefer: ?[]const u8) ?*const Entry(sdk.Quant) {
            if (quants.len == 0) return null;
            var claim: [quants.len]?sdk.Priority = undefined;
            for (&quants, &claim) |*e, *c| c.* = e.kind.claims(group, why);
            const names = comptime namesOf(&quants);
            const i = pick(&names, &claim, prefer) orelse return null;
            return &quants[i];
        }
    };
}

fn registers(comptime p: sdk.Plugin, comptime platform: Platform) bool {
    return !p.macos_only or platform.macos;
}

fn tableOf(comptime K: type, comptime field: []const u8, comptime plugins: []const sdk.Plugin, comptime platform: Platform) [countOf(field, plugins, platform)]Entry(K) {
    var out: [countOf(field, plugins, platform)]Entry(K) = undefined;
    var i: usize = 0;
    inline for (plugins) |p| {
        if (comptime registers(p, platform)) {
            if (@field(p.provides, field)) |T| {
                out[i] = .{ .plugin = p.name, .kind = K.of(T) };
                i += 1;
            }
        }
    }
    return out;
}

fn countOf(comptime field: []const u8, comptime plugins: []const sdk.Plugin, comptime platform: Platform) usize {
    var n: usize = 0;
    inline for (plugins) |p| {
        if (registers(p, platform) and @field(p.provides, field) != null) n += 1;
    }
    return n;
}

fn namesOf(comptime table: anytype) [table.len][]const u8 {
    var out: [table.len][]const u8 = undefined;
    for (table, &out) |e, *n| n.* = e.plugin;
    return out;
}

fn route(comptime K: type, comptime table: anytype, peek: *const sdk.ConfigPeek, prefer: ?[]const u8) ?*const Entry(K) {
    if (table.len == 0) return null;
    var claim: [table.len]?sdk.Priority = undefined;
    for (table, &claim) |*e, *c| c.* = e.kind.claims(peek);
    const names = comptime namesOf(table);
    const i = pick(&names, &claim, prefer) orelse return null;
    return &table[i];
}

/// A claims round's winner: the highest priority; among the tied, the entry `prefer` names, else the first.
pub fn pick(names: []const []const u8, claim: []const ?sdk.Priority, prefer: ?[]const u8) ?usize {
    var best: ?usize = null;
    for (claim, 0..) |c, i| {
        const p = c orelse continue;
        const b = best orelse {
            best = i;
            continue;
        };
        const bp = claim[b].?;
        if (@backingInt(p) > @backingInt(bp)) {
            best = i;
        } else if (p == bp) {
            if (prefer) |want| {
                if (std.mem.eql(u8, names[i], want) and !std.mem.eql(u8, names[b], want)) best = i;
            }
        }
    }
    return best;
}

const testing = std.testing;
const FakeArch = sdk.testing.FakeArch;

const GenericSource = struct {
    pub const name = "generic-source";
    pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
        return if (p.modelType() != null) .generic else null;
    }
};
const NativeSource = struct {
    pub const name = "native-source";
    pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
        const t = p.modelType() orelse return null;
        return if (std.mem.eql(u8, t, "fake_arch")) .native else null;
    }
};

const fake_a: sdk.Plugin = .{ .name = "fake-a", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .arch = FakeArch(.{}), .source = GenericSource } };
const fake_b: sdk.Plugin = .{ .name = "fake-b", .api = .{ .major = sdk.api.major, .minor = 9 }, .mlx = sdk.mlx_pin, .macos_only = true, .provides = .{ .arch = FakeArch(.{ .model_type = "fake_b" }), .source = NativeSource } };
const fake_c: sdk.Plugin = .{ .name = "fake-c", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .source = GenericSource } };

fn peekOf(arena: std.mem.Allocator, config: []const u8) !sdk.ConfigPeek {
    return sdk.ConfigPeek.parse(arena, "/m", config);
}

test "plugins registry: one table per kind in registry order; a macOS-only plugin registers nothing elsewhere" {
    const R = Registry(&.{ fake_a, fake_b, fake_c }, .{ .macos = true });
    try testing.expectEqual(@as(usize, 2), R.archs.len);
    try testing.expectEqual(@as(usize, 3), R.sources.len);
    try testing.expectEqual(@as(usize, 0), R.quants.len + R.expert_sources.len + R.engines.len);
    try testing.expectEqualStrings("fake-b", R.archs[1].plugin);
    const Linux = Registry(&.{ fake_a, fake_b, fake_c }, .{ .macos = false });
    try testing.expectEqual(@as(usize, 1), Linux.archs.len);
    try testing.expectEqual(@as(usize, 2), Linux.sources.len);
}

test "plugins registry: claims route each model to the highest claim, ties in order unless preferred by name" {
    const R = Registry(&.{ fake_a, fake_b, fake_c }, .{ .macos = true });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("fake-a", R.arch(&try peekOf(a, "{\"model_type\":\"fake_arch\"}"), null).?.plugin);
    try testing.expectEqualStrings("fake-b", R.arch(&try peekOf(a, "{\"model_type\":\"fake_b\"}"), null).?.plugin);
    try testing.expect(R.arch(&try peekOf(a, "{\"model_type\":\"qwen3\"}"), null) == null);
    // native beats generic whatever the order or the preference
    try testing.expectEqualStrings("fake-b", R.source(&try peekOf(a, "{\"model_type\":\"fake_arch\"}"), "fake-c").?.plugin);
    // two generic claims tie: registry order, unless model-settings.json names one of them
    const llama = try peekOf(a, "{\"model_type\":\"llama\"}");
    try testing.expectEqualStrings("fake-a", R.source(&llama, null).?.plugin);
    try testing.expectEqualStrings("fake-c", R.source(&llama, "fake-c").?.plugin);
    try testing.expectEqualStrings("fake-a", R.source(&llama, "fake-b").?.plugin);
}

test "plugins registry: a registry without plugins claims nothing (the host built without its plugins)" {
    const R = Registry(&.{}, .{ .macos = true });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const peek = try peekOf(arena.allocator(), "{\"model_type\":\"deepseek_v41\"}");
    try testing.expect(R.arch(&peek, null) == null and R.source(&peek, null) == null and R.engine(&peek, null) == null and R.expertSource(&peek, null) == null);
    const group: sdk.GroupPeek = .{ .quantization = .null, .hidden = 0, .inter = 0, .n_experts = 0, .n_layers = 0, .layers = &.{} };
    try testing.expect(R.quant(&group, null, null) == null);
}

test "plugins registry: the winner of a claims round" {
    const names = [_][]const u8{ "a", "b", "c", "d" };
    try testing.expectEqual(@as(?usize, null), pick(&names, &.{ null, null, null, null }, null));
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .native, null }, null));
    try testing.expectEqual(@as(?usize, 2), pick(&names, &.{ .generic, .native, .native, null }, "c"));
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .generic, null }, "c"));
    try testing.expectEqual(@as(?usize, 3), pick(&names, &.{ .generic, null, .generic, .native }, "a"));
}

test "plugins conformance: every registered plugin's kinds decline what is not theirs" {
    const near_misses = [_]sdk.testing.ClaimCase{
        .{ .config = "{}", .want = null },
        .{ .config = "{\"model_type\":\"__no_such_model__\"}", .want = null },
    };
    inline for (registry.archs) |e| try sdk.testing.expectClaims(e.kind.claims, &near_misses);
    inline for (registry.expert_sources) |e| try sdk.testing.expectClaims(e.kind.claims, &near_misses);
    // a weight group no quant's format describes
    const group_near_misses = [_]sdk.testing.GroupClaimCase{
        .{ .quantization = "{}", .hidden = 5120, .inter = 2304, .want = null },
        .{ .quantization = "{\"mode\":\"__no_such_format__\",\"bits\":4,\"group_size\":32}", .hidden = 5120, .inter = 2304, .want = null },
    };
    inline for (registry.quants) |e| try sdk.testing.expectGroupClaims(e.kind.claims, &group_near_misses);
}

test "plugins conformance: mlx-stream registers its EXL3 quant, pinned by the kernel registry's manifest" {
    const R = Registry(&.{@import("mlx_stream.zig").plugin}, .{ .macos = true });
    try testing.expectEqual(@as(usize, 1), R.quants.len);
    try testing.expectEqualStrings("exl3-mul1-k3", R.quants[0].kind.name);
    try testing.expectEqualStrings(@import("exl3_kernels.zig").manifest_sha256, R.quants[0].kind.kernels.?.manifest_sha256);
}

test "plugins conformance: mlx-stream registers its EXL3 source, its capabilities and the one reader" {
    const R = Registry(&.{@import("mlx_stream.zig").plugin}, .{ .macos = true });
    try testing.expectEqual(@as(usize, 1), R.expert_sources.len);
    const k = R.expert_sources[0].kind;
    try testing.expectEqualStrings("exl3-stream", k.name);
    try testing.expect(k.caps.two_phase and k.caps.transient_release and k.caps.event_gates and !k.caps.construction_reset);
    try testing.expect(@import("expert_stream.zig").uses_reader);
}

// Declared last so it runs after every other conformance test (the CPU lane's bar).
test "plugins conformance: the CPU lane created no Metal device" {
    try sdk.testing.expectNoDevice();
}

// The import boundary's own test runs with the conformance suite ("plugins import probe").
comptime {
    if (@import("builtin").is_test) _ = @import("mlx_stream_imports.zig");
}
