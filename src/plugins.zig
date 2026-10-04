//! The plugin registry (docs/plugins.md): one line per plugin, negotiated at compile time, routed by `claims`.
//! The host asks it once per model (discovery, load) and keeps the tables it resolves: nothing here runs per
//! request, token or layer.

const std = @import("std");
const sdk = @import("sdk");
const build_options = @import("build_options");

/// One line per plugin: `@import("<its module>").plugin`. Each plugin is its own repo, a pinned submodule under lib/
/// (`-D<name>-dir` builds against a checkout instead); a plugin left out (`-Dmlx-stream=false`) leaves none of its
/// files in the build: the host reaches a plugin through this table only.
pub const all = if (registers_mlx_stream) [_]sdk.Plugin{
    @import("mlx_stream").plugin,
} else [_]sdk.Plugin{};
/// Whether this build registers mlx-stream (lib/mlx-stream, `-Dmlx-stream`, default on; the unit-test graph follows it).
pub const registers_mlx_stream = if (@hasDecl(build_options, "plugin_mlx_stream")) build_options.plugin_mlx_stream else true;

/// The registered mlx-stream plugin's test surface (its root's `testing`), null in a build that leaves the plugin out:
/// the host's tests reach the plugin through the registry only, so such a build analyzes none of its files.
pub const mlx_stream_testing: ?type = if (registers_mlx_stream) @import("mlx_stream").testing else null;

/// The model types a known plugin serves, and that plugin: a build that leaves the plugin out refuses such a model by
/// name at config parse (`unservedPlugin`) instead of reading it as another arch.
pub const known = [_]struct { model_type: []const u8, plugin: []const u8 }{
    .{ .model_type = "deepseek_v41", .plugin = "mlx-stream" },
};

/// The plugin that serves `model_type` when this build does not register it; null when it does, or when no known
/// plugin serves that model type.
pub fn unservedPlugin(model_type: []const u8) ?[]const u8 {
    return unservedIn(registry, model_type);
}

fn unservedIn(comptime R: type, model_type: []const u8) ?[]const u8 {
    for (known) |k| {
        if (std.mem.eql(u8, k.model_type, model_type) and !R.registered(k.plugin)) return k.plugin;
    }
    return null;
}

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
        pub const archs = tableOf(sdk.Arch, "arch", plugins, platform);
        pub const engines = tableOf(sdk.Engine, "engine", plugins, platform);

        /// Whether two archs can tie on a model in this build (two or more registered). Only then does the host read
        /// the model's `plugin` setting (model-settings.json) to break the tie: a build with one arch never reads it.
        pub const arch_ties_possible = archs.len > 1;

        /// Whether `name` is a plugin this build registers.
        pub fn registered(name: []const u8) bool {
            inline for (plugins) |p| {
                if (comptime registers(p, platform)) {
                    if (std.mem.eql(u8, p.name, name)) return true;
                }
            }
            return false;
        }

        /// What served a model, as `/v1/models` and `/props` carry it: `,"plugins":[...]`, the arch's object naming its
        /// plugin. Built at compile time; "" for a model no registered arch serves (an in-tree arch, an embedded engine), which leaves those
        /// bodies as they were.
        pub fn servedJson(a: ?*const sdk.Arch) []const u8 {
            const want = a orelse return "";
            for (&archs, &arch_served) |*e, j| {
                if (&e.kind == want) return j;
            }
            return "";
        }
        const arch_served = servedOf(plugins, platform);

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

/// Each registered arch's `plugins` fragment, in its table's order.
fn servedOf(comptime plugins: []const sdk.Plugin, comptime platform: Platform) [countOf("arch", plugins, platform)][]const u8 {
    comptime {
        var out: [countOf("arch", plugins, platform)][]const u8 = undefined;
        var i: usize = 0;
        for (plugins) |p| {
            if (!registers(p, platform)) continue;
            const A = p.provides.arch orelse continue;
            out[i] = ",\"plugins\":[" ++ servedItem(p.name, "arch", A.name) ++ "]";
            i += 1;
        }
        const final = out;
        return final;
    }
}

fn servedItem(comptime plugin: []const u8, comptime kind: []const u8, comptime name: []const u8) []const u8 {
    return "{\"plugin\":\"" ++ jsonSafe(plugin) ++ "\",\"kind\":\"" ++ kind ++ "\",\"name\":\"" ++ jsonSafe(name) ++ "\"}";
}

/// A name spliced into JSON verbatim: printable ASCII without a quote or a backslash, else a compile error.
fn jsonSafe(comptime s: []const u8) []const u8 {
    for (s) |c| {
        if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') @compileError("plugin name not JSON-safe: " ++ s);
    }
    return s;
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
    try testing.expectEqual(@as(usize, 0), R.engines.len);
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
    try testing.expect(R.arch(&peek, null) == null and R.source(&peek, null) == null and R.engine(&peek, null) == null);
}

test "plugins registry: a model only a plugin serves is refused by that plugin's name on a build without it" {
    const none = Registry(&.{}, .{ .macos = true });
    try testing.expectEqualStrings("mlx-stream", unservedIn(none, "deepseek_v41").?);
    try testing.expect(unservedIn(none, "qwen3") == null);
    // This build registers mlx-stream exactly when -Dmlx-stream is on.
    try testing.expectEqual(!registers_mlx_stream, unservedPlugin("deepseek_v41") != null);
}

test "plugins registry: the winner of a claims round" {
    const names = [_][]const u8{ "a", "b", "c", "d" };
    try testing.expectEqual(@as(?usize, null), pick(&names, &.{ null, null, null, null }, null));
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .native, null }, null));
    try testing.expectEqual(@as(?usize, 2), pick(&names, &.{ .generic, .native, .native, null }, "c"));
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .generic, null }, "c"));
    try testing.expectEqual(@as(?usize, 3), pick(&names, &.{ .generic, null, .generic, .native }, "a"));
}


const fake_d: sdk.Plugin = .{ .name = "fake-d", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .arch = FakeArch(.{ .model_type = "fake_d" }) } };

test "plugins registry: what served a model names its arch with its plugin; nothing for a model no arch serves" {
    const R = Registry(&.{ fake_a, fake_d }, .{ .macos = true });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = R.arch(&try peekOf(a, "{\"model_type\":\"fake_d\"}"), null).?;
    try testing.expectEqualStrings(",\"plugins\":[{\"plugin\":\"fake-d\",\"kind\":\"arch\",\"name\":\"fake-arch\"}]", R.servedJson(&d.kind));
    const f = R.arch(&try peekOf(a, "{\"model_type\":\"fake_arch\"}"), null).?;
    try testing.expectEqualStrings(",\"plugins\":[{\"plugin\":\"fake-a\",\"kind\":\"arch\",\"name\":\"fake-arch\"}]", R.servedJson(&f.kind));
    try testing.expectEqualStrings("", R.servedJson(null));
    // An arch table that is not this registry's entry (an in-tree arch's own) is served by no plugin.
    const loose = comptime sdk.Arch.of(FakeArch(.{}));
    try testing.expectEqualStrings("", R.servedJson(&loose));
    // Two archs can tie here, so the host reads the model's plugin setting; a one-arch registry never does.
    try testing.expect(R.arch_ties_possible and R.registered("fake-d") and !R.registered("mlx-stream"));
    try testing.expect(!Registry(&.{ fake_a, fake_c }, .{ .macos = true }).arch_ties_possible);
}

fn FakeKind(comptime name_: []const u8, comptime model_type: []const u8, comptime priority: sdk.Priority) type {
    return struct {
        pub const name = name_;
        pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
            const t = p.modelType() orelse return null;
            return if (std.mem.eql(u8, t, model_type)) priority else null;
        }
    };
}

const kind_a: sdk.Plugin = .{ .name = "kind-a", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .engine = FakeKind("engine-a", "gguf_x", .generic) } };
const kind_b: sdk.Plugin = .{ .name = "kind-b", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .engine = FakeKind("engine-b", "gguf_x", .generic), .source = FakeKind("source-b", "gguf_x", .generic) } };
const kind_n: sdk.Plugin = .{ .name = "kind-n", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .source = FakeKind("source-n", "own_arch", .native), .arch = FakeArch(.{ .model_type = "own_arch" }) } };

test "plugins registry: engines and sources route like archs; a macOS-only plugin is not registered elsewhere" {
    const mac_b: sdk.Plugin = .{ .name = kind_b.name, .api = kind_b.api, .mlx = kind_b.mlx, .macos_only = true, .provides = kind_b.provides };
    const R = Registry(&.{ kind_a, mac_b, kind_n }, .{ .macos = true });
    const Linux = Registry(&.{ kind_a, mac_b, kind_n }, .{ .macos = false });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gguf = try peekOf(arena.allocator(), "{\"model_type\":\"gguf_x\"}");
    try testing.expectEqualStrings("engine-a", R.engine(&gguf, null).?.kind.name);
    try testing.expectEqualStrings("engine-b", R.engine(&gguf, "kind-b").?.kind.name);
    try testing.expectEqualStrings("engine-a", Linux.engine(&gguf, "kind-b").?.kind.name);
    try testing.expectEqualStrings("source-b", R.source(&gguf, null).?.kind.name);
    try testing.expect(Linux.source(&gguf, null) == null);
    try testing.expectEqualStrings("source-n", R.source(&try peekOf(arena.allocator(), "{\"model_type\":\"own_arch\"}"), "kind-b").?.kind.name);
    try testing.expect(R.registered("kind-b") and !Linux.registered("kind-b") and Linux.registered("kind-a"));
    try testing.expect(!R.registered("") and !R.registered("kind") and !R.registered("kind-a "));
    try testing.expect(Linux.sources.len == 1 and Linux.engines.len == 1 and Linux.archs.len == 1);
}

test "plugins registry: what served a model names the arch's plugin; ties are possible only where two archs register" {
    const R = Registry(&.{ kind_a, kind_n }, .{ .macos = true });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const e = R.arch(&try peekOf(arena.allocator(), "{\"model_type\":\"own_arch\"}"), null).?;
    try testing.expectEqualStrings(",\"plugins\":[{\"plugin\":\"kind-n\",\"kind\":\"arch\",\"name\":\"fake-arch\"}]", R.servedJson(&e.kind));
    // the only arch: no tie is possible, whatever the other kinds register
    try testing.expect(!R.arch_ties_possible);
    // a second arch that registers only on macOS makes ties possible there and nowhere else
    const mac_arch: sdk.Plugin = .{ .name = "mac-arch", .api = sdk.api, .mlx = sdk.mlx_pin, .macos_only = true, .provides = .{ .arch = FakeArch(.{ .model_type = "own_arch" }) } };
    try testing.expect(Registry(&.{ kind_n, mac_arch }, .{ .macos = true }).arch_ties_possible);
    try testing.expect(!Registry(&.{ kind_n, mac_arch }, .{ .macos = false }).arch_ties_possible);
}

test "plugins registry: the claims round's edge cases (one entry, a preference for a lower claim, an unknown or repeated preference)" {
    const names = [_][]const u8{ "a", "b", "c" };
    try testing.expectEqual(@as(?usize, 0), pick(names[0..1], &.{.generic}, null));
    try testing.expectEqual(@as(?usize, null), pick(&.{}, &.{}, "a"));
    try testing.expectEqual(@as(?usize, 2), pick(&names, &.{ null, null, .generic }, null));
    // the preference breaks ties only: it never lifts a lower claim
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .generic }, "a"));
    try testing.expectEqual(@as(?usize, 1), pick(&names, &.{ .generic, .native, .generic }, "c"));
    // an unknown preference keeps registry order; one naming the first of the tied keeps it
    try testing.expectEqual(@as(?usize, 0), pick(&names, &.{ .native, .native, .native }, "zz"));
    try testing.expectEqual(@as(?usize, 0), pick(&names, &.{ .native, .native, .native }, "a"));
    try testing.expectEqual(@as(?usize, 2), pick(&names, &.{ .native, .native, .native }, "c"));
    // a preferred entry that is outclaimed later loses to the higher claim
    try testing.expectEqual(@as(?usize, 2), pick(&names, &.{ .generic, .generic, .native }, "b"));
    // two entries with the preferred name (two quants of one plugin): the first of them wins the tie
    try testing.expectEqual(@as(?usize, 1), pick(&.{ "x", "p", "p" }, &.{ .generic, .generic, .generic }, "p"));
}

test "plugins registry: model-settings.json's plugin breaks a tie between two archs; an unregistered or empty name changes nothing" {
    const model_settings = @import("model_settings.zig");
    const twin_a: sdk.Plugin = .{ .name = "twin-a", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .arch = FakeArch(.{ .model_type = "twin" }) } };
    const twin_b: sdk.Plugin = .{ .name = "twin-b", .api = sdk.api, .mlx = sdk.mlx_pin, .provides = .{ .arch = FakeArch(.{ .model_type = "twin" }) } };
    const R = Registry(&.{ twin_a, twin_b }, .{ .macos = true });
    try testing.expect(R.arch_ties_possible);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const peek = try peekOf(a, "{\"model_type\":\"twin\"}");
    const Case = struct { entry: []const u8, want: []const u8 };
    for ([_]Case{
        .{ .entry = "{\"plugin\":\"twin-b\"}", .want = "twin-b" },
        .{ .entry = "{\"plugin\":\"twin-a\"}", .want = "twin-a" },
        .{ .entry = "{\"plugin\":\"mlx-stream-gone\"}", .want = "twin-a" },
        .{ .entry = "{\"plugin\":\"\"}", .want = "twin-a" },
        .{ .entry = "{\"plugin\":7}", .want = "twin-a" },
        .{ .entry = "{}", .want = "twin-a" },
    }) |c| {
        const entry = try std.json.parseFromSliceLeaky(std.json.Value, a, c.entry, .{});
        try testing.expectEqualStrings(c.want, R.arch(&peek, model_settings.pluginOf(entry)).?.plugin);
    }
}

test "plugins registry: this build's registry follows -Dmlx-stream (the plugin's tables, its tests and its name together)" {
    try testing.expectEqual(registers_mlx_stream, all.len == 1);
    try testing.expectEqual(registers_mlx_stream, mlx_stream_testing != null);
    try testing.expectEqual(registers_mlx_stream and build_options.macos_engines, registry.registered("mlx-stream"));
    const n: usize = if (registers_mlx_stream and build_options.macos_engines) 1 else 0;
    try testing.expectEqual(n, registry.archs.len);
    try testing.expect(registry.sources.len == 0 and registry.engines.len == 0 and !registry.arch_ties_possible);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v41 = try peekOf(arena.allocator(), "{\"model_type\":\"deepseek_v41\"}");
    try testing.expectEqual(n == 1, registry.arch(&v41, null) != null);
    inline for (all) |p| try sdk.negotiate(p, sdk.host);
}

test "plugins conformance: the host's weight map a plugin binds owns its keys and handles; replace and drop touch only present names" {
    // empty handles: freeing one is a no-op in mlx-c, so the map's ownership runs without an array
    var w = sdk.Weights.init(testing.allocator);
    defer w.deinit();
    try w.map.put(try testing.allocator.dupe(u8, "a.weight"), .{});
    try w.map.put(try testing.allocator.dupe(u8, "b.weight"), .{});
    try testing.expectEqual(@as(u32, 2), w.count());
    try testing.expect(w.get("a.weight") != null and w.get("c.weight") == null);
    w.replace("c.weight", .{});
    try testing.expectEqual(@as(u32, 2), w.count());
    w.replace("a.weight", .{});
    w.drop("c.weight");
    w.drop("b.weight");
    try testing.expect(w.count() == 1 and w.get("b.weight") == null);
}


test "plugins conformance: the host's MLX pin is the MLX this binary links, and every registered plugin's matches it" {
    var v = sdk.mlx.mlx_string_new();
    defer _ = sdk.mlx.mlx_string_free(v);
    try testing.expectEqual(@as(c_int, 0), sdk.mlx.mlx_version(&v));
    const linked = std.mem.span(sdk.mlx.mlx_string_data(v));
    try testing.expectEqualStrings(linked, sdk.mlx_pin[1..]);
    inline for (all) |p| try testing.expectEqualStrings(sdk.mlx_pin, p.mlx);
}

// Declared last so it runs after every other conformance test (the CPU lane's bar).
test "plugins conformance: the CPU lane created no Metal device" {
    // Only the conformance run (`zig build conformance`: the registry's tests alone) is the CPU lane. The unit-test
    // binary reaches this file too, after device tests in the same process.
    for (@import("builtin").test_functions) |t| if (std.mem.indexOf(u8, t.name, "plugins ") == null) return error.SkipZigTest;
    try sdk.testing.expectNoDevice();
}
