//! mlx-stream's import boundary, checked on every build (build.zig runs this file as a host tool over src/mlx_stream/): a
//! package file imports `sdk`, the shared named modules and its own files (its directory's); host files only through the
//! test-only bridge.

const std = @import("std");


/// Named modules any package file may import: the SDK, the modules it shares with the host, and the build's options
/// (the plugin's profile flags ride `build_options`, mlx_stream_options.zig).
pub const named = [_][]const u8{ "std", "builtin", "sdk", "mlx", "log", "io_util", "ngram", "build_options" };

/// The only path imports of a non-package file, each by the one package file allowed to make it.
pub const Allowed = struct { file: []const u8, import: []const u8 };
pub const allowed = [_]Allowed{
    // The harness and test bridge: it refuses to compile outside a test build.
    .{ .file = "deepseek_v41_host.zig", .import = "../model.zig" },
    .{ .file = "deepseek_v41_host.zig", .import = "../gpu_ceiling.zig" },
    .{ .file = "deepseek_v41_host.zig", .import = "../transformer.zig" },
    // The profile build's guard reads the root's declarations (profile builds only).
    .{ .file = "dsv41_profile.zig", .import = "root" },
    // build.zig's view of the plugin's build options (std only).
    .{ .file = "mlx_stream_options.zig", .import = "../sdk/build_option.zig" },
};

/// A file of the package's directory: every .zig file there, and an import of a sibling (no path separator).
pub fn isPackageFile(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".zig") and std.mem.indexOfScalar(u8, name, '/') == null;
}

/// Whether package file `file` may import `import` (an `@import` string).
pub fn allows(file: []const u8, import: []const u8) bool {
    for (named) |n| if (std.mem.eql(u8, n, import)) return true;
    if (isPackageFile(import)) return true;
    for (allowed) |a| if (std.mem.eql(u8, a.file, file) and std.mem.eql(u8, a.import, import)) return true;
    return false;
}

/// Calls `found(ctx, import)` for every `@import("...")` in `source` (comments and strings are not imports).
pub fn eachImport(source: [:0]const u8, ctx: anytype, comptime found: fn (@TypeOf(ctx), []const u8) void) void {
    var t = std.zig.Tokenizer.init(source);
    var prev2: std.zig.Token.Tag = .eof;
    var prev1: std.zig.Token.Tag = .eof;
    var builtin_import = false;
    while (true) {
        const tok = t.next();
        if (tok.tag == .eof) break;
        if (tok.tag == .string_literal and prev1 == .l_paren and prev2 == .builtin and builtin_import) {
            found(ctx, source[tok.loc.start + 1 .. tok.loc.end - 1]);
        }
        if (tok.tag == .builtin) builtin_import = std.mem.eql(u8, source[tok.loc.start..tok.loc.end], "@import");
        prev2 = prev1;
        prev1 = tok.tag;
    }
}

/// `mlx-stream-import-probe <src dir>`: exits 1 naming every package import outside the boundary.
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const a = init.arena.allocator();
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    _ = args.next();
    const src_path = args.next() orelse return error.MissingSrcDir;
    var dir = try std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true });
    defer dir.close(io);
    const Found = struct {
        file: []const u8,
        bad: *usize,
        dir: std.Io.Dir,
        io: std.Io,
        fn check(f: @This(), import: []const u8) void {
            // A sibling import must name a file of the package's directory (a host file is "../").
            const sibling_ok = !isPackageFile(import) or if (f.dir.access(f.io, import, .{})) |_| true else |_| false;
            if (allows(f.file, import) and sibling_ok) return;
            std.debug.print("mlx-stream import outside its boundary (src/mlx_stream/mlx_stream_imports.zig): src/mlx_stream/{s}: @import(\"{s}\")\n", .{ f.file, import });
            f.bad.* += 1;
        }
    };
    var bad: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !isPackageFile(entry.name)) continue;
        const file = try a.dupe(u8, entry.name);
        const source = try dir.readFileAllocOptions(io, file, a, .limited(64 << 20), .of(u8), 0);
        eachImport(source, Found{ .file = file, .bad = &bad, .dir = dir, .io = io }, Found.check);
    }
    if (bad > 0) std.process.exit(1);
}

const testing = std.testing;

test "plugins import probe: the package's own files, the SDK and the shared modules pass; a host file does not, but through the bridge" {
    try testing.expect(allows("deepseek_v41_module.zig", "sdk"));
    try testing.expect(allows("deepseek_v41_module.zig", "ngram"));
    try testing.expect(allows("deepseek_v41_module.zig", "expert_bank.zig"));
    try testing.expect(!allows("deepseek_v41_module.zig", "../model.zig"));
    try testing.expect(!allows("deepseek_v41_module.zig", "../status.zig"));
    try testing.expect(!allows("deepseek_v41_ar.zig", "../gpu_ceiling.zig"));
    try testing.expect(allows("deepseek_v41_host.zig", "../model.zig"));
    try testing.expect(!allows("deepseek_v41_host.zig", "../status.zig"));
    try testing.expect(!isPackageFile("../model.zig") and isPackageFile("exl3_quant.zig"));
    const src =
        \\const a = @import("sdk");
        \\// @import("../model.zig") in a comment
        \\const s = "@import(\"status.zig\")";
        \\test { _ = @import("qwen4_exp.zig"); }
    ;
    var got: std.ArrayList([]const u8) = .empty;
    defer got.deinit(testing.allocator);
    const Ctx = struct {
        list: *std.ArrayList([]const u8),
        fn add(c: @This(), name: []const u8) void {
            c.list.append(testing.allocator, name) catch unreachable;
        }
    };
    eachImport(src, Ctx{ .list = &got }, Ctx.add);
    try testing.expectEqual(@as(usize, 2), got.items.len);
    try testing.expectEqualStrings("sdk", got.items[0]);
    try testing.expectEqualStrings("qwen4_exp.zig", got.items[1]);
}
