//! Seams between the lanes merged on the integration branch, checked at
//! compile time: the model's graph backends carry the methods the kernels'
//! routes call (exl3_kernel_ops' backend contract), with its exact types.

const std = @import("std");
const ops = @import("deepseek_v41_ops.zig");
const xk = @import("exl3_kernels.zig");
const xo = @import("exl3_kernel_ops.zig");

/// `G.name` takes exactly `params` after `*G` (`*const G` when `self_const`)
/// and returns `Payload`, through an error union when `errors`.
fn hasMethod(comptime G: type, comptime name: []const u8, comptime self_const: bool, comptime params: []const type, comptime Payload: type, comptime errors: bool) bool {
    if (!@hasDecl(G, name)) return false;
    const info = @typeInfo(@TypeOf(@field(G, name)));
    if (info != .@"fn") return false;
    const f = info.@"fn";
    if (f.param_types.len != params.len + 1) return false;
    if ((f.param_types[0] orelse return false) != (if (self_const) *const G else *G)) return false;
    inline for (params, f.param_types[1..]) |want, got| if ((got orelse return false) != want) return false;
    const ret = f.return_type orelse return false;
    if (!errors) return ret == Payload;
    const r = @typeInfo(ret);
    return r == .error_union and r.error_union.payload == Payload;
}

/// The prefill wave route's four methods (kernels contract, phase 3b).
pub fn hasWaveMethods(comptime G: type) bool {
    const T = G.T;
    return hasMethod(G, "evalAll", false, &.{[]const T}, void, true) and
        hasMethod(G, "asyncEval", false, &.{[]const T}, void, true) and
        hasMethod(G, "concat", false, &.{ []const T, c_int }, T, true) and
        hasMethod(G, "take", false, &.{ T, T, c_int }, T, true);
}

/// Every route's launch (kernels contract, phase 3):
/// `launch(g: *G, k: Kernel, inputs: []const T, cfg: *const LaunchConfig, out: []T) !void`.
pub fn hasLaunch(comptime G: type) bool {
    const T = G.T;
    return hasMethod(G, "launch", false, &.{ xk.Kernel, []const T, *const xk.LaunchConfig, []T }, void, true);
}

/// The prefill wave lifecycle (`mark(g: *const G) Mark`, `resetTo(g: *G, m: Mark) void`: MlxOps keeps every op
/// output live until a reset, so a wave's intermediates are freed by truncating to its mark). The kernels'
/// DigXPrefill does not compile without them.
pub fn hasWaveLifecycle(comptime G: type) bool {
    return hasMethod(G, "mark", true, &.{}, ops.Mark, false) and hasMethod(G, "resetTo", false, &.{ops.Mark}, void, false);
}

/// The first method the binding's routes call that backend `G` lacks (or has with other types), or null:
/// the kernels' route methods (`exl3_kernel_ops.missingBackendMethod`), the typed launch, the wave methods,
/// the wave lifecycle and, on MLX, the launcher the accepted kernels are installed in.
pub fn missing(comptime G: type) ?[]const u8 {
    if (xo.missingBackendMethod(G)) |m| return m;
    if (!hasLaunch(G)) return "launch (the kernels contract's types)";
    const T = G.T;
    if (!hasMethod(G, "evalAll", false, &.{[]const T}, void, true)) return "evalAll";
    if (!hasMethod(G, "asyncEval", false, &.{[]const T}, void, true)) return "asyncEval";
    if (!hasMethod(G, "concat", false, &.{ []const T, c_int }, T, true)) return "concat";
    if (!hasMethod(G, "take", false, &.{ T, T, c_int }, T, true)) return "take";
    if (!hasMethod(G, "mark", true, &.{}, ops.Mark, false)) return "mark";
    if (!hasMethod(G, "resetTo", false, &.{ops.Mark}, void, false)) return "resetTo";
    if (G == ops.MlxOps and !(@hasField(G, "launcher") and @FieldType(G, "launcher") == ?*const xk.Bound)) return "launcher";
    return null;
}

test "dsv41 integration: the model's backends carry the kernel routes' methods with the contract's types" {
    comptime std.debug.assert(hasWaveMethods(ops.MlxOps));
    comptime std.debug.assert(hasWaveMethods(ops.TraceOps));
    comptime std.debug.assert(missing(ops.MlxOps) == null);
    comptime std.debug.assert(missing(ops.TraceOps) == null);
    // A signature change is caught: a method with other parameters does not count.
    const Wrong = struct {
        pub const T = u32;
        pub fn evalAll(_: *@This(), _: []T) !void {}
    };
    comptime std.debug.assert(!hasMethod(Wrong, "evalAll", false, &.{[]const u32}, void, true));
    std.debug.print("backend launch (every route): MlxOps {}, TraceOps {}; wave lifecycle mark / resetTo: MlxOps {}, TraceOps {}\n", .{
        comptime hasLaunch(ops.MlxOps), comptime hasLaunch(ops.TraceOps), comptime hasWaveLifecycle(ops.MlxOps), comptime hasWaveLifecycle(ops.TraceOps),
    });
}

test "dsv41 integration: a backend without one of the routes' methods is named" {
    try std.testing.expectEqualStrings("launch", comptime missing(struct {
        pub const T = u32;
    }).?);
    const Base = struct {
        pub const T = u32;
        pub fn launch(_: *@This(), _: xk.Kernel, _: []const T, _: *const xk.LaunchConfig, _: []T) !void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
        pub fn evalAll(_: *@This(), _: []const T) !void {}
        pub fn asyncEval(_: *@This(), _: []const T) !void {}
        pub fn concat(_: *@This(), _: []const T, _: c_int) !T {
            return 0;
        }
        pub fn take(_: *@This(), _: T, _: T, _: c_int) !T {
            return 0;
        }
    };
    // Every route method and the wave methods, but no wave lifecycle.
    try std.testing.expectEqualStrings("mark", comptime missing(Base).?);
    // A launch without the kernels' LaunchConfig.
    const OldLaunch = struct {
        pub const T = u32;
        pub fn launch(_: *@This(), _: xk.Kernel, _: []const T, _: []T) !void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    try std.testing.expectEqualStrings("launch (the kernels contract's types)", comptime missing(OldLaunch).?);
}
