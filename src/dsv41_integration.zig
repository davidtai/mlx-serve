//! Seams between the lanes merged on the integration branch, checked at
//! compile time: the model's graph backends carry the methods the kernels'
//! routes call (exl3_kernel_ops' backend contract).

const std = @import("std");
const ops = @import("deepseek_v41_ops.zig");

/// `G.name` exists with parameters `params` (after `*G`) and returns an error
/// union whose payload is `Payload`.
fn hasMethod(comptime G: type, comptime name: []const u8, comptime params: []const type, comptime Payload: type) bool {
    if (!@hasDecl(G, name)) return false;
    const f = @typeInfo(@TypeOf(@field(G, name))).@"fn";
    if (f.param_types.len != params.len + 1 or f.param_types[0] != *G) return false;
    for (params, f.param_types[1..]) |want, got| if (got != want) return false;
    const ret = @typeInfo(f.return_type.?);
    return ret == .error_union and ret.error_union.payload == Payload;
}

/// The prefill wave route's four methods (kernels contract, phase 3b).
pub fn hasWaveMethods(comptime G: type) bool {
    const T = G.T;
    return hasMethod(G, "evalAll", &.{[]const T}, void) and
        hasMethod(G, "asyncEval", &.{[]const T}, void) and
        hasMethod(G, "concat", &.{ []const T, c_int }, T) and
        hasMethod(G, "take", &.{ T, T, c_int }, T);
}

/// Every route's launch (kernels contract, phase 3): owed by the model lane.
pub fn hasLaunch(comptime G: type) bool {
    return @hasDecl(G, "launch");
}

/// The prefill wave lifecycle (`mark(g: *const G) Mark`, `resetTo(g, m) void`: MlxOps keeps every op output live
/// until a reset, so a wave's intermediates are freed by truncating to its mark). The kernels' DigXPrefill refuses to
/// compile without them; owed by the model lane.
pub fn hasWaveLifecycle(comptime G: type) bool {
    return @hasDecl(G, "mark") and @hasDecl(G, "resetTo");
}

test "dsv41 integration: the model's backends carry the kernel routes' four wave methods" {
    comptime std.debug.assert(hasWaveMethods(ops.MlxOps));
    comptime std.debug.assert(hasWaveMethods(ops.TraceOps));
    // A signature change is caught: a method with other parameters does not count.
    const Wrong = struct {
        pub const T = u32;
        pub fn evalAll(_: *@This(), _: []T) !void {}
    };
    comptime std.debug.assert(!hasMethod(Wrong, "evalAll", &.{[]const u32}, void));
    std.debug.print("backend launch (every route): MlxOps {}, TraceOps {}; wave lifecycle mark / resetTo: MlxOps {}, TraceOps {}\n", .{
        hasLaunch(ops.MlxOps), hasLaunch(ops.TraceOps), hasWaveLifecycle(ops.MlxOps), hasWaveLifecycle(ops.TraceOps),
    });
}
