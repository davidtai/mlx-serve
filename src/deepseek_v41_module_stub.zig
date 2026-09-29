//! Stand-in for `deepseek_v41_module.zig` on graphs built without its macOS sources (Linux, iOS:
//! `build_options.macos_engines` false), as `arch/ds4_stub.zig`: the arch refuses to load, by name.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_io = @import("model.zig");

pub const Module = struct {
    pub fn init(_: std.mem.Allocator, _: std.Io, _: *const model_io.ModelConfig, _: *model_io.Weights, _: mlx.mlx_stream) !*Module {
        return error.Dsv41NotBuiltForThisTarget;
    }
    pub fn deinit(_: *Module) void {}
    pub fn prefill(_: *Module, _: []const u32) !mlx.mlx_array {
        return error.Dsv41NotBuiltForThisTarget;
    }
    pub fn extend(_: *Module, _: []const u32) !mlx.mlx_array {
        return error.Dsv41NotBuiltForThisTarget;
    }
};
