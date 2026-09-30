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
    pub fn prefill(_: *Module, _: []const u32, _: u64) !mlx.mlx_array {
        return error.Dsv41NotBuiltForThisTarget;
    }
    pub fn extend(_: *Module, _: []const u32) !mlx.mlx_array {
        return error.Dsv41NotBuiltForThisTarget;
    }
    pub const DsparkRound = struct {
        tokens: []u32,
        accepted: u32,
        next_token: u32,
        pub fn deinit(self: *DsparkRound, a: std.mem.Allocator) void {
            a.free(self.tokens);
        }
    };
    pub fn draftBlockSize(_: *const Module) u32 {
        return 0;
    }
    pub fn decodeLane(_: *const Module) []const u8 {
        return "serial";
    }
    pub fn position(_: *const Module) u64 {
        return 0;
    }
    pub fn dsparkRound(_: *Module, _: std.mem.Allocator, _: u32, _: u32) !DsparkRound {
        return error.Dsv41NotBuiltForThisTarget;
    }
};
