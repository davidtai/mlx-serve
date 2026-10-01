//! mlx-stream: the native streaming stack as one plugin. The DeepSeek-V4.1 arch today; its EXL3 quant and its
//! expert source join as their kinds land. Built only where the macOS-only sources are.

const sdk = @import("sdk");

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 1, .minor = 0 },
    .mlx = "v0.32.2",
    .macos_only = true,
    .provides = .{ .arch = @import("deepseek_v41_plugin.zig") },
};
