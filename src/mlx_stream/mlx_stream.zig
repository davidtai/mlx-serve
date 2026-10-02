//! mlx-stream: the native streaming stack as one plugin: the DeepSeek-V4.1 arch, the EXL3 quant it binds (the routed
//! experts' kernels over the package's pinned registry) and the EXL3 expert source that streams them. Built only where
//! the macOS-only sources are.

const sdk = @import("sdk");

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 1, .minor = 0 },
    .mlx = "v0.32.2",
    .macos_only = true,
    .provides = .{
        .arch = @import("deepseek_v41_plugin.zig"),
        .quant = @import("exl3_quant.zig"),
        .expert_source = @import("exl3_source.zig"),
    },
};
