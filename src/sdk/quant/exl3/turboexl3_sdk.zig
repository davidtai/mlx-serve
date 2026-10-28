//! turboexl3's `sdk` (its README and CONSUMING.md) over mlx-serve's own: the root's MLX bindings, quant ABI and `Shared`
//! (turboexl3's module globals are the host's under `-Dunits`: `src/shared_all.zig` registers them), read through
//! `mlx_host` so no file is in two modules, turboexl3's own launcher (`turboexl3_kit`) and MLX's steel NAX headers. A
//! module of its own (build/libs.zig `addTurboExl3`): nothing imports this file by path.

pub const mlx = @import("mlx_host").mlx;
pub const quant_abi = @import("mlx_host").quant_abi;
pub const shared = @import("mlx_host").shared;
pub const kernel_launch = @import("turboexl3_kit").kernel_launch;
pub const steel = @import("mlx_steel_sources");
