//! turboexl3's `mlx_serve`, the plugin kit's root, which its launcher and quant table read (`abi.common`, `abi.host`,
//! `host.Host`, `host.digest`): mlx-serve's own kit files, through the root (`mlx_host`). A module of its own
//! (build/libs.zig `addTurboExl3`): nothing imports this file by path.

pub const abi = @import("mlx_host").kit.abi;
pub const host = @import("mlx_host").kit.host;
