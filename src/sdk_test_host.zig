//! The host files a plugin's test build reaches beside `sdk` (the plugin's harnesses and bank tests): the config parse
//! through the registry, the loaders, the static memory ceiling and the transformer's constants. Imported only by a
//! plugin's test module, as `mlx_serve_host` (build.zig `addMlxStreamTests`); served code reaches the host through
//! `sdk` alone. Its module imports the plugin under test as `mlx_stream`, so the registry here registers that build.

comptime {
    if (!@import("builtin").is_test) @compileError("sdk_test_host.zig is a plugin test build's bridge to the host");
}

pub const model = @import("model.zig");
pub const gpu_ceiling = @import("gpu_ceiling.zig");
pub const transformer = @import("transformer.zig");
pub const plugins = @import("plugins.zig");
