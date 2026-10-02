//! The loaded weights a plugin binds (the host's weight map, one concrete type: no view, no indirection per lookup) and
//! the host's loaders it reaches through `LoadCtx.loader` (called once per file at load).

const std = @import("std");
const mlx = @import("mlx");

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }

    /// Forget `name`, freeing the map's handle (arrays built from it keep what they read).
    pub fn drop(self: *Weights, name: []const u8) void {
        if (self.map.fetchRemove(name)) |kv| {
            _ = mlx.mlx_array_free(kv.value);
            self.allocator.free(kv.key);
        }
    }
};

/// How a load treats stored dtypes. `keep_f16`: a pack whose activation dtype
/// is f16 (Prism Hadamard packs) keeps its f16 side tensors and tables as
/// stored; narrowing them to bf16 drops 3 mantissa bits of every group scale.
/// `nocache`: read the shards past the page cache (`nocache_reader`): the
/// load keeps no file pages next to the array buffers.
pub const LoadOpts = struct { vision: bool = false, keep_f16: bool = false, nocache: bool = false };

/// The host's safetensors loaders (`model.zig`'s): a model directory's shards, and one file (a sidecar the index does
/// not name) into an existing map.
pub const WeightLoader = struct {
    dir: *const fn (io: std.Io, gpa: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) anyerror!Weights,
    file: *const fn (gpa: std.mem.Allocator, weights: *Weights, path: [*:0]const u8, s: mlx.mlx_stream, opts: LoadOpts) anyerror!void,
};
