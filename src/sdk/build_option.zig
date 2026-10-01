//! G7: a build option a plugin declares (docs/plugins.md). build.zig imports the declarations directly (std only,
//! no SDK import), registers each as `-D<name>` and hands its value to the sources as `build_options.<field>`.

/// A boolean build option. Profile code is compiled in when set and out of every served build (`default` false).
pub const BuildOption = struct {
    /// The `-D` flag.
    name: []const u8,
    /// The `build_options` field the sources read.
    field: []const u8,
    description: []const u8,
    default: bool = false,
};
