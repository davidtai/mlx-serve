//! The installed outside quant plugins' tables (dylibs from the plugin kit): the host's loader sets them once at
//! startup, before any load reads them (`host/plugins/loader.zig`), and `formats.claimant` weighs them against the
//! built-in formats. MLX-free, so the loader reaches it without the formats' kernels.

const abi = @import("../../abi/quant.zig");

/// The tables in a struct: `Shared` hands its storage out as `*anyopaque`, which a pointer to a slice does not coerce to.
pub const Tables = struct { list: []const *const abi.QuantTable = &.{} };

/// Process-wide under `-Dunits` (`sdk/util/shared.zig`); empty when no quant plugin is loaded.
pub const tables = @import("../util/shared.zig").Shared(Tables, "outside_quant_tables", .{});

/// The bytes the installed plugins' resident matrices are owed and do not hold yet (`resident_pending`), which the GPU
/// ceiling holds aside until they are kept.
pub fn residentPending() u64 {
    var n: u64 = 0;
    for (tables.get().list) |t| if (t.resident_pending) |f| {
        n +|= f();
    };
    return n;
}
