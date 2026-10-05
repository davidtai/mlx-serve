//! The plugin SDK (docs/plugins.md): the one module a plugin imports. The small SDK: what a module-owned arch needs
//! from the host (G1 module-owned decode state, G2 the phase change, G3 spec decode, G4 phase bills, the process
//! claim, the weight loader and the memory ledgers) as optional declarations of the kinds below. A seam only one
//! plugin consumes (an expert source, a kernel registry, a quant contract) stays in that plugin until a second
//! consumer exists. The registry (src/plugins.zig) builds each kind's table once, at compile time, and the host
//! resolves a model's tables once at load: a hook runs per request, step or round, never per layer.
//!
//! `KVCache`, `ForwardCtx` and `Linear` (an arch over the host's cache) land with their first consumer: the
//! archs registered so far own their decode state (G1).

const std = @import("std");

/// The SDK's version. A plugin built against another major is refused at compile time; a newer minor on either
/// side is compatible (newer hooks are optional).
pub const api: Version = .{ .major = 1, .minor = 0 };

/// The MLX this binary links (lib/mlx-src 64ea011cb: v0.32.3). One MLX per process: a plugin tested on another is
/// refused at compile time, so an MLX bump is one change that moves this pin and every plugin's.
pub const mlx_pin = "v0.32.3";
/// Compile the registered plugins' profile probes in (`-Dplugin-profile=true`); off in every served build.
pub const plugin_profile: bool = @import("sdk_build").plugin_profile;

pub const mlx = @import("mlx");
pub const log = @import("log");
pub const io_util = @import("io_util");

const plugin = @import("sdk/plugin.zig");
pub const Version = plugin.Version;
pub const Plugin = plugin.Plugin;
pub const Provides = plugin.Provides;
pub const Host = plugin.Host;
pub const NegotiationError = plugin.NegotiationError;
pub const negotiate = plugin.negotiate;
/// What this host checks every plugin against.
pub const host: Host = .{ .api = api, .mlx = mlx_pin };

const peek = @import("sdk/peek.zig");
pub const Priority = peek.Priority;
pub const Diag = peek.Diag;
pub const ConfigPeek = peek.ConfigPeek;
pub const GroupPeek = peek.GroupPeek;
pub const LayerPeek = peek.LayerPeek;
pub const Segment = peek.Segment;

const arch = @import("sdk/arch.zig");
pub const Arch = arch.Arch;
pub const ArchInstance = arch.ArchInstance;
pub const Caps = arch.Caps;
pub const Shell = arch.Shell;
pub const LoadFacts = arch.LoadFacts;
pub const LoadCtx = arch.LoadCtx;
pub const RequestShape = arch.RequestShape;
pub const DecodeHandover = arch.DecodeHandover;

const spec = @import("sdk/spec.zig");
pub const Spec = spec.Spec;
pub const DraftLane = spec.DraftLane;
pub const ArmRequest = spec.ArmRequest;
pub const DraftArm = spec.DraftArm;
pub const DraftRound = spec.DraftRound;
pub const DraftStats = spec.DraftStats;

const memory_bill = @import("sdk/memory_bill.zig");
pub const MemoryBill = memory_bill.MemoryBill;
pub const BillRequest = memory_bill.BillRequest;
pub const Rows = memory_bill.Rows;
pub const fill = memory_bill.fill;
pub const admit = memory_bill.admit;
pub const checkMeasured = memory_bill.checkMeasured;
pub const checkRows = memory_bill.checkRows;
pub const checkConstruction = memory_bill.checkConstruction;

pub const lifecycle = @import("sdk/lifecycle.zig");
pub const PhaseObserver = lifecycle.PhaseObserver;

const kinds = @import("sdk/kinds.zig");
pub const Source = kinds.Source;
pub const Engine = kinds.Engine;

/// Process and box memory readings (the kernel's ledgers) that bills and construction checks compare against.
pub const memory = @import("sdk/memory.zig");
const weights = @import("sdk/weights.zig");
pub const Weights = weights.Weights;
pub const LoadOpts = weights.LoadOpts;
pub const WeightLoader = weights.WeightLoader;
pub const QuantMode = @import("sdk/quant_mode.zig").QuantMode;

/// The comptime interface checks the kinds run on a plugin's namespaces; a plugin's own contracts reuse them.
pub const check = @import("sdk/check.zig");

/// The MTP acceptance modes the host serves (`Mode`, `DEFAULT_TYPICAL_DELTA`, `typicalThreshold`).
pub const acceptance = @import("mtp_acceptance");

/// Conformance (docs/plugins.md): every check declares its lane.
pub const testing = @import("sdk/testing.zig");

test {
    std.testing.refAllDecls(@This());
    _ = plugin;
    _ = peek;
    _ = arch;
    _ = spec;
    _ = memory_bill;
    _ = lifecycle;
    _ = kinds;
    _ = testing;
}
