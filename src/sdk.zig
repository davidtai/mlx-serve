//! The plugin SDK (docs/plugins.md): the one module a plugin imports. One SDK: the hooks a native streaming arch
//! needs (G1 module-owned decode state, G2 the phase change, G3 spec decode, G4 phase bills, G5 the kernel
//! registry, G6 the expert source, G7 profile build options) are optional declarations of the kinds below. The
//! registry (src/plugins.zig) builds each kind's table once, at compile time, and the host resolves a model's
//! tables once at load: a hook runs per request, step or round, never per layer.
//!
//! `KVCache`, `ForwardCtx` and `Linear` (an arch over the host's cache) land with their first consumer: the
//! archs registered so far own their decode state (G1).

const std = @import("std");

/// The SDK's version. A plugin built against another major is refused at compile time; a newer minor on either
/// side is compatible (newer hooks are optional).
pub const api: Version = .{ .major = 1, .minor = 0 };

/// The MLX this binary links (lib/mlx-src d73eb752: v0.32.2 with the gather_qmm NAX fix). One MLX per process.
pub const mlx_pin = "v0.32.2";

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
pub const MtpHeadOps = spec.MtpHeadOps;
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
pub const Freed = lifecycle.Freed;

const kinds = @import("sdk/kinds.zig");
pub const Source = kinds.Source;
pub const Quant = kinds.Quant;
pub const ExpertSource = kinds.ExpertSource;
pub const Engine = kinds.Engine;

/// The expert_source kind's shared surface; the reader, the event gate, the policy and the source contract
/// join it with the kind's first registered consumer.
pub const expert = struct {
    pub const Caps = kinds.ExpertCaps;
};

pub const kernels = @import("sdk/kernels.zig");
/// The `quant` kind's contract (C2): the routed-expert quant a weight group's claim binds, and the generic
/// gather quant (`GatherQmm` through `FromGatherMatmul`).
pub const quant = @import("sdk/quant.zig");
/// G7: a plugin's profile probes, injected by its arch's backend type (`of(Backend)`); off everywhere else.
pub const profile = @import("sdk/profile.zig");
pub const BuildOption = @import("sdk/build_option.zig").BuildOption;
pub const QuantMode = @import("sdk/quant_mode.zig").QuantMode;

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
    _ = profile;
    _ = testing;
}
