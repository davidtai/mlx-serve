//! G2, the phase change's memory lifecycle: the order every module-owned arch keeps at its handover, the stages an
//! arch's handover supplies, and the harness-only observer. Every refusal is named and sticky; nothing here runs per
//! token, per layer or per cycle.

const std = @import("std");

/// The steps of a phase change, in the one order the memory proofs rely on.
pub const Step = enum {
    synchronize,
    /// harness only: the observer's mark before any free
    observe_start,
    /// the arch's frees (the embedding fence)
    fence,
    /// the transient release, when its route is installed
    release,
    cache_clear,
    /// decode's cache limit
    cache_limit,
    synchronize_freed,
    settle,
    check_freed,
    /// harness only: after the release, the clear and the boundary check, before the grow allocates
    observe_released,
    grow,
    /// harness only: after the grow and the grown banks' check
    observe_grown,
};

/// The contract's order; the optional steps may be absent, never moved.
pub const order = [_]struct { step: Step, optional: bool = false }{
    .{ .step = .synchronize },
    .{ .step = .observe_start, .optional = true },
    .{ .step = .fence },
    .{ .step = .release, .optional = true },
    .{ .step = .cache_clear },
    .{ .step = .cache_limit },
    .{ .step = .synchronize_freed },
    .{ .step = .settle },
    .{ .step = .check_freed },
    .{ .step = .observe_released, .optional = true },
    .{ .step = .grow },
    .{ .step = .observe_grown, .optional = true },
};

/// A recorded phase change against the contract's order: every required step once, in order, optional steps only
/// in their places. On a violation `at` names the step the contract expected.
pub fn checkOrder(steps: []const Step, at: *?Step) error{PhaseOrderViolated}!void {
    var i: usize = 0;
    for (order) |o| {
        if (i < steps.len and steps[i] == o.step) {
            i += 1;
        } else if (!o.optional) {
            at.* = o.step;
            return error.PhaseOrderViolated;
        }
    }
    if (i != steps.len) {
        at.* = steps[i];
        return error.PhaseOrderViolated;
    }
}

/// What an arch's release stage freed, for the host's boundary check.
pub const Freed = struct {
    /// device bytes the arch freed, the transient release's included
    device_bytes: u64,
    /// the transient release's share
    transient_bytes: u64,
    /// decode's MLX cache limit
    decode_cache_limit: usize,
};

/// A harness's observer of the phase change (the window's box proofs), set before the first request; the served
/// path passes none. A mark records only; it never refuses inside the phase change.
pub const PhaseObserver = struct {
    ctx: *anyopaque,
    mark: *const fn (ctx: *anyopaque, stage: Stage) anyerror!void,

    pub const Stage = enum { start, released, grown, tail };
};

const testing = std.testing;

test "sdk lifecycle: the phase change's order holds with or without the harness marks and the release route" {
    var at: ?Step = null;
    try checkOrder(&.{ .synchronize, .fence, .cache_clear, .cache_limit, .synchronize_freed, .settle, .check_freed, .grow }, &at);
    try checkOrder(&.{ .synchronize, .observe_start, .fence, .release, .cache_clear, .cache_limit, .synchronize_freed, .settle, .check_freed, .observe_released, .grow, .observe_grown }, &at);
    // A grow before the boundary check, a release after the clear and a second grow are refused, naming what was due.
    try testing.expectError(error.PhaseOrderViolated, checkOrder(&.{ .synchronize, .fence, .cache_clear, .cache_limit, .synchronize_freed, .grow, .settle, .check_freed }, &at));
    try testing.expectEqual(Step.settle, at.?);
    try testing.expectError(error.PhaseOrderViolated, checkOrder(&.{ .synchronize, .fence, .cache_clear, .release, .cache_limit, .synchronize_freed, .settle, .check_freed, .grow }, &at));
    try testing.expectEqual(Step.cache_limit, at.?);
    try testing.expectError(error.PhaseOrderViolated, checkOrder(&.{ .synchronize, .fence, .cache_clear, .cache_limit, .synchronize_freed, .settle, .check_freed, .grow, .grow }, &at));
}
