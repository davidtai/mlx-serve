//! The DSV4.1 prompt pass's routed-call split, for PROFILE builds only (`-Ddsv41-prefill-timers=true`): host
//! time in the wide lane's steps and the DIG-X dispatcher's waves, accumulated per process. In every
//! other build `enabled` is false and each call below compiles to nothing (the timed path carries no
//! timer, branch or counter).

const std = @import("std");
const bo = @import("build_options");

pub const enabled: bool = if (@hasDecl(bo, "dsv41_prefill_timers")) bo.dsv41_prefill_timers else false;

/// Where the routed call's host time goes: the routing barrier (the ids to the host), the stream's route
/// (reads issued), the read waits, the waves' encode (graphs, host tables, submission), the drains (the
/// host blocked on GPU compute), the join (the outputs' concatenate / take).
pub const Bucket = enum { barrier, route, read_wait, encode, drain, join };
const n_buckets = @typeInfo(Bucket).@"enum".field_names.len;

pub var ns: [n_buckets]u64 = @splat(0);
pub var waves: u64 = 0;
pub var calls: u64 = 0;
pub var launches: u64 = 0;

pub const Stamp = if (enabled) u64 else void;

pub inline fn now() Stamp {
    if (comptime !enabled) return {};
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Charge the time since `t0` to `b`.
pub inline fn charge(b: Bucket, t0: Stamp) void {
    if (comptime !enabled) return;
    ns[@intFromEnum(b)] += now() - t0;
}

pub inline fn count(w: u64, l: u64) void {
    if (comptime !enabled) return;
    waves += w;
    launches += l;
}

pub inline fn countCall() void {
    if (comptime !enabled) return;
    calls += 1;
}

pub fn reset() void {
    ns = @splat(0);
    waves = 0;
    calls = 0;
    launches = 0;
}

pub fn seconds(b: Bucket) f64 {
    return @as(f64, @floatFromInt(ns[@intFromEnum(b)])) / 1e9;
}
