//! The GPU memory ceiling's pure helpers and knobs (moved whole from server.zig, which re-exports them):
//! Metal's working-set limit and its static override, the free-RAM term, the raised wired limit's floor and
//! its margin, the OS reserve. Lower in the import graph than the server, so a module-owned arch sizes itself
//! against the same ceiling the server's admission reads.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx");
const status = @import("status.zig");

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process: ggml's
/// global std::terminate handler prints the backtrace, but the throw is MLX's).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
pub fn getGpuWorkingSetLimit() u64 {
    if (static_ceiling_override) |v| return v;
    const max_rec = mlx.maxRecommendedWorkingSet();
    return if (max_rec > 0) max_rec else getMetalBufferLimit();
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process (the
/// libllama frames in the backtrace are just its global std::terminate handler;
/// the throw is MLX's). Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
pub fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// How far under the enforced wired limit a plan may reach: past the limit Metal returns
/// zeros before an uncatchable abort, so a real transient's worth of margin stays unplanned.
pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 8 << 30;

/// PURE: the floor an explicitly raised `iogpu.wired_limit_mb` puts under the ceiling; 0 when
/// the sysctl is absent or at the macOS default (75% of RAM), which leaves the ceiling as is.
/// A floor, not a cap: the working-set term already tracks the sysctl. The term it lifts is
/// the free-RAM one, which counts other processes' pages as gone; the operator who raised the
/// enforced limit has said the GPU may claim that much. `margin` (`--wired-margin-gib`)
/// is what stays unplanned under the limit: other processes' wired GPU memory, the
/// allocator cache and the estimator's slack; past the limit Metal returns zeros.
pub fn wiredLimitFloor(wired_limit: u64, total_ram: u64, margin: u64) u64 {
    if (wired_limit == 0 or total_ram == 0) return 0;
    if (wired_limit <= total_ram * 75 / 100) return 0; // the macOS default: no declaration
    return @min(wired_limit -| margin, total_ram -| margin);
}

pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
    if (n < 2 or n > 32) return error.InvalidWiredMargin;
    return @as(u64, n) << 30;
}

/// `--wired-margin-gib` in bytes: what stays unplanned under a raised wired limit.
pub var wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES;

/// `--wired-margin <size>` (upstream's size syntax: bytes, or KB / MB / GB binary multiples): the margin at byte
/// granularity, so a margin stated in decimal bytes (e.g. 2.0 GB) reaches the plan exactly. 1..32 GiB:
/// `--wired-margin-gib`'s range with its floor at 1 GiB, since a 2.0 GB decimal stop is 1.86 GiB.
pub fn wiredMarginFromBytes(bytes: u64) error{InvalidWiredMargin}!u64 {
    if (bytes < 1 << 30 or bytes > 32 << 30) return error.InvalidWiredMargin;
    return bytes;
}

/// PURE: the ceiling with the wired-limit floor applied. `wired_floor == 0` returns
/// `physicalMemoryCeiling` byte for byte.
pub fn gpuCeilingWithWiredFloor(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
) u64 {
    const physical = physicalMemoryCeiling(working_set_limit, mlx_footprint, free_system);
    return @max(physical, wired_floor);
}

/// Test seam: `iogpu.wired_limit_mb` in MB; `null` reads the machine.
pub var wired_limit_mb_override: ?u64 = null;

var wired_limit_bytes_cached: u64 = 0;
var wired_limit_read: bool = false;

/// `iogpu.wired_limit_mb` in bytes, read ONCE per process: re-reading it per admission would
/// let the ceiling move under a live request. 0 when the OID is absent.
pub fn wiredLimitBytes() u64 {
    if (wired_limit_mb_override) |mb| return mb *| (1024 * 1024);
    // iogpu.wired_limit_mb is an Apple GPU sysctl; Linux has no wired-limit
    // concept, so the query reads as absent (0), same as a non-Apple-Silicon Mac.
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    if (wired_limit_read) return wired_limit_bytes_cached;
    var v: u32 = 0;
    var len: usize = @sizeOf(u32);
    wired_limit_bytes_cached = if (sysctlbyname("iogpu.wired_limit_mb", @ptrCast(&v), &len, null, 0) == 0)
        @as(u64, v) * 1024 * 1024
    else
        0;
    wired_limit_read = true;
    return wired_limit_bytes_cached;
}

/// The ceiling term that does not move with instantaneous free RAM: Metal's recommended
/// working set (or the wired limit). The load-time hot-cache clamp bills against this and
/// nothing else: two boots 11 minutes apart resolved the same ask to 1076 and 9757 MB off the
/// live term. Request-time admission still reads live memory.
/// Stands in for the machine's working-set limit: tests (CI runners have 7 GB) and `applyGpuCeilingEnv`.
pub var static_ceiling_override: ?u64 = null;

pub fn staticGpuMemoryCeiling() u64 {
    return getGpuWorkingSetLimit();
}

/// Free RAM the plan never touches. MLX wires what it allocates, so planning down to the last
/// free page leaves the OS nothing to reclaim: a 16 GB Mac died on wired memory (15.4 GB wired,
/// 14 MB free) with every request admitted.
pub fn osReserveBytes(total_ram: u64) u64 {
    if (os_reserve_override) |v| return v;
    return std.math.clamp(total_ram / 8, 2 << 30, 8 << 30);
}

/// `--os-reserve-gib N` in bytes; 0 turns the reserve off. Null = the automatic eighth.
pub var os_reserve_override: ?u64 = null;

pub fn parseOsReserveGib(raw: []const u8) error{InvalidOsReserve}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidOsReserve;
    if (n > 64) return error.InvalidOsReserve;
    return @as(u64, n) << 30;
}

/// Get the max buffer allocation limit (~75% of system unified memory).
/// `hw.memsize` on Darwin; on Linux there is no unified-memory sysctl, so the
/// same 75%-of-physical-RAM heuristic runs against /proc/meminfo.
pub fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    if (comptime builtin.os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    } else {
        mem = status.getTotalMemBytes();
    }
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

// ── Tests (pure arithmetic and the static knobs; no device query: the working-set term is overridden) ──

const testing = std.testing;
const GiB: u64 = 1 << 30;

test "gpu ceiling: the physical ceiling is the smaller of the working set and what MLX can reach now, saturating" {
    try testing.expectEqual(@as(u64, 100 * GiB), physicalMemoryCeiling(100 * GiB, 60 * GiB, 50 * GiB));
    try testing.expectEqual(@as(u64, 70 * GiB), physicalMemoryCeiling(100 * GiB, 60 * GiB, 10 * GiB));
    try testing.expectEqual(@as(u64, 5), physicalMemoryCeiling(5, std.math.maxInt(u64), 1));
    // No floor: the physical ceiling byte for byte; a floor above it lifts it, one below leaves it.
    try testing.expectEqual(physicalMemoryCeiling(100 * GiB, 60 * GiB, 10 * GiB), gpuCeilingWithWiredFloor(100 * GiB, 60 * GiB, 10 * GiB, 0));
    try testing.expectEqual(@as(u64, 90 * GiB), gpuCeilingWithWiredFloor(100 * GiB, 60 * GiB, 10 * GiB, 90 * GiB));
    try testing.expectEqual(@as(u64, 70 * GiB), gpuCeilingWithWiredFloor(100 * GiB, 60 * GiB, 10 * GiB, 20 * GiB));
}

test "gpu ceiling: the wired floor is 0 at the macOS default or with the sysctl absent, else the limit less the margin, capped by RAM" {
    const ram = 128 * GiB;
    try testing.expectEqual(@as(u64, 0), wiredLimitFloor(0, ram, 8 * GiB));
    try testing.expectEqual(@as(u64, 0), wiredLimitFloor(112 * GiB, 0, 8 * GiB));
    try testing.expectEqual(@as(u64, 0), wiredLimitFloor(ram * 75 / 100, ram, 8 * GiB));
    try testing.expectEqual(@as(u64, 104 * GiB), wiredLimitFloor(112 * GiB, ram, 8 * GiB));
    // A limit past RAM is capped by RAM; a margin past both saturates at 0.
    try testing.expectEqual(@as(u64, 120 * GiB), wiredLimitFloor(200 * GiB, ram, 8 * GiB));
    try testing.expectEqual(@as(u64, 0), wiredLimitFloor(112 * GiB, ram, 300 * GiB));
}

test "gpu ceiling: default refuses, pinned admits: the default 8 GiB margin refuses a plan a 2.0 GB margin admits" {
    const ram = 128 * GiB;
    const limit = 112 * GiB; // iogpu.wired_limit_mb 114688
    const need = 108 * GiB;
    const ceiling = struct {
        fn at(margin: u64) u64 {
            return gpuCeilingWithWiredFloor(115 * GiB, 30 * GiB, 20 * GiB, wiredLimitFloor(limit, ram, margin));
        }
    }.at;
    try testing.expect(ceiling(WIRED_LIMIT_MARGIN_BYTES) < need);
    const pinned = try wiredMarginFromBytes(2_000_000_000);
    try testing.expect(ceiling(pinned) >= need);
    try testing.expectEqual(limit - 2_000_000_000, ceiling(pinned));
}

test "gpu ceiling: margin, reserve and override knobs parse within their ranges and refuse the rest by name" {
    try testing.expectEqual(@as(u64, 2 * GiB), try parseWiredMarginGib("2"));
    try testing.expectEqual(@as(u64, 32 * GiB), try parseWiredMarginGib("32"));
    for ([_][]const u8{ "1", "33", "-2", "", "two", "8.5" }) |bad| try testing.expectError(error.InvalidWiredMargin, parseWiredMarginGib(bad));
    try testing.expectEqual(@as(u64, GiB), try wiredMarginFromBytes(GiB));
    try testing.expectEqual(@as(u64, 32 * GiB), try wiredMarginFromBytes(32 * GiB));
    try testing.expectError(error.InvalidWiredMargin, wiredMarginFromBytes(GiB - 1));
    try testing.expectError(error.InvalidWiredMargin, wiredMarginFromBytes(32 * GiB + 1));
    try testing.expectEqual(@as(u64, 0), try parseOsReserveGib("0"));
    try testing.expectEqual(@as(u64, 64 * GiB), try parseOsReserveGib("64"));
    for ([_][]const u8{ "65", "-1", "x" }) |bad| try testing.expectError(error.InvalidOsReserve, parseOsReserveGib(bad));
    // The OS reserve: an eighth of RAM within 2..8 GiB, unless overridden (0 turns it off).
    const saved = os_reserve_override;
    defer os_reserve_override = saved;
    os_reserve_override = null;
    try testing.expectEqual(@as(u64, 2 * GiB), osReserveBytes(8 * GiB));
    try testing.expectEqual(@as(u64, 4 * GiB), osReserveBytes(32 * GiB));
    try testing.expectEqual(@as(u64, 8 * GiB), osReserveBytes(512 * GiB));
    os_reserve_override = 0;
    try testing.expectEqual(@as(u64, 0), osReserveBytes(128 * GiB));
}

test "gpu ceiling: the static term and the wired limit read their overrides, never the device" {
    const saved_ceiling = static_ceiling_override;
    const saved_wired = wired_limit_mb_override;
    defer {
        static_ceiling_override = saved_ceiling;
        wired_limit_mb_override = saved_wired;
    }
    static_ceiling_override = 7 * GiB;
    try testing.expectEqual(@as(u64, 7 * GiB), getGpuWorkingSetLimit());
    try testing.expectEqual(@as(u64, 7 * GiB), staticGpuMemoryCeiling());
    wired_limit_mb_override = 114688;
    try testing.expectEqual(@as(u64, 112 * GiB), wiredLimitBytes());
    wired_limit_mb_override = std.math.maxInt(u64);
    try testing.expectEqual(@as(u64, std.math.maxInt(u64)), wiredLimitBytes());
    // The machine's own reads: the wired limit once per process (stable across calls), the buffer limit 75% of RAM.
    wired_limit_mb_override = null;
    try testing.expectEqual(wiredLimitBytes(), wiredLimitBytes());
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    try testing.expectEqual(mem * 75 / 100, getMetalBufferLimit());
}
