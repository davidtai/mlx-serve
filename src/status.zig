const std = @import("std");
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// ── macOS Mach externs ──

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

// ── IOKit / CoreFoundation externs ──

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;

// ── Mach struct layouts (extern = C ABI) ──

const TaskBasicInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_sec: i32,
    user_time_usec: i32,
    sys_time_sec: i32,
    sys_time_usec: i32,
    policy: i32,
    suspend_count: i32,
};

/// task_vm_info through the rev3 ledger block. Field order matches <mach/task_info.h>
/// exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 84 == TASK_VM_INFO_REV3_COUNT. The count
/// is in/out: a kernel below rev3 fills fewer fields and leaves the rest zero.
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    // rev2
    min_address: u64,
    max_address: u64,
    // rev3: the ledger block
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_novolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
};

comptime {
    std.debug.assert(@sizeOf(TaskVmInfo) / @sizeOf(i32) == 84); // TASK_VM_INFO_REV3_COUNT
}

const CpuLoadInfo = extern struct {
    ticks: [4]u32, // user, system, idle, nice
};

const VmStats64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

// ── CPU delta tracking (module-level state) ──
var prev_ticks: [4]u64 = @splat(0);

// ── Public metric helpers ──

pub fn getAppRssMb() u32 {
    if (comptime !builtin.os.tag.isDarwin())
        return @intCast(linuxProcStatusKib("VmRSS:") / 1024);
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

/// Process physical memory footprint in MB (TASK_VM_INFO flavor 22). Unlike
/// resident_size, this includes MLX's Metal/IOKit + compressed memory — the
/// only figure that reflects a loaded model's true footprint on Apple Silicon.
/// On Linux the honest analog is VmRSS (unified-memory footprint has no
/// equivalent); swap-paged-out pages are invisible to it.
pub fn getAppMemFootprintMb() u32 {
    if (comptime !builtin.os.tag.isDarwin())
        return @intCast(linuxProcStatusKib("VmRSS:") / 1024);
    const info = taskVmInfo() orelse return 0;
    return @intCast(info.phys_footprint / (1024 * 1024));
}

fn taskVmInfo() ?TaskVmInfo {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32); // 84 = TASK_VM_INFO_REV3_COUNT (in/out)
    if (task_info(mach_task_self_, 22, @ptrCast(&info), &count) != 0) return null;
    return info;
}

/// This process's phys_footprint (MLX's Metal / IOKit memory included) and its lifetime peak, in bytes.
pub const Footprint = struct { now: u64, peak: u64 };

pub fn footprint() Footprint {
    const info = taskVmInfo() orelse return .{ .now = 0, .peak = 0 };
    return .{ .now = info.phys_footprint, .peak = if (info.ledger_phys_footprint_peak > 0) @intCast(info.ledger_phys_footprint_peak) else info.phys_footprint };
}

/// rusage_info_v4 (<sys/resource.h>): the footprint's interval high-water mark, which
/// `proc_reset_footprint_interval` (libsystem_kernel) restarts: the kernel's own ledger, no sampling.
const RusageInfoV4 = extern struct {
    uuid: [16]u8,
    f: [35]u64,
    const lifetime_max_phys_footprint = 28;
    const interval_max_phys_footprint = 33;
};
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *RusageInfoV4) c_int;
extern "c" fn proc_reset_footprint_interval(pid: c_int) c_int;
extern "c" fn getpid() c_int;

/// This process's memory from the kernel's ledgers (task_vm_info rev3 and rusage v4), in bytes: the
/// footprint, its interval and lifetime peaks, and how it splits. Zero where the platform has no ledger.
pub const ProcessMemory = struct {
    /// phys_footprint now (Metal / IOAccelerator included).
    footprint: u64 = 0,
    /// The footprint's high-water mark since the previous `startFootprintInterval`.
    footprint_interval_peak: u64 = 0,
    footprint_lifetime_peak: u64 = 0,
    /// Anonymous pages resident; compressed pages this task owns.
    internal: u64 = 0,
    compressed: u64 = 0,
    /// File-backed pages this task maps resident (outside the footprint).
    external: u64 = 0,
    /// IOKit graphics memory (Metal buffers): in the footprint / outside it.
    graphics_footprint: u64 = 0,
    graphics_nofootprint: u64 = 0,
    /// Volatile purgeable memory (outside the footprint).
    purgeable_volatile: u64 = 0,
};

pub fn processMemory() ProcessMemory {
    const info = taskVmInfo() orelse return .{};
    const pos = struct {
        fn f(x: i64) u64 {
            return if (x > 0) @intCast(x) else 0;
        }
    }.f;
    var m: ProcessMemory = .{
        .footprint = info.phys_footprint,
        .footprint_lifetime_peak = pos(info.ledger_phys_footprint_peak),
        .internal = info.internal,
        .compressed = info.compressed,
        .external = info.external,
        .graphics_footprint = pos(info.ledger_tag_graphics_footprint),
        .graphics_nofootprint = pos(info.ledger_tag_graphics_nofootprint),
        .purgeable_volatile = pos(info.ledger_purgeable_volatile),
    };
    var ru = std.mem.zeroes(RusageInfoV4);
    if (proc_pid_rusage(getpid(), 4, &ru) == 0) {
        m.footprint_interval_peak = ru.f[RusageInfoV4.interval_max_phys_footprint];
        m.footprint_lifetime_peak = @max(m.footprint_lifetime_peak, ru.f[RusageInfoV4.lifetime_max_phys_footprint]);
    }
    return m;
}

/// Restarts the footprint's interval high-water mark (`ProcessMemory.footprint_interval_peak`).
pub fn startFootprintInterval() void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    _ = proc_reset_footprint_interval(getpid());
}

/// The box's page counts (`VmStats64`), in bytes. Zero off Darwin.
pub const VmBytes = struct { free: u64 = 0, active: u64 = 0, inactive: u64 = 0, wired: u64 = 0, purgeable: u64 = 0, speculative: u64 = 0, compressor: u64 = 0, external: u64 = 0, internal: u64 = 0 };

pub fn vmBytes() VmBytes {
    if (comptime !builtin.os.tag.isDarwin()) return .{};
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return .{};
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return .{};
    const pg: u64 = page;
    return .{
        .free = vm.free_count * pg,
        .active = vm.active_count * pg,
        .inactive = vm.inactive_count * pg,
        .wired = vm.wire_count * pg,
        .purgeable = vm.purgeable_count * pg,
        .speculative = vm.speculative_count * pg,
        .compressor = vm.compressor_page_count * pg,
        .external = vm.external_page_count * pg,
        .internal = vm.internal_page_count * pg,
    };
}

/// vm_stat's wired + active + inactive + compressor-occupied pages: the "physical used" an external
/// memory guard reads (file cache included; free and speculative pages excluded).
pub fn physicalUsedBytes(v: VmBytes) u64 {
    return v.wired + v.active + v.inactive + v.compressor;
}

/// Bytes of physical memory available for new allocation without heavy
/// Pure: bytes available for a new large allocation given the live page counts.
///
/// Subtracts the genuinely non-reclaimable set: `wired` (pinned), `compressor`
/// (already-compressed app data), and `internal` (anonymous app pages — crucially
/// INCLUDING a resident MLX model). File-backed cache (`external`) plus
/// free/speculative/purgeable pages are NOT subtracted: macOS evicts them the
/// instant a big allocation lands, so they don't block a load. That keeps a 12B
/// (~7.7 GB) loading on a 16 GB Mac that shows only ~7.8 GB *instantaneous* free
/// (the rest is reclaimable file cache). It also fixes the #45 OOM: a prior model
/// still resident lives in the anonymous (`internal`) set, NOT necessarily in
/// `wired` (verified live: a 5 GB resident model with only ~2.8 GB total wired) —
/// so counting `internal` makes a second large load correctly fail the guard.
/// `purgeable` anon pages (caches an app explicitly marked discardable — e.g.
/// image/tile caches) are a SUBSET of `internal` that macOS drops the instant a
/// big allocation lands, so they must NOT count as used; subtracting them back
/// out of the internal set is the accuracy fix. Still slightly conservative
/// (wired-anonymous pages can appear in both `wired` and `internal`), which is
/// the safe direction for an OOM guard (`--skip-mem-preflight` overrides).
/// Returns 0 when total is 0 or used ≥ total (a failed query must never block).
fn computeAvailableBytes(total_mem: u64, wire_pages: u64, compressor_pages: u64, internal_pages: u64, purgeable_pages: u64, page: u64) u64 {
    // Purgeable is reclaimable, so exclude it from the resident anon set. Saturate
    // (never underflow) in case the counters momentarily disagree.
    const resident_anon: u64 = internal_pages -| purgeable_pages;
    const used: u64 = (wire_pages + compressor_pages + resident_anon) * page;
    if (total_mem == 0 or used >= total_mem) return 0;
    return total_mem - used;
}

extern "c" fn os_proc_available_memory() usize;

/// Per-process memory headroom before jetsam (iOS only — the entitlement-
/// aware figure that actually governs whether a big allocation survives).
/// Returns 0 on macOS, where the concept doesn't apply; the symbol is only
/// referenced on iOS builds so macOS links are unaffected.
pub fn getProcAvailableMemBytes() u64 {
    if (comptime builtin.os.tag != .ios) return 0;
    return @intCast(os_proc_available_memory());
}

/// Total physical RAM (hw.memsize on Darwin; /proc/meminfo on Linux). 0 on failure.
pub fn getTotalMemBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return linuxMemInfoKib("MemTotal:") * 1024;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

pub fn getAvailableMemBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return linuxMemInfoKib("MemAvailable:") * 1024;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    return computeAvailableBytes(total_mem, vm.wire_count, vm.compressor_page_count, vm.internal_page_count, vm.purgeable_count, page);
}

/// One `/proc/meminfo` field in KiB ("MemTotal:", "MemAvailable:", …).
/// 0 when missing or unreadable — callers treat 0 as "unknown".
fn linuxMemInfoKib(key: []const u8) u64 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile("/proc/meminfo", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        const val = std.mem.trim(u8, line[key.len..], " \t");
        const end = std.mem.indexOf(u8, val, " kB") orelse val.len;
        return std.fmt.parseInt(u64, std.mem.trim(u8, val[0..end], " \t"), 10) catch 0;
    }
    return 0;
}

/// One `/proc/self/status` field in KiB ("VmRSS:", "VmHWM:", …). 0 unknown.
fn linuxProcStatusKib(key: []const u8) u64 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile("/proc/self/status", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        const val = std.mem.trim(u8, line[key.len..], " \t");
        const end = std.mem.indexOf(u8, val, " kB") orelse val.len;
        return std.fmt.parseInt(u64, std.mem.trim(u8, val[0..end], " \t"), 10) catch 0;
    }
    return 0;
}

/// Whole-file read of a small procfs file. Returns null when missing/unreadable.
fn readProcFile(path: []const u8, buf: []u8) ?usize {
    if (path.len >= std.fs.max_path_bytes) return null;
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    const fd = std.c.open(pbuf[0..path.len :0].ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n <= 0) return null;
    return @intCast(n);
}

test "computeAvailableBytes counts the resident anon set, not file cache or purgeable" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page; // pages per GB

    // 16 GB Mac, light anon load: 3 GB wired, 1 GB compressed, 2 GB anonymous app
    // pages, no purgeable; the remaining ~10 GB is free + reclaimable file cache,
    // which must NOT count against availability. Available = 16 − (3+1+2) = 10 GB.
    // (The old `active`-subtracting formula counted file cache and wrongly refused
    // loads that fit; later dropping `active` entirely wrongly ignored resident
    // models.)
    try std.testing.expectEqual(@as(u64, 10 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 2 * ppg, 0, page));

    // #45 OOM guard: a prior 7 GB model is resident. It lives in the anonymous
    // (`internal`) set — here 9 GB = 2 GB apps + 7 GB model — NOT in `wired`. So
    // available = 16 − (3+1+9) = 3 GB, and a second 7 GB load is correctly refused.
    try std.testing.expectEqual(@as(u64, 3 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 9 * ppg, 0, page));

    // Purgeable is reclaimable: 2 GB of the 9 GB internal set is a discardable
    // cache, so it should NOT count as used. Available = 16 − (3+1+(9−2)) = 5 GB,
    // up from the 3 GB the old formula reported — the accuracy fix.
    try std.testing.expectEqual(@as(u64, 5 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 9 * ppg, 2 * ppg, page));

    // Purgeable never underflows the anon set even if the counters disagree.
    try std.testing.expectEqual(@as(u64, 12 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 1 * ppg, 5 * ppg, page));

    // Degenerate guards: failed query (total 0) and used ≥ total → 0, never block.
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(0, 1, 1, 1, 0, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(8 * GB, 4 * ppg, 0, 5 * ppg, 0, page));
}

pub fn getSysMemPct() u32 {
    if (comptime !builtin.os.tag.isDarwin()) {
        // (total - available) / total — available already excludes reclaimable
        // page cache, so this matches the Mach wire+compressor+anon intent.
        const total = linuxMemInfoKib("MemTotal:");
        if (total == 0) return 0;
        const avail = linuxMemInfoKib("MemAvailable:");
        return @intCast((total -| avail) * 100 / total);
    }
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    const used: u64 = (@as(u64, vm.active_count) + vm.wire_count + vm.compressor_page_count) * page;
    if (total_mem == 0) return 0;
    return @intCast(used * 100 / total_mem);
}

pub fn getCpuPct() u32 {
    if (comptime !builtin.os.tag.isDarwin()) {
        // /proc/stat aggregate line: "cpu  user nice system idle iowait irq
        // softirq steal ...". Delta over the calls, idle = nice-adjusted idle
        // columns, mirroring the Mach ticks loop below.
        var buf: [512]u8 = undefined;
        const n = readProcFile("/proc/stat", &buf) orelse return 0;
        const line_end = std.mem.indexOfScalar(u8, buf[0..n], '\n') orelse n;
        var it = std.mem.tokenizeAny(u8, buf[0..line_end], " \t");
        _ = it.next(); // "cpu"
        var ticks: [4]u64 = @splat(0);
        var all: [10]u64 = @splat(0);
        var i: usize = 0;
        while (it.next()) |tok| : (i += 1) {
            if (i >= 10) break;
            all[i] = std.fmt.parseInt(u64, tok, 10) catch 0;
        }
        ticks[0] = all[0] + all[1] + all[2]; // user+nice+system
        ticks[1] = 0;
        ticks[2] = 0;
        ticks[3] = all[3] + all[4]; // idle + iowait
        var total: u64 = 0;
        var idle: u64 = 0;
        for (0..4) |j| {
            const delta = ticks[j] -| prev_ticks[j];
            total += delta;
            if (j == 3) idle = delta;
            prev_ticks[j] = ticks[j];
        }
        if (total == 0) return 0;
        return @intCast((total - idle) * 100 / total);
    }
    var info = std.mem.zeroes(CpuLoadInfo);
    var count: u32 = 4;
    if (host_statistics(mach_host_self(), 3, @ptrCast(&info), &count) != 0) return 0;

    var total: u64 = 0;
    var idle: u64 = 0;
    for (0..4) |i| {
        const cur: u64 = info.ticks[i];
        const delta = cur -| prev_ticks[i];
        total += delta;
        if (i == 2) idle = delta;
        prev_ticks[i] = cur;
    }
    if (total == 0) return 0;
    return @intCast((total - idle) * 100 / total);
}

pub fn getGpuPct() u32 {
    // IOKit's IOServiceMatching/AGXAccelerator path is macOS-only; the symbols
    // aren't in the public iOS SDK (and apps are sandboxed from the GPU service
    // registry anyway). On iOS we report 0 — the value is only a log-line stat.
    if (comptime !is_macos) return 0;
    const matching = IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = IOObjectRelease(iter);

    const entry = IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = IOObjectRelease(entry);

    var props: ?*anyopaque = null;
    if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| CFRelease(p);

    const perf = cfDictGet(props, "PerformanceStatistics") orelse return 0;
    const util = cfDictGet(perf, "Device Utilization %") orelse return 0;

    var value: i64 = 0;
    _ = CFNumberGetValue(util, 4, @ptrCast(&value));
    return if (value >= 0 and value <= 100) @intCast(value) else 0;
}

fn cfDictGet(dict: ?*const anyopaque, key_name: [*:0]const u8) ?*const anyopaque {
    const key = CFStringCreateWithCString(null, key_name, 0x08000100) orelse return null;
    defer CFRelease(key);
    return CFDictionaryGetValue(dict, key);
}

test "getAppMemFootprintMb returns a plausible nonzero footprint" {
    const fp = getAppMemFootprintMb();
    // The test process itself footprints several MB; a wrong flavor/offset
    // would yield 0 or absurd garbage.
    try std.testing.expect(fp > 0);
    try std.testing.expect(fp < 1024 * 1024); // < 1 TB sanity bound
}

test "status: the process ledgers, and the footprint's interval peak restarts and then holds a touched allocation" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const m0 = processMemory();
    try std.testing.expect(m0.footprint > 0 and m0.internal > 0);
    try std.testing.expect(m0.footprint_lifetime_peak >= m0.footprint);
    try std.testing.expect(footprint().now > 0);
    startFootprintInterval();
    const m1 = processMemory();
    try std.testing.expect(m1.footprint_interval_peak <= m1.footprint_lifetime_peak);
    try std.testing.expect(m1.footprint_interval_peak + (8 << 20) >= m1.footprint);
    // 64 MiB touched then freed: the interval peak holds it after the free.
    const buf = try std.heap.page_allocator.alloc(u8, 64 << 20);
    @memset(buf, 1);
    const touched = processMemory().footprint;
    std.heap.page_allocator.free(buf);
    const m2 = processMemory();
    try std.testing.expect(touched >= m1.footprint + (60 << 20));
    try std.testing.expect(m2.footprint_interval_peak >= touched);
    try std.testing.expect(m2.footprint + (60 << 20) <= m2.footprint_interval_peak);
    // The megabyte reader agrees with the byte ledger.
    try std.testing.expect(@as(u64, getAppMemFootprintMb()) * (1 << 20) <= footprint().now + (16 << 20));
}

test "status: the box's typed page counts hold this process's footprint" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const v = vmBytes();
    try std.testing.expect(v.wired > 0 and v.active > 0 and v.free > 0);
    try std.testing.expect(physicalUsedBytes(v) > footprint().now);
    try std.testing.expect(physicalUsedBytes(v) + v.free <= getTotalMemBytes() + (1 << 30));
}
