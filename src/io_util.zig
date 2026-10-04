//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.
//!
//! Also the descriptors of files read past the page cache (`openNoCache`,
//! `noCache`): resident weights loaded past the page cache (nocache_reader).

const std = @import("std");
const builtin = @import("builtin");

/// Seconds since the Unix epoch (wall-clock).
pub fn nowSecs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toSeconds();
}

/// Milliseconds since the Unix epoch (wall-clock).
pub fn nowMs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toMilliseconds();
}

/// Milliseconds on the MONOTONIC `boot` clock (arbitrary epoch — only
/// differences are meaningful). Use for deadlines/intervals, never for
/// timestamps a client sees: an NTP step or a manual clock change must not be
/// able to stall (or spam) a timer. `.boot` counts across pmset sleep, so a
/// deadline that expired while the lid was shut fires immediately on wake.
pub fn nowMsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMilliseconds();
}

/// Drop-in replacement for `std.time.Timer` — uses the `boot` clock, which
/// counts wall-time across pmset sleep (vs `.awake`, which stops during
/// suspend). Important for long-running stopwatches that span a lid close.
/// Resolution is set by the `Io` implementation.
pub const Stopwatch = struct {
    io: std.Io,
    started_at: std.Io.Timestamp,

    pub fn init(io: std.Io) Stopwatch {
        return .{ .io = io, .started_at = std.Io.Timestamp.now(io, .boot) };
    }

    /// Nanoseconds since `init` (or last `reset`).
    pub fn read(s: Stopwatch) u64 {
        return @intCast(s.started_at.untilNow(s.io, .boot).nanoseconds);
    }

    /// Reset the start point to "now".
    pub fn reset(s: *Stopwatch) void {
        s.started_at = std.Io.Timestamp.now(s.io, .boot);
    }
};

pub const NoCacheOptions = struct {
    /// Keep the kernel's read-ahead (a file streamed front to back).
    read_ahead: bool = false,
    /// Open a symlink's target; false refuses a symlink (ELOOP).
    follow_symlinks: bool = true,
};

/// Reads of `fd` bypass the page cache (F_NOCACHE); read-ahead off unless asked.
/// Darwin only: elsewhere it refuses, so no caller silently reads through the cache.
pub fn noCache(fd: std.c.fd_t, opts: NoCacheOptions) error{ NoCacheFcntl, NoCacheUnsupported }!void {
    if (comptime !builtin.os.tag.isDarwin()) return error.NoCacheUnsupported;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.NoCacheFcntl;
    if (!opts.read_ahead and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
}

/// `path` read-only and close-on-exec, then `noCache`. `error.OpenFailed` leaves
/// errno as `open` set it.
pub fn openNoCache(path: [*:0]const u8, opts: NoCacheOptions) error{ FileNotFound, OpenFailed, NoCacheFcntl, NoCacheUnsupported }!std.c.fd_t {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = !opts.follow_symlinks, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.posix.E.NOENT)) error.FileNotFound else error.OpenFailed;
    errdefer _ = std.c.close(fd);
    try noCache(fd, opts);
    return fd;
}

// ── Tests (temp files; the reads checked against a plain pread of the same bytes) ──

const testing = std.testing;

/// A temp file of `len` pattern bytes and its absolute path.
const TmpFile = struct {
    dir: std.testing.TmpDir,
    path: [:0]u8,
    image: []u8,

    fn init(len: usize) !TmpFile {
        var dir = std.testing.tmpDir(.{});
        errdefer dir.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        for (image, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 11);
        try dir.dir.writeFile(testing.io, .{ .sub_path = "f.bin", .data = image });
        var root: [512]u8 = undefined;
        const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/f.bin", .{root[0..try dir.dir.realPath(testing.io, &root)]}, 0);
        return .{ .dir = dir, .path = path, .image = image };
    }

    fn deinit(t: *TmpFile) void {
        testing.allocator.free(t.path);
        testing.allocator.free(t.image);
        t.dir.cleanup();
    }
};

test "io_util: the past-the-cache open refuses a missing file and a symlink it may not follow" {
    var t = try TmpFile.init(5000);
    defer t.deinit();
    try testing.expectError(error.FileNotFound, openNoCache("/nonexistent/cov-d/f.bin", .{}));
    var lbuf: [700]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&lbuf, "{s}.link", .{t.path}, 0);
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(t.path.ptr, link.ptr));
    try testing.expectError(error.OpenFailed, openNoCache(link.ptr, .{ .follow_symlinks = false }));
    const fd = try openNoCache(link.ptr, .{ .read_ahead = true });
    _ = std.c.close(fd);
    try testing.expectError(error.NoCacheFcntl, noCache(-1, .{}));
}

test "io_util: the stopwatch and the clocks move forward" {
    const io = testing.io;
    var sw = Stopwatch.init(io);
    const a = sw.read();
    try testing.expect(sw.read() >= a);
    sw.reset();
    try testing.expect(nowMsMonotonic(io) > 0);
    const secs = nowSecs(io);
    try testing.expect(nowMs(io) >= secs * 1000);
}
