//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.
//!
//! Also the descriptors of files read past the page cache (`openNoCache`,
//! `noCache`): streamed weights, expert banks, on-disk row tables.

const std = @import("std");

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
pub fn noCache(fd: std.c.fd_t, opts: NoCacheOptions) error{NoCacheFcntl}!void {
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.NoCacheFcntl;
    if (!opts.read_ahead and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
}

/// `path` read-only and close-on-exec, then `noCache`. `error.OpenFailed` leaves
/// errno as `open` set it.
pub fn openNoCache(path: [*:0]const u8, opts: NoCacheOptions) error{ FileNotFound, OpenFailed, NoCacheFcntl }!std.c.fd_t {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = !opts.follow_symlinks, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.posix.E.NOENT)) error.FileNotFound else error.OpenFailed;
    errdefer _ = std.c.close(fd);
    try noCache(fd, opts);
    return fd;
}
