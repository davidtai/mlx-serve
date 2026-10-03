//! The streamed-expert read pool's scheduling knobs: a dependency-free type the model settings, the config and the pool
//! share (`expert_io.Pool` hands it to the C pool at start).

const std = @import("std");

/// The pool's scheduling, fixed at start (`ReaderSched` of the model settings; all false = the stock pool, its threads
/// inheriting the creating thread's QoS): `qos` runs the demand workers and the watchdog USER_INTERACTIVE and the
/// speculative workers UTILITY, every thread named; `qos_demand` (not with `qos`) runs only the demand workers and the
/// watchdog USER_INTERACTIVE, the speculative workers keeping the stock pool's inherited class (no UTILITY anywhere),
/// every thread named; `spin` (with `qos`) adds a bounded 30 us spin before a demand worker or a `wait` sleeps;
/// `demand_first` starts no unclaimed speculative chunk while a demand job is queued or executing.
pub const Sched = struct {
    qos: bool = false,
    qos_demand: bool = false,
    spin: bool = false,
    demand_first: bool = false,

    pub fn bits(s: Sched) i32 {
        return @as(i32, @intFromBool(s.qos)) | @as(i32, @intFromBool(s.spin)) << 1 | @as(i32, @intFromBool(s.demand_first)) << 2 | @as(i32, @intFromBool(s.qos_demand)) << 3;
    }

    /// "off", or the set knobs joined by commas in this order (qos, qosdemand, spin, demandfirst).
    pub fn name(s: Sched, buf: *[24]u8) []const u8 {
        if (!s.qos and !s.qos_demand and !s.spin and !s.demand_first) return "off";
        var n: usize = 0;
        inline for (.{ .{ s.qos, "qos" }, .{ s.qos_demand, "qosdemand" }, .{ s.spin, "spin" }, .{ s.demand_first, "demandfirst" } }) |kv| if (kv[0]) {
            if (n > 0) {
                buf[n] = ',';
                n += 1;
            }
            @memcpy(buf[n..][0..kv[1].len], kv[1]);
            n += kv[1].len;
        };
        return buf[0..n];
    }

    /// "off" or a comma list of qos, qosdemand, spin, demandfirst (spin only with qos; qos and qosdemand not together);
    /// null for anything else.
    pub fn parse(text: []const u8) ?Sched {
        if (std.mem.eql(u8, text, "off")) return .{};
        var s: Sched = .{};
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |t| {
            if (std.mem.eql(u8, t, "qos") and !s.qos) s.qos = true else if (std.mem.eql(u8, t, "qosdemand") and !s.qos_demand) s.qos_demand = true else if (std.mem.eql(u8, t, "spin") and !s.spin) s.spin = true else if (std.mem.eql(u8, t, "demandfirst") and !s.demand_first) s.demand_first = true else return null;
        }
        if (s.spin and !s.qos) return null;
        if (s.qos and s.qos_demand) return null;
        return s;
    }
};
