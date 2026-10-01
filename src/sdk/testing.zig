//! Conformance (docs/plugins.md): the checks a plugin's kinds run against the SDK's contracts, and the
//! fakes the host's own tests drive. Every check declares its lane:
//! - `cpu`: `zig build conformance` with no device; the lane fails if a Metal device was created.
//! - `gpu_small`: fixture shapes on an author's Mac, lock-held on our box.
//! - `window`: ours only (the real bank at the box ceiling), never in the suite.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx");
const peek = @import("peek.zig");
const arch = @import("arch.zig");
const spec = @import("spec.zig");
const bill = @import("memory_bill.zig");
const lifecycle = @import("lifecycle.zig");

pub const Lane = enum { cpu, gpu_small, window };

// ── The CPU lane's device check ──

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

/// Whether this process created a Metal device: only that maps a GPU driver bundle (AGXMetal*).
pub fn deviceCreated() bool {
    if (builtin.os.tag != .macos) return false;
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

/// The CPU lane's last check: nothing before it created a device (any MLX array in the Metal build does).
pub fn expectNoDevice() error{DeviceCreatedInCpuLane}!void {
    if (deviceCreated()) return error.DeviceCreatedInCpuLane;
}

// ── Claims (cpu) ──

/// A fixture config and the claim it must get: the plugin's own config at its priority, near misses declined.
pub const ClaimCase = struct { config: []const u8, want: ?peek.Priority };

pub fn expectClaims(claims: *const fn (*const peek.ConfigPeek) ?peek.Priority, cases: []const ClaimCase) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const p = try peek.ConfigPeek.parse(arena.allocator(), "/fixture", c.config);
        std.testing.expectEqual(c.want, claims(&p)) catch |e| {
            std.debug.print("claims: {s}\n", .{c.config});
            return e;
        };
    }
}

// ── Bills, fills and admission (cpu) ──

/// A (baseline, ceiling, stop) triple and the fill it must reach, or the refusal it must name.
pub const AdmissionCase = struct {
    baseline: u64,
    ceiling: u64,
    stop: u64,
    want: union(enum) { rows: bill.Rows, refused: anyerror },
};

/// The fill at each triple, then the admission of the filled rows (each phase within one row of the target).
pub fn expectAdmission(b: bill.MemoryBill, n_experts: u32, min_rows: u32, cases: []const AdmissionCase) !void {
    for (cases) |c| {
        const target = c.ceiling -| c.stop;
        const got = bill.fill(b, c.baseline, target, n_experts, min_rows);
        switch (c.want) {
            .refused => |e| try std.testing.expectError(e, got),
            .rows => |want| {
                const rows = try got;
                try std.testing.expectEqual(want, rows);
                try bill.admit(b, c.baseline, rows, target);
                if (rows.prompt < rows.decode) try std.testing.expect(b.total(.prompt, c.baseline, rows.prompt + 1) > target);
                if (rows.decode < n_experts) try std.testing.expect(b.total(.decode, c.baseline, rows.decode + 1) > target);
            },
        }
    }
}

/// A route override changes exactly the terms its owner bills: `a` and `b` (one bill at two routes) differ in the
/// `owned` terms only, and keep the same term names in the same order.
pub fn expectRouteFollowing(a: bill.MemoryBill, b: bill.MemoryBill, owned: []const []const u8) !void {
    try std.testing.expectEqual(a.terms.len, b.terms.len);
    var moved = false;
    for (a.terms, b.terms) |x, y| {
        try std.testing.expectEqualStrings(x.name, y.name);
        const is_owned = for (owned) |o| {
            if (std.mem.eql(u8, o, x.name)) break true;
        } else false;
        if (std.mem.eql(u64, &x.bytes, &y.bytes)) continue;
        if (!is_owned) {
            std.debug.print("route moved a term it does not own: {s}\n", .{x.name});
            return error.RouteMovedForeignTerm;
        }
        moved = true;
    }
    if (!moved) return error.RouteMovedNothing;
}

// ── The phase change (cpu) ──

/// A recorded phase change keeps the contract's order (lifecycle.order).
pub fn expectPhaseOrder(steps: []const lifecycle.Step) !void {
    var at: ?lifecycle.Step = null;
    lifecycle.checkOrder(steps, &at) catch |e| {
        std.debug.print("phase change out of order: {s} was due\n", .{@tagName(at.?)});
        return e;
    };
}

/// A refused boundary is sticky: after `refuse`, every later `request` is refused by name (PhaseChangeRefused), so
/// no retry grows over what the refused check saw. `Gate` declares `request` and `refuse`.
pub fn expectStickyRefusal(comptime Gate: type) !void {
    var g: Gate = .{};
    try g.request();
    g.refuse(error.PhaseChangeFootprintNotFreed);
    for (0..3) |_| try std.testing.expectError(error.PhaseChangeRefused, g.request());
}

// ── The draft lane (gpu_small on a real module; cpu on a fake) ──

/// Each round returns [t1, <= cap accepted drafts] and its next token; the committed position advances by the
/// tokens kept; the next round starts from the next token. One round per entry of `caps`.
pub fn expectRoundInvariant(vt: *const arch.Arch, module: *anyopaque, t1: u32, caps: []const u32) !void {
    const lane = switch (vt.spec) {
        .draft_lane => |l| l,
        else => return error.NoDraftLane,
    };
    var t = t1;
    for (caps) |cap| {
        const before = vt.position(module);
        var r = try lane.round(module, std.testing.allocator, t, cap);
        defer r.deinit(std.testing.allocator);
        try std.testing.expect(r.tokens.len >= 1 and r.tokens[0] == t);
        try std.testing.expect(r.accepted <= cap and r.tokens.len == r.accepted + 1);
        try std.testing.expectEqual(before + r.tokens.len, vt.position(module));
        t = r.next_token;
    }
}

// ── Receipts (cpu) ──

/// A receipt carries every stamp the judges read (a JSON null is a stamp: arm 1's `cell_arm`); a dotted name walks
/// nested objects ("decode_stream.misses").
pub fn expectStamps(receipt_json: []const u8, stamps: []const []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), receipt_json, .{});
    for (stamps) |name| {
        var v: ?std.json.Value = root;
        var it = std.mem.splitScalar(u8, name, '.');
        while (it.next()) |key| {
            const cur = v orelse break;
            v = if (cur == .object) cur.object.get(key) else null;
        }
        if (v == null) {
            std.debug.print("receipt lacks stamp {s}\n", .{name});
            return error.ReceiptStampMissing;
        }
    }
}

// ── Fakes for the host's tests (cpu) ──

pub const FakeOptions = struct {
    caps: arch.Caps = .{ .owns_decode_state = true, .prefill_whole_prompt = true, .prefill_yields_last_logits = true },
    /// The model_type the fake claims.
    model_type: []const u8 = "fake_arch",
    handover: bool = true,
    /// Rows a round verifies; 0 = no draft lane.
    block_size: u32 = 0,
    /// The prompt admission's bytes; null = the host's estimator.
    prompt_bytes: ?u64 = null,
};

/// Every call a fake arch's module received.
pub const FakeCalls = struct {
    init: u32 = 0,
    deinit: u32 = 0,
    prefill: u32 = 0,
    step: u32 = 0,
    handover: u32 = 0,
    rounds: u32 = 0,
};

/// An arch for host tests: no MLX (its logits are empty handles), caps per test, every call counted.
pub fn FakeArch(comptime opts: FakeOptions) type {
    return struct {
        pub const name = "fake-arch";
        pub const caps = opts.caps;
        pub const Config = struct { settings_applied: u32 = 0 };
        pub const Module = struct {
            gpa: std.mem.Allocator,
            calls: *FakeCalls,
            position: u64 = 0,
            last_handover: ?arch.DecodeHandover = null,
            last_request: ?arch.RequestShape = null,
        };
        /// The counters every module of this fake writes; reset per test.
        pub var calls: FakeCalls = .{};

        pub fn claims(p: *const peek.ConfigPeek) ?peek.Priority {
            const t = p.modelType() orelse return null;
            return if (std.mem.eql(u8, t, opts.model_type)) .native else null;
        }
        pub fn parse(gpa: std.mem.Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) !*Config {
            if (p.int("refuse") != null) {
                diag.set("fake arch: refused by the fixture", .{});
                return error.FakeArchRefused;
            }
            const c = try gpa.create(Config);
            c.* = .{};
            return c;
        }
        pub fn freeConfig(gpa: std.mem.Allocator, c: *Config) void {
            gpa.destroy(c);
        }
        pub fn shell(_: *const Config) arch.Shell {
            return .{ .num_experts = 4, .num_layers = 2 };
        }
        pub fn applySettings(c: *Config, _: std.json.Value) void {
            c.settings_applied += 1;
        }
        pub fn loadBytes(_: std.mem.Allocator, _: std.Io, _: *const Config, _: *const arch.LoadFacts, _: u64) !u64 {
            return 1_000_000_000;
        }
        pub const promptBytes = if (opts.prompt_bytes) |n| struct {
            fn f(_: *const Config, _: u64, _: u32) u64 {
                return n;
            }
        }.f else {};
        pub fn init(load: *const arch.LoadCtx, _: *const Config) !*Module {
            const m = try load.gpa.create(Module);
            m.* = .{ .gpa = load.gpa, .calls = &calls };
            calls.init += 1;
            return m;
        }
        pub fn deinit(m: *Module) void {
            m.calls.deinit += 1;
            m.gpa.destroy(m);
        }
        pub fn prefill(m: *Module, ids: []const u32, req: arch.RequestShape) !mlx.mlx_array {
            m.calls.prefill += 1;
            m.position = ids.len;
            m.last_request = req;
            return .{};
        }
        pub fn step(m: *Module, ids: []const u32) !mlx.mlx_array {
            m.calls.step += 1;
            m.position += ids.len;
            return .{};
        }
        pub fn position(m: *const Module) u64 {
            return m.position;
        }
        pub const handover = if (opts.handover) struct {
            fn f(m: *Module, h: arch.DecodeHandover) !void {
                m.calls.handover += 1;
                m.last_handover = h;
            }
        }.f else {};
        pub const draft_lane = if (opts.block_size > 0) struct {
            pub fn blockSize(_: *const Module) u32 {
                return opts.block_size;
            }
            pub fn laneName(_: *const Module) []const u8 {
                return "fake lane";
            }
            pub fn arm(_: *const Module, req: spec.ArmRequest) spec.DraftArm {
                return if (req.greedy and req.clean) .typical else .off;
            }
            /// Keeps min(cap, block - 1) drafts: [t1, t1 + 1, ...], the next token after them.
            pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32) !spec.DraftRound {
                const k = @min(accepted_cap, opts.block_size - 1);
                const toks = try a.alloc(u32, k + 1);
                for (toks, 0..) |*t, i| t.* = t1 + @as(u32, @intCast(i));
                m.calls.rounds += 1;
                m.position += toks.len;
                return .{ .tokens = toks, .accepted = k, .next_token = t1 + k + 1 };
            }
            pub fn stats(m: *const Module) spec.DraftStats {
                return .{ .rounds = m.calls.rounds };
            }
        } else {};
    };
}

const testing = std.testing;

test "sdk testing: the fake arch's table counts every call, and its optional hooks follow its options" {
    const Fake = FakeArch(.{ .block_size = 5, .prompt_bytes = 7 });
    Fake.calls = .{};
    const vt = comptime arch.Arch.of(Fake);
    try testing.expect(vt.caps.owns_decode_state and vt.handover != null and vt.prompt_bytes != null and vt.bill == null);
    var diag: peek.Diag = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\"}");
    try testing.expectEqual(@as(?peek.Priority, .native), vt.claims(&p));
    const cfg = try vt.parse(testing.allocator, &p, &diag);
    defer vt.free_config(testing.allocator, cfg);
    vt.apply_settings(cfg, .null);
    try testing.expectEqual(@as(u32, 2), vt.shell(cfg).num_layers);
    try testing.expectEqual(@as(u64, 7), vt.prompt_bytes.?(cfg, 16384, 1024));
    const load: arch.LoadCtx = .{ .gpa = testing.allocator, .io = testing.io, .stream = .{}, .weights = undefined, .facts = .{}, .ceiling = 0 };
    const m = try vt.init(&load, cfg);
    _ = try vt.prefill(m, &.{ 1, 2, 3 }, .{ .prompt_tokens = 3, .max_tokens = 8, .host_context = 4096 });
    try vt.handover.?(m, .{ .prompt_tokens = 3, .reserved_tokens = 0, .native_draft = true });
    try expectRoundInvariant(&vt, m, 11, &.{ 4, 0, 2 });
    _ = try vt.step(m, &.{9});
    vt.deinit(m);
    try testing.expectEqual(FakeCalls{ .init = 1, .deinit = 1, .prefill = 1, .step = 1, .handover = 1, .rounds = 3 }, Fake.calls);

    const Bare = FakeArch(.{ .handover = false, .caps = .{} });
    const bare = comptime arch.Arch.of(Bare);
    try testing.expect(bare.handover == null and bare.prompt_bytes == null and bare.spec == .none and !bare.caps.owns_decode_state);
    try testing.expectError(error.FakeArchRefused, vt.parse(testing.allocator, &(try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\",\"refuse\":1}")), &diag));
    try testing.expectEqualStrings("fake arch: refused by the fixture", diag.message());
}

test "sdk testing: claims fixtures, admission triples and route-following run on any bill" {
    try expectClaims(FakeArch(.{}).claims, &.{
        .{ .config = "{\"model_type\":\"fake_arch\"}", .want = .native },
        .{ .config = "{\"model_type\":\"deepseek_v4\"}", .want = null },
        .{ .config = "{\"architectures\":[\"X\"]}", .want = null },
    });
    const gb: u64 = 1_000_000_000;
    const terms = [_]bill.MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb } }, .{ .name = "waves", .bytes = .{ 14 * gb, 2 * gb } } };
    const b: bill.MemoryBill = .{ .terms = &terms, .per_row = gb / 4 };
    try expectAdmission(b, 128, 16, &.{
        // the decode rows capped by the experts, the prompt rows by the decode rows
        .{ .baseline = 9 * gb, .ceiling = 120 * gb, .stop = 2 * gb, .want = .{ .rows = .{ .prompt = 128, .decode = 128 } } },
        .{ .baseline = 9 * gb, .ceiling = 100 * gb, .stop = 2 * gb, .want = .{ .rows = .{ .prompt = 60, .decode = 112 } } },
        // the upstream 8 GiB margin in place of the 2 GB stop costs 27 rows a phase
        .{ .baseline = 9 * gb, .ceiling = 100 * gb, .stop = 8 * (1 << 30), .want = .{ .rows = .{ .prompt = 33, .decode = 85 } } },
        .{ .baseline = 21 * gb, .ceiling = 100 * gb, .stop = 2 * gb, .want = .{ .refused = error.NativeBillDoesNotFit } },
    });
    const release = [_]bill.MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb } }, .{ .name = "waves", .bytes = .{ 14 * gb, 1 * gb } } };
    try expectRouteFollowing(b, .{ .terms = &release, .per_row = gb / 4 }, &.{"waves"});
    try testing.expectError(error.RouteMovedForeignTerm, expectRouteFollowing(b, .{ .terms = &release, .per_row = gb / 4 }, &.{"residents"}));
    try testing.expectError(error.RouteMovedNothing, expectRouteFollowing(b, b, &.{"waves"}));
}

test "sdk testing: the phase change's order and a sticky refusal; a receipt's stamps" {
    try expectPhaseOrder(&.{ .synchronize, .fence, .cache_clear, .cache_limit, .synchronize_freed, .settle, .check_freed, .grow });
    const Gate = struct {
        refused: ?anyerror = null,
        fn request(g: *const @This()) error{PhaseChangeRefused}!void {
            if (g.refused != null) return error.PhaseChangeRefused;
        }
        fn refuse(g: *@This(), e: anyerror) void {
            if (g.refused == null) g.refused = e;
        }
    };
    try expectStickyRefusal(Gate);
    const receipt = "{\"decode_lane\":\"dspark typical 0.3\",\"cell_arm\":null,\"decode_stream\":{\"misses\":7952}}";
    try expectStamps(receipt, &.{ "decode_lane", "cell_arm", "decode_stream.misses" });
    try testing.expectError(error.ReceiptStampMissing, expectStamps(receipt, &.{"head_mode"}));
    try testing.expectError(error.ReceiptStampMissing, expectStamps(receipt, &.{"decode_stream.bytes"}));
    try testing.expectError(error.ReceiptStampMissing, expectStamps(receipt, &.{"decode_lane.x"}));
}
