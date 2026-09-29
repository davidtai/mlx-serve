//! The served binding of the native deepseek_v41 arm: which decode loop,
//! which routed-expert math and which residents the server constructs, fixed
//! at compile time, and the one construction that builds them on the
//! inference thread before the first request. Nothing here runs per token.
//!
//! Bindings:
//!   `stand_in`  the arm's `StandIn` decode over `StandInMath`: seeded routes
//!               through the real hook, no model math, no kernels, no trunk
//!               (the gates measure construction and reads with it).
//!   `dspark`    the DSpark loop (`deepseek_v41_dspark_serve.Dspark`) over the
//!               trunk and draft head it opens (resident weights, the Engram
//!               rows over the exported token map, the stock routes); the
//!               routed experts' math is `EagerChain(MlxOps, MlxGemv)` over
//!               the kernels' decode GEMV.
//! `serving` follows `deepseek_v41_arm.serving_decode`: the stand-in until
//! the DSpark loop's gate passes; that one line then binds `dspark` in the
//! server (`openServing`), the receipt's `decode_binding` and `model.zig`'s
//! config route.
//!
//! Construction (`Construction(b, G).create`; `openMlx` on MLX), in order:
//!   1. the kernels (math `.kernels`): `exl3_kernel_ops.acceptAtStartup`
//!      checks the registry (every text against the pinned manifest), builds
//!      every kernel on the device, runs and judges the device self-check
//!      plan, installs the accepted kernels in the backend's launcher and
//!      builds the GEMV route; the MLX cache is cleared after it, so the
//!      admission below reads the box without the self-check's buffers;
//!   2. the arm (`deepseek_v41_arm.Arm`): config, bank, admission, the stream
//!      at the admitted rows on MLX slot memory, the hook over the binding's
//!      math (the kernels' GEMV bound once: `Accepted.gemvRoute`);
//!   3. the banks (math `.kernels`): every bank the hook bound (base and
//!      transient, per layer) holds what the kernels read; the arm runs the
//!      same check over every bank once more at the phase change, before it
//!      counts as grown (`Arm.grown_check`: the grown banks);
//!   4. the decode's residents (a decode with `open`: the DSpark loop's trunk,
//!      draft head and Engram rows; the Engram token map is the model
//!      directory's `engram_token_map_file`, `engram-token-map.u32`);
//!   5. the session (`deepseek_v41_serve.Session`), released in reverse: the
//!      decode's residents, the arm, the kernels (after their stream drained),
//!      the backend.
//!
//! Per request (the DSpark adapter's): the prompt runs in forwards of 8 rows
//! (the DIG-X prefill route is not bound yet); the last request's target
//! state and draft windows (its KV cache) stay live until the next request's
//! prefill or the engine's release.
//!
//! Every refusal is named at construction: a backend without a method the
//! routes call does not compile (`requireBackend` names it); the kernels
//! refuse a text or manifest that is not the pinned one (TextSha256Mismatch,
//! ManifestNotPinned, LanePinMismatch, ...), a stream that is not a GPU
//! stream, a kernel MLX cannot create, and a device self-check plan with a
//! failure (SelfCheckFailed: the kernel / check / site named); a bank that is
//! not the kernels' layout (RouteInput: the layer, bank and kernel input
//! named; at the phase change the refusal is logged and every later request
//! is refused); the trunk's own refusals (weights; the Engram rows, e.g.
//! EngramTokenMap when the model directory has no exported token map;
//! routes).

const std = @import("std");
const mlx = @import("mlx.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const xk = @import("exl3_kernels.zig");
const xo = @import("exl3_kernel_ops.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const serve = @import("deepseek_v41_serve.zig");
const dsp = @import("deepseek_v41_dspark_serve.zig");
const integration = @import("dsv41_integration.zig");

const log = std.log.scoped(.dsv41);

/// The routed experts' math of a binding.
pub const Math = enum {
    /// zeros of the math's output shapes (`deepseek_v41_arm.StandInMath`)
    stand_in,
    /// the exact tier's op chain around the kernels' EXL3 decode GEMV
    /// (`deepseek_v41_experts.EagerChain` over `KernelGemv`)
    kernels,
};

pub const Binding = struct {
    /// The receipt's `decode_binding`.
    tag: arm_mod.DecodeBinding,
    /// The decode seam over arm `A`: `init(seed, rows, max_cycles)`,
    /// `begin(cfg)`, `prefill(arm, g, prompt)`, `cycle(arm, g, a, out)`,
    /// `stats()`; `open(a, io, arm, g, token_map, diag)` and `deinit(g)` when
    /// it owns residents.
    decode: fn (type) type,
    math: Math,
};

pub const stand_in: Binding = .{ .tag = .stand_in, .decode = arm_mod.StandIn, .math = .stand_in };
pub const dspark: Binding = .{ .tag = .dspark, .decode = dsp.Dspark, .math = .kernels };

/// What the server constructs (`deepseek_v41_arm.serving_decode`).
pub const serving: Binding = switch (arm_mod.serving_decode) {
    .stand_in => stand_in,
    .dspark => dspark,
};

/// The kernels' GEMV route as `EagerChain`'s `Gemv` over backend `G`:
/// `experts.MlxGemv` on MLX, the same shape on any other backend.
pub fn KernelGemv(comptime G: type) type {
    if (G == ops.MlxOps) return xp.MlxGemv;
    return struct {
        ctx: *const anyopaque,
        project_fn: *const fn (ctx: *const anyopaque, g: *G, k: u32, out_dim: u32, xh: G.T, ids: G.T, code: G.T) anyerror!G.T,

        pub fn project(self: @This(), g: *G, k: u32, out_dim: u32, xh: G.T, ids: G.T, code: G.T) !G.T {
            return self.project_fn(self.ctx, g, k, out_dim, xh, ids, code);
        }
    };
}

/// The arm's routed-expert math under binding `b`.
pub fn MathOf(comptime b: Binding, comptime G: type) type {
    return switch (b.math) {
        .stand_in => arm_mod.StandInMath(G),
        .kernels => xp.EagerChain(G, KernelGemv(G)),
    };
}

/// Compile time: backend `G` carries every method the binding's routes call,
/// with the contract's types (the missing one is named).
pub fn requireBackend(comptime G: type) void {
    if (integration.missing(G)) |m| @compileError("dsv41 binding: the backend " ++ @typeName(G) ++ " lacks " ++ m);
}

fn refuse(diag: *arm_mod.Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    const s = std.fmt.bufPrint(&diag.buf, fmt, args) catch diag.buf[0..];
    diag.len = s.len;
    return err;
}

/// Step 1: the kernels, accepted once before the arm allocates
/// (`exl3_kernel_ops.acceptAtStartup`); the kernels' message in `diag`.
pub fn acceptKernels(comptime G: type, a: std.mem.Allocator, g: *G, opts: xo.StartupOptions, diag: *arm_mod.Diag) !*xo.Accepted(G) {
    var kd: xk.Diag = .{};
    return xo.acceptAtStartup(G, a, g, opts, &kd) catch |e| return refuse(diag, e, "kernels: {s}", .{kd.message()});
}

/// A bank's gate / up / down arrays are what the kernels read.
pub fn checkBank(comptime G: type, g: *G, reg: *const xk.Registry, bank: xp.BankArraysOf(G.T), diag: *xk.Diag) xo.Refusal!void {
    inline for (.{ .{ xo.Proj.gate, bank.gate }, .{ xo.Proj.up, bank.up }, .{ xo.Proj.down, bank.down } }) |pb| {
        const p = pb[1];
        try xo.checkBank(G, g, reg, pb[0], .{ .code = p.code, .rout = p.rout, .rin = p.rin }, diag);
    }
}

/// Step 3: every bank arm `arm`'s hook has bound (base and transient from
/// construction; the grown banks too after the phase change) is the kernels'
/// layout.
pub fn checkArmBanks(comptime A: type, arm: *A, g: *A.Backend, reg: *const xk.Registry, diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        checkBank(A.Backend, g, reg, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @enumFromInt(kind)), kd.message() });
    };
}

/// Arm `A`'s grown check over the registry at `ctx` (`Arm.grown_check`): the
/// phase change's banks, once; a refusal is logged by name.
pub fn GrownBanks(comptime A: type) type {
    return struct {
        pub fn check(ctx: *const anyopaque, arm: *A, g: *A.Backend) anyerror!void {
            const reg: *const xk.Registry = @ptrCast(@alignCast(ctx));
            var diag: arm_mod.Diag = .{};
            checkArmBanks(A, arm, g, reg, &diag) catch |e| {
                log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
                return e;
            };
        }
    };
}

/// The Engram token map's home in the model directory: the converter's output
/// (`exl3/runtime/export_dsv41_engram_token_map.py --bank <dir> --out
/// <dir>/engram-token-map.u32`, its `.json` sidecar beside it). A top-level
/// file: a bank's `engram/` may be a link into another bank's directory.
pub const engram_token_map_file = "engram-token-map.u32";

/// Binding `b`'s construction over backend `G` (MLX serving; the trace
/// backend in host tests, where no decode with residents binds).
pub fn Construction(comptime b: Binding, comptime G: type) type {
    comptime requireBackend(G);
    return struct {
        const Self = @This();
        pub const M = MathOf(b, G);
        pub const A = arm_mod.Arm(G, M);
        pub const D = b.decode(A);
        pub const S = serve.Session(A, D);
        const Kernels = if (b.math == .kernels) *xo.Accepted(G) else void;

        session: S,
        g: G,
        kernels: Kernels,

        /// Steps 1-5 on backend `g` (owned from the call: released with the
        /// session, or here on a refusal); `startup` is the kernels' device.
        pub fn create(a: std.mem.Allocator, io: std.Io, g: G, startup: xo.StartupOptions, arm_opt: arm_mod.Options, opts: serve.Options, diag: *arm_mod.Diag) !*Self {
            var backend = g;
            const self = a.create(Self) catch |e| {
                backend.deinit();
                return e;
            };
            errdefer a.destroy(self);
            self.g = backend;
            errdefer self.g.deinit();
            if (b.math == .kernels) {
                self.kernels = try acceptKernels(G, a, &self.g, startup, diag);
                if (G == ops.MlxOps) _ = mlx.mlx_clear_cache();
            } else self.kernels = {};
            errdefer if (b.math == .kernels) self.dropKernels();
            const arm = try A.init(a, io, &self.g, self.mathArg(), arm_opt, diag);
            errdefer arm.deinit();
            if (b.math == .kernels) {
                try checkArmBanks(A, arm, &self.g, &self.kernels.reg, diag);
                arm.grown_check = .{ .ctx = &self.kernels.reg, .check = GrownBanks(A).check };
            }
            self.session = .{ .a = a, .io = io, .arm = arm, .g = &self.g, .decode = D.init(0, opts.depth + 1, 0), .opts = opts, .release = release };
            if (@hasDecl(D, "bind")) self.session.decode.bind(&self.g);
            if (@hasDecl(D, "open")) {
                if (G != ops.MlxOps) @compileError("dsv41 binding: " ++ @typeName(D) ++ " opens its residents on MLX only");
                const map = try std.fmt.allocPrint(a, "{s}/" ++ engram_token_map_file, .{arm.model_dir});
                defer a.free(map);
                var vd: v41.Diag = .{};
                self.session.decode.open(a, io, arm, &self.g, map, &vd) catch |e| return refuse(diag, e, "trunk: {s}", .{vd.message()});
            }
            return self;
        }

        fn mathArg(self: *const Self) if (b.math == .kernels) KernelGemv(G) else void {
            return if (b.math == .kernels) self.kernels.gemvRoute(KernelGemv(G)) else {};
        }

        /// The kernels go after the last launch drained.
        fn dropKernels(self: *Self) void {
            if (G == ops.MlxOps) _ = mlx.mlx_synchronize(self.g.s);
            self.kernels.deinit(&self.g);
        }

        fn release(s: *S) void {
            const self: *Self = @fieldParentPtr("session", s);
            if (@hasDecl(D, "deinit")) s.decode.deinit(&self.g);
            s.arm.deinit();
            if (b.math == .kernels) self.dropKernels();
            self.g.deinit();
            s.a.destroy(self);
        }
    };
}

/// The served engine: `serving`'s binding on MLX. Refused by name while it is
/// the stand-in, before anything is opened.
pub fn openServing(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, stream: mlx.mlx_stream, arm_opt: arm_mod.Options, opts: serve.Options, diag: *arm_mod.Diag) !serve.Engine {
    if (serving.tag == .stand_in) return error.Dsv41DecodeNotBound;
    return openMlx(serving, a, io, model_dir, stream, arm_opt, opts, diag);
}

/// Binding `b`'s engine on MLX `stream`: the kernels on that stream, the arm's
/// slot banks as MLX arrays (the gates open the bindings directly).
pub fn openMlx(comptime b: Binding, a: std.mem.Allocator, io: std.Io, model_dir: []const u8, stream: mlx.mlx_stream, arm_opt: arm_mod.Options, opts: serve.Options, diag: *arm_mod.Diag) !serve.Engine {
    var o = arm_opt;
    o.model_dir = model_dir;
    o.slot_memory = .{ .mlx = stream };
    const c = try Construction(b, ops.MlxOps).create(a, io, try ops.MlxOps.init(a, stream), .{ .device = .{ .stream = stream } }, o, opts, diag);
    return c.session.engine();
}

// ── Tests (host: the trace backend, the kernels on the stub device; no MLX array) ──

const testing = std.testing;
const TraceOps = ops.TraceOps;

/// A bank of `cap` slots in the kernels' layout (the bank of record's
/// geometry: hidden 5120, inter 2304, K 3), as trace inputs.
fn kernelBank(g: *TraceOps, cap: c_int) !xp.BankArraysOf(u32) {
    const proj = struct {
        fn f(t: *TraceOps, n: c_int, in: c_int, out: c_int) !xp.ProjOf(u32) {
            return .{
                .code = try t.input(&.{ n, @divExact(in, 16), @divExact(out, 16), 48 }, .int16),
                .rout = try t.input(&.{ n, out }, .float16),
                .rin = try t.input(&.{ n, in }, .float16),
            };
        }
    }.f;
    return .{ .gate = try proj(g, cap, 5120, 2304), .up = try proj(g, cap, 5120, 2304), .down = try proj(g, cap, 2304, 5120) };
}

test "dsv41 bind: the server binds the stand-in until the flip; the DSpark binding carries the adapter and the kernels' math" {
    const MlxDspark = Construction(dspark, ops.MlxOps);
    comptime {
        std.debug.assert(serving.tag == arm_mod.serving_decode);
        std.debug.assert(MlxDspark.M == xp.EagerChain(ops.MlxOps, xp.MlxGemv));
        std.debug.assert(MlxDspark.D == dsp.Dspark(arm_mod.Arm(ops.MlxOps, xp.EagerChain(ops.MlxOps, xp.MlxGemv))));
        std.debug.assert(Construction(stand_in, ops.MlxOps).M == arm_mod.StandInMath(ops.MlxOps));
        std.debug.assert(Construction(stand_in, ops.MlxOps).D == arm_mod.StandIn(arm_mod.Arm(ops.MlxOps, arm_mod.StandInMath(ops.MlxOps))));
    }
    var diag: arm_mod.Diag = .{};
    const s: mlx.mlx_stream = .{ .ctx = null };
    if (serving.tag == .stand_in)
        try testing.expectError(error.Dsv41DecodeNotBound, openServing(testing.allocator, testing.io, "/nonexistent", s, .{ .model_dir = "", .baseline_bytes = null, .slot_memory = .host }, .{}, &diag));
}

test "dsv41 bind: the kernels accepted at startup give the served math the kernels' GEMV (stub device, trace backend)" {
    const a = testing.allocator;
    var g = TraceOps.init(a);
    defer g.deinit();
    var diag: arm_mod.Diag = .{};
    const acc = try acceptKernels(TraceOps, a, &g, .{ .device = .{ .stub = .{} } }, &diag);
    defer acc.deinit(&g);
    try testing.expect(acc.report.results.items.len >= 100);
    try testing.expectEqual(@as(usize, 0), acc.report.failures());
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    const math = MathOf(dspark, TraceOps).init(acc.gemvRoute(KernelGemv(TraceOps)), &c);
    try testing.expectEqual(@as(u32, 5120), math.hidden);
    try testing.expectEqual(@as(u32, 2304), math.inter);
    const bank = try kernelBank(&g, 150);
    const x = try g.input(&.{ 3, 5120 }, .bfloat16);
    const ids = try g.input(&.{3}, .uint32);
    const n0 = g.nodes.items.len;
    const h = try math.gateUp(&g, x, ids, bank.gate, bank.up);
    const y = try math.down(&g, h, ids, bank.down);
    const sh = g.shapeOf(h);
    const sy = g.shapeOf(y);
    try testing.expectEqualSlices(c_int, &.{ 3, 2304 }, sh.slice());
    try testing.expectEqualSlices(c_int, &.{ 3, 5120 }, sy.slice());
    // The chain's only custom kernels are the GEMV launches: gate, up (out
    // 2304), then down (out 5120), f32 rows.
    var outs: std.ArrayList(c_int) = .empty;
    defer outs.deinit(a);
    for (g.nodes.items[n0..]) |nd| if (nd.op == .kernel) {
        try testing.expect(nd.dtype == .float32 and nd.shape.n == 2 and nd.shape.d[0] == 3);
        try outs.append(a, nd.shape.d[1]);
    };
    try testing.expectEqualSlices(c_int, &.{ 2304, 2304, 5120 }, outs.items);
    // The route takes only what the bank of record holds.
    const xh = try g.input(&.{ 3, 5120 }, .float32);
    try testing.expectError(error.GemvKNotRegistered, math.gemv.project(&g, 2, 2304, xh, ids, bank.gate.code));
    try testing.expectError(error.GemvOutDim, math.gemv.project(&g, 3, 1280, xh, ids, bank.gate.code));
}

test "dsv41 bind: the kernels' acceptance refuses by name, its message in the arm's diag" {
    const a = testing.allocator;
    var g = TraceOps.init(a);
    defer g.deinit();
    var diag: arm_mod.Diag = .{};
    // A failing device self-check names its kernel and check.
    try testing.expectError(error.SelfCheckFailed, acceptKernels(TraceOps, a, &g, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .dsv41_exl3_mul1h_k3_5120, .check = .decode_table } } } }, &diag));
    try testing.expect(std.mem.startsWith(u8, diag.message(), "kernels: "));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "dsv41_exl3_mul1h_k3_5120 decode_table") != null);
    // A kernel text that is not the pinned manifest's.
    var texts = xk.embedded;
    const k = xk.Kernel.dsv41_exl3_mul1h_k3_2304;
    const bad = try a.dupeSentinel(u8, xk.embedded.sources[@backingInt(k)], 0);
    defer a.free(bad);
    bad[bad.len / 2] ^= 0x01;
    texts.sources[@backingInt(k)] = bad;
    try testing.expectError(error.TextSha256Mismatch, acceptKernels(TraceOps, a, &g, .{ .device = .{ .stub = .{} }, .texts = &texts }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), @tagName(k)) != null);
    // A manifest that is not the pinned one.
    const zero_pin: [64]u8 = @splat('0');
    try testing.expectError(error.ManifestNotPinned, acceptKernels(TraceOps, a, &g, .{ .device = .{ .stub = .{} }, .pin = &zero_pin }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "is not the pinned") != null);
}

test "dsv41 bind: a bank that is not the kernels' layout is refused, by kernel input" {
    const a = testing.allocator;
    var g = TraceOps.init(a);
    defer g.deinit();
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, null);
    defer reg.deinit();
    var kd: xk.Diag = .{};
    const ok = try kernelBank(&g, 150);
    try checkBank(TraceOps, &g, &reg, ok, &kd);
    var f32_rout = ok;
    f32_rout.up.rout = try g.input(&.{ 150, 2304 }, .float32);
    try testing.expectError(error.RouteInput, checkBank(TraceOps, &g, &reg, f32_rout, &kd));
    try testing.expect(std.mem.indexOf(u8, kd.message(), "q3_exl3_prep_gu_epi input rg") != null);
    var swapped = ok;
    swapped.down = ok.gate;
    try testing.expectError(error.RouteInput, checkBank(TraceOps, &g, &reg, swapped, &kd));
    try testing.expect(std.mem.indexOf(u8, kd.message(), "dsv41_exl3_mul1h_k3_5120 input code") != null);
    const wide = try kernelBank(&g, 4097);
    try testing.expectError(error.RouteInput, checkBank(TraceOps, &g, &reg, wide, &kd));
    try testing.expect(std.mem.indexOf(u8, kd.message(), "4097 slots") != null);
}

/// The kernels' math over the stand-in decode: the construction's steps 1-3
/// on the trace backend.
const kernels_stand_in: Binding = .{ .tag = .stand_in, .decode = arm_mod.StandIn, .math = .kernels };

test "dsv41 bind: the construction refuses an arm whose banks the kernels cannot read, and releases what it built" {
    const tm = try arm_mod.TestModel.create(true);
    defer tm.destroy();
    var diag: arm_mod.Diag = .{};
    // The mini model's bank (hidden 64): the kernels accept (stub device), the
    // arm builds, the bank check refuses its first bank; testing.allocator
    // proves every step before it released.
    try testing.expectError(error.RouteInput, Construction(kernels_stand_in, TraceOps).create(testing.allocator, testing.io, TraceOps.init(testing.allocator), .{ .device = .{ .stub = .{} } }, tm.options(), .{}, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "kernels: layer 0 base bank: exl3 kernel ops: dsv41_exl3_mul1h_k3_2304 input code") != null);
    // A failing self-check refuses before the arm opens anything.
    try testing.expectError(error.SelfCheckFailed, Construction(kernels_stand_in, TraceOps).create(testing.allocator, testing.io, TraceOps.init(testing.allocator), .{ .device = .{ .stub = .{ .fail = .{ .kernel = .q3rc_router_tail, .check = .f64 } } } }, tm.options(), .{}, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_router_tail f64") != null);
}

test "dsv41 bind: the arm checks its banks at the phase change; a refusal leaves it ungrown and refuses every later growth" {
    const tm = try arm_mod.TestModel.create(true);
    defer tm.destroy();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: arm_mod.Diag = .{};
    const A = arm_mod.Arm(TraceOps, arm_mod.StandInMath(TraceOps));
    const arm = try A.init(testing.allocator, testing.io, &g, {}, tm.options(), &diag);
    defer arm.deinit();
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, null);
    defer reg.deinit();
    // The mini bank (hidden 64) is not the kernels' layout: the check the
    // binding installs refuses the grown arm's banks.
    arm.grown_check = .{ .ctx = &reg, .check = GrownBanks(A).check };
    try testing.expectError(error.RouteInput, arm.grow(&g));
    try testing.expect(!arm.grown);
    try testing.expectError(error.AlreadyGrown, arm.grow(&g));
    try testing.expect(!arm.grown);
    // Every bound bank is checked, as at construction.
    try testing.expectError(error.RouteInput, checkArmBanks(A, arm, &g, &reg, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "kernels: layer 0 base bank: exl3 kernel ops: dsv41_exl3_mul1h_k3_2304 input code") != null);
}

test "dsv41 bind: the stand-in binding constructs on the trace backend, serves a request and releases everything" {
    const tm = try arm_mod.TestModel.create(true);
    defer tm.destroy();
    var diag: arm_mod.Diag = .{};
    const C = Construction(stand_in, TraceOps);
    const c = C.create(testing.allocator, testing.io, TraceOps.init(testing.allocator), .{ .device = .{ .stub = .{} } }, tm.options(), .{}, &diag) catch |e| {
        std.debug.print("dsv41 bind: {s}\n", .{diag.message()});
        return e;
    };
    const e = c.session.engine();
    defer e.deinit();
    const Collect = struct {
        n: u32 = 0,
        fn push(ctx: *anyopaque, _: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.n += 1;
        }
    };
    var out: Collect = .{};
    try e.begin(.{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 6, .stop_ids = &.{} });
    while (try e.step(.{ .ctx = &out, .push = Collect.push }) == null) {}
    var run = try e.end();
    defer run.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 6), out.n);
    try testing.expectEqual(@as(usize, 6), run.generated.len);
    try testing.expect(c.session.arm.grown);
}

test "dsv41 bind: a served request's receipt names the DSpark decode and the rows its loop verifies" {
    const a = testing.allocator;
    const mdl = @import("deepseek_v41_model.zig");
    const routes = @import("deepseek_v41_routes.zig");
    const cell = @import("deepseek_v41_cell.zig");
    const A = arm_mod.Arm(TraceOps, arm_mod.StandInMath(TraceOps));
    const D = dsp.Dspark(A);
    const tm = try arm_mod.TestModel.create(true);
    defer tm.destroy();
    const mini = try mdl.Mini.init();
    defer mini.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    var diag: arm_mod.Diag = .{};
    const arm = try A.init(a, testing.io, &g, {}, tm.options(), &diag);
    defer arm.deinit();
    var lookup: mdl.SpecLookup = .{ .g = &g, .spec = mini.spec };
    const model = try D.Loop.M.init(a, &g, mini.c, try routes.parse(&.{}, null), &lookup, &mini.src);
    defer model.deinit(&g);
    const head = try D.Loop.H.init(a, &g, mini.c, .{}, &lookup);
    defer head.deinit(&g);
    const noRelease = struct {
        fn f(_: *serve.Session(A, D)) void {}
    }.f;
    var session: serve.Session(A, D) = .{ .a = a, .io = testing.io, .arm = arm, .g = &g, .decode = D.init(0, 6, 0), .opts = .{}, .release = noRelease };
    try session.decode.attach(a, model, head);
    defer session.decode.deinit(&g);
    const e = session.engine();
    defer e.deinit();
    // The prefill's host reads: k distinct experts per routed row, the pick.
    const Script = struct {
        n: u16,
        k: u16,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const sc: *@This() = @ptrCast(@alignCast(ctx));
            for (out, 0..) |*o, i| o.* = @intCast((i / sc.k + i % sc.k) % sc.n);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return 3;
        }
    };
    var sc: Script = .{ .n = @intCast(mini.c.n_routed_experts), .k = @intCast(mini.c.n_experts_per_tok) };
    g.host_values = .{ .ctx = &sc, .ids = Script.ids, .argmax = Script.argmax };
    try e.begin(.{ .prompt = &.{ 7, 8, 9 }, .max_tokens = 1, .stop_ids = &.{} });
    const Drop = struct {
        fn push(_: *anyopaque, _: u32) void {}
    };
    try testing.expectEqual(@as(?serve.Finish, .length), try e.step(.{ .ctx = &sc, .push = Drop.push }));
    var run = try e.end();
    defer run.deinit(a);
    try testing.expectEqualSlices(u32, &.{3}, run.generated);
    const path = try std.fmt.allocPrint(a, "{s}/bind-1.comparison.json", .{tm.root});
    defer a.free(path);
    var lines: std.Io.Writer.Allocating = .init(a);
    defer lines.deinit();
    try e.receipt(&run, path, &lines.writer);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 20));
    defer a.free(text);
    const Back = struct { kind: []const u8, decode_binding: []const u8, selection: struct { policy: struct { verify_rows: u32 } } };
    const back = try std.json.parseFromSlice(Back, a, text, .{ .ignore_unknown_fields = true });
    defer back.deinit();
    try testing.expectEqualStrings(cell.receipt_kind, back.value.kind);
    try testing.expectEqualStrings("dspark", back.value.decode_binding);
    // The mini head drafts blocks of 2 (depth 5 capped), no lookup extension:
    // 3 rows, the loop's own bound (8 at the tier's depth 5), not depth + 1.
    try testing.expectEqual(session.decode.req.?.loop.max_rows, back.value.selection.policy.verify_rows);
    try testing.expectEqual(@as(u32, 3), back.value.selection.policy.verify_rows);
}

// DSV41_BANK=<the 3.0 bank dir>: the plan G6 constructs at a box baseline, on the CPU (config, bank, admission;
// no slot memory): the slot rows and bytes behind G6's static peak table. [DSV41_BIND_BASELINE_GB=7.755397656]
// [DSV41_BIND_WIRED_GB=3.377741824] (the pass-2 reference receipt's; the window's own come from its guard)
// [DSV41_BIND_ROWS=<forced decode rows>].
test "dsv41 bind: the served plan on the real bank at a box baseline (G6's static table, CPU)" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const gbOf = struct {
        fn f(name: [*:0]const u8, default: f64) !u64 {
            const v = if (std.c.getenv(name)) |x| try std.fmt.parseFloat(f64, std.mem.span(x)) else default;
            return @intFromFloat(@round(v * 1e9));
        }
    }.f;
    var diag: arm_mod.Diag = .{};
    var p = arm_mod.planRows(testing.allocator, testing.io, .{
        .model_dir = dir,
        .baseline_bytes = try gbOf("DSV41_BIND_BASELINE_GB", 7.755397656),
        .wired_bytes = try gbOf("DSV41_BIND_WIRED_GB", 3.377741824),
        .fixed_rows = if (std.c.getenv("DSV41_BIND_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else null,
        .slot_memory = .host,
    }, &diag) catch |e| {
        std.debug.print("dsv41 bind plan: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    const adm = p.plan.admission;
    var widest: u64 = 0;
    for (p.bank.layers) |l| widest = @max(widest, l.logical_bytes);
    // The stream's slot memory: every layer's rows of its record, plus the transient rows of the widest record.
    const slots = struct {
        fn f(bank: *const @import("expert_bank.zig").Bank, rows: u32, transient_record: u64) u64 {
            var sum: u64 = @as(u64, @import("expert_policy.zig").max_route_ids) * transient_record;
            for (bank.layers) |l| sum += @as(u64, rows) * l.logical_bytes;
            return sum;
        }
    }.f;
    const prefill_slots = slots(&p.bank, p.prefill_rows, widest);
    const decode_slots = slots(&p.bank, p.decode_rows, widest);
    std.debug.print("dsv41 bind plan: baseline {d} B, wired {d} B -> prefill {d} / decode {d} rows; slot memory {d} B (prefill) / {d} B (decode); record {d} B; admission final bank {d} B, active bound {d} B, physical bound {d} B, modeled peak {d} B\n", .{
        adm.baseline_bytes, p.inputs.wired_bytes, p.prefill_rows, p.decode_rows, prefill_slots, decode_slots, widest,
        adm.final_bank_bytes, adm.active_bound_bytes, adm.physical_bound_bytes,
        if (p.plan.peak_fill) |pf| pf.modeled_peak_bytes else adm.physical_bound_bytes,
    });
    try testing.expect(p.decode_rows >= p.prefill_rows);
}

/// A window's prompt ids from a reference file: its `prompt` (a DSpark
/// reference) or `prompt_ids` (an AR reference).
fn promptFromFile(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(text);
    const P = struct { prompt: ?[]const u32 = null, prompt_ids: ?[]const u32 = null };
    const p = try std.json.parseFromSlice(P, a, text, .{ .ignore_unknown_fields = true });
    defer p.deinit();
    const ids = p.value.prompt orelse p.value.prompt_ids orelse return error.PromptFileHasNoIds;
    if (ids.len == 0) return error.PromptFileHasNoIds;
    return a.dupe(u32, ids);
}

// Guarded window only (G6: the served path on the bank, the DSpark binding without the flip): the kernels' device
// self-check, the slot banks at the admitted rows, the trunk + draft head + Engram rows resident, one request.
// _GPU_WINDOW_LOCKED=1 DSV41_BIND_MODEL=<model dir, with engram-token-map.u32 and its .json>
// DSV41_BIND_OUT=<receipt path; must not exist> MTPLX_DSV41_BOX_BASELINE_GB=<the guard's baseline>
// DSV41_BIND_PROMPT=<json: `prompt` (a DSpark reference) or `prompt_ids` (an AR reference); unset: seeded ids>
// [DSV41_BIND_ROWS=<forced decode rows; unset: the admission's own>] [DSV41_BIND_PROMPT_TOKENS=64 (seeded only)]
// [DSV41_BIND_MAX_TOKENS=100]
test "dsv41 bind: the DSpark binding constructs on the bank and serves one request (the served path's GPU gate)" {
    const envOf = struct {
        fn f(name: [*:0]const u8) ?[]const u8 {
            return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
        }
    }.f;
    const model_dir = envOf("DSV41_BIND_MODEL") orelse return error.SkipZigTest;
    const out = envOf("DSV41_BIND_OUT") orelse return error.SkipZigTest;
    if (envOf("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const a = testing.allocator;
    const io = testing.io;
    const cell = @import("deepseek_v41_cell.zig");
    const baseline_gb = envOf("MTPLX_DSV41_BOX_BASELINE_GB");
    const max_tokens = if (envOf("DSV41_BIND_MAX_TOKENS")) |v| try std.fmt.parseInt(u32, v, 10) else 100;
    const prompt = if (envOf("DSV41_BIND_PROMPT")) |path| try promptFromFile(a, io, path) else blk: {
        const n = if (envOf("DSV41_BIND_PROMPT_TOKENS")) |v| try std.fmt.parseInt(u32, v, 10) else 64;
        const ids = try a.alloc(u32, n);
        var rng = std.Random.DefaultPrng.init(0x5eed);
        for (ids) |*t| t.* = rng.random().uintLessThan(u32, 100_000);
        break :blk ids;
    };
    defer a.free(prompt);
    const prompt_sha = try cell.idsSha256(a, prompt);
    std.debug.print("DSV41_BIND_PROMPT {{\"tokens\": {d}, \"sha256\": \"{s}\", \"source\": \"{s}\"}}\n", .{ prompt.len, &prompt_sha, envOf("DSV41_BIND_PROMPT") orelse "seeded" });

    var diag: arm_mod.Diag = .{};
    const arm_opt: arm_mod.Options = .{
        .model_dir = model_dir,
        .baseline_bytes = if (baseline_gb) |v| @intFromFloat(@round(try std.fmt.parseFloat(f64, v) * 1e9)) else null,
        .fixed_rows = if (envOf("DSV41_BIND_ROWS")) |v| try std.fmt.parseInt(u32, v, 10) else null,
        .slot_memory = .host,
    };
    // The plan at the box now (CPU: config, bank, admission; no slot memory), for the window's child cap.
    {
        var planned = arm_mod.planRows(a, io, arm_opt, &diag) catch |err| {
            std.debug.print("DSV41_BIND_REFUSED {s}: {s}\n", .{ @errorName(err), diag.message() });
            return err;
        };
        defer planned.bank.deinit();
        const adm = planned.plan.admission;
        const process_bound = adm.physical_bound_bytes -| adm.baseline_bytes;
        std.debug.print("DSV41_BIND_PLAN {{\"prefill_rows\": {d}, \"decode_rows\": {d}, \"slot_bank_bytes\": {d}, \"active_bound_bytes\": {d}, \"physical_bound_bytes\": {d}, \"baseline_bytes\": {d}, \"process_bound_bytes\": {d}, \"child_cap_bytes\": {d}, \"modeled_peak_physical_bytes\": {d}}}\n", .{
            planned.prefill_rows,   planned.decode_rows,       adm.final_bank_bytes, adm.active_bound_bytes, adm.physical_bound_bytes,
            adm.baseline_bytes,     process_bound,             process_bound + 3_000_000_000,
            if (planned.plan.peak_fill) |pf| pf.modeled_peak_bytes else adm.physical_bound_bytes,
        });
    }

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const t0 = std.Io.Timestamp.now(io, .boot);
    const e = openMlx(dspark, a, io, model_dir, s, arm_opt, .{}, &diag) catch |err| {
        std.debug.print("DSV41_BIND_REFUSED {s}: {s}\n", .{ @errorName(err), diag.message() });
        return err;
    };
    defer e.deinit();
    const built_s = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e9;
    const fp = arm_mod.footprint();
    std.debug.print("DSV41_BIND_BUILT {{\"construction_s\": {d:.1}, \"footprint_bytes\": {d}, \"footprint_peak_bytes\": {d}}}\n", .{ built_s, fp.now, fp.peak });

    // One request as the scheduler serves it: the serve defaults (typical 0.3, depth 5), no stop ids.
    const Count = struct {
        n: u32 = 0,
        fn push(ctx: *anyopaque, _: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.n += 1;
        }
    };
    var sink: Count = .{};
    try e.begin(.{ .prompt = prompt, .max_tokens = max_tokens, .stop_ids = &.{} });
    while (try e.step(.{ .ctx = &sink, .push = Count.push }) == null) {}
    var run = try e.end();
    defer run.deinit(a);
    var line_buf: [256]u8 = undefined;
    var line: std.Io.Writer = .fixed(&line_buf);
    try serve.writeRequestLine(&run, &line);
    std.debug.print("{s}", .{line.buffered()});
    var lines: std.Io.Writer.Allocating = .init(a);
    defer lines.deinit();
    try e.receipt(&run, out, &lines.writer);
    std.debug.print("{s}", .{lines.written()});
    std.debug.print("DSV41_BIND_DONE {{\"generated\": {d}, \"cycles\": {d}, \"prompt_eval_time_s\": {d:.3}, \"decode_wall_s\": {d:.3}, \"process_footprint_peak_bytes\": {d}, \"mlx_peak_bytes\": {d}}}\n", .{
        run.generated.len, run.stats.cycles, run.prompt_eval_s, run.decode_wall_s, run.footprint.peak, run.mlx_peak_bytes orelse 0,
    });
    // The receipt is a DSpark measurement: the math is real.
    const text = try std.Io.Dir.cwd().readFileAlloc(io, out, a, .limited(1 << 20));
    defer a.free(text);
    const Back = struct { decode_binding: []const u8, measurement_valid: bool };
    const back = try std.json.parseFromSlice(Back, a, text, .{ .ignore_unknown_fields = true });
    defer back.deinit();
    try testing.expectEqualStrings("dspark", back.value.decode_binding);
    try testing.expect(back.value.measurement_valid);
    try testing.expectEqual(@as(usize, max_tokens), run.generated.len);
}
