//! Kernel routes of the DeepSeek-V4.1 tier: the kernel side of the model's lever routes.
//! One construction-time route per lever family over the pinned registry (`exl3_kernels`):
//! built once (static inputs, per-M launch tables, the weights it binds checked against the
//! kernel signature and refused by name), then called with the per-call arrays only. A call
//! launches exactly what the Python lane's kernel call launches (inputs in the kernel's order,
//! the lane's grid / threadgroup / template) and returns the same outputs.
//!
//! Routes are generic over the model's graph backend `G`, which provides `T`, `shapeOf`
//! (a value with `slice()`), `dtypeOf`, `hostArray`, `keep`, `release`, `reshape`, `astype`
//! and one launch:
//!     launch(g: *G, k: Kernel, inputs: []const G.T, cfg: *const LaunchConfig, out: []G.T) !void
//! (a real backend: `Bound.apply`, the outputs owned by the backend's scope; a trace backend:
//! nodes of `cfg`'s output shapes and dtypes).

const std = @import("std");
const mlx = @import("mlx.zig");
const xk = @import("exl3_kernels.zig");
const selfcheck = @import("exl3_selfcheck.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Entry = xk.Entry;
const Vars = xk.Vars;
const LaunchConfig = xk.LaunchConfig;
const Dtype = mlx.mlx_dtype;

pub const Refusal = error{
    /// A bound array (weight, bias, bank) whose dtype or shape is not the kernel's signature.
    RouteInput,
    /// A template the registry does not carry (its self-check never ran that instantiation).
    TemplateNotRegistered,
    /// A row count outside a plan kernel's table (the caller's phase route owns those rows).
    RowsOutOfPlan,
    /// A routed row naming a slot outside its call's bank.
    SlotOutOfBank,
};

fn refuse(diag: ?*xk.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

fn argOf(e: *const Entry, comptime name: []const u8) *const xk.Arg {
    for (e.inputs) |*a| if (std.mem.eql(u8, a.name, name)) return a;
    unreachable;
}

fn dims(comptime G: type, g: *G, x: G.T) Shape {
    const sh = g.shapeOf(x);
    return Shape.of(sh.slice());
}

/// A shape copied out of the backend (the backend's own shape value may be a temporary).
pub const Shape = struct {
    n: u8 = 0,
    d: [xk.max_rank + 2]c_int = @splat(0),

    pub fn of(s: []const c_int) Shape {
        var r: Shape = .{ .n = @intCast(s.len) };
        @memcpy(r.d[0..s.len], s);
        return r;
    }

    pub fn slice(self: *const Shape) []const c_int {
        return self.d[0..self.n];
    }
};

/// `x` has input `name`'s dtype and its shape at `vars` (construction time, once).
fn expectInput(comptime G: type, g: *G, e: *const Entry, comptime name: []const u8, x: G.T, vars: *const Vars, diag: ?*xk.Diag) Refusal!void {
    const a = argOf(e, name);
    const got = dims(G, g, x);
    const dt = g.dtypeOf(x);
    var ok = dt == a.dtype and got.n == a.shape.len;
    if (ok) for (a.shape, got.slice()) |dim, v| {
        ok = ok and dim.eval(vars) == @as(u64, @intCast(v));
    };
    if (!ok) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} input {s} is {t} {any}, the kernel reads {t} [{f}]", .{ e.kernel, name, dt, got.slice(), a.dtype, shapeFmt(a, vars) });
}

fn shapeFmt(a: *const xk.Arg, vars: *const Vars) std.fmt.Alt(ShapeAt, ShapeAt.format) {
    return .{ .data = .{ .a = a, .vars = vars } };
}

const ShapeAt = struct {
    a: *const xk.Arg,
    vars: *const Vars,

    fn format(s: ShapeAt, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (s.a.shape, 0..) |dim, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", dim.eval(s.vars) });
    }
};

/// One launch of a rule kernel at `vars`.
fn launchRule(comptime G: type, g: *G, e: *const Entry, vars: *const Vars, inputs: []const G.T, out: []G.T) !void {
    const cfg = xk.launchFor(e, vars, null) catch unreachable;
    try g.launch(e.kernel, inputs, &cfg, out[0..cfg.n_out]);
}

fn rowsOf(comptime G: type, g: *G, x: G.T, axis: usize) u64 {
    return @intCast(dims(G, g, x).slice()[axis]);
}

/// The static inputs of `e` (role static: the lane's own constants, from the manifest's values),
/// built once and kept; `at(i)` is input i's array (undefined for non-static inputs).
fn Statics(comptime G: type) type {
    return struct {
        const Self = @This();
        arrays: [16]G.T = undefined,
        mask: u16 = 0,

        fn init(g: *G, e: *const Entry) !Self {
            var s: Self = .{};
            errdefer s.deinit(g);
            for (e.inputs, 0..) |*a, i| {
                if (a.role != .static) continue;
                var buf: [1024]u8 = undefined;
                const shape, const bytes = staticBytes(a, &buf);
                s.arrays[i] = g.keep(try g.hostArray(bytes, shape.slice(), a.dtype));
                s.mask |= @as(u16, 1) << @intCast(i);
            }
            return s;
        }

        fn deinit(s: *Self, g: *G) void {
            for (0..16) |i| {
                if (s.mask & (@as(u16, 1) << @intCast(i)) != 0) g.release(s.arrays[i]);
            }
            s.mask = 0;
        }
    };
}

/// A static input's shape and bytes (little endian): the manifest's `values`, or zeros.
fn staticBytes(a: *const xk.Arg, buf: *[1024]u8) struct { Shape, []const u8 } {
    const vars: Vars = .initFill(0);
    var shape: Shape = .{ .n = @intCast(a.shape.len) };
    var n: usize = 1;
    for (a.shape, 0..) |dim, i| {
        shape.d[i] = @intCast(dim.eval(&vars));
        n *= @intCast(shape.d[i]);
    }
    const size = dtypeSize(a.dtype);
    const bytes = buf[0 .. n * size];
    @memset(bytes, 0);
    switch (a.domain.kind) {
        .zeros => {},
        .values => for (0..n) |i| {
            if (a.domain.ints.len > 0) {
                const v = a.domain.ints[i % a.domain.ints.len];
                switch (a.dtype) {
                    .uint32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @intCast(v), .little),
                    .int32 => std.mem.writeInt(i32, bytes[i * 4 ..][0..4], @intCast(v), .little),
                    else => unreachable,
                }
            } else {
                const v = a.domain.floats[i % a.domain.floats.len];
                switch (a.dtype) {
                    .float32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little),
                    else => unreachable,
                }
            }
        },
        else => unreachable,
    }
    return .{ shape, bytes };
}

fn dtypeSize(dt: Dtype) usize {
    return switch (dt) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}

const no_vars: Vars = .initFill(0);

fn rowsVars(rows: u64) Vars {
    var v: Vars = .initFill(0);
    v.set(.rows, rows);
    return v;
}

/// `G` declares `name` as a function (a wrapper backend declares an absent capability as `{}`).
fn declaresFn(comptime G: type, comptime name: []const u8) bool {
    return @hasDecl(G, name) and @typeInfo(@TypeOf(@field(G, name))) == .@"fn";
}

/// A decode kernel's launch at every row count 1..n (the kernel's `rows` bound), built once at a
/// route's construction; a call indexes it by M, the one launch value that varies per call. A
/// backend that prepares launches (`Prepared`, `prepareLaunch`, `launchPrepared`,
/// `releasePrepared`; chosen at compile time) gets each launch's mlx config built here too, so a
/// call builds none; any other backend takes the per-call `launch` (deprecated for decode).
fn RowPlans(comptime G: type, comptime n: usize) type {
    const prepared = declaresFn(G, "prepareLaunch");
    return struct {
        const Self = @This();
        e: *const Entry,
        cfg: [n]LaunchConfig,
        prep: if (prepared) [n]G.Prepared else void,

        /// `site`: a plan kernel's site (rcproj sites; "" for sinkhorn), null for a rule kernel.
        fn init(g: *G, e: *const Entry, site: ?[]const u8, diag: ?*xk.Diag) !Self {
            const b = e.bounds.get(.rows) orelse return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} has no row bound", .{e.kernel});
            if (b[0] != 1 or b[1] != n) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} takes rows {d}..{d}, the route's table 1..{d}", .{ e.kernel, b[0], b[1], n });
            var p: Self = .{ .e = e, .cfg = undefined, .prep = undefined };
            for (&p.cfg, 1..) |*c, m| c.* = xk.launchFor(e, &rowsVars(m), site) catch return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} has no plan at site {?s} rows {d}", .{ e.kernel, site, m });
            if (prepared) {
                var built: usize = 0;
                errdefer for (p.prep[0..built]) |*x| g.releasePrepared(x);
                for (&p.prep, &p.cfg) |*x, *c| {
                    x.* = try g.prepareLaunch(e.kernel, c);
                    built += 1;
                }
            }
            return p;
        }

        fn deinit(p: *Self, g: *G) void {
            if (prepared) for (&p.prep) |*x| g.releasePrepared(x);
        }

        /// The launch at M rows; RowsOutOfPlan outside 1..n (the caller's phase route owns those).
        fn at(p: *const Self, m: u64) Refusal!*const LaunchConfig {
            if (m < 1 or m > n) return error.RowsOutOfPlan;
            return &p.cfg[m - 1];
        }

        fn launch(p: *const Self, g: *G, m: u64, inputs: []const G.T, out: []G.T) !void {
            if (m < 1 or m > n) return error.RowsOutOfPlan;
            const k = m - 1;
            if (prepared) {
                try g.launchPrepared(&p.prep[k], inputs, out[0..p.cfg[k].n_out]);
            } else {
                try g.launch(p.e.kernel, inputs, &p.cfg[k], out[0..p.cfg[k].n_out]);
            }
        }
    };
}

// ── RCTAIL (DSV41_DECODE_RCTAIL = router, hcpremix, sinkhorn) ──

/// router: the MoE gate on q3rc_gate_part (split-K logits) + q3rc_router_tail (sqrt(softplus),
/// + bias, top-6, weights / sum x 1.5): `RouterKernels.run`, decode / verify rows (M <= 8).
pub fn Router(comptime G: type) type {
    return struct {
        const Self = @This();
        part: *const Entry,
        tail: *const Entry,
        part_p: RowPlans(G, 8),
        tail_p: RowPlans(G, 8),
        w: G.T,
        bias: G.T,

        /// `w` the gate weight (bf16 [N, 5120]), `bias` the selection bias (f32 [N]): N 384 = the
        /// verify router (top-6), N 128 = the DSpark draft router (DRAFTRC member router: the
        /// __n128 / __n128_top3 variants, top-3). Another N is refused (RouteInput).
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, bias: G.T, diag: ?*xk.Diag) !Self {
            const n = rowsOf(G, g, w, 0);
            const part, const tail = switch (n) {
                384 => .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) },
                128 => .{ reg.get(.q3rc_gate_part__n128), reg.get(.q3rc_router_tail__n128_top3) },
                else => return refuse(diag, error.RouteInput, "exl3 kernel ops: a router of {d} experts (the registry carries 384 and the draft's 128)", .{n}),
            };
            try expectInput(G, g, part, "w", w, &no_vars, diag);
            try expectInput(G, g, tail, "bias", bias, &no_vars, diag);
            var part_p: RowPlans(G, 8) = try .init(g, part, null, diag);
            errdefer part_p.deinit(g);
            const tail_p: RowPlans(G, 8) = try .init(g, tail, null, diag);
            return .{ .part = part, .tail = tail, .part_p = part_p, .tail_p = tail_p, .w = g.keep(w), .bias = g.keep(bias) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.part_p.deinit(g);
            self.tail_p.deinit(g);
            g.release(self.w);
            g.release(self.bias);
        }

        /// The `_gate_topk_impl` seam at rows <= 8: `run(xf.astype(f32), gweight, gbias)`.
        pub fn gateTopk(self: *const Self, g: *G, xf: G.T) ![2]G.T {
            return self.call(g, try g.astype(xf, .float32));
        }

        /// x [M, 5120] f32 -> (weights [M, top-k] f32, indices [M, top-k] i32): top-6 / top-3.
        pub fn call(self: *const Self, g: *G, x: G.T) ![2]G.T {
            const m = rowsOf(G, g, x, 0);
            var part: [1]G.T = undefined;
            try self.part_p.launch(g, m, &.{ x, self.w }, &part);
            var out: [2]G.T = undefined;
            try self.tail_p.launch(g, m, &.{ part[0], self.bias }, &out);
            return out;
        }
    };
}

/// hcpremix: the HC premix GEMV on q3rc_premix_part + q3rc_premix_fin: `PremixKernels.run`.
pub fn Premix(comptime G: type) type {
    return struct {
        const Self = @This();
        part: *const Entry,
        fin: *const Entry,
        part_p: RowPlans(G, 8),
        fin_p: RowPlans(G, 8),
        w: G.T,

        /// `w` the premix weight (f32 [24, 20480]).
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, diag: ?*xk.Diag) !Self {
            const part = reg.get(.q3rc_premix_part);
            const fin = reg.get(.q3rc_premix_fin);
            try expectInput(G, g, part, "w", w, &no_vars, diag);
            var part_p: RowPlans(G, 8) = try .init(g, part, null, diag);
            errdefer part_p.deinit(g);
            const fin_p: RowPlans(G, 8) = try .init(g, fin, null, diag);
            return .{ .part = part, .fin = fin, .part_p = part_p, .fin_p = fin_p, .w = g.keep(w) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.part_p.deinit(g);
            self.fin_p.deinit(g);
            g.release(self.w);
        }

        /// The `_q3_attnkernel_mm(a, w)` seam at <= 8 rows: a [..., 20480] f32 -> [..., 24] f32
        /// (a 1-d lead stays the kernel's [M, 24]).
        pub fn mm(self: *const Self, g: *G, a: G.T) !G.T {
            const sh = dims(G, g, a);
            const lead = sh.slice()[0 .. sh.n - 1];
            var m: c_int = 1;
            for (lead) |d| m *= d;
            const out = try self.call(g, try g.reshape(a, &.{ m, sh.slice()[sh.n - 1] }));
            if (lead.len == 1) return out;
            var shape: Shape = .of(lead);
            shape.d[shape.n] = @intCast(argOf(self.fin, "part").shape[2].m);
            shape.n += 1;
            return g.reshape(out, shape.slice());
        }

        /// x [M, 20480] f32 -> out [M, 24] f32.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            const m = rowsOf(G, g, x, 0);
            var part: [1]G.T = undefined;
            try self.part_p.launch(g, m, &.{ x, self.w }, &part);
            var out: [1]G.T = undefined;
            try self.fin_p.launch(g, m, &.{part[0]}, &out);
            return out[0];
        }
    };
}

/// SINKHORN_METAL: `_sinkhorn_kernel_apply` as the RCTAIL sinkhorn member rebinds it: up to 32
/// matrices on the 16-lane kernel (`Sinkhorn16.run`, bitwise the stock result), more (prefill
/// chunks) on the stock K3 text (`deepseek_v4._sinkhorn_kernel_apply`).
pub fn Sinkhorn(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_mats = 32;
        e: *const Entry,
        k3: *const Entry,
        plans: RowPlans(G, max_mats),
        nmat: [max_mats]G.T,

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const e = reg.get(.q3dk_sinkhorn16_hc4_it20);
            var s: Self = .{ .e = e, .k3 = reg.get(.mtplx_dsv4_sinkhorn_hc4_it20), .plans = try .init(g, e, "", null), .nmat = undefined };
            errdefer s.plans.deinit(g);
            var built: usize = 0;
            errdefer for (s.nmat[0..built]) |a| g.release(a);
            for (0..max_mats) |i| {
                const n: i32 = @intCast(i + 1);
                s.nmat[i] = g.keep(try g.hostArray(std.mem.asBytes(&n), &.{}, .int32));
                built += 1;
            }
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            for (self.nmat) |a| g.release(a);
        }

        /// comb [..., 4, 4] f32 (n = numel / 16 matrices) -> the Sinkhorn projection, comb's shape.
        pub fn call(self: *const Self, g: *G, comb: G.T) !G.T {
            const sh = dims(G, g, comb);
            var numel: usize = 1;
            for (sh.slice()) |v| numel *= @intCast(v);
            const n = numel / 16;
            const c3 = try g.reshape(comb, &.{ @intCast(n), 4, 4 });
            var out: [1]G.T = undefined;
            if (n <= max_mats) {
                try self.plans.launch(g, n, &.{ c3, self.nmat[n - 1] }, &out);
            } else {
                const count: i32 = @intCast(n);
                const nm = try g.hostArray(std.mem.asBytes(&count), &.{}, .int32);
                try launchRule(G, g, self.k3, &rowsVars(n), &.{ c3, nm }, &out);
            }
            return g.reshape(out[0], sh.slice());
        }
    };
}

// ── RCPROJ (DSV41_DECODE_RCPROJ = mxfp8, woarc) and DRAFT_HEAD = both (the head site) ──

/// The decode-once mxfp8 FMA sites: the four verify projections, the o-LoRA wo_a (8 groups) and
/// the head (DRAFT_HEAD = both: the bf16 head quantized once to mxfp8 gs 32).
pub const RcSite = enum { wq_a, wkv, wq_b, wo_b, woa, head };

/// One site on q3rc_mxfp8_fma at its pinned geometry, a launch plan per M = 1..8: `Kernels.run`.
pub fn RcProj(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        site: RcSite,
        plans: RowPlans(G, max_rows),
        w: G.T,
        scales: G.T,

        /// `w` the packed mxfp8 weight (u32 [G N, K / 4]), `scales` its e8m0 scales (u8 [G N, K / 32]).
        pub fn init(g: *G, reg: *const xk.Registry, site: RcSite, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const e = reg.get(.q3rc_mxfp8_fma);
            const s = e.site(@tagName(site)) orelse unreachable;
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            try expectInput(G, g, e, "scales", scales, &vars, diag);
            return .{ .e = e, .site = site, .plans = try .init(g, e, @tagName(site), diag), .w = g.keep(w), .scales = g.keep(scales) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
            g.release(self.scales);
        }

        /// The `QuantizedLinear.__call__` seam (these linears carry no bias): x [..., K] with
        /// 1..8 rows -> y [..., N].
        pub fn linear(self: *const Self, g: *G, x: G.T) !G.T {
            const sh = dims(G, g, x);
            const lead = sh.slice()[0 .. sh.n - 1];
            var m: c_int = 1;
            for (lead) |d| m *= d;
            const y = try self.call(g, try g.reshape(x, &.{ m, sh.slice()[sh.n - 1] }));
            var shape: Shape = .of(lead);
            shape.d[shape.n] = @intCast(self.plans.cfg[0].out_shapes[0][1]);
            shape.n += 1;
            return g.reshape(y, shape.slice());
        }

        /// x [M, G K] bf16 (row-contiguous), M = 1..8 -> y [M, G N] bf16.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, self.scales, x }, &out);
            return out[0];
        }
    };
}

// ── DRAFTRC proj (DSV41_DECODE_DRAFTRC member proj): the DSpark draft block's rcproj sites ──

/// The draft's rcproj sites (`q3_decode_draftrc_candidate.DRAFT_SHAPES`): the verify sites the
/// draft stages share, and the draft-only main_proj / shared expert.
pub const DraftSite = enum { wq_a, wkv, wq_b, wo_b, woa, main_proj, shared_w13, shared_w2 };

/// One draft site at one x dtype (`DraftProjKernels.run`): bf16 x -> the registered FMA text
/// (q3rc_mxfp8_fma at the verify sites, its plan variant q3rc_mxfp8_fma__draft at the
/// draft-only sites), f32 x -> q3drc_mxfp8_fma_f32x; the output has x's dtype; a plan per M =
/// 1..8 at the site's pinned geometry. The dtype is the call site's (fixed at construction).
pub fn DraftProj(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        site: DraftSite,
        plans: RowPlans(G, max_rows),
        w: G.T,
        scales: G.T,

        /// `x_dtype` bf16 or f32 (the call site's activations), `w` the packed mxfp8 weight (u32
        /// [G N, K / 4]), `scales` its e8m0 scales (u8 [G N, K / 32]).
        pub fn init(g: *G, reg: *const xk.Registry, site: DraftSite, x_dtype: Dtype, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const draft_only = switch (site) {
                .main_proj, .shared_w13, .shared_w2 => true,
                else => false,
            };
            const e = switch (x_dtype) {
                .bfloat16 => reg.get(if (draft_only) .q3rc_mxfp8_fma__draft else .q3rc_mxfp8_fma),
                .float32 => reg.get(.q3drc_mxfp8_fma_f32x),
                else => return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: draft {t} at x {t}: the registry carries bf16 and f32 x", .{ site, x_dtype }),
            };
            const s = e.site(@tagName(site)) orelse unreachable;
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            try expectInput(G, g, e, "scales", scales, &vars, diag);
            return .{ .e = e, .site = site, .plans = try .init(g, e, @tagName(site), diag), .w = g.keep(w), .scales = g.keep(scales) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
            g.release(self.scales);
        }

        /// The draft's `QuantizedLinear.__call__` seam (`Tr.qlinear`): x [..., G K] with 1..8 rows ->
        /// y [..., G N] at x's dtype.
        pub fn linear(self: *const Self, g: *G, x: G.T) !G.T {
            const sh = dims(G, g, x);
            const lead = sh.slice()[0 .. sh.n - 1];
            var m: c_int = 1;
            for (lead) |d| m *= d;
            const y = try self.call(g, try g.reshape(x, &.{ m, sh.slice()[sh.n - 1] }));
            var shape: Shape = .of(lead);
            shape.d[shape.n] = @intCast(self.plans.cfg[0].out_shapes[0][1]);
            shape.n += 1;
            return g.reshape(y, shape.slice());
        }

        /// x [M, G K] (row-contiguous, the construction dtype), M = 1..8 -> y [M, G N].
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, self.scales, x }, &out);
            return out[0];
        }
    };
}

// ── HCTAPE (DSV41_DECODE_HCTAPE = all) ──

/// The HC tail kernels: `HcTapeKernels` (combine, collapse_norm, combine_collapse_norm,
/// mixfin). The stream dtype is a template (OT): bf16 = the verify barrier's texts, f32 = the
/// DSpark draft stages' variants (DRAFTRC member tape; the norm weight stays bf16). The f32
/// stream's fused call on a bf16 residual is its own route (`HcTapeMixed`).
pub fn HcTape(comptime G: type) type {
    return struct {
        const Self = @This();
        combine_e: *const Entry,
        collapse_e: *const Entry,
        fused_e: *const Entry,
        mixfin_e: *const Entry,
        combine_p: RowPlans(G, 8),
        collapse_p: RowPlans(G, 8),
        fused_p: RowPlans(G, 8),
        mixfin_p: RowPlans(G, 8),

        pub fn init(g: *G, reg: *const xk.Registry, stream: Dtype, diag: ?*xk.Diag) !Self {
            const ce, const le, const fe = switch (stream) {
                .bfloat16 => .{ reg.get(.q3ht_combine), reg.get(.q3ht_collapse_norm), reg.get(.q3ht_combine_collapse_norm) },
                .float32 => .{ reg.get(.q3ht_combine__f32), reg.get(.q3ht_collapse_norm__f32), reg.get(.q3ht_combine_collapse_norm__f32) },
                else => return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: HCTAPE stream {t} (OT): the registry carries bf16 and f32", .{stream}),
            };
            const me = reg.get(.q3ht_mixfin);
            var combine_p: RowPlans(G, 8) = try .init(g, ce, null, diag);
            errdefer combine_p.deinit(g);
            var collapse_p: RowPlans(G, 8) = try .init(g, le, null, diag);
            errdefer collapse_p.deinit(g);
            var fused_p: RowPlans(G, 8) = try .init(g, fe, null, diag);
            errdefer fused_p.deinit(g);
            return .{
                .combine_e = ce,
                .collapse_e = le,
                .fused_e = fe,
                .mixfin_e = me,
                .combine_p = combine_p,
                .collapse_p = collapse_p,
                .fused_p = fused_p,
                .mixfin_p = try .init(g, me, null, diag),
            };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.combine_p.deinit(g);
            self.collapse_p.deinit(g);
            self.fused_p.deinit(g);
            self.mixfin_p.deinit(g);
        }

        /// x [M, D], r [M, 4, D], post [M, 4] f32, comb [M, 16] f32 -> h [M, 4, D].
        pub fn combine(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.combine_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb }, &out);
            return out[0];
        }

        /// s [M, 4, D], pre [M, 4] f32, w [D] -> (sf [M, 4 D] f32, ssq [M] f32, y [M, D]).
        pub fn collapseNorm(self: *const Self, g: *G, s: G.T, pre: G.T, w: G.T) ![3]G.T {
            var out: [3]G.T = undefined;
            try self.collapse_p.launch(g, rowsOf(G, g, s, 0), &.{ s, pre, w }, &out);
            return out;
        }

        /// -> (h [M, 4, D], hf [M, 4 D] f32, ssq [M] f32, y [M, D]).
        pub fn combineCollapseNorm(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T, pre: G.T, w: G.T) ![4]G.T {
            var out: [4]G.T = undefined;
            try self.fused_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb, pre, w }, &out);
            return out;
        }

        /// mm [M, 24] f32, ssq [M] f32, scale [3] f32, base [24] f32 -> (pre [M, 4], post [M, 4], comb [M, 16]) f32.
        pub fn mixfin(self: *const Self, g: *G, mm: G.T, ssq: G.T, scale: G.T, base: G.T) ![3]G.T {
            var out: [3]G.T = undefined;
            try self.mixfin_p.launch(g, rowsOf(G, g, mm, 0), &.{ mm, ssq, scale, base }, &out);
            return out;
        }
    };
}

/// The f32 stream's fused call on a bf16 residual (x f32, r [M, 4, D] bf16: the DSpark draft's
/// stage-0 first ffn prep): `q3ht_combine_collapse_norm__f32_rbf16`, its own route so that no
/// call checks which stream it serves.
pub fn HcTapeMixed(comptime G: type) type {
    return struct {
        const Self = @This();
        fused_p: RowPlans(G, 8),

        pub fn init(g: *G, reg: *const xk.Registry, diag: ?*xk.Diag) !Self {
            return .{ .fused_p = try .init(g, reg.get(.q3ht_combine_collapse_norm__f32_rbf16), null, diag) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.fused_p.deinit(g);
        }

        /// -> (h [M, 4, D] f32, hf [M, 4 D] f32, ssq [M] f32, y [M, D] f32).
        pub fn call(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T, pre: G.T, w: G.T) ![4]G.T {
            var out: [4]G.T = undefined;
            try self.fused_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb, pre, w }, &out);
            return out;
        }
    };
}

// ── ATTN_FUSED_PROJ (K36) ──

pub const RopeDir = enum { fwd, inv };

/// The decode / verify projection-chain glue (`deepseek_v41_fused_proj_kernels`, rows <= 8 by
/// `_fused_proj_use`): the q-latent RMSNorm, the KV RMSNorm + k_pe RoPE, the query RoPE (bf16
/// in) and the attention output's inverse RoPE (f32 in); all store bf16.
pub fn FusedProj(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        rms: *const Entry,
        rms_rope: *const Entry,
        fwd: *const Entry,
        inv: *const Entry,
        q_norm: G.T,
        kv_norm: G.T,
        rms_statics: Statics(G),
        rope_statics: Statics(G),
        ints: [max_rows]G.T,
        rms_p: RowPlans(G, max_rows),
        rms_rope_p: RowPlans(G, max_rows),
        fwd_p: RowPlans(G, max_rows),
        inv_p: RowPlans(G, max_rows),

        /// `q_norm` / `kv_norm`: the layer's q_norm (bf16 [1280]) and kv_norm (bf16 [512]) weights.
        pub fn init(g: *G, reg: *const xk.Registry, q_norm: G.T, kv_norm: G.T, diag: ?*xk.Diag) !Self {
            const rms = reg.get(.mtplx_dsv41_fp_rmsnorm_tg128_d1280);
            const rms_rope = reg.get(.mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64);
            try expectInput(G, g, rms, "weight", q_norm, &no_vars, diag);
            try expectInput(G, g, rms_rope, "weight", kv_norm, &no_vars, diag);
            const fwd = reg.get(.mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd);
            const inv = reg.get(.mtplx_dsv41_fp_rope_h64_hd512_rd64_inv);
            var s: Self = .{ .rms = rms, .rms_rope = rms_rope, .fwd = fwd, .inv = inv, .q_norm = undefined, .kv_norm = undefined, .rms_statics = undefined, .rope_statics = undefined, .ints = undefined, .rms_p = undefined, .rms_rope_p = undefined, .fwd_p = undefined, .inv_p = undefined };
            s.rms_p = try .init(g, rms, null, diag);
            errdefer s.rms_p.deinit(g);
            s.rms_rope_p = try .init(g, rms_rope, null, diag);
            errdefer s.rms_rope_p.deinit(g);
            s.fwd_p = try .init(g, fwd, null, diag);
            errdefer s.fwd_p.deinit(g);
            s.inv_p = try .init(g, inv, null, diag);
            errdefer s.inv_p.deinit(g);
            s.rms_statics = try Statics(G).init(g, rms);
            errdefer s.rms_statics.deinit(g);
            s.rope_statics = try Statics(G).init(g, rms_rope);
            errdefer s.rope_statics.deinit(g);
            var built: usize = 0;
            errdefer for (s.ints[0..built]) |a| g.release(a);
            for (0..max_rows) |i| {
                const v: i32 = @intCast(i + 1);
                s.ints[i] = g.keep(try g.hostArray(std.mem.asBytes(&v), &.{}, .int32));
                built += 1;
            }
            s.q_norm = g.keep(q_norm);
            s.kv_norm = g.keep(kv_norm);
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.rms_p.deinit(g);
            self.rms_rope_p.deinit(g);
            self.fwd_p.deinit(g);
            self.inv_p.deinit(g);
            self.rms_statics.deinit(g);
            self.rope_statics.deinit(g);
            for (self.ints) |a| g.release(a);
            g.release(self.q_norm);
            g.release(self.kv_norm);
        }

        fn rowsIn(g: *G, x: G.T, width: usize) !struct { Shape, u64 } {
            const sh = dims(G, g, x);
            var numel: usize = 1;
            for (sh.slice()) |v| numel *= @intCast(v);
            const rows = numel / width;
            if (rows < 1 or rows > max_rows) return error.RowsOutOfPlan;
            return .{ sh, rows };
        }

        /// seq rows of cos / sin (`S`; 0 rows = the call's rows).
        fn seqOf(g: *G, cos: G.T, rows: u64) !u64 {
            const s: u64 = rowsOf(G, g, cos, 0);
            const seq = if (s == 0) rows else s;
            if (seq > max_rows) return error.RowsOutOfPlan;
            return seq;
        }

        /// `rmsnorm(wq_a(x), q_norm_weight, eps)`: x [..., 1280] bf16 -> x's shape, bf16.
        pub fn qNorm(self: *const Self, g: *G, x: G.T) !G.T {
            const sh, const rows = try rowsIn(g, x, 1280);
            var out: [1]G.T = undefined;
            try self.rms_p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 1280 }), self.q_norm, self.rms_statics.arrays[2] }, &out);
            return g.reshape(out[0], sh.slice());
        }

        /// `rmsnorm_rope(wkv(x), kv_norm_weight, eps, cos, sin)`: x [..., 512] bf16, cos / sin f32 [S, 32].
        pub fn kvNormRope(self: *const Self, g: *G, x: G.T, cos: G.T, sin: G.T) !G.T {
            const sh, const rows = try rowsIn(g, x, 512);
            const seq = try seqOf(g, cos, rows);
            var out: [1]G.T = undefined;
            try self.rms_rope_p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 512 }), self.kv_norm, self.rope_statics.arrays[2], cos, sin, self.ints[seq - 1] }, &out);
            return g.reshape(out[0], sh.slice());
        }

        /// `rope_heads(x [..., 64, 512], cos, sin, inverse)` -> x's shape, bf16.
        pub fn ropeHeads(self: *const Self, g: *G, x: G.T, cos: G.T, sin: G.T, dir: RopeDir) !G.T {
            const sh, const rows = try rowsIn(g, x, 64 * 512);
            const seq = try seqOf(g, cos, rows);
            var out: [1]G.T = undefined;
            const p = if (dir == .fwd) &self.fwd_p else &self.inv_p;
            try p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 64, 512 }), cos, sin, self.ints[rows - 1], self.ints[seq - 1] }, &out);
            return g.reshape(out[0], sh.slice());
        }
    };
}

// ── The expert path over the streamer's slot banks (DSV41_EXL3_BANK = on, PREP = rin) ──

pub const Proj = enum { gate, up, down };

/// One projection's slot-bank arrays, as the streamer's `ProjArrays`: code i16 [rows, in/16,
/// out/16, 48], rout f16 [rows, out], rin f16 [rows, in] (row = slot).
pub fn ProjArrays(comptime T: type) type {
    return struct { code: T, rout: T, rin: T };
}

/// A layer bank's projection arrays are what the kernels read (bind time, once per bank).
pub fn checkBank(comptime G: type, g: *G, reg: *const xk.Registry, proj: Proj, a: ProjArrays(G.T), diag: ?*xk.Diag) Refusal!void {
    var vars: Vars = .initFill(0);
    const cap = rowsOf(G, g, a.code, 0);
    const bound = reg.get(.dsv41_exl3_mul1h_k3_2304).bounds.get(.cap).?;
    if (cap < bound[0] or cap > bound[1]) return refuse(diag, error.RouteInput, "exl3 kernel ops: a bank of {d} slots (the kernels take {d}..{d})", .{ cap, bound[0], bound[1] });
    vars.set(.cap, cap);
    switch (proj) {
        .gate, .up => {
            try expectInput(G, g, reg.get(.dsv41_exl3_mul1h_k3_2304), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_gu_epi), "rg", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_in_rin), "rg", a.rin, &vars, diag);
        },
        .down => {
            try expectInput(G, g, reg.get(.dsv41_exl3_mul1h_k3_5120), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_moeprep_dpost), "rd", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_din_rin), "rn", a.rin, &vars, diag);
        },
    }
}

/// The decode GEMVs (form mul1h, K 3): `Provider.project`.
pub fn Gemv(comptime G: type) type {
    return struct {
        const Self = @This();
        gu: *const Entry,
        dn: *const Entry,
        gu_statics: Statics(G),
        dn_statics: Statics(G),
        gu_p: RowPlans(G, 48),
        dn_p: RowPlans(G, 48),

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const gu = reg.get(.dsv41_exl3_mul1h_k3_2304);
            const dn = reg.get(.dsv41_exl3_mul1h_k3_5120);
            var gu_p: RowPlans(G, 48) = try .init(g, gu, null, null);
            errdefer gu_p.deinit(g);
            var dn_p: RowPlans(G, 48) = try .init(g, dn, null, null);
            errdefer dn_p.deinit(g);
            var gs = try Statics(G).init(g, gu);
            errdefer gs.deinit(g);
            return .{ .gu = gu, .dn = dn, .gu_statics = gs, .dn_statics = try Statics(G).init(g, dn), .gu_p = gu_p, .dn_p = dn_p };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.gu_p.deinit(g);
            self.dn_p.deinit(g);
            self.gu_statics.deinit(g);
            self.dn_statics.deinit(g);
        }

        /// xh [rows, in] f32 (rotated), ids [rows] u32 (each row's slot), code = the projection's
        /// bank code -> z [rows, out] f32 (gate / up: out 2304, down: out 5120).
        pub fn project(self: *const Self, g: *G, proj: Proj, xh: G.T, ids: G.T, code: G.T) !G.T {
            const e, const st, const p = if (proj == .down) .{ self.dn, &self.dn_statics, &self.dn_p } else .{ self.gu, &self.gu_statics, &self.gu_p };
            var ins: [9]G.T = undefined;
            ins[0] = xh;
            ins[1] = ids;
            ins[2] = code;
            for (3..e.inputs.len) |i| ins[i] = st.arrays[i];
            var out: [1]G.T = undefined;
            try p.launch(g, rowsOf(G, g, xh, 0), ins[0..e.inputs.len], &out);
            return out[0];
        }
    };
}

/// PREP = rin: the rin / rout stages around the GEMVs (`build_kernels` of the rinprep lane).
pub fn RinPrep(comptime G: type) type {
    return struct {
        const Self = @This();
        in_rin_e: *const Entry,
        gu_epi_e: *const Entry,
        din_rin_e: *const Entry,
        dpost_e: *const Entry,
        in_rin_p: RowPlans(G, 48),
        gu_epi_p: RowPlans(G, 48),
        din_rin_p: RowPlans(G, 48),
        dpost_p: RowPlans(G, 48),

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const a, const b, const c, const d = .{ reg.get(.q3_exl3_prep_in_rin), reg.get(.q3_exl3_prep_gu_epi), reg.get(.q3_exl3_prep_din_rin), reg.get(.q3_moeprep_dpost) };
            var pa: RowPlans(G, 48) = try .init(g, a, null, null);
            errdefer pa.deinit(g);
            var pb: RowPlans(G, 48) = try .init(g, b, null, null);
            errdefer pb.deinit(g);
            var pc: RowPlans(G, 48) = try .init(g, c, null, null);
            errdefer pc.deinit(g);
            return .{ .in_rin_e = a, .gu_epi_e = b, .din_rin_e = c, .dpost_e = d, .in_rin_p = pa, .gu_epi_p = pb, .din_rin_p = pc, .dpost_p = try .init(g, d, null, null) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.in_rin_p.deinit(g);
            self.gu_epi_p.deinit(g);
            self.din_rin_p.deinit(g);
            self.dpost_p.deinit(g);
        }

        /// x [tokens, 5120] bf16, tok [rows] i32, rg / ru = gate / up rin, ids [rows] u32 ->
        /// (xg, xu) [rows, 5120] f32 = t128(x[tok] * rin[ids]).
        pub fn inRin(self: *const Self, g: *G, x: G.T, tok: G.T, rg: G.T, ru: G.T, ids: G.T) ![2]G.T {
            var out: [2]G.T = undefined;
            try self.in_rin_p.launch(g, rowsOf(G, g, tok, 0), &.{ x, tok, rg, ru, ids }, &out);
            return out;
        }

        /// zg / zu [rows, 2304] f32, rg / ru = gate / up rout -> clamped SwiGLU [rows, 2304] f32.
        pub fn guEpi(self: *const Self, g: *G, zg: G.T, zu: G.T, rg: G.T, ru: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.gu_epi_p.launch(g, rowsOf(G, g, zg, 0), &.{ zg, zu, rg, ru, ids }, &out);
            return out[0];
        }

        /// hid [rows, 2304] f32, rn = down rin -> t128(hid * rin[ids]) [rows, 2304] f32.
        pub fn dinRin(self: *const Self, g: *G, hid: G.T, rn: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.din_rin_p.launch(g, rowsOf(G, g, hid, 0), &.{ hid, rn, ids }, &out);
            return out[0];
        }

        /// zd [rows, 5120] f32, rd = down rout -> t128(zd) * rout[ids] [rows, 5120] f32.
        pub fn dpost(self: *const Self, g: *G, zd: G.T, rd: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.dpost_p.launch(g, rowsOf(G, g, zd, 0), &.{ zd, rd, ids }, &out);
            return out[0];
        }
    };
}

// ── Prefill: REBUILD (DSV41_PREFILL_FUSED_BANK = exl3, mul1lut) and DIG-X (DIG = 1, DIG2 = onepass) ──

pub const wave_max = 16;

/// The rebuild's slot table: the wave's slots, padded with the first (the lane's `slots8`).
pub fn rebuildSlots(slots: []const u32) [wave_max]i32 {
    var t: [wave_max]i32 = undefined;
    for (&t, 0..) |*v, i| v.* = @intCast(slots[if (i < slots.len) i else 0]);
    return t;
}

/// The EXL3 rebuild of a wave's experts into bf16 weights: `Exl3Rebuild3Kernel.launch`.
pub fn Rebuild(comptime G: type) type {
    return struct {
        const Self = @This();
        e: *const Entry,

        pub fn init(reg: *const xk.Registry) Self {
            return .{ .e = reg.get(.q3_prefill_fused_exl3x3_mul1lut_k3_bf16) };
        }

        /// slots i32 [16] (`rebuildSlots`), `experts` of them used -> (og, ou [n, 5120, 2304], od [n, 2304, 5120]) bf16.
        pub fn call(self: *const Self, g: *G, gate: ProjArrays(G.T), up: ProjArrays(G.T), down: ProjArrays(G.T), slots: G.T, experts: u32) ![3]G.T {
            var vars: Vars = .initFill(0);
            vars.set(.experts, experts);
            var out: [3]G.T = undefined;
            try launchRule(G, g, self.e, &vars, &.{ gate.code, gate.rout, gate.rin, up.code, up.rout, up.rin, down.code, down.rout, down.rin, slots }, &out);
            return out;
        }
    };
}

pub const WaveExpert = struct { slot: u32, rows: u32 };
pub const DigTable = struct { table: [80]i32, tgs: u32 };

/// q3_prefill_dig_candidate.wave_table: per expert (<= 16, rows grouped by expert in order)
/// slot, first row, rows, first threadgroup (unused: INT32_MAX); [64] experts, [65] threadgroups.
/// `tiles` = the GEMM's threadgroups per 64-row tile (`digTiles`).
pub fn digTable(experts: []const WaveExpert, tiles: u32) DigTable {
    var t: [80]i32 = @splat(0);
    var row0: i64 = 0;
    var tg: i64 = 0;
    for (0..wave_max) |j| {
        if (j >= experts.len) {
            t[48 + j] = std.math.maxInt(i32);
            continue;
        }
        t[j] = @intCast(experts[j].slot);
        t[16 + j] = @intCast(row0);
        t[32 + j] = @intCast(experts[j].rows);
        t[48 + j] = @intCast(tg);
        row0 += experts[j].rows;
        tg += @as(i64, @intCast((experts[j].rows + 63) / 64)) * tiles;
    }
    t[64] = @intCast(experts.len);
    t[65] = @intCast(tg);
    return .{ .table = t, .tgs = @intCast(tg) };
}

/// DIG-X on DIG2 onepass: the NAX GEMMs whose B loader decodes the trellis, the rotation
/// stages and the onepass SwiGLU (`DigGemmX`, `RotKernelsX`, `Dig2OnePassX`).
pub fn DigX(comptime G: type) type {
    return struct {
        const Self = @This();
        gemm_gu: *const Entry,
        gemm_dn: *const Entry,
        take2_e: *const Entry,
        roundx_e: *const Entry,
        onepass_e: *const Entry,
        widen2_e: *const Entry,
        widen1_e: *const Entry,

        pub fn init(reg: *const xk.Registry) Self {
            return .{
                .gemm_gu = reg.get(.q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3),
                .gemm_dn = reg.get(.q3_prefill_dig_gemm_2304x5120_xmul1hk3),
                .take2_e = reg.get(.q3_prefill_dig_rot_take2_5120),
                .roundx_e = reg.get(.q3_prefill_dig_rot_roundx_2304),
                .onepass_e = reg.get(.q3_prefill_dig2_swiglu_2304_x),
                .widen2_e = reg.get(.q3_prefill_dig_rot_widen2_2304),
                .widen1_e = reg.get(.q3_prefill_dig_rot_widen1_5120),
            };
        }

        /// The GEMMs' threadgroups per 64-row tile: gate|up (both operands), down.
        pub fn digTiles(self: *const Self, proj: enum { gate_up, down }) u32 {
            const e = if (proj == .down) self.gemm_dn else self.gemm_gu;
            return argOf(e, "tbl").domain.tiles;
        }

        /// x0 / x1 [rows, 1, 5120] f16 (take2), gate / up code, the gate|up table -> (zg, zu) [rows, 2304] f32.
        pub fn gemmGateUp(self: *const Self, g: *G, x0: G.T, x1: G.T, code_g: G.T, code_u: G.T, tbl: G.T, tgs: u32) ![2]G.T {
            var vars = rowsVars(rowsOf(G, g, x0, 0));
            vars.set(.tgs, tgs);
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.gemm_gu, &vars, &.{ x0, x1, code_g, code_u, tbl }, &out);
            return out;
        }

        /// x [rows, 1, 2304] f16 (onepass), down code, the down table -> z [rows, 5120] f32.
        pub fn gemmDown(self: *const Self, g: *G, x: G.T, code_d: G.T, tbl: G.T, tgs: u32) !G.T {
            var vars = rowsVars(rowsOf(G, g, x, 0));
            vars.set(.tgs, tgs);
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.gemm_dn, &vars, &.{ x, code_d, tbl }, &out);
            return out[0];
        }

        /// act [A, 5120] bf16 rows ridx [rows] i32, rhs [rows] u32 (expert per row), slots = a
        /// table -> (f16(t128(act * rin_g[slot])), f16(t128(act * rin_u[slot]))) [rows, 1, 5120].
        pub fn take2(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, slots: G.T, rin_g: G.T, rin_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, ridx, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.take2_e, &vars, &.{ act, ridx, rhs, slots, rin_g, rin_u }, &out);
            return out;
        }

        /// act [rows, 1, 2304] f32 -> f16(t128(act * rin_d[slot])) [rows, 1, 2304].
        pub fn roundx(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.roundx_e, &vars, &.{ act, rhs, slots, rin_d }, &out);
            return out[0];
        }

        /// zg / zu [rows, 2304] f32 -> hd = f16(t128(clamped SwiGLU(t128(zg) rout_g, t128(zu) rout_u) rin_d)) [rows, 1, 2304].
        pub fn onePass(self: *const Self, g: *G, zg: G.T, zu: G.T, rhs: G.T, tbl: G.T, rout_g: G.T, rout_u: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, zg, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.onepass_e, &vars, &.{ zg, zu, rhs, tbl, rout_g, rout_u, rin_d }, &out);
            return out[0];
        }

        /// act_g / act_u [rows, 2304] f32 -> t128(act) * rout[slot] [rows, 1, 2304] f32 each.
        pub fn widen2(self: *const Self, g: *G, act_g: G.T, act_u: G.T, rhs: G.T, slots: G.T, rout_g: G.T, rout_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, act_g, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.widen2_e, &vars, &.{ act_g, act_u, rhs, slots, rout_g, rout_u }, &out);
            return out;
        }

        /// act [rows, 5120] f32 -> t128(act) * rout_d[slot] [rows, 1, 5120] f32.
        pub fn widen1(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rout_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.widen1_e, &vars, &.{ act, rhs, slots, rout_d }, &out);
            return out[0];
        }

        /// `Dig2GemmOnePassX`: take2 -> the gate|up GEMM -> onepass: (hd, zg, zu).
        pub fn gateUpOnePass(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, gu: DigTableArray(G), gate: ProjArrays(G.T), up: ProjArrays(G.T), down_rin: G.T) ![3]G.T {
            const x = try self.take2(g, act, ridx, rhs, gu.tbl, gate.rin, up.rin);
            const z = try self.gemmGateUp(g, x[0], x[1], gate.code, up.code, gu.tbl, gu.tgs);
            const hd = try self.onePass(g, z[0], z[1], rhs, gu.tbl, gate.rout, up.rout, down_rin);
            return .{ hd, z[0], z[1] };
        }
    };
}

/// A wave table on the device and its threadgroup count (`digTable`).
pub fn DigTableArray(comptime G: type) type {
    return struct { tbl: G.T, tgs: u32 };
}

// ── Prefill: the routed wave dispatch (DIG-X, DIG2 onepass, SHAPE balance / rebuildahead / carry) ──

/// The installed prefill wave shape: the fused point's wave / inflight / row budget
/// (Q3_PREFILL_FUSED_INSTALL config) and SHAPE's carry rows (Q3_PREFILL_SHAPE_INSTALL).
pub const PrefillShape = struct {
    /// experts per wave (1..16, the wave table's expert rows)
    wave: u32,
    /// waves in flight (>= 2, SHAPE's overlap route): a wave first waits for all but `inflight - 1` older ones
    inflight: u32,
    /// assignment rows per wave; an expert above it forms a wave alone (solo: drained before, evaluated after)
    row_budget: u32,
    /// a call of at most `carry_rows` rows leaves its last waves in flight (to the next call or `finish`)
    carry_rows: u32,

    /// Record 3 (pass3r-record3-fast-typical-exl3-30-guard-20260928.log).
    pub const tier: PrefillShape = .{ .wave = 4, .inflight = 2, .row_budget = 7168, .carry_rows = 8192 };
};

/// A layer bank's three projections (the streamer's `BankArrays`).
pub fn BankArrays(comptime T: type) type {
    return struct { gate: ProjArrays(T), up: ProjArrays(T), down: ProjArrays(T) };
}

/// One call's routed rows: `slot[i]` = assignment row i's bank slot (its binding's bank_index).
/// Row i reads act row i (the switch's `selected`), or act row `act_row[i]` when given (the
/// chunk's tokens and position / top_k: the same words without the switch's row take).
pub const PrefillRows = struct { slot: []const u32, act_row: ?[]const u32 = null };

/// The lane of record's `Tcq3FusedPrefillDispatch...__dig2_onepass_exl3.__call__` over one call's
/// routed rows (one route per layer, as the lane has one dispatcher per layer): experts grouped
/// by slot in first-appearance order, snake-ordered (largest, smallest, ...; stable), packed
/// greedily into waves of <= `wave` experts and <= `row_budget` rows; per wave rot_take2 -> the
/// gate|up GEMM (72-tile table) -> dig2 onepass -> the down GEMM (80-tile table) -> rot_widen1;
/// then `take(concatenate(waves), argsort(positions))`, the permutation made on the host. The
/// eval schedule is SHAPE's: a wave waits for all but `inflight - 1` older waves (a solo wave for
/// all, and is evaluated at once); a call above `carry_rows` rows drains every wave and evaluates
/// the join and the result, a carried call leaves its last waves in flight and async-evaluates
/// the result; `finish` (the prefill boundary) drains them. The lane's rebuild-ahead is the next
/// wave's two host tables (no GPU work) and has no counterpart here.
///
/// Wave lifecycle (the backend's `mark` / `resetTo`: `resetTo` frees every array tracked since
/// the mark, kept handles survive): each wave's ops run between a mark and a `resetTo` right
/// after the wave is submitted, so its intermediates live only while the GPU still needs them
/// (a pending graph holds its inputs); the wave's output survives as two kept handles (the
/// join's, the in-flight queue's). The join runs between its own mark and `resetTo`; `call`
/// returns a KEPT result (the caller releases it). Nothing else the call builds outlives it.
pub fn DigXPrefill(comptime G: type) type {
    if (!@hasDecl(G, "mark") or !@hasDecl(G, "resetTo"))
        @compileError("exl3 kernel ops: DigXPrefill needs a backend with mark / resetTo (the wave lifecycle)");
    return struct {
        const Self = @This();
        const hidden = 5120;
        dig: DigX(G),
        shape: PrefillShape,
        tiles_gu: u32,
        tiles_dn: u32,
        rows_hi: u64,
        a: Allocator,
        diag: ?*xk.Diag,
        /// waves in flight, oldest first (kept handles; persist across carried calls)
        flight: std.ArrayList(G.T) = .empty,
        parts: std.ArrayList(G.T) = .empty,
        // host scratch, reused across calls
        group_of: std.ArrayList(i32) = .empty,
        gslot: std.ArrayList(u32) = .empty,
        gcount: std.ArrayList(u32) = .empty,
        gnext: std.ArrayList(u32) = .empty,
        snake: std.ArrayList(u32) = .empty,
        sorted: std.ArrayList(u32) = .empty,
        grows: std.ArrayList(u32) = .empty,
        pos: std.ArrayList(u32) = .empty,
        ridx: std.ArrayList(i32) = .empty,
        rhs: std.ArrayList(u32) = .empty,
        inv: std.ArrayList(u32) = .empty,

        /// `diag` (optional) receives the refusal messages of `init` and of every call.
        pub fn init(a: Allocator, reg: *const xk.Registry, shape: PrefillShape, diag: ?*xk.Diag) Refusal!Self {
            if (shape.wave < 1 or shape.wave > wave_max or shape.inflight < 2 or shape.row_budget < 1)
                return refuse(diag, error.RouteInput, "exl3 kernel ops: prefill shape wave {d} (1..{d}), inflight {d} (>= 2: SHAPE's overlap route), row budget {d} (>= 1)", .{ shape.wave, wave_max, shape.inflight, shape.row_budget });
            const dig = DigX(G).init(reg);
            return .{ .dig = dig, .shape = shape, .tiles_gu = dig.digTiles(.gate_up), .tiles_dn = dig.digTiles(.down), .rows_hi = dig.gemm_gu.bounds.get(.rows).?[1], .a = a, .diag = diag };
        }

        /// Releases the waves still in flight (without evaluating them) and the scratch.
        pub fn deinit(self: *Self, g: *G) void {
            for (self.flight.items) |x| g.release(x);
            for (self.parts.items) |x| g.release(x);
            self.flight.deinit(self.a);
            self.parts.deinit(self.a);
            inline for (.{ &self.group_of, &self.gslot, &self.gcount, &self.gnext, &self.snake, &self.sorted, &self.grows, &self.pos, &self.ridx, &self.rhs, &self.inv }) |l| l.deinit(self.a);
        }

        /// The prefill boundary: every wave still in flight evaluated, oldest first.
        pub fn finish(self: *Self, g: *G) !void {
            while (self.flight.items.len > 0) try self.drainOne(g);
        }

        fn drainOne(self: *Self, g: *G) !void {
            const x = self.flight.orderedRemove(0);
            defer g.release(x);
            try g.evalAll(&.{x});
        }

        /// act bf16 [a_rows, 5120], `rows` (A assignment rows), the call's bank -> a KEPT f32
        /// [A, 5120] in assignment-row order (the lane's `result`; release it). Refused: A outside 1..the kernels' row
        /// bound (RowsOutOfPlan), a slot outside the bank (SlotOutOfBank), act rows that are not
        /// A (no act_row) or an act_row outside act (RouteInput).
        pub fn call(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T)) !G.T {
            const n_rows = rows.slot.len;
            if (n_rows == 0 or n_rows > self.rows_hi) return refuse(self.diag, error.RowsOutOfPlan, "exl3 kernel ops: a prefill call of {d} rows (1..{d})", .{ n_rows, self.rows_hi });
            const a_rows = rowsOf(G, g, act, 0);
            if (rows.act_row == null and a_rows != n_rows) return refuse(self.diag, error.RouteInput, "exl3 kernel ops: prefill act has {d} rows for {d} routed rows", .{ a_rows, n_rows });
            if (rows.act_row) |ar| if (ar.len != n_rows) return refuse(self.diag, error.RouteInput, "exl3 kernel ops: {d} act rows for {d} routed rows", .{ ar.len, n_rows });
            try self.group(rows, rowsOf(G, g, bank.gate.code, 0), a_rows);
            const carried = n_rows <= self.shape.carry_rows;
            const a = self.a;
            const cnt = self.gcount.items;
            const order = self.snake.items;
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            var i: usize = 0;
            var off: usize = 0;
            while (i < order.len) {
                const first = i;
                var wave_rows: usize = cnt[order[i]];
                i += 1;
                while (i < order.len and i - first < self.shape.wave and wave_rows + cnt[order[i]] <= self.shape.row_budget) : (i += 1) wave_rows += cnt[order[i]];
                const solo = wave_rows > self.shape.row_budget;
                const keep: usize = if (solo) 0 else self.shape.inflight - 1;
                while (self.flight.items.len > keep) try self.drainOne(g);
                var ex: [wave_max]WaveExpert = undefined;
                for (order[first..i], 0..) |gi, j| {
                    ex[j] = .{ .slot = self.gslot.items[gi], .rows = cnt[gi] };
                    @memset(self.rhs.items[off..][0..cnt[gi]], @intCast(j));
                    off += cnt[gi];
                }
                const r0 = off - wave_rows;
                try self.parts.ensureUnusedCapacity(a, 1);
                try self.flight.ensureUnusedCapacity(a, 1);
                const m = g.mark();
                const y = try self.submit(g, act, bank, ex[0 .. i - first], self.ridx.items[r0..off], self.rhs.items[r0..off]);
                self.parts.appendAssumeCapacity(g.keep(y));
                if (solo) {
                    try g.evalAll(&.{y});
                } else {
                    try g.asyncEval(&.{y});
                    self.flight.appendAssumeCapacity(g.keep(y));
                }
                g.resetTo(m);
            }
            if (!carried) while (self.flight.items.len > 0) try self.drainOne(g);
            const m = g.mark();
            const joined = try g.concat(self.parts.items, 0);
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            if (!carried) try g.evalAll(&.{joined});
            for (self.pos.items, 0..) |p, j| self.inv.items[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{@intCast(n_rows)}, .uint32);
            const result = try g.take(joined, ord, 0);
            const kept = g.keep(result);
            errdefer g.release(kept);
            if (carried) try g.asyncEval(&.{result}) else try g.evalAll(&.{result});
            g.resetTo(m);
            return kept;
        }

        /// One wave's five launches -> its rows' output f32 [R, 5120] (the lane's `y.reshape(-1, H)`).
        fn submit(self: *Self, g: *G, act: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, ridx: []const i32, rhs: []const u32) !G.T {
            const n: c_int = @intCast(ridx.len);
            const tg = digTable(ex, self.tiles_gu);
            const td = digTable(ex, self.tiles_dn);
            const tgu = try g.hostArray(std.mem.sliceAsBytes(&tg.table), &.{80}, .int32);
            const tdn = try g.hostArray(std.mem.sliceAsBytes(&td.table), &.{80}, .int32);
            const ridx_a = try g.hostArray(std.mem.sliceAsBytes(ridx), &.{n}, .int32);
            const rhs_a = try g.hostArray(std.mem.sliceAsBytes(rhs), &.{n}, .uint32);
            const hz = try self.dig.gateUpOnePass(g, act, ridx_a, rhs_a, .{ .tbl = tgu, .tgs = tg.tgs }, bank.gate, bank.up, bank.down.rin);
            const zd = try self.dig.gemmDown(g, hz[0], bank.down.code, tdn, td.tgs);
            const y = try self.dig.widen1(g, zd, rhs_a, tdn, bank.down.rout);
            return g.reshape(y, &.{ n, hidden });
        }

        /// The host plan: groups (slot -> rows, first-appearance order, rows ascending), the snake
        /// order, and per wave-ordered row its assignment row (`pos`) and act row (`ridx`).
        fn group(self: *Self, rows: PrefillRows, cap: u64, a_rows: u64) !void {
            const a = self.a;
            const n = rows.slot.len;
            try self.group_of.resize(a, @intCast(cap));
            @memset(self.group_of.items, -1);
            self.gslot.clearRetainingCapacity();
            self.gcount.clearRetainingCapacity();
            for (rows.slot, 0..) |s, i| {
                if (s >= cap) return refuse(self.diag, error.SlotOutOfBank, "exl3 kernel ops: prefill row {d} names slot {d} of a {d}-slot bank", .{ i, s, cap });
                if (rows.act_row) |ar| if (ar[i] >= a_rows) return refuse(self.diag, error.RouteInput, "exl3 kernel ops: prefill row {d} reads act row {d} of {d}", .{ i, ar[i], a_rows });
                var gi = self.group_of.items[s];
                if (gi < 0) {
                    gi = @intCast(self.gslot.items.len);
                    self.group_of.items[s] = gi;
                    try self.gslot.append(a, s);
                    try self.gcount.append(a, 0);
                }
                self.gcount.items[@intCast(gi)] += 1;
            }
            const ng = self.gslot.items.len;
            // each group's rows, ascending (the lane appends rows in order)
            try self.gnext.resize(a, ng);
            var start: u32 = 0;
            for (self.gcount.items, self.gnext.items) |c, *nx| {
                nx.* = start;
                start += c;
            }
            try self.grows.resize(a, n);
            for (rows.slot, 0..) |s, i| {
                const gi: usize = @intCast(self.group_of.items[s]);
                self.grows.items[self.gnext.items[gi]] = @intCast(i);
                self.gnext.items[gi] += 1;
            }
            // snake over the rows-descending order (stable: ties keep first appearance)
            try self.sorted.resize(a, ng);
            for (self.sorted.items, 0..) |*v, i| v.* = @intCast(i);
            std.mem.sort(u32, self.sorted.items, @as([]const u32, self.gcount.items), struct {
                fn more(c: []const u32, x: u32, y: u32) bool {
                    return c[x] > c[y];
                }
            }.more);
            try self.snake.resize(a, ng);
            var lo: usize = 0;
            var hi: usize = ng;
            var k: usize = 0;
            while (lo < hi) {
                self.snake.items[k] = self.sorted.items[lo];
                k += 1;
                lo += 1;
                if (lo < hi) {
                    hi -= 1;
                    self.snake.items[k] = self.sorted.items[hi];
                    k += 1;
                }
            }
            // rows in wave order: the snake order's groups, each group's rows ascending
            try self.pos.resize(a, n);
            try self.ridx.resize(a, n);
            try self.rhs.resize(a, n);
            try self.inv.resize(a, n);
            var off: usize = 0;
            for (self.snake.items) |gi| {
                const c = self.gcount.items[gi];
                const first = self.gnext.items[gi] - c;
                for (self.grows.items[first..][0..c]) |row| {
                    self.pos.items[off] = row;
                    self.ridx.items[off] = @intCast(if (rows.act_row) |ar| ar[row] else row);
                    off += 1;
                }
            }
        }
    };
}

// ── Startup acceptance: the arm's one entry (registry, device self-check, typed handles) ──

/// What the startup acceptance runs on.
pub const Device = union(enum) {
    /// the arm's GPU stream: every kernel is built on it and the self-check plan runs there
    stream: mlx.mlx_stream,
    /// host tests: no kernel object and no MLX array; the plan's results are scripted
    stub: StubDevice,
};

/// A scripted device: every (kernel, check) of the plan passes, except `fail` when set.
pub const StubDevice = struct {
    fail: ?struct { kernel: Kernel, check: xk.Check } = null,
};

pub const StartupOptions = struct {
    device: Device,
    /// the texts the registry checks against the pin (production: the embedded ones)
    texts: *const xk.Texts = &xk.embedded,
    pin: []const u8 = xk.manifest_sha256,
};

/// The backend methods every route uses (the phase-3 contract); the prefill route adds its
/// own (evalAll, asyncEval, concat, take, mark, resetTo) and checks them itself.
const backend_methods = [_][]const u8{ "launch", "shapeOf", "dtypeOf", "hostArray", "keep", "release", "reshape", "astype" };

/// Where acceptAtStartup installs the launcher: the backend's `launcher: ?*const xk.Bound`
/// field, or its base backend's under a wrapper that declares `Inner` and `base()`
/// (`dsv41_profile.Profiled`); null when neither has one.
fn launcherSlot(comptime G: type, g: *G) ?*?*const xk.Bound {
    if (@hasField(G, "launcher")) return &g.launcher;
    if (@hasDecl(G, "Inner")) return launcherSlot(G.Inner, g.base());
    return null;
}

/// The first route method `G` lacks, or null.
pub fn missingBackendMethod(comptime G: type) ?[]const u8 {
    inline for (backend_methods) |m| if (!@hasDecl(G, m)) return m;
    return null;
}

/// The accepted registry on one device: the typed handles the arm keeps (heap-allocated: the
/// backend's launcher and the routes point into it). `deinit` after the last launch drained.
pub fn Accepted(comptime G: type) type {
    return struct {
        const Self = @This();
        a: Allocator,
        reg: xk.Registry,
        bound: xk.Bound,
        /// the self-check plan's results (its receipt: `report.writeJsonLines(a)`)
        report: selfcheck.Report,
        gemv: Gemv(G),

        /// The EXL3 decode GEMV as the caller's `GemvT` = { ctx, project_fn(ctx, g: *G, k, out_dim, xh,
        /// ids, code) anyerror!G.T } (EagerChain's MlxGemv): xh f32 [rows, in] rotated, ids u32 [rows]
        /// (each row's slot), code the projection's bank code; out_dim 2304 (gate / up) or 5120
        /// (down); k 3 (the bank). The output is a launch output the backend owns.
        pub fn gemvRoute(self: *const Self, comptime GemvT: type) GemvT {
            return .{ .ctx = &self.gemv, .project_fn = GemvFn(G).project };
        }

        /// Releases the GEMV route (statics, prepared configs), clears the backend's launcher,
        /// then frees the kernels, the plan's results and the registry.
        pub fn deinit(self: *Self, g: *G) void {
            self.gemv.deinit(g);
            if (launcherSlot(G, g)) |s| s.* = null;
            self.bound.deinit();
            self.report.deinit(self.a);
            self.reg.deinit();
            self.a.destroy(self);
        }
    };
}

fn GemvFn(comptime G: type) type {
    return struct {
        fn project(ctx: *const anyopaque, g: *G, k: u32, out_dim: u32, xh: G.T, ids: G.T, code: G.T) anyerror!G.T {
            const r: *const Gemv(G) = @ptrCast(@alignCast(ctx));
            if (k != xk.bank_ks[0]) return error.GemvKNotRegistered;
            const proj: Proj = switch (out_dim) {
                2304 => .gate,
                5120 => .down,
                else => return error.GemvOutDim,
            };
            return r.project(g, proj, xh, ids, code);
        }
    };
}

/// The stub device's plan: one scripted result per (kernel, check) of the registry's plan.
fn stubPlan(a: Allocator, reg: *const xk.Registry, stub: StubDevice, report: *selfcheck.Report) !void {
    for (&reg.entries) |*e| {
        var it = e.checks.iterator();
        while (it.next()) |c| {
            const fails = if (stub.fail) |f| f.kernel == e.kernel and f.check == c else false;
            try report.results.append(a, .{ .kernel = e.kernel, .check = c, .words = 1, .ok = !fails, .err = if (fails) "stub device: scripted failure" else "" });
        }
    }
}

/// The arm's one startup entry: builds the registry (every text against the pinned manifest),
/// builds every kernel on the device, runs the device self-check plan (the KSELF plan of this
/// registry, ~30 s and ~1.6 GB on the GPU: before the model loads) and judges it, points the
/// backend's `launcher: ?*const xk.Bound` at the accepted kernels (when the backend has that
/// field: `deepseek_v41_ops.MlxOps` at m1 c8411cf+), and builds the GEMV route (its statics and,
/// on a backend that prepares launches, its prepared configs).
/// Refused by name: a registry refusal (xk.Refusal: TextSha256Mismatch, ManifestNotPinned,
/// LanePinMismatch, ...; `diag` names the text), NotGpuStream / KernelCreateFailed at bind,
/// SelfCheckFailed (`diag` names the first failing kernel / check / site); a backend without a
/// route method does not compile (the method is named).
pub fn acceptAtStartup(comptime G: type, a: Allocator, g: *G, opts: StartupOptions, diag: *xk.Diag) !*Accepted(G) {
    comptime {
        if (missingBackendMethod(G)) |m| @compileError("exl3 kernel ops: the backend " ++ @typeName(G) ++ " lacks " ++ m ++ " (acceptAtStartup)");
    }
    const acc = try a.create(Accepted(G));
    errdefer a.destroy(acc);
    acc.* = .{ .a = a, .reg = undefined, .bound = undefined, .report = .{}, .gemv = undefined };
    acc.reg = try xk.Registry.init(a, opts.texts, opts.pin, diag);
    errdefer acc.reg.deinit();
    acc.bound = switch (opts.device) {
        .stream => |s| try acc.reg.bind(s, diag),
        .stub => .{ .reg = &acc.reg, .stream = .{}, .kernels = @splat(.{}) },
    };
    errdefer acc.bound.deinit();
    errdefer acc.report.deinit(a);
    switch (opts.device) {
        .stream => try selfcheck.runAll(a, &acc.reg, &acc.bound, &acc.report),
        .stub => |st| try stubPlan(a, &acc.reg, st, &acc.report),
    }
    try selfcheck.judge(&acc.report, diag);
    // the launcher first: a backend that prepares launches prepares them through it
    const slot = launcherSlot(G, g);
    if (slot) |s| s.* = &acc.bound;
    errdefer if (slot) |s| {
        s.* = null;
    };
    acc.gemv = try Gemv(G).init(g, &acc.reg);
    return acc;
}

// ── Tests ──

const testing = std.testing;

/// Host-only backend: nodes of shape + dtype (host arrays keep their bytes), every launch
/// recorded with its inputs and outputs, every node with its origin, and the launches, evals
/// and joins in one ordered log. Nothing reaches MLX.
const Trace = struct {
    pub const T = u32;
    const Origin = union(enum) { none, host, ext: []const u8, out: [2]u32, view: T, cat: u32, take: u32 };
    const Node = struct { shape: Shape, dtype: Dtype, bytes: []u8, origin: Origin = .none };
    const Launch = struct { k: Kernel, cfg: LaunchConfig, inputs: [16]T = undefined, n_in: usize, outs: [xk.max_outputs]T = undefined, prepared: bool = false };
    const Ev = union(enum) { launch: u32, eval: []T, async_eval: []T, concat: []T, take: [2]T };
    const Freed = struct { from: u32, to: u32, at: u32 };

    a: Allocator,
    nodes: std.ArrayList(Node) = .empty,
    launches: std.ArrayList(Launch) = .empty,
    log: std.ArrayList(Ev) = .empty,
    cats: u32 = 0,
    takes: u32 = 0,
    keeps: isize = 0,
    /// node ranges a `resetTo` freed, [from, to), with the log length when it ran, in call order
    freed: std.ArrayList(Freed) = .empty,
    /// nodes with a kept handle (one entry per keep)
    held: std.ArrayList(T) = .empty,
    /// every keep, in order, with the number of resets before it (append-only)
    kept: std.ArrayList(struct { node: T, resets: u32 }) = .empty,
    /// the model backend's kernel launcher (`MlxOps.launcher: ?*const xk.Bound`), set by acceptAtStartup
    launcher: ?*const xk.Bound = null,
    /// prepared launches not yet released; launches made from a prepared config
    prepared_live: isize = 0,
    prepared_launches: usize = 0,

    fn deinit(t: *Trace) void {
        for (t.nodes.items) |n| t.a.free(n.bytes);
        t.nodes.deinit(t.a);
        t.launches.deinit(t.a);
        for (t.log.items) |e| switch (e) {
            .eval, .async_eval, .concat => |xs| t.a.free(xs),
            else => {},
        };
        t.log.deinit(t.a);
        t.freed.deinit(t.a);
        t.held.deinit(t.a);
        t.kept.deinit(t.a);
    }

    pub fn mark(t: *const Trace) u32 {
        return @intCast(t.nodes.items.len);
    }

    pub fn resetTo(t: *Trace, m: u32) void {
        t.freed.append(t.a, .{ .from = m, .to = @intCast(t.nodes.items.len), .at = @intCast(t.log.items.len) }) catch @panic("trace: out of memory");
    }

    /// A node no `resetTo` freed and no kept handle holds (a leak past the call).
    fn leaked(t: *const Trace, x: T) bool {
        for (t.freed.items) |r| if (x >= r.from and x < r.to) return false;
        return std.mem.indexOfScalar(T, t.held.items, x) == null;
    }

    /// The node a view resolves to (reshape chains to their source).
    fn root(t: *const Trace, x: T) T {
        var n = x;
        while (t.nodes.items[n].origin == .view) n = t.nodes.items[n].origin.view;
        return n;
    }

    fn node(t: *Trace, shape: []const c_int, dt: Dtype, bytes: []const u8) !T {
        const copy = try t.a.dupe(u8, bytes);
        errdefer t.a.free(copy);
        try t.nodes.append(t.a, .{ .shape = Shape.of(shape), .dtype = dt, .bytes = copy });
        return @intCast(t.nodes.items.len - 1);
    }

    fn with(t: *Trace, x: T, origin: Origin) T {
        t.nodes.items[x].origin = origin;
        return x;
    }

    /// A caller array named `name` (a lane's ext: reference).
    fn ext(t: *Trace, name: []const u8, shape: []const c_int, dt: Dtype) !T {
        return t.with(try t.node(shape, dt, &.{}), .{ .ext = name });
    }

    pub fn shapeOf(t: *Trace, x: T) Shape {
        return t.nodes.items[x].shape;
    }

    pub fn dtypeOf(t: *Trace, x: T) Dtype {
        return t.nodes.items[x].dtype;
    }

    pub fn hostArray(t: *Trace, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        return t.with(try t.node(shape, dt, bytes), .host);
    }

    pub fn keep(t: *Trace, x: T) T {
        t.keeps += 1;
        t.held.append(t.a, x) catch @panic("trace: out of memory");
        t.kept.append(t.a, .{ .node = x, .resets = @intCast(t.freed.items.len) }) catch @panic("trace: out of memory");
        return x;
    }

    pub fn release(t: *Trace, x: T) void {
        t.keeps -= 1;
        if (std.mem.indexOfScalar(T, t.held.items, x)) |i| _ = t.held.swapRemove(i);
    }

    pub fn reshape(t: *Trace, x: T, shape: []const c_int) !T {
        return t.with(try t.node(shape, t.nodes.items[x].dtype, &.{}), .{ .view = x });
    }

    pub fn astype(t: *Trace, x: T, dt: Dtype) !T {
        return t.node(t.nodes.items[x].shape.slice(), dt, &.{});
    }

    /// A prepared launch (the model backend's `xk.Prepared`): the config, kept by value.
    pub const Prepared = struct { k: Kernel, cfg: LaunchConfig };

    pub fn prepareLaunch(t: *Trace, k: Kernel, cfg: *const LaunchConfig) !Prepared {
        t.prepared_live += 1;
        return .{ .k = k, .cfg = cfg.* };
    }

    /// Records exactly what `launch(p.k, inputs, &p.cfg, out)` records, marked prepared.
    pub fn launchPrepared(t: *Trace, p: *const Prepared, inputs: []const T, out: []T) !void {
        t.prepared_launches += 1;
        try t.launch(p.k, inputs, &p.cfg, out);
        t.launches.items[t.launches.items.len - 1].prepared = true;
    }

    pub fn releasePrepared(t: *Trace, _: *Prepared) void {
        t.prepared_live -= 1;
    }

    pub fn launch(t: *Trace, k: Kernel, inputs: []const T, cfg: *const LaunchConfig, out: []T) !void {
        var l: Launch = .{ .k = k, .cfg = cfg.*, .n_in = inputs.len };
        @memcpy(l.inputs[0..inputs.len], inputs);
        const li: u32 = @intCast(t.launches.items.len);
        for (out, 0..) |*o, i| {
            o.* = t.with(try t.node(cfg.out_shapes[i][0..cfg.out_ranks[i]], cfg.out_dtypes[i], &.{}), .{ .out = .{ li, @intCast(i) } });
            l.outs[i] = o.*;
        }
        try t.launches.append(t.a, l);
        try t.log.append(t.a, .{ .launch = li });
    }

    pub fn evalAll(t: *Trace, xs: []const T) !void {
        const d = try t.a.dupe(T, xs);
        errdefer t.a.free(d);
        try t.log.append(t.a, .{ .eval = d });
    }

    pub fn asyncEval(t: *Trace, xs: []const T) !void {
        const d = try t.a.dupe(T, xs);
        errdefer t.a.free(d);
        try t.log.append(t.a, .{ .async_eval = d });
    }

    pub fn concat(t: *Trace, xs: []const T, axis: c_int) !T {
        std.debug.assert(axis == 0);
        var s = t.nodes.items[xs[0]].shape;
        s.d[0] = 0;
        for (xs) |x| s.d[0] += t.nodes.items[x].shape.d[0];
        const d = try t.a.dupe(T, xs);
        errdefer t.a.free(d);
        try t.log.append(t.a, .{ .concat = d });
        t.cats += 1;
        return t.with(try t.node(s.slice(), t.nodes.items[xs[0]].dtype, &.{}), .{ .cat = t.cats - 1 });
    }

    pub fn take(t: *Trace, x: T, idx: T, axis: c_int) !T {
        std.debug.assert(axis == 0);
        var s = t.nodes.items[x].shape;
        s.d[0] = t.nodes.items[idx].shape.d[0];
        try t.log.append(t.a, .{ .take = .{ x, idx } });
        t.takes += 1;
        return t.with(try t.node(s.slice(), t.nodes.items[x].dtype, &.{}), .{ .take = t.takes - 1 });
    }

    /// A caller array: `e`'s input `name` at `vars`.
    fn arg(t: *Trace, e: *const Entry, comptime name: []const u8, vars: *const Vars) !T {
        const a = argOf(e, name);
        var shape: [xk.max_rank]c_int = undefined;
        for (a.shape, 0..) |d, i| shape[i] = @intCast(d.eval(vars));
        return t.node(shape[0..a.shape.len], a.dtype, &.{});
    }

    fn back(t: *const Trace, n: usize) *const Launch {
        return &t.launches.items[t.launches.items.len - n];
    }
};

fn testRegistry() !xk.Registry {
    var diag: xk.Diag = .{};
    return xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
}

/// A kernel a route tables per M (rows 1..8 / 32 / 48: the decode kernels); the prefill kernels
/// (rows up to 2^20, or no row bound) launch per call.
fn tabled(e: *const Entry) bool {
    const b = e.bounds.get(.rows) orelse return false;
    return b[1] <= 48;
}

/// Every decode launch came from a prepared config, every prefill launch from a per-call one.
fn expectPreparedDecode(t: *const Trace, reg: *const xk.Registry) !void {
    for (t.launches.items) |l| if (tabled(reg.get(l.k)) != l.prepared) {
        std.debug.print("launch of {t}: prepared {}\n", .{ l.k, l.prepared });
        return error.TestUnexpectedResult;
    };
}

fn sampleAt(e: *const Entry, site: ?[]const u8, rows: u64) *const xk.Sample {
    for (e.samples) |*s| {
        const same_site = if (site) |a| (if (s.site) |b| std.mem.eql(u8, a, b) else false) else s.site == null;
        if (same_site and s.vars.get(.rows) == rows) return s;
    }
    unreachable;
}

/// `l` launched `e` with `inputs` (in the kernel's order) at the lane's own launch `s`.
fn expectLaunch(l: *const Trace.Launch, e: *const Entry, s: *const xk.Sample, inputs: []const Trace.T) !void {
    try testing.expectEqual(e.kernel, l.k);
    try testing.expectEqualSlices(Trace.T, inputs, l.inputs[0..l.n_in]);
    const cfg = &l.cfg;
    try testing.expectEqual(s.grid, cfg.grid);
    try testing.expectEqual(s.threadgroup, cfg.threadgroup);
    try testing.expectEqual(s.output_shapes.len, cfg.n_out);
    for (s.output_shapes, s.output_dtypes, 0..) |shape, dt, i| {
        try testing.expectEqual(shape.len, cfg.out_ranks[i]);
        for (shape, 0..) |d, j| try testing.expectEqual(@as(c_int, @intCast(d)), cfg.out_shapes[i][j]);
        try testing.expectEqual(dt, cfg.out_dtypes[i]);
    }
    try testing.expectEqual(s.template.len, cfg.template.len);
    for (s.template, cfg.template) |want, got| {
        try testing.expectEqualStrings(want.name, got.name);
        try testing.expectEqual(want.value, got.value);
    }
}

test "dsv41 kernels ops: every route launches its lane's calls at the lane's own sizes, arguments in order" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var hit: std.EnumSet(Kernel) = .empty;

    {
        const pe, const te = .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) };
        const w, const bias = .{ try t.arg(pe, "w", &no_vars), try t.arg(te, "bias", &no_vars) };
        var r = try Router(Trace).init(&t, &reg, w, bias, null);
        defer r.deinit(&t);
        for (pe.samples) |*s| {
            const x = try t.arg(pe, "x", &s.vars);
            _ = try r.call(&t, x);
            try expectLaunch(t.back(2), pe, s, &.{ x, w });
            try expectLaunch(t.back(1), te, sampleAt(te, null, s.vars.get(.rows)), &.{ t.back(2).outs[0], bias });
        }
    }
    {
        const pe, const fe = .{ reg.get(.q3rc_premix_part), reg.get(.q3rc_premix_fin) };
        const w = try t.arg(pe, "w", &no_vars);
        var r = try Premix(Trace).init(&t, &reg, w, null);
        defer r.deinit(&t);
        for (pe.samples) |*s| {
            const x = try t.arg(pe, "x", &s.vars);
            _ = try r.call(&t, x);
            try expectLaunch(t.back(2), pe, s, &.{ x, w });
            try expectLaunch(t.back(1), fe, sampleAt(fe, null, s.vars.get(.rows)), &.{t.back(2).outs[0]});
        }
    }
    {
        const e = reg.get(.q3dk_sinkhorn16_hc4_it20);
        var r = try Sinkhorn(Trace).init(&t, &reg);
        defer r.deinit(&t);
        for (e.samples) |*s| {
            const n: c_int = @intCast(s.vars.get(.rows));
            const comb = try t.node(&.{ n, 16 }, .float32, &.{});
            const y = try r.call(&t, comb);
            const l = t.back(1);
            try expectLaunch(l, e, s, &.{ l.inputs[0], r.nmat[@intCast(n - 1)] });
            try testing.expectEqualSlices(c_int, &.{ n, 4, 4 }, t.shapeOf(l.inputs[0]).slice());
            try testing.expectEqual(n, std.mem.bytesToValue(i32, t.nodes.items[l.inputs[1]].bytes));
            try testing.expectEqualSlices(c_int, &.{ n, 16 }, t.shapeOf(y).slice());
        }
        for (r.k3.samples) |*s| {
            const n: c_int = @intCast(s.vars.get(.rows));
            if (n <= Sinkhorn(Trace).max_mats) continue;
            _ = try r.call(&t, try t.node(&.{ n, 4, 4 }, .float32, &.{}));
            const l = t.back(1);
            try expectLaunch(l, r.k3, s, &.{ l.inputs[0], l.inputs[1] });
            try testing.expectEqual(n, std.mem.bytesToValue(i32, t.nodes.items[l.inputs[1]].bytes));
        }
    }
    {
        const q_norm, const kv_norm = .{ try t.node(&.{1280}, .bfloat16, &.{}), try t.node(&.{512}, .bfloat16, &.{}) };
        var r = try FusedProj(Trace).init(&t, &reg, q_norm, kv_norm, null);
        defer r.deinit(&t);
        for (r.rms.samples) |*s| {
            const m: c_int = @intCast(s.vars.get(.rows));
            const x = try t.node(&.{ 1, m, 1280 }, .bfloat16, &.{});
            const y = try r.qNorm(&t, x);
            const l = t.back(1);
            try expectLaunch(l, r.rms, s, &.{ l.inputs[0], q_norm, r.rms_statics.arrays[2] });
            try testing.expectEqualSlices(c_int, &.{ 1, m, 1280 }, t.shapeOf(y).slice());
            const cos, const sin = .{ try t.node(&.{ m, 32 }, .float32, &.{}), try t.node(&.{ m, 32 }, .float32, &.{}) };
            _ = try r.kvNormRope(&t, try t.node(&.{ 1, m, 512 }, .bfloat16, &.{}), cos, sin);
            const k = t.back(1);
            try expectLaunch(k, r.rms_rope, sampleAt(r.rms_rope, null, @intCast(m)), &.{ k.inputs[0], kv_norm, r.rope_statics.arrays[2], cos, sin, r.ints[@intCast(m - 1)] });
            for ([_]RopeDir{ .fwd, .inv }) |dir| {
                const e = if (dir == .fwd) r.fwd else r.inv;
                _ = try r.ropeHeads(&t, try t.node(&.{ 1, m, 64, 512 }, if (dir == .fwd) .bfloat16 else .float32, &.{}), cos, sin, dir);
                const h = t.back(1);
                try expectLaunch(h, e, sampleAt(e, null, @intCast(m)), &.{ h.inputs[0], cos, sin, r.ints[@intCast(m - 1)], r.ints[@intCast(m - 1)] });
            }
        }
    }
    {
        const e = reg.get(.q3rc_mxfp8_fma);
        for (std.enums.values(RcSite)) |site| {
            var vars: Vars = .initFill(0);
            xk.siteVars(e.site(@tagName(site)).?, &vars);
            const w, const sc = .{ try t.arg(e, "w", &vars), try t.arg(e, "scales", &vars) };
            var r = try RcProj(Trace).init(&t, &reg, site, w, sc, null);
            defer r.deinit(&t);
            var n: usize = 0;
            for (e.samples) |*s| {
                if (!std.mem.eql(u8, s.site.?, @tagName(site))) continue;
                const x = try t.arg(e, "x", &s.vars);
                _ = try r.call(&t, x);
                try expectLaunch(t.back(1), e, s, &.{ w, sc, x });
                n += 1;
            }
            try testing.expect(n >= 4);
        }
    }
    {
        const ce, const le, const fe, const me = .{ reg.get(.q3ht_combine), reg.get(.q3ht_collapse_norm), reg.get(.q3ht_combine_collapse_norm), reg.get(.q3ht_mixfin) };
        var r = try HcTape(Trace).init(&t, &reg, .bfloat16, null);
        defer r.deinit(&t);
        for (ce.samples) |*s| {
            const v = &s.vars;
            const x, const rr, const post, const comb = .{ try t.arg(ce, "x", v), try t.arg(ce, "r", v), try t.arg(ce, "post", v), try t.arg(ce, "comb", v) };
            const pre, const w = .{ try t.arg(fe, "pre", v), try t.arg(fe, "w", v) };
            _ = try r.combine(&t, x, rr, post, comb);
            try expectLaunch(t.back(1), ce, s, &.{ x, rr, post, comb });
            _ = try r.collapseNorm(&t, rr, pre, w);
            try expectLaunch(t.back(1), le, sampleAt(le, null, v.get(.rows)), &.{ rr, pre, w });
            _ = try r.combineCollapseNorm(&t, x, rr, post, comb, pre, w);
            try expectLaunch(t.back(1), fe, sampleAt(fe, null, v.get(.rows)), &.{ x, rr, post, comb, pre, w });
            const mm, const ssq, const scale, const base = .{ try t.arg(me, "mm", v), try t.arg(me, "ssq", v), try t.arg(me, "scale", v), try t.arg(me, "base", v) };
            _ = try r.mixfin(&t, mm, ssq, scale, base);
            try expectLaunch(t.back(1), me, sampleAt(me, null, v.get(.rows)), &.{ mm, ssq, scale, base });
        }
    }
    {
        var r = try Gemv(Trace).init(&t, &reg);
        defer r.deinit(&t);
        for ([_]Proj{ .gate, .down }) |proj| {
            const e = if (proj == .down) r.dn else r.gu;
            const st = if (proj == .down) &r.dn_statics else &r.gu_statics;
            for (e.samples) |*s| {
                const xh, const ids, const code = .{ try t.arg(e, "xh", &s.vars), try t.arg(e, "ids", &s.vars), try t.arg(e, "code", &s.vars) };
                _ = try r.project(&t, proj, xh, ids, code);
                var want: [9]Trace.T = undefined;
                want[0..3].* = .{ xh, ids, code };
                @memcpy(want[3..], st.arrays[3..9]);
                try expectLaunch(t.back(1), e, s, &want);
            }
        }
    }
    {
        var r = try RinPrep(Trace).init(&t, &reg);
        defer r.deinit(&t);
        for (r.in_rin_e.samples) |*s| {
            const e, const v = .{ r.in_rin_e, &s.vars };
            const x, const tok, const rg, const ru, const ids = .{ try t.arg(e, "x", v), try t.arg(e, "tok", v), try t.arg(e, "rg", v), try t.arg(e, "ru", v), try t.arg(e, "ids", v) };
            _ = try r.inRin(&t, x, tok, rg, ru, ids);
            try expectLaunch(t.back(1), e, s, &.{ x, tok, rg, ru, ids });
        }
        for (r.gu_epi_e.samples) |*s| {
            const e, const v = .{ r.gu_epi_e, &s.vars };
            const zg, const zu, const rg, const ru, const ids = .{ try t.arg(e, "zg", v), try t.arg(e, "zu", v), try t.arg(e, "rg", v), try t.arg(e, "ru", v), try t.arg(e, "ids", v) };
            _ = try r.guEpi(&t, zg, zu, rg, ru, ids);
            try expectLaunch(t.back(1), e, s, &.{ zg, zu, rg, ru, ids });
        }
        for (r.din_rin_e.samples) |*s| {
            const e, const v = .{ r.din_rin_e, &s.vars };
            const hid, const rn, const ids = .{ try t.arg(e, "hid", v), try t.arg(e, "rn", v), try t.arg(e, "ids", v) };
            _ = try r.dinRin(&t, hid, rn, ids);
            try expectLaunch(t.back(1), e, s, &.{ hid, rn, ids });
        }
        for (r.dpost_e.samples) |*s| {
            const e, const v = .{ r.dpost_e, &s.vars };
            const zd, const rd, const ids = .{ try t.arg(e, "zd", v), try t.arg(e, "rd", v), try t.arg(e, "ids", v) };
            _ = try r.dpost(&t, zd, rd, ids);
            try expectLaunch(t.back(1), e, s, &.{ zd, rd, ids });
        }
    }
    {
        const r = Rebuild(Trace).init(&reg);
        const e = r.e;
        for (e.samples) |*s| {
            const v = &s.vars;
            const gate: ProjArrays(Trace.T) = .{ .code = try t.arg(e, "code_g", v), .rout = try t.arg(e, "rout_g", v), .rin = try t.arg(e, "rin_g", v) };
            const up: ProjArrays(Trace.T) = .{ .code = try t.arg(e, "code_u", v), .rout = try t.arg(e, "rout_u", v), .rin = try t.arg(e, "rin_u", v) };
            const down: ProjArrays(Trace.T) = .{ .code = try t.arg(e, "code_d", v), .rout = try t.arg(e, "rout_d", v), .rin = try t.arg(e, "rin_d", v) };
            const slots = try t.arg(e, "slots", v);
            _ = try r.call(&t, gate, up, down, slots, @intCast(v.get(.experts)));
            try expectLaunch(t.back(1), e, s, &.{ gate.code, gate.rout, gate.rin, up.code, up.rout, up.rin, down.code, down.rout, down.rin, slots });
        }
    }
    {
        const r = DigX(Trace).init(&reg);
        for (r.gemm_gu.samples) |*s| {
            const e, const v = .{ r.gemm_gu, &s.vars };
            const x0, const x1, const c0, const c1, const tbl = .{ try t.arg(e, "x0", v), try t.arg(e, "x1", v), try t.arg(e, "code0", v), try t.arg(e, "code1", v), try t.arg(e, "tbl", v) };
            _ = try r.gemmGateUp(&t, x0, x1, c0, c1, tbl, @intCast(v.get(.tgs)));
            try expectLaunch(t.back(1), e, s, &.{ x0, x1, c0, c1, tbl });
        }
        for (r.gemm_dn.samples) |*s| {
            const e, const v = .{ r.gemm_dn, &s.vars };
            const x, const c0, const tbl = .{ try t.arg(e, "x", v), try t.arg(e, "code0", v), try t.arg(e, "tbl", v) };
            _ = try r.gemmDown(&t, x, c0, tbl, @intCast(v.get(.tgs)));
            try expectLaunch(t.back(1), e, s, &.{ x, c0, tbl });
        }
        for (r.take2_e.samples) |*s| {
            const e, const v = .{ r.take2_e, &s.vars };
            const act, const ridx, const rhs, const slots, const rg, const ru = .{ try t.arg(e, "act", v), try t.arg(e, "ridx", v), try t.arg(e, "rhs", v), try t.arg(e, "slots", v), try t.arg(e, "rin_g", v), try t.arg(e, "rin_u", v) };
            _ = try r.take2(&t, act, ridx, rhs, slots, rg, ru);
            try expectLaunch(t.back(1), e, s, &.{ act, ridx, rhs, slots, rg, ru });
        }
        for (r.roundx_e.samples) |*s| {
            const e, const v = .{ r.roundx_e, &s.vars };
            const act, const rhs, const slots, const rin = .{ try t.arg(e, "act", v), try t.arg(e, "rhs", v), try t.arg(e, "slots", v), try t.arg(e, "rin", v) };
            _ = try r.roundx(&t, act, rhs, slots, rin);
            try expectLaunch(t.back(1), e, s, &.{ act, rhs, slots, rin });
        }
        for (r.onepass_e.samples) |*s| {
            const e, const v = .{ r.onepass_e, &s.vars };
            const z0, const z1, const rhs, const tbl, const r0, const r1, const r2 = .{ try t.arg(e, "z0", v), try t.arg(e, "z1", v), try t.arg(e, "rhs", v), try t.arg(e, "tbl", v), try t.arg(e, "rout0", v), try t.arg(e, "rout1", v), try t.arg(e, "rin2", v) };
            _ = try r.onePass(&t, z0, z1, rhs, tbl, r0, r1, r2);
            try expectLaunch(t.back(1), e, s, &.{ z0, z1, rhs, tbl, r0, r1, r2 });
        }
        for (r.widen2_e.samples) |*s| {
            const e, const v = .{ r.widen2_e, &s.vars };
            const ag, const au, const rhs, const slots, const rg, const ru = .{ try t.arg(e, "act_g", v), try t.arg(e, "act_u", v), try t.arg(e, "rhs", v), try t.arg(e, "slots", v), try t.arg(e, "rout_g", v), try t.arg(e, "rout_u", v) };
            _ = try r.widen2(&t, ag, au, rhs, slots, rg, ru);
            try expectLaunch(t.back(1), e, s, &.{ ag, au, rhs, slots, rg, ru });
        }
        for (r.widen1_e.samples) |*s| {
            const e, const v = .{ r.widen1_e, &s.vars };
            const act, const rhs, const slots, const rout = .{ try t.arg(e, "act", v), try t.arg(e, "rhs", v), try t.arg(e, "slots", v), try t.arg(e, "rout", v) };
            _ = try r.widen1(&t, act, rhs, slots, rout);
            try expectLaunch(t.back(1), e, s, &.{ act, rhs, slots, rout });
        }
    }
    for (t.launches.items) |l| hit.insert(l.k);
    // every kernel of record is a route's except the DIG-X golden-tile texts (install self-check
    // only) and the DRAFTRC entries (the draft routes' own test covers those)
    for (reg.entries) |e| {
        const unrouted = std.mem.startsWith(u8, @tagName(e.kernel), "q3_exl3_dig_decmat_") or std.mem.startsWith(u8, e.family, "draftrc_");
        try testing.expectEqual(!unrouted, hit.contains(e.kernel));
    }
    // the decode routes launched their prepared configs only (no config built per call)
    try expectPreparedDecode(&t, &reg);
    try testing.expect(t.prepared_launches > 0);
    try testing.expectEqual(@as(isize, 0), t.keeps);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

test "dsv41 kernels ops: a bound array of another dtype or shape is refused, by name" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var diag: xk.Diag = .{};
    const f32w = try t.node(&.{ 384, 5120 }, .float32, &.{});
    const bias = try t.node(&.{384}, .float32, &.{});
    try testing.expectError(error.RouteInput, Router(Trace).init(&t, &reg, f32w, bias, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_gate_part input w") != null);
    const short_w = try t.node(&.{ 24, 10240 }, .float32, &.{});
    try testing.expectError(error.RouteInput, Premix(Trace).init(&t, &reg, short_w, &diag));
    const w = try t.node(&.{ 1280, 1280 }, .uint32, &.{});
    const sc = try t.node(&.{ 1280, 40 }, .uint8, &.{});
    try testing.expectError(error.RouteInput, RcProj(Trace).init(&t, &reg, .wq_a, w, sc, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_mxfp8_fma input scales") != null);
    try testing.expectError(error.TemplateNotRegistered, HcTape(Trace).init(&t, &reg, .float16, &diag));
    const qn32 = try t.node(&.{1280}, .float32, &.{});
    try testing.expectError(error.RouteInput, FusedProj(Trace).init(&t, &reg, qn32, try t.node(&.{512}, .bfloat16, &.{}), &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "mtplx_dsv41_fp_rmsnorm_tg128_d1280 input weight") != null);
    const bank: ProjArrays(Trace.T) = .{
        .code = try t.node(&.{ 4, 144, 320, 48 }, .int16, &.{}),
        .rout = try t.node(&.{ 4, 5120 }, .float32, &.{}),
        .rin = try t.node(&.{ 4, 2304 }, .float16, &.{}),
    };
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .down, bank, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3_moeprep_dpost input rd") != null);
    const ok: ProjArrays(Trace.T) = .{ .code = bank.code, .rout = try t.node(&.{ 4, 5120 }, .float16, &.{}), .rin = bank.rin };
    try checkBank(Trace, &t, &reg, .down, ok, &diag);
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .gate, ok, &diag));
    try testing.expectEqual(@as(isize, 0), t.keeps);
}

test "dsv41 kernels ops: plan routes refuse rows outside their tables" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    const w = try t.node(&.{ 512, 1280 }, .uint32, &.{});
    const sc = try t.node(&.{ 512, 160 }, .uint8, &.{});
    var r = try RcProj(Trace).init(&t, &reg, .wkv, w, sc, null);
    defer r.deinit(&t);
    try testing.expectError(error.RowsOutOfPlan, r.call(&t, try t.node(&.{ 9, 5120 }, .bfloat16, &.{})));
    var fp = try FusedProj(Trace).init(&t, &reg, try t.node(&.{1280}, .bfloat16, &.{}), try t.node(&.{512}, .bfloat16, &.{}), null);
    defer fp.deinit(&t);
    try testing.expectError(error.RowsOutOfPlan, fp.qNorm(&t, try t.node(&.{ 9, 1280 }, .bfloat16, &.{})));
    try testing.expectEqual(@as(usize, 0), t.launches.items.len);
    // 33 matrices leave the 16-lane plans for the stock K3 text at threadgroup min(n, 256)
    var s = try Sinkhorn(Trace).init(&t, &reg);
    defer s.deinit(&t);
    _ = try s.call(&t, try t.node(&.{ 33, 4, 4 }, .float32, &.{}));
    try testing.expectEqual(Kernel.mtplx_dsv4_sinkhorn_hc4_it20, t.back(1).k);
    try testing.expectEqual([3]u32{ 33, 1, 1 }, t.back(1).cfg.threadgroup);
}

test "dsv41 kernels ops: the routes carry the lanes' installed configuration" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    // Q3_DECODE_RCPROJ_INSTALL geometry: (R, KS, RG, KV) per site, R 1 at M 7 / 8; the head = wq_b's form
    const geo = [_]struct { RcSite, [4]i32 }{
        .{ .wq_a, .{ 2, 2, 1, 4 } }, .{ .wkv, .{ 2, 4, 1, 4 } }, .{ .wq_b, .{ 2, 1, 2, 4 } },
        .{ .wo_b, .{ 2, 2, 1, 4 } }, .{ .woa, .{ 2, 1, 2, 4 } }, .{ .head, .{ 2, 1, 2, 4 } },
    };
    const e = reg.get(.q3rc_mxfp8_fma);
    for (geo) |row| {
        var vars: Vars = .initFill(0);
        xk.siteVars(e.site(@tagName(row[0])).?, &vars);
        var r = try RcProj(Trace).init(&t, &reg, row[0], try t.arg(e, "w", &vars), try t.arg(e, "scales", &vars), null);
        defer r.deinit(&t);
        for (&r.plans.cfg, 1..) |*p, m| {
            const want = [4]i32{ if (m >= 7) 1 else row[1][0], row[1][1], row[1][2], row[1][3] };
            for ([_][]const u8{ "R", "KS", "RG", "KV" }, want) |name, v| try testing.expectEqual(v, templateInt(p.template, name));
            try testing.expectEqual(@as(i32, @intCast(m)), templateInt(p.template, "M"));
        }
    }
    // Q3_DECODE_RCTAIL_INSTALL router plan: N 384, K 5120, KP 512, P 10, top-6
    const part = reg.get(.q3rc_gate_part).template;
    const tail = reg.get(.q3rc_router_tail).template;
    try testing.expectEqual(@as(i32, 384), templateInt(part, "N"));
    try testing.expectEqual(@as(i32, 5120), templateInt(part, "K"));
    try testing.expectEqual(@as(i32, 512), templateInt(part, "KP"));
    try testing.expectEqual(@as(i32, 10), templateInt(tail, "P"));
    try testing.expectEqual(@as(i32, 6), templateInt(tail, "TOPK"));
    // q3_exl3_decode_kernels: the lane_uint launch's constants (MCG_MULT, 0, LOP3_MASK, LOP3_XOR)
    var gv = try Gemv(Trace).init(&t, &reg);
    defer gv.deinit(&t);
    const cb = t.nodes.items[gv.gu_statics.arrays[3]].bytes;
    var cbv: [4]u32 = undefined;
    for (&cbv, 0..) |*v, i| v.* = std.mem.readInt(u32, cb[i * 4 ..][0..4], .little);
    try testing.expectEqualSlices(u32, &.{ 0xCBAC1FED, 0, 0x8FFF8FFF, 0x3B603B60 }, &cbv);
    for (gv.dn_statics.arrays[4..9]) |z| try testing.expect(std.mem.allEqual(u8, t.nodes.items[z].bytes, 0));
    // config.json rms_norm_eps 1e-20 (a 0-d f32 input); every K36 kernel stores bf16
    var fp = try FusedProj(Trace).init(&t, &reg, try t.node(&.{1280}, .bfloat16, &.{}), try t.node(&.{512}, .bfloat16, &.{}), null);
    defer fp.deinit(&t);
    const eps = std.mem.bytesToValue(f32, t.nodes.items[fp.rms_statics.arrays[2]].bytes);
    try testing.expectEqual(@as(f32, 1e-20), eps);
    for ([_]*const Entry{ fp.rms, fp.rms_rope, fp.fwd, fp.inv }) |k36| try testing.expectEqual(Dtype.bfloat16, k36.template[0].value.dtype);
}

fn templateInt(tmpl: []const xk.TemplateArg, name: []const u8) i32 {
    for (tmpl) |a| if (std.mem.eql(u8, a.name, name)) return a.value.int;
    unreachable;
}

test "dsv41 kernels ops: wave tables are the lanes' (DIG wave_table, rebuild slots8)" {
    var reg = try testRegistry();
    defer reg.deinit();
    const dx = DigX(Trace).init(&reg);
    try testing.expectEqual(@as(u32, 72), dx.digTiles(.gate_up));
    try testing.expectEqual(@as(u32, 80), dx.digTiles(.down));
    const w = digTable(&.{ .{ .slot = 3, .rows = 70 }, .{ .slot = 0, .rows = 37 }, .{ .slot = 2, .rows = 20 } }, 72);
    try testing.expectEqualSlices(i32, &.{ 3, 0, 2, 0 }, w.table[0..4]);
    try testing.expectEqualSlices(i32, &.{ 0, 70, 107, 0 }, w.table[16..20]);
    try testing.expectEqualSlices(i32, &.{ 70, 37, 20, 0 }, w.table[32..36]);
    try testing.expectEqualSlices(i32, &.{ 0, 144, 216, std.math.maxInt(i32) }, w.table[48..52]);
    try testing.expectEqual(@as(i32, 3), w.table[64]);
    try testing.expectEqual(@as(u32, 288), w.tgs);
    try testing.expectEqual(@as(i32, 288), w.table[65]);
    try testing.expectEqualSlices(i32, &.{ 3, 0, 2, 3, 3 }, rebuildSlots(&.{ 3, 0, 2 })[0..5]);
}

test "dsv41 kernels ops: the seam calls cast and reshape as the lanes' seams do" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var router = try Router(Trace).init(&t, &reg, try t.node(&.{ 384, 5120 }, .bfloat16, &.{}), try t.node(&.{384}, .float32, &.{}), null);
    defer router.deinit(&t);
    const out = try router.gateTopk(&t, try t.node(&.{ 6, 5120 }, .bfloat16, &.{}));
    try testing.expectEqual(Dtype.float32, t.dtypeOf(t.back(2).inputs[0]));
    try testing.expectEqualSlices(c_int, &.{ 6, 6 }, t.shapeOf(out[1]).slice());
    var premix = try Premix(Trace).init(&t, &reg, try t.node(&.{ 24, 20480 }, .float32, &.{}), null);
    defer premix.deinit(&t);
    const mixes = try premix.mm(&t, try t.node(&.{ 1, 6, 20480 }, .float32, &.{}));
    try testing.expectEqualSlices(c_int, &.{ 6, 20480 }, t.shapeOf(t.back(2).inputs[0]).slice());
    try testing.expectEqualSlices(c_int, &.{ 1, 6, 24 }, t.shapeOf(mixes).slice());
    const flat = try premix.mm(&t, try t.node(&.{ 5, 20480 }, .float32, &.{}));
    try testing.expectEqual(t.back(1).outs[0], flat);
    var wq_b = try RcProj(Trace).init(&t, &reg, .wq_b, try t.node(&.{ 32768, 320 }, .uint32, &.{}), try t.node(&.{ 32768, 40 }, .uint8, &.{}), null);
    defer wq_b.deinit(&t);
    const q = try wq_b.linear(&t, try t.node(&.{ 1, 7, 1280 }, .bfloat16, &.{}));
    try testing.expectEqualSlices(c_int, &.{ 7, 1280 }, t.shapeOf(t.back(1).inputs[2]).slice());
    try testing.expectEqualSlices(c_int, &.{ 1, 7, 32768 }, t.shapeOf(q).slice());
    try testing.expectEqual(@as(i32, 7), templateInt(t.back(1).cfg.template, "M"));
}

// ── The prefill wave route vs the lane of record's own dispatch (dump_prefill_waves.py --samples) ──

const prefill_samples = @embedFile("fixtures/dsv41_prefill_wave_samples.json");
const JRoute = struct { seed: u64, slots: []const u32, counts: []const u32 };
const JCall = struct { name: []const u8, a_rows: u32, route: JRoute, events: []const []const u8, ret: []const u8, ret_shape: []const i64 };
const JShapeCfg = struct { wave: u32, inflight: u32, row_budget: u32, carry_rows: u32 };
const JSampleCase = struct { case: []const u8, shape: JShapeCfg, cap: u32, calls: []const JCall, finish: []const []const u8 };
const JSamples = struct { format: []const u8, cases: []const JSampleCase };

/// dump_prefill_waves.route_rows: slot j repeated counts[j] times, then Fisher-Yates from the end
/// with j = splitmix64(seed) output k % (i + 1), k = 0, 1, ... as i runs A - 1 .. 1.
fn routeRows(a: Allocator, seed: u64, slots: []const u32, counts: []const u32) ![]u32 {
    var n: usize = 0;
    for (counts) |c| n += c;
    const rows = try a.alloc(u32, n);
    var k: usize = 0;
    for (slots, counts) |s, c| for (0..c) |_| {
        rows[k] = s;
        k += 1;
    };
    var st = seed;
    var i = n;
    while (i > 1) {
        i -= 1;
        const j: usize = @intCast(xk.splitmix64(&st) % (i + 1));
        std.mem.swap(u32, &rows[i], &rows[j]);
    }
    return rows;
}

fn shapeStr(out: *std.ArrayList(u8), a: Allocator, s: []const c_int) !void {
    try out.append(a, '[');
    for (s, 0..) |d, i| try out.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
    try out.append(a, ']');
}

/// The lane samples' canonical reference of a node: ext:<name>, host:<dtype>:<shape>:<sha256[0..16]>,
/// L<launch>.<output>, C<concat>, T<take>; a view appends @<its shape>.
fn traceRef(t: *const Trace, x: Trace.T, out: *std.ArrayList(u8)) !void {
    const a = t.a;
    var n = x;
    var view: ?Shape = null;
    while (t.nodes.items[n].origin == .view) {
        if (view == null) view = t.nodes.items[n].shape;
        n = t.nodes.items[n].origin.view;
    }
    const nd = &t.nodes.items[n];
    switch (nd.origin) {
        .host => {
            var d: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(nd.bytes, &d, .{});
            const hex = std.fmt.bytesToHex(d, .lower);
            try out.print(a, "host:{t}:", .{nd.dtype});
            try shapeStr(out, a, nd.shape.slice());
            try out.print(a, ":{s}", .{hex[0..16]});
        },
        .ext => |name| try out.print(a, "ext:{s}", .{name}),
        .out => |o| try out.print(a, "L{d}.{d}", .{ o[0], o[1] }),
        .cat => |i| try out.print(a, "C{d}", .{i}),
        .take => |i| try out.print(a, "T{d}", .{i}),
        .none, .view => return error.UntracedNode,
    }
    if (view) |s| {
        try out.append(a, '@');
        try shapeStr(out, a, s.slice());
    }
}

fn traceRefs(t: *const Trace, xs: []const Trace.T, out: *std.ArrayList(u8)) !void {
    for (xs, 0..) |x, i| {
        if (i > 0) try out.append(t.a, ';');
        try traceRef(t, x, out);
    }
}

fn traceEvent(t: *const Trace, e: Trace.Ev, out: *std.ArrayList(u8)) !void {
    const a = t.a;
    switch (e) {
        .launch => |li| {
            const l = &t.launches.items[li];
            const c = &l.cfg;
            try out.print(a, "launch {t} g={d},{d},{d} t={d},{d},{d} in=", .{ l.k, c.grid[0], c.grid[1], c.grid[2], c.threadgroup[0], c.threadgroup[1], c.threadgroup[2] });
            try traceRefs(t, l.inputs[0..l.n_in], out);
            try out.appendSlice(a, " out=");
            for (0..c.n_out) |i| {
                if (i > 0) try out.append(a, ';');
                try out.print(a, "{t}", .{c.out_dtypes[i]});
                try shapeStr(out, a, c.out_shapes[i][0..c.out_ranks[i]]);
            }
            if (c.template.len > 0) return error.TemplateOnPrefillRoute;
        },
        .eval => |xs| {
            try out.appendSlice(a, "eval ");
            try traceRefs(t, xs, out);
        },
        .async_eval => |xs| {
            try out.appendSlice(a, "async ");
            try traceRefs(t, xs, out);
        },
        .concat => |xs| {
            try out.appendSlice(a, "concat ");
            try traceRefs(t, xs, out);
        },
        .take => |xi| {
            try out.appendSlice(a, "take ");
            try traceRef(t, xi[0], out);
            try out.append(a, ' ');
            try traceRef(t, xi[1], out);
        },
    }
}

/// The trace's log from `from` on is the lane's `want`, event for event.
fn expectEvents(t: *const Trace, from: usize, want: []const []const u8, case: []const u8, what: []const u8) !void {
    const got = t.log.items[from..];
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.a);
    for (got[0..@min(got.len, want.len)], want[0..@min(got.len, want.len)], 0..) |e, w, i| {
        buf.clearRetainingCapacity();
        try traceEvent(t, e, &buf);
        if (!std.mem.eql(u8, buf.items, w)) {
            std.debug.print("prefill {s} {s} event {d}:\n  route: {s}\n  lane:  {s}\n", .{ case, what, i, buf.items, w });
            return error.TestExpectedEqual;
        }
    }
    if (got.len != want.len) {
        std.debug.print("prefill {s} {s}: {d} events, the lane {d}\n", .{ case, what, got.len, want.len });
        return error.TestExpectedEqual;
    }
}

/// Each wave's `resetTo` runs right after the wave's five launches and its eval / async_eval
/// (only drains of older waves may come between) and frees those launches' outputs; the last
/// `resetTo` of the call follows the join (concatenate, take, the result's eval) and frees the
/// concatenation and the take.
fn expectLifecycle(t: *const Trace, log0: usize, resets0: usize, waves: usize, case: []const u8, what: []const u8) !void {
    var at = log0;
    for (t.freed.items[resets0..][0..waves], 0..) |r, w| {
        var launches: usize = 0;
        var last_out: ?Trace.T = null;
        var own_eval = false;
        for (t.log.items[at..r.at]) |e| switch (e) {
            .launch => |li| {
                launches += 1;
                const l = &t.launches.items[li];
                last_out = l.outs[0];
                for (l.outs[0..l.cfg.n_out]) |o| if (o < r.from or o >= r.to) {
                    std.debug.print("prefill {s} {s} wave {d}: launch output {d} outside its reset [{d}, {d})\n", .{ case, what, w, o, r.from, r.to });
                    return error.TestUnexpectedResult;
                };
            },
            // the wave's own eval / async_eval: its rot_widen1 output (through the reshape)
            .eval, .async_eval => |xs| own_eval = own_eval or (launches == 5 and xs.len == 1 and last_out != null and t.root(xs[0]) == last_out.?),
            .concat, .take => {
                std.debug.print("prefill {s} {s} wave {d}: a join inside a wave's reset\n", .{ case, what, w });
                return error.TestUnexpectedResult;
            },
        };
        if (launches != 5 or !own_eval) {
            std.debug.print("prefill {s} {s} wave {d}: {d} launches, own eval {} before its reset\n", .{ case, what, w, launches, own_eval });
            return error.TestUnexpectedResult;
        }
        // its output kept (the join's handle, and the in-flight queue's unless solo) before its reset
        var keeps: usize = 0;
        for (t.kept.items) |k| keeps += @intFromBool(k.resets == resets0 + w and t.root(k.node) == last_out.?);
        if (keeps < 1 or keeps > 2) {
            std.debug.print("prefill {s} {s} wave {d}: its output kept {d} times before its reset\n", .{ case, what, w, keeps });
            return error.TestUnexpectedResult;
        }
        at = r.at;
    }
    const j = t.freed.items[resets0 + waves];
    var cat = false;
    var tk = false;
    for (t.log.items[at..j.at]) |e| switch (e) {
        .launch => return error.TestUnexpectedResult,
        .concat => cat = true,
        .take => tk = true,
        else => {},
    };
    const last = t.nodes.items.len - 1;
    // the result (the take, the call's last node) kept before the join's reset
    var result_kept = false;
    for (t.kept.items) |k| result_kept = result_kept or (k.node == last and k.resets == resets0 + waves);
    if (!cat or !tk or j.to != last + 1 or !result_kept) {
        std.debug.print("prefill {s} {s}: the join's reset [{d}, {d}) (concat {}, take {}, result kept {})\n", .{ case, what, j.from, j.to, cat, tk, result_kept });
        return error.TestUnexpectedResult;
    }
}

fn testBank(t: *Trace, cap: c_int) !BankArrays(Trace.T) {
    return .{
        .gate = .{ .code = try t.ext("gate_proj.code", &.{ cap, 320, 144, 48 }, .int16), .rout = try t.ext("gate_proj.rout", &.{ cap, 2304 }, .float16), .rin = try t.ext("gate_proj.rin", &.{ cap, 5120 }, .float16) },
        .up = .{ .code = try t.ext("up_proj.code", &.{ cap, 320, 144, 48 }, .int16), .rout = try t.ext("up_proj.rout", &.{ cap, 2304 }, .float16), .rin = try t.ext("up_proj.rin", &.{ cap, 5120 }, .float16) },
        .down = .{ .code = try t.ext("down_proj.code", &.{ cap, 144, 320, 48 }, .int16), .rout = try t.ext("down_proj.rout", &.{ cap, 5120 }, .float16), .rin = try t.ext("down_proj.rin", &.{ cap, 2304 }, .float16) },
    };
}

test "dsv41 kernels ops: the prefill wave route replays the lane's own launches, evals and joins (lane samples)" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    const parsed = try std.json.parseFromSlice(JSamples, a, prefill_samples, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx-serve-exl3-prefill-wave-samples-v1", parsed.value.format);
    var n_calls: usize = 0;
    var n_waves: usize = 0;
    for (parsed.value.cases) |*cs| {
        var t: Trace = .{ .a = a };
        defer t.deinit();
        const shape: PrefillShape = .{ .wave = cs.shape.wave, .inflight = cs.shape.inflight, .row_budget = cs.shape.row_budget, .carry_rows = cs.shape.carry_rows };
        var r = try DigXPrefill(Trace).init(a, &reg, shape, null);
        defer r.deinit(&t);
        const bank = try testBank(&t, @intCast(cs.cap));
        var mark: usize = 0;
        for (cs.calls) |*cl| {
            const slots = try routeRows(a, cl.route.seed, cl.route.slots, cl.route.counts);
            defer a.free(slots);
            try testing.expectEqual(@as(usize, cl.a_rows), slots.len);
            const act = try t.ext("act", &.{ @intCast(slots.len), 5120 }, .bfloat16);
            const launches0 = t.launches.items.len;
            const nodes0 = t.nodes.items.len;
            const resets0 = t.freed.items.len;
            const res = try r.call(&t, act, .{ .slot = slots }, bank);
            try expectEvents(&t, mark, cl.events, cs.case, cl.name);
            // the wave lifecycle: one mark / resetTo per wave and one around the join; nothing the
            // call built outlives it but the kept result and the waves still in flight
            const waves = (t.launches.items.len - launches0) / 5;
            try testing.expectEqual(waves + 1, t.freed.items.len - resets0);
            try expectLifecycle(&t, mark, resets0, waves, cs.case, cl.name);
            for (nodes0..t.nodes.items.len) |x| {
                if (!t.leaked(@intCast(x))) continue;
                std.debug.print("prefill {s} {s}: node {d} outlives the call\n", .{ cs.case, cl.name, x });
                return error.TestUnexpectedResult;
            }
            try testing.expectEqual(@as(usize, 1 + r.flight.items.len), t.held.items.len);
            try testing.expect(std.mem.indexOfScalar(Trace.T, t.held.items, res) != null);
            mark = t.log.items.len;
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(a);
            try traceRef(&t, res, &buf);
            try testing.expectEqualStrings(cl.ret, buf.items);
            const rs = t.shapeOf(res);
            for (cl.ret_shape, rs.slice()) |w, d| try testing.expectEqual(w, @as(i64, d));
            n_calls += 1;
            n_waves += waves;
            t.release(res);
        }
        try r.finish(&t);
        try expectEvents(&t, mark, cs.finish, cs.case, "finish");
        try testing.expectEqual(@as(isize, 0), t.keeps);
    }
    try testing.expect(n_calls >= 10 and n_waves >= 60);
}

test "dsv41 kernels ops: the prefill wave route refuses by name, before any launch" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    var diag: xk.Diag = .{};
    const tier = PrefillShape.tier;
    var bads = [_]PrefillShape{ tier, tier, tier, tier };
    bads[0].wave = 0;
    bads[1].wave = 17;
    bads[2].inflight = 1;
    bads[3].row_budget = 0;
    for (bads) |s| try testing.expectError(error.RouteInput, DigXPrefill(Trace).init(a, &reg, s, &diag));
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var r = try DigXPrefill(Trace).init(a, &reg, tier, &diag);
    defer r.deinit(&t);
    const bank = try testBank(&t, 8);
    const act4 = try t.ext("act", &.{ 4, 5120 }, .bfloat16);
    try testing.expectError(error.RowsOutOfPlan, r.call(&t, act4, .{ .slot = &.{} }, bank));
    try testing.expectError(error.SlotOutOfBank, r.call(&t, act4, .{ .slot = &.{ 1, 2, 8, 3 } }, bank));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "slot 8 of a 8-slot bank") != null);
    try testing.expectError(error.RouteInput, r.call(&t, act4, .{ .slot = &.{ 1, 2, 3 } }, bank));
    try testing.expectError(error.RouteInput, r.call(&t, act4, .{ .slot = &.{ 1, 2, 3 }, .act_row = &.{ 0, 4, 1 } }, bank));
    try testing.expectError(error.RouteInput, r.call(&t, act4, .{ .slot = &.{ 1, 2, 3 }, .act_row = &.{ 0, 1 } }, bank));
    try testing.expectEqual(@as(usize, 0), t.launches.items.len);
    try testing.expectEqual(@as(usize, 0), t.log.items.len);
    // a bank above the kernels' slot bound is refused where the model binds it
    const big: ProjArrays(Trace.T) = .{ .code = try t.ext("c", &.{ 4097, 320, 144, 48 }, .int16), .rout = try t.ext("r", &.{ 4097, 2304 }, .float16), .rin = try t.ext("i", &.{ 4097, 5120 }, .float16) };
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .gate, big, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "a bank of 4097 slots") != null);
    try checkBank(Trace, &t, &reg, .gate, bank.gate, &diag);
    try checkBank(Trace, &t, &reg, .down, bank.down, &diag);
}

test "dsv41 kernels ops: prefill rows read by act_row take the same act words (tokens, position / top_k)" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    const counts = [_]u32{ 900, 700, 300, 297, 40, 1 };
    const slots = try routeRows(a, 77, &.{ 5, 0, 2, 7, 3, 6 }, &counts);
    defer a.free(slots);
    const n = slots.len;
    try testing.expectEqual(@as(usize, 2238), n);
    const act_row = try a.alloc(u32, n);
    defer a.free(act_row);
    for (act_row, 0..) |*v, i| v.* = @intCast(i / 6);
    var ta: Trace = .{ .a = a };
    defer ta.deinit();
    var tb: Trace = .{ .a = a };
    defer tb.deinit();
    var ra = try DigXPrefill(Trace).init(a, &reg, .tier, null);
    defer ra.deinit(&ta);
    var rb = try DigXPrefill(Trace).init(a, &reg, .tier, null);
    defer rb.deinit(&tb);
    _ = try ra.call(&ta, try ta.ext("act", &.{ @intCast(n), 5120 }, .bfloat16), .{ .slot = slots }, try testBank(&ta, 8));
    _ = try rb.call(&tb, try tb.ext("tokens", &.{ @intCast(n / 6), 5120 }, .bfloat16), .{ .slot = slots, .act_row = act_row }, try testBank(&tb, 8));
    try testing.expectEqual(ta.launches.items.len, tb.launches.items.len);
    try testing.expectEqual(ta.log.items.len, tb.log.items.len);
    for (ta.launches.items, tb.launches.items) |la, lb| {
        try testing.expectEqual(la.k, lb.k);
        try testing.expectEqual(la.cfg.grid, lb.cfg.grid);
        if (la.k != .q3_prefill_dig_rot_take2_5120) continue;
        // ridx: assignment row p in A reads act row p; in B, act_row[p] = p / 6
        const ba, const bb = .{ ta.nodes.items[la.inputs[1]].bytes, tb.nodes.items[lb.inputs[1]].bytes };
        try testing.expectEqual(ba.len, bb.len);
        var k: usize = 0;
        while (k < ba.len) : (k += 4) try testing.expectEqual(@divTrunc(std.mem.readInt(i32, ba[k..][0..4], .little), 6), std.mem.readInt(i32, bb[k..][0..4], .little));
    }
}

// ── Startup acceptance (stub device) ──

/// `deepseek_v41_experts.MlxGemv`'s shape on the trace backend.
const TestGemv = struct {
    ctx: *const anyopaque,
    project_fn: *const fn (ctx: *const anyopaque, g: *Trace, k: u32, out_dim: u32, xh: Trace.T, ids: Trace.T, code: Trace.T) anyerror!Trace.T,

    fn project(self: TestGemv, g: *Trace, k: u32, out_dim: u32, xh: Trace.T, ids: Trace.T, code: Trace.T) !Trace.T {
        return self.project_fn(self.ctx, g, k, out_dim, xh, ids, code);
    }
};

test "dsv41 kernels ops: acceptAtStartup (stub device) judges the plan, installs the launcher, hands out the GEMV route" {
    const a = testing.allocator;
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var diag: xk.Diag = .{};
    const acc = try acceptAtStartup(Trace, a, &t, .{ .device = .{ .stub = .{} } }, &diag);
    // the plan: one result per (kernel, check) of the registry, all passed
    var n: usize = 0;
    for (&acc.reg.entries) |*e| n += e.checks.count();
    try testing.expect(n >= 100);
    try testing.expectEqual(n, acc.report.results.items.len);
    try testing.expectEqual(@as(usize, 0), acc.report.failures());
    const receipt = try acc.report.writeJsonLines(a);
    defer a.free(receipt);
    try testing.expectEqual(n, std.mem.count(u8, receipt, "\n"));
    // the backend's launcher: the accepted kernels
    try testing.expectEqual(@as(*const xk.Bound, &acc.bound), t.launcher.?);
    try testing.expectEqual(&acc.reg, acc.bound.reg);
    // the GEMV route's configs: prepared once here (gate|up and down, M 1..48 each)
    try testing.expectEqual(@as(isize, 2 * 48), t.prepared_live);
    const prepared0 = t.prepared_launches;
    var calls: usize = 0;
    // the GEMV route through the caller's type: EagerChain's (k, out_dim, xh, ids, code) -> the lane's launch
    const gemv = acc.gemvRoute(TestGemv);
    try testing.expectEqual(@as(*const anyopaque, &acc.gemv), gemv.ctx);
    for ([_]struct { e: *const Entry, out: u32, st: *const Statics(Trace) }{
        .{ .e = acc.gemv.gu, .out = 2304, .st = &acc.gemv.gu_statics },
        .{ .e = acc.gemv.dn, .out = 5120, .st = &acc.gemv.dn_statics },
    }) |c| for (c.e.samples) |*s| {
        const xh, const ids, const code = .{ try t.arg(c.e, "xh", &s.vars), try t.arg(c.e, "ids", &s.vars), try t.arg(c.e, "code", &s.vars) };
        const z = try gemv.project(&t, 3, c.out, xh, ids, code);
        var want: [9]Trace.T = undefined;
        want[0..3].* = .{ xh, ids, code };
        @memcpy(want[3..], c.st.arrays[3..9]);
        try expectLaunch(t.back(1), c.e, s, &want);
        try testing.expectEqual(t.back(1).outs[0], z);
        try testing.expect(t.back(1).prepared);
        calls += 1;
    };
    try testing.expectEqual(prepared0 + calls, t.prepared_launches);
    const e = acc.gemv.gu;
    const s = &e.samples[0];
    const xh, const ids, const code = .{ try t.arg(e, "xh", &s.vars), try t.arg(e, "ids", &s.vars), try t.arg(e, "code", &s.vars) };
    const launches = t.launches.items.len;
    try testing.expectError(error.GemvKNotRegistered, gemv.project(&t, 2, 2304, xh, ids, code));
    try testing.expectError(error.GemvOutDim, gemv.project(&t, 3, 1280, xh, ids, code));
    try testing.expectEqual(launches, t.launches.items.len);
    try testing.expect(t.keeps > 0);
    acc.deinit(&t);
    try testing.expectEqual(@as(isize, 0), t.keeps);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
    try testing.expect(t.launcher == null);
}

test "dsv41 kernels ops: acceptAtStartup refuses by name (text, pin, self-check) and a backend without a route method" {
    const a = testing.allocator;
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var diag: xk.Diag = .{};
    // a self-check failure names its kernel and check; nothing is handed out
    try testing.expectError(error.SelfCheckFailed, acceptAtStartup(Trace, a, &t, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .q3rc_router_tail__n128_top3, .check = .f64 } } } }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_router_tail__n128_top3 f64") != null);
    try testing.expect(t.launcher == null);
    try testing.expectEqual(@as(isize, 0), t.keeps);
    // a text that is not the pinned manifest's
    var texts = xk.embedded;
    const k = Kernel.q3drc_mxfp8_fma_f32x;
    const bad = try a.dupeSentinel(u8, xk.embedded.sources[@backingInt(k)], 0);
    defer a.free(bad);
    bad[bad.len / 3] ^= 0x04;
    texts.sources[@backingInt(k)] = bad;
    try testing.expectError(error.TextSha256Mismatch, acceptAtStartup(Trace, a, &t, .{ .device = .{ .stub = .{} }, .texts = &texts }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), @tagName(k)) != null);
    // another pin
    try testing.expectError(error.ManifestNotPinned, acceptAtStartup(Trace, a, &t, .{ .device = .{ .stub = .{} }, .pin = "0000000000000000000000000000000000000000000000000000000000000000" }, &diag));
    try testing.expect(t.launcher == null);
    // the route methods: the first one a backend lacks (acceptAtStartup names it at compile time)
    try testing.expect(missingBackendMethod(Trace) == null);
    try testing.expectEqualStrings("launch", missingBackendMethod(struct {}).?);
    const NoRelease = struct {
        pub const T = u32;
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    try testing.expectEqualStrings("release", missingBackendMethod(NoRelease).?);
}

test "dsv41 kernels ops: the DRAFTRC routes launch the draft's variants at the lane's own sizes, by site and dtype" {
    var reg = try testRegistry();
    defer reg.deinit();
    const a = testing.allocator;
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var diag: xk.Diag = .{};
    // proj: every draft site at both x dtypes (the entry the dtype and site select)
    for (std.enums.values(DraftSite)) |site| for ([_]Dtype{ .bfloat16, .float32 }) |dt| {
        const draft_only = site == .main_proj or site == .shared_w13 or site == .shared_w2;
        const want_k: Kernel = if (dt == .float32) .q3drc_mxfp8_fma_f32x else if (draft_only) .q3rc_mxfp8_fma__draft else .q3rc_mxfp8_fma;
        const e = reg.get(want_k);
        const s0 = sampleAt(e, @tagName(site), 1);
        const w, const sc = .{ try t.arg(e, "w", &s0.vars), try t.arg(e, "scales", &s0.vars) };
        var r = try DraftProj(Trace).init(&t, &reg, site, dt, w, sc, &diag);
        defer r.deinit(&t);
        try testing.expectEqual(want_k, r.e.kernel);
        var n: usize = 0;
        for (e.samples) |*s| {
            if (!std.mem.eql(u8, s.site.?, @tagName(site))) continue;
            const x = try t.arg(e, "x", &s.vars);
            try testing.expectEqual(dt, t.dtypeOf(x));
            _ = try r.call(&t, x);
            try expectLaunch(t.back(1), e, s, &.{ w, sc, x });
            n += 1;
        }
        try testing.expect(n >= 3);
        try testing.expectError(error.RowsOutOfPlan, r.call(&t, try t.node(&.{ 9, 5120 }, dt, &.{})));
    };
    // router: the draft gate weight (N 128) selects the N 128 / top-3 variants
    {
        const pe, const te = .{ reg.get(.q3rc_gate_part__n128), reg.get(.q3rc_router_tail__n128_top3) };
        const w, const bias = .{ try t.arg(pe, "w", &no_vars), try t.arg(te, "bias", &no_vars) };
        var r = try Router(Trace).init(&t, &reg, w, bias, &diag);
        defer r.deinit(&t);
        for (pe.samples) |*s| {
            const x = try t.arg(pe, "x", &s.vars);
            const out = try r.call(&t, x);
            try expectLaunch(t.back(2), pe, s, &.{ x, w });
            try expectLaunch(t.back(1), te, sampleAt(te, null, s.vars.get(.rows)), &.{ t.back(2).outs[0], bias });
            try testing.expectEqual(@as(c_int, 3), t.shapeOf(out[0]).d[1]);
        }
        const w200 = try t.node(&.{ 200, 5120 }, .bfloat16, &.{});
        try testing.expectError(error.RouteInput, Router(Trace).init(&t, &reg, w200, bias, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "a router of 200 experts") != null);
    }
    // tape: the f32 stream's variants, and the f32-x / bf16-residual fused call
    {
        const ce, const le, const fe, const fr = .{ reg.get(.q3ht_combine__f32), reg.get(.q3ht_collapse_norm__f32), reg.get(.q3ht_combine_collapse_norm__f32), reg.get(.q3ht_combine_collapse_norm__f32_rbf16) };
        var r = try HcTape(Trace).init(&t, &reg, .float32, &diag);
        defer r.deinit(&t);
        var mixed = try HcTapeMixed(Trace).init(&t, &reg, &diag);
        defer mixed.deinit(&t);
        for (ce.samples) |*s| {
            const v = &s.vars;
            const x, const rr, const post, const comb = .{ try t.arg(ce, "x", v), try t.arg(ce, "r", v), try t.arg(ce, "post", v), try t.arg(ce, "comb", v) };
            const pre, const w = .{ try t.arg(fe, "pre", v), try t.arg(fe, "w", v) };
            const rb = try t.arg(fr, "r", v);
            _ = try r.combine(&t, x, rr, post, comb);
            try expectLaunch(t.back(1), ce, s, &.{ x, rr, post, comb });
            _ = try r.collapseNorm(&t, rr, pre, w);
            try expectLaunch(t.back(1), le, sampleAt(le, null, v.get(.rows)), &.{ rr, pre, w });
            _ = try r.combineCollapseNorm(&t, x, rr, post, comb, pre, w);
            try expectLaunch(t.back(1), fe, sampleAt(fe, null, v.get(.rows)), &.{ x, rr, post, comb, pre, w });
            _ = try mixed.call(&t, x, rb, post, comb, pre, w);
            try expectLaunch(t.back(1), fr, sampleAt(fr, null, v.get(.rows)), &.{ x, rb, post, comb, pre, w });
        }
    }
    // every DRAFTRC entry is some route's
    var hit: std.EnumSet(Kernel) = .empty;
    for (t.launches.items) |l| hit.insert(l.k);
    for (&reg.entries) |*e| if (std.mem.startsWith(u8, e.family, "draftrc_")) try testing.expect(hit.contains(e.kernel));
    // refusals: another x dtype, a weight of another site
    const fe = reg.get(.q3drc_mxfp8_fma_f32x);
    const s0 = sampleAt(fe, "wq_a", 1);
    const w, const sc = .{ try t.arg(fe, "w", &s0.vars), try t.arg(fe, "scales", &s0.vars) };
    try testing.expectError(error.TemplateNotRegistered, DraftProj(Trace).init(&t, &reg, .wq_a, .float16, w, sc, &diag));
    try testing.expectError(error.RouteInput, DraftProj(Trace).init(&t, &reg, .main_proj, .float32, w, sc, &diag));
    // every draft launch came from a prepared config
    try expectPreparedDecode(&t, &reg);
    try testing.expectEqual(t.launches.items.len, t.prepared_launches);
    try testing.expectEqual(@as(isize, 0), t.keeps);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

/// A backend without prepared launches (the per-call path), recording through a Trace.
const TracePerCall = struct {
    pub const T = u32;
    t: *Trace,

    pub fn launch(p: *TracePerCall, k: Kernel, inputs: []const T, cfg: *const LaunchConfig, out: []T) !void {
        return p.t.launch(k, inputs, cfg, out);
    }
};

/// Every M of `e`'s table: the prebuilt config == the per-call `launchFor`, and the prepared
/// launch records exactly the launch the per-call path records.
fn expectRowPlans(comptime n: usize, t: *Trace, e: *const Entry, site: ?[]const u8) !usize {
    var pc: TracePerCall = .{ .t = t };
    var prep = try RowPlans(Trace, n).init(t, e, site, null);
    defer prep.deinit(t);
    var per = try RowPlans(TracePerCall, n).init(&pc, e, site, null);
    defer per.deinit(&pc);
    const ins: [16]Trace.T = @splat(0);
    var o1: [xk.max_outputs]Trace.T = undefined;
    var o2: [xk.max_outputs]Trace.T = undefined;
    for (1..n + 1) |m| {
        const want = try xk.launchFor(e, &rowsVars(m), site);
        try testing.expect(std.meta.eql(want, (try prep.at(m)).*));
        const before = t.prepared_launches;
        try prep.launch(t, m, ins[0..e.inputs.len], &o1);
        try testing.expectEqual(before + 1, t.prepared_launches);
        const a = t.back(1).*;
        try per.launch(&pc, m, ins[0..e.inputs.len], &o2);
        try testing.expectEqual(before + 1, t.prepared_launches);
        const b = t.back(1).*;
        try testing.expect(a.prepared and !b.prepared);
        try testing.expect(a.k == b.k and a.n_in == b.n_in and std.mem.eql(Trace.T, a.inputs[0..a.n_in], b.inputs[0..b.n_in]));
        try testing.expect(std.meta.eql(a.cfg, b.cfg) and std.meta.eql(a.cfg, want));
    }
    try testing.expectError(error.RowsOutOfPlan, prep.at(0));
    try testing.expectError(error.RowsOutOfPlan, prep.launch(t, n + 1, ins[0..e.inputs.len], &o1));
    return n;
}

test "dsv41 kernels ops: the prepared per-M launches are the per-call launches they replace (every decode kernel, every M)" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var rule: usize = 0;
    var launches: usize = 0;
    for (&reg.entries) |*e| {
        switch (e.launch) {
            .rule => {
                const b = e.bounds.get(.rows) orelse continue;
                launches += switch (b[1]) {
                    8 => try expectRowPlans(8, &t, e, null),
                    48 => try expectRowPlans(48, &t, e, null),
                    else => continue, // prefill rules (rows up to 2^20): their launch stays per call
                };
                rule += 1;
            },
            .plans => {
                if (e.kernel == .q3dk_sinkhorn16_hc4_it20) {
                    launches += try expectRowPlans(32, &t, e, "");
                    continue;
                }
                for (e.sites) |s| launches += try expectRowPlans(8, &t, e, s.name);
            },
        }
    }
    // router 2 + 2 draft, premix 2, HCTAPE 4 + 4 draft, K36 4, GEMV 2, PREP 4
    try testing.expectEqual(@as(usize, 24), rule);
    // + the plan kernels: rcproj 6 sites, the draft variant 3, f32-x 8, sinkhorn16 (32 n)
    try testing.expectEqual(@as(usize, 6 * 48 + 18 * 8 + (6 + 3 + 8) * 8 + 32), launches);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
    // a route refuses M outside its table before any launch; its prepared configs are released
    const n_launch = t.launches.items.len;
    const pe, const te = .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) };
    var r = try Router(Trace).init(&t, &reg, try t.arg(pe, "w", &no_vars), try t.arg(te, "bias", &no_vars), null);
    try testing.expectEqual(@as(isize, 16), t.prepared_live);
    try testing.expectError(error.RowsOutOfPlan, r.call(&t, try t.node(&.{ 9, 5120 }, .float32, &.{})));
    r.deinit(&t);
    var gv = try Gemv(Trace).init(&t, &reg);
    const ge = reg.get(.dsv41_exl3_mul1h_k3_2304);
    const s = &ge.samples[0];
    try testing.expectError(error.RowsOutOfPlan, gv.project(&t, .gate, try t.node(&.{ 49, 5120 }, .float32, &.{}), try t.node(&.{49}, .uint32, &.{}), try t.arg(ge, "code", &s.vars)));
    gv.deinit(&t);
    try testing.expectEqual(n_launch, t.launches.items.len);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

test "dsv41 kernels ops: the routes launch the same through the profiling backend (dsv41_profile.Profiled over the host trace)" {
    const prof = @import("dsv41_profile.zig");
    const P = prof.Profiled(Trace);
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    var clock: u64 = 0;
    var pt = try P.init(a, .{ .a = a }, .{ .manual = &clock }, 2);
    defer pt.inner.deinit();
    defer pt.deinit();
    var t: Trace = .{ .a = a };
    defer t.deinit();
    // startup through the wrapper: the launcher lands on the wrapped backend; the GEMV route
    // prepares its 96 configs through the wrapper (RowPlans sees the wrapped capability)
    var diag: xk.Diag = .{};
    const acc = try acceptAtStartup(P, a, &pt, .{ .device = .{ .stub = .{} } }, &diag);
    try testing.expectEqual(@as(*const xk.Bound, &acc.bound), pt.inner.launcher.?);
    try testing.expectEqual(@as(isize, 96), pt.inner.prepared_live);
    // the router at the lane's samples, wrapped (inside a phase) and not
    const pe, const te = .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) };
    var rp = try Router(P).init(&pt, &reg, try pt.inner.arg(pe, "w", &no_vars), try pt.inner.arg(te, "bias", &no_vars), null);
    var rt = try Router(Trace).init(&t, &reg, try t.arg(pe, "w", &no_vars), try t.arg(te, "bias", &no_vars), null);
    try testing.expectEqual(@as(isize, 96 + 16), pt.inner.prepared_live);
    prof.cycleBegin(&pt);
    prof.beginPhase(&pt, .barrier);
    for (pe.samples) |*s| {
        _ = try rp.call(&pt, try pt.inner.arg(pe, "x", &s.vars));
        _ = try rt.call(&t, try t.arg(pe, "x", &s.vars));
    }
    prof.endPhase(&pt);
    prof.cycleEnd(&pt);
    // the same launches in the same order (kernel, inputs, config, prepared), none added
    try testing.expect(t.launches.items.len >= 2);
    try testing.expectEqual(t.launches.items.len, pt.inner.launches.items.len);
    for (t.launches.items, pt.inner.launches.items) |x, y| {
        try testing.expect(x.k == y.k and x.n_in == y.n_in and x.prepared and y.prepared);
        try testing.expect(std.meta.eql(x.cfg, y.cfg));
    }
    // each counted once, in the phase that was open
    const c = pt.prof.stored[0].tags[@backingInt(prof.Tag.barrier)];
    try testing.expectEqual(@as(u32, @intCast(t.launches.items.len)), c.launches);
    try testing.expectEqual(@as(u32, 1), c.calls);
    rp.deinit(&pt);
    rt.deinit(&t);
    acc.deinit(&pt);
    try testing.expect(pt.inner.launcher == null);
    try testing.expectEqual(@as(isize, 0), pt.inner.prepared_live);
}
