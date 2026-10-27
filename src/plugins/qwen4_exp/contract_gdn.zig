//! qwen4_exp's GDN kernels under the kernel contract. The core prework and recurrence register through the gdn area's
//! test types (`sdk/blocks/gdn/contract.zig`) as replacements, bit-equal to the packed prework and the pipelined
//! kernel on the same inputs; the GDN tail's fused prepare (`gdn_out_prep`) against its float64 reference: the
//! format's finish of z, the norm-gate and the format's prepare (`abi/quant.zig`'s `linear_*` contract), at Flash
//! Next's 48 value heads of 128.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const gc = @import("../../sdk/blocks/gdn/contract.zig");
const fa = @import("../../sdk/blocks/mlp/fused_activations.zig");
const gdn_core = @import("trunk/gdn_core.zig");
const gdn_tail = @import("trunk/gdn_tail.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Val = host.Val;
const U = host.U;
const Shape = spec.Shape;

const core_recur_launch: metal.Launch = .{
    .name = "msv_gdn_core_recur",
    .inputs = &.{ "q", "k", "v", "g", "beta", "state_in", "T" },
    .outputs = &.{ "y", "state_out" },
    .source = gdn_core.RECUR_SOURCE,
    .header = gdn_core.RECUR_HEADER,
    // k_s, q_s [16][128] bf16, v_s [16][32] f32, g_s, b_s [16] f32, y_s [16][32] bf16
    .tg_bytes = gdn_core.recurTgBytes(128, 2, 2),
    .m1 = 1024,
    .m2 = 1024,
};

const core_prework_launch: metal.Launch = .{
    .name = "msv_gdn_core_prework",
    .inputs = &.{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" },
    .outputs = &.{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" },
    .source = gdn_core.PREWORK_SOURCE,
    .header = gdn_core.PREWORK_HEADER,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const core_geo: gc.CoreGeo = .{ .tb = gdn_core.TB, .db = gdn_core.DB, .kqf = gdn_core.KQF, .rpt = gdn_core.RPT, .nt = gdn_core.NT };

// ── the GDN tail's fused prepare ──

const HV = 48;
const DV = 128;
const VD = HV * DV;
/// The format's 1 / sqrt(128), as the kernel holds it (f32).
const HAD_SCALE: f64 = @as(f32, 0.08838834764831845);

const out_prep_launch: metal.Launch = .{
    .name = "mlxserve_gdn_out_prep",
    .inputs = &.{ "y", "zq", "svh", "norm_w", "eps", "sigtab", "suh" },
    .outputs = &.{"xp"},
    .source = gdn_tail.OUT_PREP_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// A shape's rows are the chunk's rows (one batch row); the head geometry is fixed, the gate qwen4_exp's sigmoid.
const OutPrep = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &out_prep_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    y: []f64 = &.{},
    zq: []f64 = &.{},
    svh: []f64 = &.{},
    suh: []f64 = &.{},
    nw: []f64 = &.{},
    eps: f64 = 0,
    args: [7]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) OutPrep {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *OutPrep) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// The wrapper's launch (`gdn_tail.outPrep`): one simdgroup per (row, head), NSG heads a threadgroup, the sigmoid
    /// from the table. The wrapper takes it from 17 rows; the kernel runs at any row count, so the small-shape gates
    /// run it below.
    pub fn geometry(_: *const OutPrep, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1) return null;
        const rows: c_int = @intCast(shape.rows);
        const nsg = gdn_tail.outPrepNsg(HV);
        var g = metal.Geometry{ .grid = .{ 32 * nsg, rows, @divExact(HV, nsg) }, .tg = .{ 32 * nsg, 1, 1 } };
        g.out(&.{ rows, VD }, .float16);
        g.dtype("T", .bfloat16);
        g.int("HV", HV);
        g.int("DV", DV);
        g.int("TAB", 1);
        g.int("SWISH", 0);
        g.int("NSG", nsg);
        return g;
    }

    pub fn prepare(self: *OutPrep, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.y = try host.uniform(a, rng, rows * VD, -3, 3, .bfloat16);
        self.zq = try host.uniform(a, rng, rows * VD, -8, 8, .float16);
        self.svh = try host.uniform(a, rng, VD, -0.2, 0.2, .float16);
        self.suh = try a.alloc(f64, VD);
        for (self.suh) |*x| x.* = host.rt(.float16, (if (rng.boolean()) @as(f64, 1) else -1) * (0.5 + rng.float(f64)));
        self.nw = try host.uniform(a, rng, DV, 0.5, 1.5, .bfloat16);
        self.eps = host.rt(.float32, 1e-6);
        switch (fill) {
            .random => {},
            .nonfinite => {
                host.plant(self.y, rows, VD);
                self.zq[5 * DV + 3] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: integer y (the head's sum of squares is exact in fp32) and z, unit vectors and weights.
                for (self.y) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                for (self.zq) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.svh, 1);
                @memset(self.suh, 1);
                @memset(self.nw, 1);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const r: c_int = @intCast(rows);
        const sig = try host.read(a, try fa.sigmoidTableFor(rig.stream(), .bfloat16), rig.stream());
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, r, HV, DV }, self.y),
            try rig.input(.float16, &.{ r, VD }, self.zq),
            try rig.input(.float16, &.{VD}, self.svh),
            try rig.input(.bfloat16, &.{DV}, self.nw),
            try rig.scalar(@floatCast(self.eps)),
            try rig.input(.bfloat16, &.{65536}, sig),
            try rig.input(.float16, &.{VD}, self.suh),
        };
        return true;
    }

    pub fn run(self: *OutPrep) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const OutPrep) usize {
        return self.shape.rows * VD;
    }

    /// The Hadamard transform of a block in place: float64 values, each output's error the inputs' plus 7 levels of
    /// fp32 sums.
    fn hadamard(v: []Val, tmp: []f64) void {
        var e: f64 = 0;
        var m: f64 = 0;
        for (v, tmp) |x, *t| {
            t.* = x.v;
            e += x.e;
            m += @abs(x.v);
        }
        host.fwht(tmp);
        const err = e + spec.gamma(7, U) * m;
        for (v, tmp) |*x, t| x.* = .{ .v = t, .e = err };
    }

    /// x * s in fp32 and its rounding, unrounded to the storage type.
    fn scaled(x: Val, s: f64) Val {
        const p = host.mul(x, .exact(s));
        return .{ .v = p.v, .e = p.e + U * @abs(p.v) };
    }

    /// Every output: per (row, head), z's finish, the norm-gate on it, out_proj's prepare of the gated rows.
    fn emit(self: *const OutPrep, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        var arena = std.heap.ArenaAllocator.init(self.rig.a);
        defer arena.deinit();
        const a = arena.allocator();
        const blk = try a.alloc(Val, DV);
        const tmp = try a.alloc(f64, DV);
        const xs = try a.alloc(Val, DV);
        const terms = try a.alloc(f64, DV);
        for (0..self.shape.rows) |r| for (0..HV) |h| {
            const base = r * VD + h * DV;
            // in_proj_z's finish: f16(H(f16(zq)) * HAD_SCALE * svh), then the activation dtype
            for (blk, self.zq[base..][0..DV]) |*o, z| o.* = .exact(z);
            hadamard(blk, tmp);
            for (blk, self.svh[h * DV ..][0..DV]) |*o, sv| o.* = host.round(.bfloat16, host.round(.float16, scaled(scaled(o.*, HAD_SCALE), sv)));
            // the norm-gate: rms over the head's y, `norm_w * T(y * inv)`, sigmoid gate
            for (xs, self.y[base..][0..DV]) |*o, y| o.* = .exact(y);
            const inv = gc.rmsInv(xs, 1.0 / @as(f64, DV), self.eps, acc, order, terms);
            for (blk, xs, self.nw) |*o, x, w| {
                const r1 = host.round(.bfloat16, host.mul(x, inv));
                const normed = host.round(.bfloat16, .{ .v = w * r1.v, .e = @abs(w) * r1.e });
                const gated = host.round(.bfloat16, host.mul(normed, host.mlxSig(.bfloat16, o.*)));
                o.* = gated;
            }
            // out_proj's prepare: f16(H(f16(gated) * suh) * HAD_SCALE)
            for (blk, self.suh[h * DV ..][0..DV]) |*o, su| o.* = scaled(host.round(.float16, o.*), su);
            hadamard(blk, tmp);
            for (blk, 0..) |x, c| {
                const out = host.round(.float16, scaled(x, HAD_SCALE));
                want[base + c] = out.v;
                if (mag) |m| m[base + c] = host.allowance(out.e);
            }
        };
    }

    pub fn reference(self: *const OutPrep, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const OutPrep, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, acc, order);
    }
};

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const f32s = mlx.mlx_dtype.float32;
    const replaces_recur = "gated_delta_pipelined_tb12 (the pipelined recurrence)";
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, f32 state (y)", .y, f32s, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, f32s, .y), sp, null);
    }
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, f32 state (state)", .states, f32s, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, f32s, .states), sp, null);
    }
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, bf16 state", .both, .bfloat16, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, .bfloat16, .both), sp, null);
    }
    const pw = struct {
        const act: gc.Cfg = .{ .kind = .prework, .st = .float32, .reads = .act, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
        const wide: gc.Cfg = .{ .kind = .prework, .st = .float32, .reads = .wide, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
        const bf16: gc.Cfg = .{ .kind = .prework, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
    };
    inline for (.{ .{ "msv_gdn_core_prework bf16, f32 gate (bf16 outputs)", pw.act }, .{ "msv_gdn_core_prework bf16, f32 gate (g)", pw.wide }, .{ "msv_gdn_core_prework bf16, bf16 gate", pw.bf16 } }) |r| {
        var sp = gc.decSpec(r[0], r[1], gc.tokens(&.{ 17, 33, 64 }));
        sp.replaces = "gdn_prework2 (the packed prework)";
        try set.add(gc.Dec, .init(a, &core_prework_launch, r[1]), sp, null);
    }
    try set.add(OutPrep, .init(a), .{
        .name = "gdn_out_prep bf16 sigmoid gate (z's finish, norm-gate, out_proj's prepare)",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f16,
        // a 17-row chunk (the table's first width), a 64-row one, two rows past a power of two
        .served = &.{ .{ .rows = 17, .n = VD, .k = DV }, .{ .rows = 64, .n = VD, .k = DV }, .{ .rows = 130, .n = VD, .k = DV } },
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the outputs pass f16 and bf16 roundings of their own, the size of a bf16 sum's error here" },
    }, null);
}

// ── tests ──

const testing = std.testing;

/// Runs a fixture's declared launch with every output kept.
fn launchAll(rig: *metal.Rig, l: *const metal.Launch, geo: metal.Geometry, args: []const mlx.mlx_array) !void {
    var g = geo;
    g.unread = 0;
    try rig.launch(l, g, args);
}

fn expectSameBytes(rig: *metal.Rig, ours: mlx.mlx_array, theirs: mlx.mlx_array) !void {
    const a = testing.allocator;
    const x = try a.dupe(u8, try rig.bytesOf(&.{ours}));
    defer a.free(x);
    try testing.expectEqualSlices(u8, x, try rig.bytesOf(&.{theirs}));
}

test "kernel contract qwen4_exp gdn: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const shape = k.spec.served[k.spec.served.len - 1];
        if (std.mem.startsWith(u8, k.spec.name, "msv_gdn_core_recur")) {
            const t: *gc.Recur = @ptrCast(@alignCast(k.ctx));
            try testing.expect(try t.prepare(shape, .random, 7));
            try launchAll(&t.rig, t.launch, t.geometry(shape).?, t.args[0..7]);
            const g = t.args;
            const r = try gdn_core.recur(t.rig.stream(), g[0], g[1], g[2], g[3], g[4], g[5], 16, 16, 128, 128, .bfloat16);
            defer for ([_]mlx.mlx_array{ r.y, r.state }) |x| {
                _ = mlx.mlx_array_free(x);
            };
            try expectSameBytes(&t.rig, t.rig.output(0), r.y);
            try expectSameBytes(&t.rig, t.rig.output(1), r.state);
        } else if (std.mem.startsWith(u8, k.spec.name, "msv_gdn_core_prework")) {
            const t: *gc.Dec = @ptrCast(@alignCast(k.ctx));
            try testing.expect(try t.prepare(shape, .random, 7));
            try launchAll(&t.rig, t.launch, t.geometry(shape).?, t.args[0..t.n_args]);
            const g = t.args;
            const pre = (try gdn_core.prework(t.rig.stream(), .{ .qkv = g[0], .qkv_off = 0, .qkv_stride = mlx.getShape(g[0])[2], .b = g[5], .b_off = 0, .b_stride = mlx.getShape(g[5])[2], .a = g[6], .a_off = 0, .a_stride = mlx.getShape(g[6])[2], .A_log = g[7], .dt_bias = g[8], .conv_state = g[1], .conv_w = g[2], .q_scale = g[3], .k_scale = g[4], .hk = 16, .hv = 32, .dk = 128, .dv = 128, .seq = @intCast(shape.rows), .gate = true, .g_dtype = t.cfg.gt() })) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta };
            defer for (theirs) |x| {
                _ = mlx.mlx_array_free(x);
            };
            for (theirs, 0..) |x, i| try expectSameBytes(&t.rig, t.rig.output(i), x);
        } else {
            const t: *OutPrep = @ptrCast(@alignCast(k.ctx));
            try testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const g = t.args;
            const rows: c_int = @intCast(shape.rows);
            const xp = (try gdn_tail.outPrep(t.rig.stream(), g[0], g[1], g[2], g[3], g[4], g[6], false, HV, DV, 1, rows)) orelse return error.WrapperDeclined;
            defer _ = mlx.mlx_array_free(xp);
            try expectSameBytes(&t.rig, t.rig.output(0), xp);
        }
    }
}
