//! The pinned Metal kernel registry of the DeepSeek-V4.1 EXL3 lanes: the Python tier's
//! kernel texts, exported byte for byte into kernels/exl3/ with a manifest of their
//! signatures, launch geometry and self-checks. `Registry.init` checks the manifest against
//! `manifest_sha256` and every text against the manifest, once, and refuses by name;
//! `Registry.bind` builds the mlx fast kernels on a GPU stream, once. A bound kernel runs
//! with the geometry fixed here: nothing is selected or re-validated per call.

const std = @import("std");
const mlx = @import("mlx.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

/// sha256 of kernels/exl3/manifest.json: pins the manifest, which pins every text.
pub const manifest_sha256 = "156576ba2958e13e7216230163bf58548ab2c7fba6292e3cf12ecb1e9e5965d0";
pub const format = "mlx-serve-exl3-kernels-v1";
const dir = "kernels/exl3/";

/// The bank these texts decode: EXL3 codebook mul1, K = 3 on every layer.
pub const bank_codebook = "mul1";
pub const bank_multiplier: u64 = 0x83DCD12D;
pub const bank_ks = [_]u32{3};

/// Every kernel of record; the tag is the kernel's MLX name and its file name.
pub const Kernel = enum {
    dsv41_exl3_mul1h_k3_2304,
    dsv41_exl3_mul1h_k3_5120,
    q3_exl3_prep_in_rin,
    q3_exl3_prep_gu_epi,
    q3_exl3_prep_din_rin,
    q3_moeprep_dpost,
    q3rc_gate_part,
    q3rc_router_tail,
    q3rc_premix_part,
    q3rc_premix_fin,
    q3dk_sinkhorn16_hc4_it20,
    q3rc_mxfp8_fma,
    q3ht_combine,
    q3ht_collapse_norm,
    q3ht_combine_collapse_norm,
    q3ht_mixfin,
    q3_prefill_fused_exl3x3_mul1lut_k3_bf16,
    q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3,
    q3_prefill_dig_gemm_2304x5120_xmul1hk3,
    q3_prefill_dig_rot_take2_5120,
    q3_prefill_dig_rot_roundx_2304,
    q3_prefill_dig2_swiglu_2304_x,
    q3_prefill_dig_rot_widen2_2304,
    q3_prefill_dig_rot_widen1_5120,
    q3_exl3_dig_decmat_5120x2304_mul1hk3,
    q3_exl3_dig_decmat_2304x5120_mul1hk3,
    q3_exl3_dig_decmat_5120x2304_mul1k3,
    q3_exl3_dig_decmat_2304x5120_mul1k3,
    mtplx_dsv4_sinkhorn_hc4_it20,
    mtplx_dsv41_fp_rmsnorm_tg128_d1280,
    mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64,
    mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd,
    mtplx_dsv41_fp_rope_h64_hd512_rd64_inv,
};

/// Header texts shared by several kernels (file header_<tag>.metal).
pub const Header = enum { dig2_x, dig_mul1_k3, dig_mul1h_k3, hctape, rcproj, router_tail };

pub const n_kernels = std.meta.fieldNames(Kernel).len;
pub const n_headers = std.meta.fieldNames(Header).len;

/// The manifest and every text, as `Registry.init` reads them.
pub const Texts = struct {
    manifest: []const u8,
    sources: [n_kernels][:0]const u8,
    headers: [n_headers][:0]const u8,
};

pub const embedded: Texts = .{
    .manifest = @embedFile(dir ++ "manifest.json"),
    .sources = embedAll(Kernel, ""),
    .headers = embedAll(Header, "header_"),
};

fn embedAll(comptime E: type, comptime prefix: []const u8) [std.meta.fieldNames(E).len][:0]const u8 {
    const names = std.meta.fieldNames(E);
    var out: [names.len][:0]const u8 = undefined;
    inline for (names, 0..) |name, i| out[i] = @embedFile(dir ++ prefix ++ name ++ ".metal");
    return out;
}

pub const Refusal = error{
    ManifestNotPinned,
    ManifestSyntax,
    ManifestFormat,
    BankNotImplemented,
    UnknownKernel,
    MissingKernel,
    DuplicateKernel,
    UnknownHeader,
    MissingHeader,
    TextSha256Mismatch,
    LanePinMismatch,
    SchemaInvalid,
    MathModeNotSafe,
    GeometryInvalid,
    NotGpuStream,
    KernelCreateFailed,
};

/// Why the registry refused, for the one log line the caller writes.
pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

// ── Signature and geometry ──

/// Runtime sizes a launch depends on (the site shape supplies gn / k4 / k32 / gk).
pub const Var = enum { rows, cap, m_tokens, experts, tgs, a_rows, gn, k4, k32, gk, seq };
pub const Vars = std.enums.EnumArray(Var, u64);

/// One extent: m x value(v), or the constant m; at most `max` when set.
pub const Dim = struct {
    m: u32,
    v: ?Var = null,
    max: ?u32 = null,

    pub fn eval(d: Dim, vars: *const Vars) u64 {
        const x = @as(u64, d.m) * (if (d.v) |v| vars.get(v) else 1);
        return if (d.max) |cap| @min(x, cap) else x;
    }
};

/// How the self-check fills an input: `rows` inputs are sliced by rows, `bank` inputs are
/// slot banks indexed by ids, `static` inputs are the lane's own constants.
pub const Role = enum { rows, shared, bank, static, table, scalar };
pub const DomainKind = enum { normal, uniform, index, bits, zeros, values, range, signed_pow2, @"var", wave_table, slots };
pub const Domain = struct {
    kind: DomainKind,
    scale: f64 = 0,
    lo: f64 = 0,
    hi: f64 = 0,
    of: ?Var = null,
    used: ?Var = null,
    ints: []const i64 = &.{},
    floats: []const f64 = &.{},
    /// wave_table: N tiles x operands per 64-row M tile (the GEMMs' first-threadgroup column).
    tiles: u32 = 0,
};

pub const Arg = struct {
    name: [:0]const u8,
    dtype: mlx.mlx_dtype,
    shape: []const Dim,
    role: Role,
    domain: Domain,
    row_axis: u8,
};

pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };

/// An explicit launch of a plan kernel at (site, rows): rcproj per site and M, sinkhorn per n.
pub const Plan = struct {
    site: []const u8,
    rows: u32,
    grid: [3]u32,
    threadgroup: [3]u32,
    template: []const TemplateArg,
    output_shapes: []const []const u32,
};

/// A launch rule; `threadgroup_rule`, when set, replaces the fixed threadgroup (the stock K3's min(n, 256)).
pub const Rule = struct { grid: [3]Dim, threadgroup: [3]u32, threadgroup_rule: ?[3]Dim = null };
pub const Launch = union(enum) { rule: Rule, plans: []const Plan };

/// A plan kernel's weight site (rcproj): N outputs per group, K inputs, G groups, strides.
pub const Site = struct { name: []const u8, N: u32, K: u32, G: u32, XS: u32, XG: u32, YS: u32, YG: u32 };

pub const Check = enum { compile, row_invariance, decode_table, golden_tiles, mlx_chain, f64, layout_guard, composition };

/// The lane's own launch at sample sizes, captured by the extractor (the geometry's witness).
pub const Sample = struct {
    site: ?[]const u8,
    vars: Vars,
    grid: [3]u32,
    threadgroup: [3]u32,
    output_shapes: []const []const u32,
    output_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

pub const Entry = struct {
    kernel: Kernel,
    family: []const u8,
    phase: []const u8,
    source: [:0]const u8,
    header: ?Header,
    inputs: []const Arg,
    outputs: []const Arg,
    ensure_row_contiguous: bool,
    template: []const TemplateArg,
    launch: Launch,
    bounds: std.enums.EnumArray(Var, ?[2]u64),
    sites: []const Site,
    checks: std.EnumSet(Check),
    rows_max: u32,
    samples: []const Sample,

    pub fn site(e: *const Entry, name: []const u8) ?*const Site {
        for (e.sites) |*s| if (std.mem.eql(u8, s.name, name)) return s;
        return null;
    }
};

pub const max_outputs = 4;
pub const max_rank = 4;

/// Everything `Bound.apply` hands mlx-c for one launch.
pub const LaunchConfig = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    template: []const TemplateArg,
    n_out: usize,
    out_ranks: [max_outputs]usize = @splat(0),
    out_shapes: [max_outputs][max_rank]c_int = @splat(@splat(0)),
    out_dtypes: [max_outputs]mlx.mlx_dtype = @splat(.float32),
};

/// The launch of `e` at `vars` (and `site` for a plan kernel).
pub fn launchFor(e: *const Entry, vars: *const Vars, site_name: ?[]const u8) error{NoPlan}!LaunchConfig {
    var cfg: LaunchConfig = .{ .grid = undefined, .threadgroup = undefined, .template = e.template, .n_out = e.outputs.len };
    for (e.outputs, 0..) |o, i| cfg.out_dtypes[i] = o.dtype;
    switch (e.launch) {
        .rule => |r| {
            for (r.grid, 0..) |d, i| cfg.grid[i] = @intCast(d.eval(vars));
            cfg.threadgroup = r.threadgroup;
            if (r.threadgroup_rule) |tr| for (tr, 0..) |d, i| {
                cfg.threadgroup[i] = @intCast(d.eval(vars));
            };
            for (e.outputs, 0..) |o, i| {
                cfg.out_ranks[i] = o.shape.len;
                for (o.shape, 0..) |d, j| cfg.out_shapes[i][j] = @intCast(d.eval(vars));
            }
        },
        .plans => |plans| {
            const want = site_name orelse "";
            const rows = vars.get(.rows);
            const p = for (plans) |*p| {
                if (p.rows == rows and std.mem.eql(u8, p.site, want)) break p;
            } else return error.NoPlan;
            cfg.grid = p.grid;
            cfg.threadgroup = p.threadgroup;
            cfg.template = p.template;
            for (p.output_shapes, 0..) |s, i| {
                cfg.out_ranks[i] = s.len;
                for (s, 0..) |d, j| cfg.out_shapes[i][j] = @intCast(d);
            }
        },
    }
    return cfg;
}

/// The site vars a plan kernel's input shapes read (rcproj: w [G N, K/4], scales [G N, K/32], x [M, G K]).
pub fn siteVars(s: *const Site, vars: *Vars) void {
    vars.set(.gn, @as(u64, s.G) * s.N);
    vars.set(.k4, s.K / 4);
    vars.set(.k32, s.K / 32);
    vars.set(.gk, @as(u64, s.G) * s.K);
}

// ── Golden digests the decode self-checks compare against (exl3_ref, via the extractor) ──

pub const GoldenPlane = struct {
    in_dim: u32,
    out_dim: u32,
    seed: u64,
    code_sha256: [32]u8,
    w_hat_sha256: [32]u8,
    onehot_rows: u32,
    onehot_rows_sha256: [32]u8,
};

pub const Golden = struct {
    mul1_table_sha256: [32]u8,
    gate_up: GoldenPlane,
    down: GoldenPlane,
};

// ── The registry ──

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,
    entries: [n_kernels]Entry,
    headers: [n_headers][:0]const u8,
    golden: Golden,
    codebook: []const u8,
    multiplier: u64,
    ks: []const u32,

    /// Parses the manifest, checks it against `pin` (hex sha256) and every text against it,
    /// once. Production passes `embedded` and `manifest_sha256`.
    pub fn init(gpa: Allocator, texts: *const Texts, pin: []const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Registry {
        var d: [32]u8 = undefined;
        Sha256.hash(texts.manifest, &d, .{});
        const got = std.fmt.bytesToHex(d, .lower);
        if (!std.mem.eql(u8, &got, pin)) return refuse(diag, error.ManifestNotPinned, "exl3 kernels: manifest sha256 {s} is not the pinned {s}", .{ &got, pin });
        var reg: Registry = .{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .entries = undefined,
            .headers = texts.headers,
            .golden = undefined,
            .codebook = "",
            .multiplier = 0,
            .ks = &.{},
        };
        errdefer reg.arena.deinit();
        const a = reg.arena.allocator();
        const m = std.json.parseFromSliceLeaky(JManifest, a, texts.manifest, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, "exl3 kernels: manifest does not parse ({t})", .{e}),
        };
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, "exl3 kernels: format \"{s}\" is not {s}", .{ m.format, format });
        if (!std.mem.eql(u8, m.bank.codebook, bank_codebook) or m.bank.multiplier != bank_multiplier or !std.mem.eql(u32, m.bank.K, &bank_ks))
            return refuse(diag, error.BankNotImplemented, "exl3 kernels: the manifest's bank ({s}, K {any}) is not the mul1 K = 3 bank these texts decode", .{ m.bank.codebook, m.bank.K });
        reg.codebook = m.bank.codebook;
        reg.multiplier = m.bank.multiplier;
        reg.ks = m.bank.K;
        try reg.adoptHeaders(texts, m.headers, diag);
        try reg.adoptKernels(a, texts, m.kernels, diag);
        reg.golden = try adoptGolden(m.golden, diag);
        return reg;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Registry, k: Kernel) *const Entry {
        return &self.entries[@backingInt(k)];
    }

    pub fn header(self: *const Registry, e: *const Entry) [:0]const u8 {
        return if (e.header) |h| self.headers[@backingInt(h)] else "";
    }

    fn adoptHeaders(self: *Registry, texts: *const Texts, hs: []const JText, diag: ?*Diag) Refusal!void {
        _ = self;
        var seen: std.EnumSet(Header) = .empty;
        for (hs) |h| {
            const id = std.meta.stringToEnum(Header, h.id orelse "") orelse return refuse(diag, error.UnknownHeader, "exl3 kernels: manifest header \"{s}\" is not one this build embeds", .{h.id orelse ""});
            if (seen.contains(id)) return refuse(diag, error.SchemaInvalid, "exl3 kernels: header {t} listed twice", .{id});
            seen.insert(id);
            try checkText(texts.headers[@backingInt(id)], h, @tagName(id), diag);
        }
        if (seen.count() != n_headers) return refuse(diag, error.MissingHeader, "exl3 kernels: the manifest lists {d} of {d} headers", .{ seen.count(), n_headers });
    }

    fn adoptKernels(self: *Registry, a: Allocator, texts: *const Texts, ks: []const JKernel, diag: ?*Diag) (Refusal || Allocator.Error)!void {
        var seen: std.EnumSet(Kernel) = .empty;
        for (ks) |j| {
            const k = std.meta.stringToEnum(Kernel, j.name) orelse return refuse(diag, error.UnknownKernel, "exl3 kernels: manifest kernel \"{s}\" is not one this build implements", .{j.name});
            if (seen.contains(k)) return refuse(diag, error.DuplicateKernel, "exl3 kernels: {t} listed twice", .{k});
            seen.insert(k);
            self.entries[@backingInt(k)] = try adoptKernel(a, texts, k, j, self.headers, diag);
        }
        if (seen.count() != n_kernels) {
            const absent = seen.complement();
            var it = absent.iterator();
            const missing = it.next().?;
            return refuse(diag, error.MissingKernel, "exl3 kernels: the manifest does not list {t} ({d} of {d})", .{ missing, seen.count(), n_kernels });
        }
    }

    /// Builds every kernel's mlx object on `stream`, once. Only the guarded path binds:
    /// the objects reach Metal at their first launch.
    pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: ?*Diag) Refusal!Bound {
        if (!mlx.streamIsGpu(stream)) return refuse(diag, error.NotGpuStream, "exl3 kernels: bind needs a GPU stream", .{});
        var b: Bound = .{ .reg = self, .stream = stream, .kernels = @splat(.{}) };
        errdefer b.deinit();
        for (&self.entries, 0..) |*e, i| {
            var in_names: [16][*:0]const u8 = undefined;
            var out_names: [max_outputs][*:0]const u8 = undefined;
            for (e.inputs, 0..) |arg, n| in_names[n] = arg.name.ptr;
            for (e.outputs, 0..) |arg, n| out_names[n] = arg.name.ptr;
            const vin = mlx.mlx_vector_string_new_data(&in_names, e.inputs.len);
            defer _ = mlx.mlx_vector_string_free(vin);
            const vout = mlx.mlx_vector_string_new_data(&out_names, e.outputs.len);
            defer _ = mlx.mlx_vector_string_free(vout);
            b.kernels[i] = mlx.mlx_fast_metal_kernel_new(@tagName(e.kernel).ptr, vin, vout, e.source.ptr, self.header(e).ptr, e.ensure_row_contiguous, false);
            if (b.kernels[i].ctx == null) return refuse(diag, error.KernelCreateFailed, "exl3 kernels: mlx_fast_metal_kernel_new({t}) failed", .{e.kernel});
        }
        return b;
    }
};

/// The kernels built on one GPU stream; freed after that stream has drained.
pub const Bound = struct {
    reg: *const Registry,
    stream: mlx.mlx_stream,
    kernels: [n_kernels]mlx.mlx_fast_metal_kernel,

    pub fn deinit(self: *Bound) void {
        for (self.kernels) |k| {
            if (k.ctx != null) _ = mlx.mlx_fast_metal_kernel_free(k);
        }
        self.* = undefined;
    }

    /// One launch of `k` with `cfg`; `outs` receives `cfg.n_out` new arrays (caller frees).
    pub fn apply(self: *const Bound, k: Kernel, inputs: []const mlx.mlx_array, cfg: *const LaunchConfig, outs: []mlx.mlx_array) !void {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        for (0..cfg.n_out) |i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &cfg.out_shapes[i], cfg.out_ranks[i], cfg.out_dtypes[i]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(cfg.grid[0]), @intCast(cfg.grid[1]), @intCast(cfg.grid[2])));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(cfg.threadgroup[0]), @intCast(cfg.threadgroup[1]), @intCast(cfg.threadgroup[2])));
        for (cfg.template) |t| switch (t.value) {
            .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name.ptr, v)),
            .dtype => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t.name.ptr, v)),
        };
        const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = mlx.mlx_vector_array_free(vin);
        var vout = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vout);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&vout, self.kernels[@backingInt(k)], vin, c, self.stream));
        for (outs[0..cfg.n_out], 0..) |*o, i| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
        }
    }
};

// ── Manifest adoption (once, at init) ──

const JDim = struct { m: u32, v: ?[]const u8 = null, max: ?u32 = null };
const JDomain = struct {
    kind: []const u8,
    scale: ?f64 = null,
    lo: ?f64 = null,
    hi: ?f64 = null,
    of: ?[]const u8 = null,
    used: ?[]const u8 = null,
    ints: ?[]const i64 = null,
    floats: ?[]const f64 = null,
    tiles: ?u32 = null,
};
const JArg = struct { name: []const u8, dtype: []const u8, shape: []const JDim, role: ?[]const u8 = null, domain: ?JDomain = null, row_axis: u8 = 0 };
const JTemplate = struct { name: []const u8, int: ?i64 = null, dtype: ?[]const u8 = null };
const JPlan = struct { site: []const u8, rows: u32, grid: [3]u32, threadgroup: [3]u32, template: []const JTemplate, output_shapes: []const []const u32 };
const JVarBound = struct { name: []const u8, lo: u64, hi: u64 };
const JVarValue = struct { name: []const u8, value: u64 };
const JSample = struct {
    site: ?[]const u8 = null,
    vars: []const JVarValue,
    grid: [3]u32,
    threadgroup: [3]u32,
    output_shapes: []const []const u32,
    output_dtypes: []const []const u8,
    template: []const JTemplate,
};
const JText = struct { id: ?[]const u8 = null, file: []const u8, sha256: []const u8, bytes: u64 };
const JPin = struct { symbol: []const u8, hash_of: []const u8, sha256: []const u8, join: ?[]const u8 = null };
const JSelfCheck = struct { checks: []const []const u8, rows_max: u32 = 0 };
const JSite = struct { name: []const u8, N: u32, K: u32, G: u32, XS: u32, XG: u32, YS: u32, YG: u32 };
const JKernel = struct {
    name: []const u8,
    family: []const u8,
    phase: []const u8,
    source: JText,
    header: ?JText = null,
    lane_pins: []const JPin,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
    math_mode: []const u8,
    inputs: []const JArg,
    outputs: []const JArg,
    template: []const JTemplate,
    vars: []const JVarBound,
    grid: ?[3]JDim = null,
    threadgroup: ?[3]u32 = null,
    threadgroup_rule: ?[3]JDim = null,
    plans: ?[]const JPlan = null,
    sites: []const JSite = &.{},
    launch_samples: []const JSample,
    self_check: JSelfCheck,
};
const JPlane = struct { projection: [2]u32, K: u32, codebook: []const u8, seed: u64, code_sha256: []const u8, w_hat_sha256: []const u8, onehot_rows: u32, onehot_rows_sha256: []const u8, states_covered_by_onehot_rows: u32 };
const JGolden = struct {
    mul1_table: struct { states: u32, sha256: []const u8 },
    planes: struct { gate_up: JPlane, down: JPlane },
};
const JBank = struct { codebook: []const u8, multiplier: u64, K: []const u32 };
const JManifest = struct { format: []const u8, bank: JBank, headers: []const JText, kernels: []const JKernel, golden: JGolden };

fn hexSha(s: []const u8) ?[32]u8 {
    if (s.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch return null;
    return out;
}

fn checkText(text: []const u8, j: JText, what: []const u8, diag: ?*Diag) Refusal!void {
    const want = hexSha(j.sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: {s}: sha256 \"{s}\" is not 64 hex", .{ what, j.sha256 });
    var d: [32]u8 = undefined;
    Sha256.hash(text, &d, .{});
    if (!std.mem.eql(u8, &d, &want) or text.len != j.bytes)
        return refuse(diag, error.TextSha256Mismatch, "exl3 kernels: {s} text ({d} B) differs from the manifest ({d} B, sha256 {s})", .{ what, text.len, j.bytes, j.sha256 });
}

/// A lane pin (the full sha256, or the 16-hex prefix an install record prints) over the
/// text as the lane hashed it.
fn checkPin(p: JPin, source: []const u8, hdr: []const u8, k: Kernel, diag: ?*Diag) Refusal!void {
    var h = Sha256.init(.{});
    if (std.mem.eql(u8, p.hash_of, "source")) {
        h.update(source);
    } else if (std.mem.eql(u8, p.hash_of, "header")) {
        h.update(hdr);
    } else if (std.mem.eql(u8, p.hash_of, "header+source")) {
        h.update(hdr);
        h.update(source);
    } else if (std.mem.eql(u8, p.hash_of, "header+join+source")) {
        h.update(hdr);
        h.update(p.join orelse "");
        h.update(source);
    } else return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: pin over \"{s}\"", .{ k, p.hash_of });
    const got = std.fmt.bytesToHex(h.finalResult(), .lower);
    if (p.sha256.len < 16 or p.sha256.len > 64 or !std.mem.startsWith(u8, &got, p.sha256))
        return refuse(diag, error.LanePinMismatch, "exl3 kernels: {t}: {s} sha256 {s} is not the lane pin {s} ({s})", .{ k, p.hash_of, &got, p.sha256, p.symbol });
}

fn parseVar(s: []const u8, k: Kernel, diag: ?*Diag) Refusal!Var {
    return std.meta.stringToEnum(Var, s) orelse refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: unknown var \"{s}\"", .{ k, s });
}

fn parseDtype(s: []const u8, k: Kernel, diag: ?*Diag) Refusal!mlx.mlx_dtype {
    return std.meta.stringToEnum(mlx.mlx_dtype, s) orelse refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: unknown dtype \"{s}\"", .{ k, s });
}

fn adoptDims(a: Allocator, js: []const JDim, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const Dim {
    if (js.len > max_rank) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: rank {d} > {d}", .{ k, js.len, max_rank });
    const out = try a.alloc(Dim, js.len);
    for (js, out) |j, *d| {
        if (j.m == 0) return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: a zero extent", .{k});
        d.* = .{ .m = j.m, .v = if (j.v) |v| try parseVar(v, k, diag) else null, .max = j.max };
    }
    return out;
}

fn adoptTemplate(a: Allocator, js: []const JTemplate, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const TemplateArg {
    const out = try a.alloc(TemplateArg, js.len);
    for (js, out) |j, *t| {
        const value: TemplateValue = if (j.int) |v| .{ .int = std.math.cast(i32, v) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: template {s} = {d}", .{ k, j.name, v }) } else if (j.dtype) |dt| .{ .dtype = try parseDtype(dt, k, diag) } else return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: template {s} has no value", .{ k, j.name });
        t.* = .{ .name = try a.dupeSentinel(u8, j.name, 0), .value = value };
    }
    return out;
}

fn adoptArgs(a: Allocator, js: []const JArg, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const Arg {
    if (js.len == 0 or js.len > 16) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: {d} arguments", .{ k, js.len });
    const out = try a.alloc(Arg, js.len);
    for (js, out, 0..) |j, *arg, i| {
        for (js[0..i]) |prev| if (std.mem.eql(u8, prev.name, j.name)) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: argument {s} twice", .{ k, j.name });
        const dom = j.domain orelse JDomain{ .kind = "bits" };
        arg.* = .{
            .name = try a.dupeSentinel(u8, j.name, 0),
            .dtype = try parseDtype(j.dtype, k, diag),
            .shape = try adoptDims(a, j.shape, k, diag),
            .role = std.meta.stringToEnum(Role, j.role orelse "rows") orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: role \"{s}\"", .{ k, j.role.? }),
            .domain = .{
                .kind = std.meta.stringToEnum(DomainKind, dom.kind) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: domain \"{s}\"", .{ k, dom.kind }),
                .scale = dom.scale orelse 0,
                .lo = dom.lo orelse 0,
                .hi = dom.hi orelse 0,
                .of = if (dom.of) |v| try parseVar(v, k, diag) else null,
                .used = if (dom.used) |v| try parseVar(v, k, diag) else null,
                .ints = dom.ints orelse &.{},
                .floats = dom.floats orelse &.{},
                .tiles = dom.tiles orelse 0,
            },
            .row_axis = j.row_axis,
        };
        if (arg.row_axis >= @max(arg.shape.len, 1)) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: {s} row axis {d}", .{ k, j.name, arg.row_axis });
    }
    return out;
}

fn checkThreadgroup(tg: [3]u32, k: Kernel, diag: ?*Diag) Refusal!void {
    if (tg[0] == 0 or tg[1] == 0 or tg[2] == 0 or @as(u64, tg[0]) * tg[1] * tg[2] > 1024)
        return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: threadgroup {d}x{d}x{d}", .{ k, tg[0], tg[1], tg[2] });
}

fn adoptKernel(a: Allocator, texts: *const Texts, k: Kernel, j: JKernel, headers: [n_headers][:0]const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Entry {
    const source = texts.sources[@backingInt(k)];
    try checkText(source, j.source, @tagName(k), diag);
    var hdr: ?Header = null;
    if (j.header) |h| hdr = std.meta.stringToEnum(Header, h.id orelse "") orelse return refuse(diag, error.UnknownHeader, "exl3 kernels: {t}: header \"{s}\"", .{ k, h.id orelse "" });
    const hdr_text: []const u8 = if (hdr) |h| headers[@backingInt(h)] else "";
    if (j.header) |h| try checkText(hdr_text, h, @tagName(hdr.?), diag);
    if (j.lane_pins.len == 0) return refuse(diag, error.LanePinMismatch, "exl3 kernels: {t}: no lane pin", .{k});
    for (j.lane_pins) |p| try checkPin(p, source, hdr_text, k, diag);
    if (!std.mem.eql(u8, j.math_mode, "safe")) return refuse(diag, error.MathModeNotSafe, "exl3 kernels: {t}: math mode \"{s}\" (mlx-c binds only the safe default)", .{ k, j.math_mode });
    if (j.atomic_outputs) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: atomic outputs", .{k});
    var e: Entry = .{
        .kernel = k,
        .family = j.family,
        .phase = j.phase,
        .source = source,
        .header = hdr,
        .inputs = try adoptArgs(a, j.inputs, k, diag),
        .outputs = try adoptArgs(a, j.outputs, k, diag),
        .ensure_row_contiguous = j.ensure_row_contiguous,
        .template = try adoptTemplate(a, j.template, k, diag),
        .launch = undefined,
        .bounds = .initFill(null),
        .sites = &.{},
        .checks = .empty,
        .rows_max = j.self_check.rows_max,
        .samples = &.{},
    };
    if (e.outputs.len > max_outputs) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: {d} outputs", .{ k, e.outputs.len });
    for (j.vars) |vb| {
        if (vb.lo == 0 or vb.lo > vb.hi) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: var {s} bounds", .{ k, vb.name });
        e.bounds.set(try parseVar(vb.name, k, diag), .{ vb.lo, vb.hi });
    }
    if ((j.grid == null) == (j.plans == null)) return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: needs exactly one of a grid rule and plans", .{k});
    if (j.grid) |g| {
        const tg = j.threadgroup orelse return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: grid without threadgroup", .{k});
        try checkThreadgroup(tg, k, diag);
        const dims = try adoptDims(a, &g, k, diag);
        e.launch = .{ .rule = .{ .grid = dims[0..3].*, .threadgroup = tg } };
        if (j.threadgroup_rule) |tr| {
            const tdims = try adoptDims(a, &tr, k, diag);
            for (tdims) |d| if ((d.v != null and d.max == null) or (d.max orelse d.m) > 1024) return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: threadgroup rule without a bound", .{k});
            e.launch.rule.threadgroup_rule = tdims[0..3].*;
        }
    } else {
        const ps = try a.alloc(Plan, j.plans.?.len);
        for (j.plans.?, ps) |jp, *p| {
            try checkThreadgroup(jp.threadgroup, k, diag);
            if (jp.output_shapes.len != e.outputs.len) return refuse(diag, error.GeometryInvalid, "exl3 kernels: {t}: plan output count", .{k});
            p.* = .{ .site = jp.site, .rows = jp.rows, .grid = jp.grid, .threadgroup = jp.threadgroup, .template = try adoptTemplate(a, jp.template, k, diag), .output_shapes = jp.output_shapes };
        }
        e.launch = .{ .plans = ps };
    }
    const sites = try a.alloc(Site, j.sites.len);
    for (j.sites, sites) |js, *s| s.* = .{ .name = js.name, .N = js.N, .K = js.K, .G = js.G, .XS = js.XS, .XG = js.XG, .YS = js.YS, .YG = js.YG };
    e.sites = sites;
    for (j.self_check.checks) |c| e.checks.insert(std.meta.stringToEnum(Check, c) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: check \"{s}\"", .{ k, c }));
    if (!e.checks.contains(.compile)) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: no compile check", .{k});
    if (e.checks.contains(.row_invariance) and e.rows_max == 0) return refuse(diag, error.SchemaInvalid, "exl3 kernels: {t}: row invariance without rows_max", .{k});
    const samples = try a.alloc(Sample, j.launch_samples.len);
    for (j.launch_samples, samples) |js, *s| {
        var vars: Vars = .initFill(0);
        for (js.vars) |vv| vars.set(try parseVar(vv.name, k, diag), vv.value);
        const dts = try a.alloc(mlx.mlx_dtype, js.output_dtypes.len);
        for (js.output_dtypes, dts) |n, *dt| dt.* = try parseDtype(n, k, diag);
        s.* = .{ .site = js.site, .vars = vars, .grid = js.grid, .threadgroup = js.threadgroup, .output_shapes = js.output_shapes, .output_dtypes = dts, .template = try adoptTemplate(a, js.template, k, diag) };
    }
    e.samples = samples;
    return e;
}

fn adoptGolden(g: JGolden, diag: ?*Diag) Refusal!Golden {
    const bad = error.SchemaInvalid;
    return .{
        .mul1_table_sha256 = hexSha(g.mul1_table.sha256) orelse return refuse(diag, bad, "exl3 kernels: golden mul1 table sha256", .{}),
        .gate_up = try adoptPlane(g.planes.gate_up, diag),
        .down = try adoptPlane(g.planes.down, diag),
    };
}

fn adoptPlane(p: JPlane, diag: ?*Diag) Refusal!GoldenPlane {
    if (p.K != 3 or !std.mem.eql(u8, p.codebook, "mul1") or p.states_covered_by_onehot_rows != 65536 or p.onehot_rows == 0 or p.onehot_rows > p.projection[0])
        return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden plane {d}x{d}", .{ p.projection[0], p.projection[1] });
    return .{
        .in_dim = p.projection[0],
        .out_dim = p.projection[1],
        .seed = p.seed,
        .code_sha256 = hexSha(p.code_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden code sha256", .{}),
        .w_hat_sha256 = hexSha(p.w_hat_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden w_hat sha256", .{}),
        .onehot_rows = p.onehot_rows,
        .onehot_rows_sha256 = hexSha(p.onehot_rows_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden rows sha256", .{}),
    };
}

// ── The EXL3 host decode the decode self-checks compare against ──

/// exllamav3 codebook mul1 (codebook.cuh L76-89): the f16 bits a 16-bit trellis state decodes to.
pub fn mul1Decode(state: u32) u16 {
    const x: u32 = state *% 0x83DCD12D;
    const s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    const k_inv: f64 = @as(f16, @bitCast(@as(u16, 0x1EEE)));
    const k_bias: f64 = @as(f16, @bitCast(@as(u16, 0xC931)));
    // (1024 + s) k_inv + k_bias is exact in f64: one round-to-nearest-even to f16, as __hfma.
    const v: f16 = @floatCast(@as(f64, @floatFromInt(1024 + s)) * k_inv + k_bias);
    return @bitCast(v);
}

pub fn mul1Table(out: *[65536]u16) void {
    for (out, 0..) |*o, s| o.* = mul1Decode(@intCast(s));
}

/// exllamav3 tensor_core_perm (quantize.py L22-44): the row-major tile slot of code position p.
pub const tile_perm: [256]u8 = blk: {
    var perm: [256]u8 = undefined;
    for (0..32) |t| {
        const r0 = (t % 4) * 2;
        const rows = [4]usize{ r0, r0 + 1, r0 + 8, r0 + 9 };
        for (0..8) |j| perm[t * 8 + j] = @intCast(rows[j % 4] * 16 + t / 4 + (if (j < 4) 0 else 8));
    }
    break :blk perm;
};

/// The trellis states of one tile's 256 code positions (exl3_dq.cuh L15-31): the 16 stream
/// bits ending at bit (p + 1) K, circularly, of the little-endian u32 view of the tile.
fn tileStates(words: []const i16, K: usize, out: *[256]u16) void {
    const nw = 8 * K;
    var u: [24]u64 = undefined;
    for (0..nw) |m| u[m] = @as(u64, @as(u16, @bitCast(words[2 * m]))) | (@as(u64, @as(u16, @bitCast(words[2 * m + 1]))) << 16);
    for (out, 0..) |*o, p| {
        const b1 = (p + 1) * K + 256 * K;
        const hi_word = (b1 - 16) / 32;
        const lo_word = (b1 - 1) / 32;
        const s0: u6 = @intCast((lo_word + 1) * 32 - b1);
        o.* = @truncate(((u[hi_word % nw] << 32) | u[lo_word % nw]) >> s0);
    }
}

/// exl3_ref.reconstruct: code int16 [nI, nJ, 16K] -> W_hat f16 bits [16 nI, 16 nJ];
/// `states` (optional) receives each weight's trellis state in the same place.
pub fn reconstruct(code: []const i16, n_i: usize, n_j: usize, K: usize, table: *const [65536]u16, w: []u16, states: ?[]u16) void {
    const tw = 16 * K;
    const cols = n_j * 16;
    var st: [256]u16 = undefined;
    for (0..n_i) |ti| {
        for (0..n_j) |tj| {
            tileStates(code[(ti * n_j + tj) * tw ..][0..tw], K, &st);
            for (st, 0..) |s, p| {
                const pos: usize = tile_perm[p];
                const at = (ti * 16 + pos / 16) * cols + tj * 16 + pos % 16;
                w[at] = table[s];
                if (states) |o| o[at] = s;
            }
        }
    }
}

pub fn splitmix64(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// A golden plane's seeded code: the little-endian int16 view of splitmix64(seed).
pub fn synthPlane(a: Allocator, g: *const GoldenPlane) ![]i16 {
    const n = (g.in_dim / 16) * (g.out_dim / 16) * 48;
    const out = try a.alloc(i16, n);
    var s = g.seed;
    var i: usize = 0;
    while (i < n) : (i += 4) {
        const z = splitmix64(&s);
        inline for (0..4) |j| out[i + j] = @bitCast(@as(u16, @truncate(z >> (16 * j))));
    }
    return out;
}

/// W_hat of a golden plane (f16 bits, row-major [IN, OUT]) and each weight's state.
pub const HostPlane = struct {
    code: []i16,
    w: []u16,
    states: []u16,

    pub fn init(a: Allocator, g: *const GoldenPlane, table: *const [65536]u16) !HostPlane {
        const code = try synthPlane(a, g);
        errdefer a.free(code);
        const n = @as(usize, g.in_dim) * g.out_dim;
        const w = try a.alloc(u16, n);
        errdefer a.free(w);
        const states = try a.alloc(u16, n);
        reconstruct(code, g.in_dim / 16, g.out_dim / 16, 3, table, w, states);
        return .{ .code = code, .w = w, .states = states };
    }

    pub fn deinit(self: *HostPlane, a: Allocator) void {
        a.free(self.code);
        a.free(self.w);
        a.free(self.states);
        self.* = undefined;
    }
};

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

fn metalDriverLoaded() bool {
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

fn initOrPrint(texts: *const Texts, pin: []const u8) !Registry {
    var diag: Diag = .{};
    return Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
}

fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "dsv41 kernels: the embedded manifest is the pinned one and every text matches it" {
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 33), n_kernels);
    for (reg.entries, 0..) |e, i| try testing.expectEqual(@as(Kernel, @fromBackingInt(@intCast(i))), e.kernel);
    try testing.expect(reg.get(.dsv41_exl3_mul1h_k3_2304).checks.contains(.decode_table));
    try testing.expect(reg.get(.mtplx_dsv4_sinkhorn_hc4_it20).launch.rule.threadgroup_rule != null);
    try testing.expect(reg.get(.q3rc_mxfp8_fma).checks.contains(.row_invariance));
    try testing.expect(reg.get(.q3_exl3_dig_decmat_5120x2304_mul1hk3).checks.contains(.golden_tiles));
    try testing.expectEqual(Header.rcproj, reg.get(.q3rc_mxfp8_fma).header.?);
}

test "dsv41 kernels: a tampered source or header text is refused, by name" {
    const a = testing.allocator;
    var texts = embedded;
    const k = Kernel.dsv41_exl3_mul1h_k3_2304;
    const bad = try a.dupeSentinel(u8, embedded.sources[@backingInt(k)], 0);
    defer a.free(bad);
    bad[bad.len / 2] ^= 0x20;
    texts.sources[@backingInt(k)] = bad;
    var diag: Diag = .{};
    try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), @tagName(k)) != null);

    texts = embedded;
    const h = Header.dig_mul1h_k3;
    const bad_h = try a.dupeSentinel(u8, embedded.headers[@backingInt(h)], 0);
    defer a.free(bad_h);
    bad_h[0] ^= 0x01;
    texts.headers[@backingInt(h)] = bad_h;
    try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), @tagName(h)) != null);
}

test "dsv41 kernels: a manifest that is not the pinned one is refused" {
    const a = testing.allocator;
    var texts = embedded;
    const m = try a.dupe(u8, embedded.manifest);
    defer a.free(m);
    m[m.len / 2] ^= 0x01;
    texts.manifest = m;
    try testing.expectError(error.ManifestNotPinned, Registry.init(a, &texts, manifest_sha256, null));
}

test "dsv41 kernels: an unknown kernel name is refused" {
    const a = testing.allocator;
    try testing.expectEqual(@as(?Kernel, null), std.meta.stringToEnum(Kernel, "dsv41_exl3_mul1_k3_2304"));
    // A manifest naming a kernel this build lacks, pinned to itself, still refuses.
    const needle = "\"name\": \"q3_moeprep_dpost\"";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, embedded.manifest, needle));
    const m = try std.mem.replaceOwned(u8, a, embedded.manifest, needle, "\"name\": \"q3_moeprep_dpost_k2\"");
    defer a.free(m);
    var texts = embedded;
    texts.manifest = m;
    var diag: Diag = .{};
    const pin = shaHex(m);
    try testing.expectError(error.UnknownKernel, Registry.init(a, &texts, &pin, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3_moeprep_dpost_k2") != null);
}

/// `text` with the first occurrence of `needle` replaced (caller frees).
fn replaceFirst(a: Allocator, text: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, text, needle) orelse return error.NeedleMissing;
    return std.mem.concat(a, u8, &.{ text[0..at], replacement, text[at + needle.len ..] });
}

test "dsv41 kernels: every manifest refusal refuses, by name" {
    const a = testing.allocator;
    const Case = struct { needle: []const u8, replacement: []const u8, want: Refusal };
    const cases = [_]Case{
        .{ .needle = "\"format\": \"mlx-serve-exl3-kernels-v1\"", .replacement = "\"format\": \"mlx-serve-exl3-kernels-v2\"", .want = error.ManifestFormat },
        .{ .needle = "\"multiplier\": 2212286765", .replacement = "\"multiplier\": 3417055213", .want = error.BankNotImplemented },
        .{ .needle = "\"id\": \"dig2_x\"", .replacement = "\"id\": \"dig2_y\"", .want = error.UnknownHeader },
        .{ .needle = "\"name\": \"dsv41_exl3_mul1h_k3_5120\"", .replacement = "\"name\": \"dsv41_exl3_mul1h_k3_2304\"", .want = error.DuplicateKernel },
        .{ .needle = "\"sha256\": \"035ad69fda53f016", .replacement = "\"sha256\": \"135ad69fda53f016", .want = error.LanePinMismatch },
        .{ .needle = "\"math_mode\": \"safe\"", .replacement = "\"math_mode\": \"fast\"", .want = error.MathModeNotSafe },
        .{ .needle = "\n}\n", .replacement = "\n", .want = error.ManifestSyntax },
    };
    for (cases) |c| {
        const m = try replaceFirst(a, embedded.manifest, c.needle, c.replacement);
        defer a.free(m);
        var texts = embedded;
        texts.manifest = m;
        const pin = shaHex(m);
        var diag: Diag = .{};
        try testing.expectError(c.want, Registry.init(a, &texts, &pin, &diag));
        try testing.expect(diag.len > 0);
    }
}

test "dsv41 kernels: every launch rule reproduces the lane's own launches (geometry round trip)" {
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    var n: usize = 0;
    for (&reg.entries) |*e| {
        try testing.expect(e.samples.len > 0);
        for (e.samples) |*s| {
            const cfg = try launchFor(e, &s.vars, s.site);
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
            n += 1;
        }
    }
    try testing.expect(n >= 3 * n_kernels);
}

test "dsv41 kernels: the host EXL3 decode reproduces exl3_ref (codebook, seeded planes, one-hot rows)" {
    const a = testing.allocator;
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    mul1Table(table);
    var d: [32]u8 = undefined;
    Sha256.hash(std.mem.sliceAsBytes(table), &d, .{});
    try testing.expectEqualSlices(u8, &reg.golden.mul1_table_sha256, &d);
    for ([_]*const GoldenPlane{ &reg.golden.gate_up, &reg.golden.down }) |g| {
        var hp = try HostPlane.init(a, g, table);
        defer hp.deinit(a);
        Sha256.hash(std.mem.sliceAsBytes(hp.code), &d, .{});
        try testing.expectEqualSlices(u8, &g.code_sha256, &d);
        Sha256.hash(std.mem.sliceAsBytes(hp.w), &d, .{});
        try testing.expectEqualSlices(u8, &g.w_hat_sha256, &d);
        const rows_n = @as(usize, g.onehot_rows) * g.out_dim;
        Sha256.hash(std.mem.sliceAsBytes(hp.w[0..rows_n]), &d, .{});
        try testing.expectEqualSlices(u8, &g.onehot_rows_sha256, &d);
        var seen = try std.DynamicBitSet.initEmpty(a, 65536);
        defer seen.deinit();
        for (hp.states[0..rows_n]) |s| seen.set(s);
        try testing.expectEqual(@as(usize, 65536), seen.count());
    }
}

test "dsv41 kernels: the registry implements exactly the bank's codebook and K" {
    const expert_bank = @import("expert_bank.zig");
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 1), expert_bank.dsv41.codebooks.len);
    try testing.expectEqualStrings(expert_bank.dsv41.codebooks[0], reg.codebook);
    try testing.expectEqualSlices(u32, expert_bank.dsv41.k, reg.ks);
    try testing.expectEqual(expert_bank.mul1_multiplier, reg.multiplier);
}

test "dsv41 kernels: no Metal device in this process" {
    if (std.c.getenv("DSV41_KERNELS_GPU") != null) return error.SkipZigTest;
    try testing.expect(!metalDriverLoaded());
}
