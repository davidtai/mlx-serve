//! DeepSeek-V4.1 as a module-owned arch of mlx-serve (the deepseek_v4 pattern): `Transformer.dsv41`
//! holds a `Module` that `Transformer.init` builds from the loaded residents and `forwardWith` runs,
//! its per-request state rebuilt at `cache.step == 0`. Construction refuses by name, in order:
//!   1. the kernels (`exl3_kernel_ops.acceptAtStartup`: the registry against the pinned manifest,
//!      every kernel built on the device, the device self-check judged);
//!   2. the expert source (`deepseek_v41_arm.ArmWith`: bank, admission, the stream at the admitted
//!      rows on MLX slot memory, the hook over the kernels' GEMV and the DIG-X prefill route), every
//!      bank it bound checked against the kernels' layout, again at the phase change;
//!   3. the residents' rows and model (as `deepseek_v41_dspark_serve.Resources.open`, over the
//!      shell's loaded residents: the Engram sidecar, the Engram rows, the embedding rows, the trunk
//!      at the stock tier, the draft head).
//! The phase change (the embedding's host rows, the grown slot banks) runs once, at the first
//! decode-width forward after a prompt.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_io = @import("model.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const xk = @import("exl3_kernels.zig");
const xo = @import("exl3_kernel_ops.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const graph = @import("deepseek_v41_graph.zig");
const routes = @import("deepseek_v41_routes.zig");
const eng = @import("deepseek_v41_engram.zig");
const mdl = @import("deepseek_v41_model.zig");
const dh = @import("deepseek_v41_dspark_head.zig");
const qwen4 = @import("qwen4_exp.zig");
const dsp = @import("deepseek_v41_dspark_serve.zig");

const log = std.log.scoped(.dsv41);

const G = ops.MlxOps;
/// The expert source: the exact tier's op chain around the kernels' EXL3 decode GEMV, the wide
/// (prefill) routed calls on the kernels' DIG-X route.
pub const A = arm_mod.ArmWith(G, xp.EagerChain(G, xp.MlxGemv), .{ .prefill = xo.DigXPrefill(G) });
const M = mdl.Model(G);
const H = dh.Head(G);

/// Beside the model's shards: the Engram token map the converter exports.
pub const engram_token_map_file = "engram-token-map.u32";

pub const Module = struct {
    gpa: std.mem.Allocator,
    g: G,
    kernels: *xo.Accepted(G),
    arm: *A,
    weights: *model_io.Weights,
    engram: eng.RowSource,
    /// The input embedding's rows in its shard, read past the page cache once the prompt fence ran.
    embed_rows: qwen4.NgramTable,
    model: *M,
    head: *H,
    /// The request in flight (rebuilt at `cache.step == 0`).
    state: ?M.State = null,
    /// The prompt fence ran: the embedding reads its host rows from then on (per process).
    fenced: bool = false,

    /// `config` is the shell's (its bank and token-map paths, the memory baseline); `weights`
    /// the loaded residents (the Engram sidecar joins them here).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, config: *const model_io.ModelConfig, weights: *model_io.Weights, s: mlx.mlx_stream) !*Module {
        const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
        const map = config.engram_token_map_path orelse return error.Dsv41BankDir;
        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .g = try G.init(gpa, s), .kernels = undefined, .arm = undefined, .weights = weights, .engram = undefined, .embed_rows = undefined, .model = undefined, .head = undefined };
        errdefer self.g.deinit();
        var diag: arm_mod.Diag = .{};
        self.kernels = acceptKernels(gpa, &self.g, .{ .device = .{ .stream = s } }, &diag) catch |e| return refused(e, &diag);
        errdefer self.dropKernels();
        _ = mlx.mlx_clear_cache();
        self.arm = A.init(gpa, io, &self.g, self.kernels.gemvRoute(xp.MlxGemv), .{
            .model_dir = dir,
            .baseline_bytes = config.memory_baseline_bytes,
            .slot_memory = .{ .mlx = s },
            .prefill = .{ .reg = &self.kernels.reg },
            .draft_pruned_bytes = 0,
        }, &diag) catch |e| return refused(e, &diag);
        errdefer self.arm.deinit();
        checkArmBanks(self.arm, &self.g, &self.kernels.reg, &diag) catch |e| return refused(e, &diag);
        self.arm.grown_check = .{ .ctx = &self.kernels.reg, .check = GrownBanks.check };
        var vd: v41.Diag = .{};
        errdefer if (vd.len > 0) log.err("residents refused: {s}", .{vd.message()});
        const c = self.arm.config;
        if (c.engram.n_layers > 0) try loadEngramResidents(gpa, weights, dir);
        self.engram = try eng.RowSource.open(gpa, io, dir, map, &c, &vd);
        errdefer self.engram.deinit();
        self.embed_rows = try dsp.openEmbeddingRows(gpa, io, dir, &c, &vd);
        errdefer self.embed_rows.close();
        self.model = try M.init(gpa, &self.g, c, try routes.parse(&.{}, &vd), weights, &self.engram);
        errdefer self.model.deinit(&self.g);
        self.head = try H.initWith(gpa, &self.g, c, .{}, weights, .{ .subset = if (self.arm.draft_subset) |*x| x else null });
        return self;
    }

    pub fn deinit(self: *Module) void {
        const gpa = self.gpa;
        if (self.state) |*st| st.deinit(&self.g, gpa);
        self.head.deinit(&self.g);
        self.model.deinit(&self.g);
        self.embed_rows.close();
        self.engram.deinit();
        self.arm.deinit();
        self.dropKernels();
        self.g.deinit();
        gpa.destroy(self);
    }

    /// The kernels go after the last launch drained.
    fn dropKernels(self: *Module) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.kernels.deinit(&self.g);
    }

    /// A fresh request: the prompt from a new state (the model chunks it by its own rule);
    /// the last row's logits.
    pub fn prefill(self: *Module, ids: []const u32) !mlx.mlx_array {
        if (self.state) |*st| st.deinit(&self.g, self.gpa);
        self.state = null;
        self.state = try self.model.newState();
        return self.forward(ids);
    }

    /// Later positions of the request: a decode-width forward runs the phase change first, once.
    pub fn extend(self: *Module, ids: []const u32) !mlx.mlx_array {
        if (ids.len == 1 and !self.arm.grown) {
            if (!self.fenced) {
                try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
                self.fenced = true;
            }
            try self.arm.grow(&self.g);
        }
        return self.forward(ids);
    }

    fn forward(self: *Module, ids: []const u32) !mlx.mlx_array {
        const g = &self.g;
        const st = &(self.state orelse return error.Dsv41NoRequest);
        const r = try self.model.forward(g, st, ids, .{ .logits = .last }, &self.arm.hook, graph.NoProbe{});
        try M.fence(g, st, &.{r.logits.?});
        try self.arm.hook.flush();
        const out = g.keep(r.logits.?);
        g.reset();
        return out;
    }
};

/// The Engram residents' sidecar joins the loaded shards (the index names none of them).
fn loadEngramResidents(gpa: std.mem.Allocator, weights: *model_io.Weights, dir: []const u8) !void {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/" ++ dsp.engram_residents_file, .{dir}, 0);
    defer gpa.free(path);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try model_io.loadSafetensorsFile(gpa, weights, path.ptr, cpu, dsp.resident_load_opts);
}

fn refused(err: anyerror, diag: *const arm_mod.Diag) anyerror {
    log.err("refused: {s} {s}", .{ @errorName(err), diag.message() });
    return err;
}

fn refuse(diag: *arm_mod.Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    const s = std.fmt.bufPrint(&diag.buf, fmt, args) catch diag.buf[0..];
    diag.len = s.len;
    return err;
}

/// The kernels, accepted once before the expert source allocates; their message in `diag`.
fn acceptKernels(a: std.mem.Allocator, g: *G, opts: xo.StartupOptions, diag: *arm_mod.Diag) !*xo.Accepted(G) {
    var kd: xk.Diag = .{};
    return xo.acceptAtStartup(G, a, g, opts, &kd) catch |e| return refuse(diag, e, "kernels: {s}", .{kd.message()});
}

/// A bank's gate / up / down arrays are what the kernels read.
fn checkBank(g: *G, reg: *const xk.Registry, bank: xp.BankArraysOf(G.T), diag: *xk.Diag) xo.Refusal!void {
    inline for (.{ .{ xo.Proj.gate, bank.gate }, .{ xo.Proj.up, bank.up }, .{ xo.Proj.down, bank.down } }) |pb| {
        const p = pb[1];
        try xo.checkBank(G, g, reg, pb[0], .{ .code = p.code, .rout = p.rout, .rin = p.rin }, diag);
    }
}

/// Every bank the hook bound (base and transient; the grown ones after the phase change).
fn checkArmBanks(arm: *A, g: *G, reg: *const xk.Registry, diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        checkBank(g, reg, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @enumFromInt(kind)), kd.message() });
    };
}

/// The phase change's banks, once (`Arm.grown_check`); a refusal is logged by name.
const GrownBanks = struct {
    fn check(ctx: *const anyopaque, arm: *A, g: *G) anyerror!void {
        const reg: *const xk.Registry = @ptrCast(@alignCast(ctx));
        var diag: arm_mod.Diag = .{};
        checkArmBanks(arm, g, reg, &diag) catch |e| {
            log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
            return e;
        };
    }
};
