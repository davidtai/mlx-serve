//! DeepSeek-V4.1 as a module-owned arch of mlx-serve (the deepseek_v4 pattern): `Transformer.dsv41`
//! holds a `Module` that `Transformer.init` builds from the loaded residents and `forwardWith` runs,
//! its per-request state rebuilt at `cache.step == 0`. Construction refuses by name, in order:
//!   1. the kernels (C2, kernels note sec. 19: the load context's `kernel_set.Set`, the registry against the pinned manifest,
//!      every kernel built on the device, the device self-check judged);
//!   2. the expert source (`deepseek_v41_arm.ArmWith`: bank, admission, the stream at the admitted
//!      rows on MLX slot memory, the hook over the kernels' GEMV and the DIG-X prefill route), every
//!      bank it bound checked against the kernels' layout, again at the phase change;
//!   3. the residents' rows and model (as `deepseek_v41_dspark_serve.Resources.open`, over the
//!      shell's loaded residents: the Engram sidecar, the Engram rows, the embedding rows, the trunk
//!      at the served tier `routes.served`, the draft head at its draft routes), then the install
//!      warm-up: every compiled region traced once at the shapes a request reaches.
//! The phase change (the embedding's host rows, the grown slot banks) runs once, at the first
//! decode-width forward after a prompt.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_io = @import("model.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const xk = @import("exl3_kernels.zig");
const kernel_set = @import("kernel_set.zig");
const xq = @import("exl3_quant.zig");
const trunk_routes = @import("dsv41_kernel_routes.zig");
const selfcheck = @import("exl3_selfcheck.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_admission = @import("expert_admission.zig");
const graph = @import("deepseek_v41_graph.zig");
const routes = @import("deepseek_v41_routes.zig");
const eng = @import("deepseek_v41_engram.zig");
const mdl = @import("deepseek_v41_model.zig");
const dh = @import("deepseek_v41_dspark_head.zig");
const qwen4 = @import("qwen4_exp.zig");
const dsp = @import("deepseek_v41_dspark_serve.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");

const log = std.log.scoped(.dsv41);

const G = ops.MlxOps;
/// The expert source: the exact tier's op chain around the kernels' EXL3 decode GEMV, the wide
/// (prefill) routed calls on the kernels' DIG-X route.
pub const A = arm_mod.ArmWith(G, xp.QuantMath(G, xq.Accepted(G)), .{ .prefill = true });
const M = mdl.Model(G);
const H = dh.Head(G);

/// The shell's generation headroom for a request that declared no budget
/// (`transformer.KVCache.RESERVE_GEN_HEADROOM`).
pub const generation_headroom: u64 = 8192;

/// Beside the model's shards: the Engram token map the converter exports.
pub const engram_token_map_file = "engram-token-map.u32";

/// The admission's calibration; its MLX allocator cache charges are the limits the module sets per phase.
const envelope = expert_admission.Envelope.dsv41_pass2;

pub const Module = struct {
    gpa: std.mem.Allocator,
    g: G,
    /// The load context's kernel set, its launcher installed on `g`.
    set: *kernel_set.Set,
    /// The EXL3 quant accepted on the set: the served routed-expert math (C2).
    exl3: *xq.Accepted(G),
    /// The trunk routes' self-check results (their acceptance on the set).
    trunk_report: selfcheck.Report = .{},
    /// The install warm-up's per-shape MLX peaks (one per forward width, then the draft
    /// block's): the bill's transient terms (C4, P4).
    warm_peaks: []u64 = &.{},
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
    /// MLX's allocator cache limit before the module set its own (restored at deinit).
    prev_cache_limit: usize = 0,

    /// `config` is the shell's (its bank and token-map paths, the memory baseline); `weights`
    /// the loaded residents (the Engram sidecar joins them here).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, config: *const model_io.ModelConfig, weights: *model_io.Weights, s: mlx.mlx_stream) !*Module {
        const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
        const map = config.engram_token_map_path orelse return error.Dsv41BankDir;
        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .g = try G.init(gpa, s), .set = undefined, .exl3 = undefined, .arm = undefined, .weights = weights, .engram = undefined, .embed_rows = undefined, .model = undefined, .head = undefined };
        errdefer self.g.deinit();
        var diag: arm_mod.Diag = .{};
        var vd0: v41.Diag = .{};
        const c0 = v41.Config.load(gpa, io, dir, &vd0) catch |e| {
            log.err("config refused: {s}", .{vd0.message()});
            return e;
        };
        try self.acceptKernels(gpa, &c0, s, &diag);
        errdefer self.dropKernels();
        _ = mlx.mlx_clear_cache();
        // The allocator cache holds no more than the admission charges for the phase (prefill here).
        _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, envelope.prefill_cache_bytes);
        errdefer setCacheLimit(self.prev_cache_limit);
        self.arm = A.init(gpa, io, &self.g, self.exl3, .{
            .model_dir = dir,
            .envelope = envelope,
            .baseline_bytes = config.memory_baseline_bytes,
            .fixed_rows = config.expert_rows,
            .slot_memory = .{ .mlx = s },
            .draft_pruned_bytes = 0,
        }, &diag) catch |e| return refused(e, &diag);
        errdefer self.arm.deinit();
        checkArmBanks(self.arm, &self.g, self.exl3, &diag) catch |e| return refused(e, &diag);
        self.arm.grown_check = .{ .ctx = self.exl3, .check = GrownBanks.check };
        var vd: v41.Diag = .{};
        errdefer if (vd.len > 0) log.err("residents refused: {s}", .{vd.message()});
        const c = self.arm.config;
        if (c.engram.n_layers > 0) try loadEngramResidents(gpa, weights, dir);
        self.engram = try eng.RowSource.open(gpa, io, dir, map, &c, &vd);
        errdefer self.engram.deinit();
        self.embed_rows = try dsp.openEmbeddingRows(gpa, io, dir, &c, &vd);
        errdefer self.embed_rows.close();
        self.model = try M.init(gpa, &self.g, c, routes.served, weights, &self.engram);
        errdefer self.model.deinit(&self.g);
        self.head = try H.initWith(gpa, &self.g, c, routes.served.draftRoutes(), weights, .{ .subset = if (self.arm.draft_subset) |*x| x else null });
        errdefer self.head.deinit(&self.g);
        // The install warm-up (P4.3): every forward width up to the compiled regions' bound traces here,
        // never in a request (the draft block joins once the draft round, P5, serves its depth). Each
        // shape's MLX peak is kept for the bill (C4).
        self.warm_peaks = try dsl.Loop(G).warmFor(&self.g, gpa, self.model, self.head, &self.arm.hook, .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) }, graph.attn_compile_max_rows);
        errdefer gpa.free(self.warm_peaks);
        _ = mlx.mlx_clear_cache();
        log.info("warm-up: {d} widths, widest peak {d} B above the residents; built residents {d} B (W97)", .{ self.warm_peaks.len - 1, std.mem.max(u64, self.warm_peaks), self.model.builtBytes() + self.head.builtBytes() });
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
        self.gpa.free(self.warm_peaks);
        self.dropKernels();
        self.g.deinit();
        setCacheLimit(self.prev_cache_limit);
        gpa.destroy(self);
    }

    /// The kernel set, its launcher, then the quant and the trunk routes' acceptance (C2).
    fn acceptKernels(self: *Module, gpa: std.mem.Allocator, c: *const v41.Config, s: mlx.mlx_stream, diag: *arm_mod.Diag) !void {
        var kd: xk.Diag = .{};
        self.set = kernel_set.Set.init(gpa, .{ .device = .{ .stream = s } }, &kd) catch |e| return refuse(diag, e, "kernels: {s}", .{kd.message()});
        errdefer self.set.deinit();
        self.set.install(G, &self.g);
        errdefer kernel_set.Set.uninstall(G, &self.g);
        self.exl3 = xq.accept(G, gpa, &self.g, .{ .kernels = self.set }, .{
            .hidden = c.hidden_size,
            .inter = c.moe_intermediate_size,
            .top_k = c.n_experts_per_tok,
            .n_layers = c.n_layers,
            .act = .{ .swiglu_clamped = c.swiglu_limit },
            .input = .bfloat16,
        }, &kd) catch |e| return refuse(diag, e, "exl3 quant: {s}", .{kd.message()});
        errdefer self.exl3.deinit(&self.g);
        trunk_routes.accept(gpa, self.set, &self.trunk_report, &kd) catch |e| {
            self.trunk_report.deinit(gpa);
            return refuse(diag, e, "trunk routes: {s}", .{kd.message()});
        };
    }

    /// The kernels go after the last launch drained.
    fn dropKernels(self: *Module) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.trunk_report.deinit(self.gpa);
        self.exl3.deinit(&self.g);
        kernel_set.Set.uninstall(G, &self.g);
        self.set.deinit();
    }

    /// A fresh request: the prompt from a new state (the model chunks it by its own rule);
    /// the last row's logits.
    ///
    /// The request's KV lanes are bounded to its positions (M5BOUND48, the served default): the
    /// ring window, the compress / index / frontier lanes preallocated once and never grown.
    /// `reserved_tokens` is the request's KV reservation (the shell's `KVCache.reserve`: prompt +
    /// its generation budget + a chunk); 0 (none declared) bounds it at the prompt plus the shell's
    /// generation headroom. A forward past the bound is refused by name (BoundedLaneFull).
    pub fn prefill(self: *Module, ids: []const u32, reserved_tokens: u64) !mlx.mlx_array {
        if (self.state) |*st| st.deinit(&self.g, self.gpa);
        self.state = null;
        self.state = try self.model.newStateWith(self.model.boundedKv(maxPositions(ids.len, reserved_tokens)));
        return self.forward(ids);
    }

    /// Positions a request's bounded lanes hold: its reservation (else the prompt plus the shell's
    /// generation headroom), plus one verify block.
    pub fn maxPositions(prompt: usize, reserved_tokens: u64) u32 {
        const budget: u64 = if (reserved_tokens > prompt) reserved_tokens else prompt + generation_headroom;
        return @intCast(budget + mdl.Model(G).scratch_rows);
    }

    /// Later positions of the request: a decode-width forward runs the phase change first, once.
    pub fn extend(self: *Module, ids: []const u32) !mlx.mlx_array {
        if (ids.len == 1 and !self.arm.grown) {
            if (!self.fenced) {
                try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
                self.fenced = true;
            }
            // The prefill's parked buffers go back before the slot banks grow; decode keeps its own charge.
            _ = mlx.mlx_clear_cache();
            setCacheLimit(envelope.decode_cache_bytes);
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

fn setCacheLimit(limit: usize) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, limit);
}

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

/// Every bank the hook bound (base and transient; the grown ones after the phase change).
fn checkArmBanks(arm: *A, g: *G, exl3: *const xq.Accepted(G), diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        exl3.checkBank(g, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @fromBackingInt(@intCast(kind))), kd.message() });
    };
}

/// The phase change's banks, once (`Arm.grown_check`); a refusal is logged by name.
const GrownBanks = struct {
    fn check(ctx: *const anyopaque, arm: *A, g: *G) anyerror!void {
        const exl3: *const xq.Accepted(G) = @ptrCast(@alignCast(ctx));
        var diag: arm_mod.Diag = .{};
        checkArmBanks(arm, g, exl3, &diag) catch |e| {
            log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
            return e;
        };
    }
};

test "dsv41 module: a request's bounded lanes hold its reservation, else the prompt plus the shell's headroom, plus a verify block" {
    try std.testing.expectEqual(@import("transformer.zig").KVCache.RESERVE_GEN_HEADROOM, generation_headroom);
    // 16K prompt, no declared budget: 16384 + 8192 + 8.
    try std.testing.expectEqual(@as(u32, 16384 + 8192 + 8), Module.maxPositions(16384, 0));
    // A reservation (prompt + budget + chunk) is the bound.
    try std.testing.expectEqual(@as(u32, 40000 + 8), Module.maxPositions(32768, 40000));
}

test "dsv41 module: the module's construction and forwards analyse (host, nothing runs)" {
    try std.testing.expect(@TypeOf(&Module.init) != void and @TypeOf(&Module.extend) != void);
}

// DSV41_BANK=<bank> [DSV41_MODULE_BASELINE_GB=7.755397656] [DSV41_MODULE_WIRED_GB=3.377741824]
// [DSV41_MODULE_ROWS=<--expert-rows>] [DSV41_MODULE_HEAD=ceiling: the record's pruned draft head]: the module's
// expert-source plan on the real bank at a box baseline (CPU: config, bank, admission; no slot memory).
test "dsv41 module: the served plan on the real bank at a box baseline" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const gb = struct {
        fn of(name: [*:0]const u8, default: f64) !u64 {
            const v = if (std.c.getenv(name)) |x| try std.fmt.parseFloat(f64, std.mem.span(x)) else default;
            return @intFromFloat(@round(v * 1e9));
        }
    }.of;
    const a = std.testing.allocator;
    var diag: arm_mod.Diag = .{};
    var p = arm_mod.planRows(a, std.testing.io, .{
        .model_dir = bank,
        .baseline_bytes = try gb("DSV41_MODULE_BASELINE_GB", 7.755397656),
        .wired_bytes = try gb("DSV41_MODULE_WIRED_GB", 3.377741824),
        .fixed_rows = if (std.c.getenv("DSV41_MODULE_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else null,
        .slot_memory = .host,
        .draft_pruned_bytes = if (std.c.getenv("DSV41_MODULE_HEAD") != null) null else 0,
    }, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    const ad = p.plan.admission;
    const rec: u64 = 13_315_584;
    std.debug.print("\nDSV41_MODULE_PLAN {{\"prefill_rows\": {d}, \"decode_rows\": {d}, \"slot_bank_prefill_bytes\": {d}, \"slot_bank_decode_bytes\": {d}, \"active_bound_bytes\": {d}, \"physical_bound_bytes\": {d}, \"host_reserve_bytes\": {d}, \"baseline_bytes\": {d}}}\n", .{
        p.prefill_rows, p.decode_rows, (40 * @as(u64, p.prefill_rows) + 48) * rec, (40 * @as(u64, p.decode_rows) + 48) * rec, ad.active_bound_bytes, ad.physical_bound_bytes, ad.host_reserve_bytes, p.inputs.baseline_bytes,
    });
    try std.testing.expect(p.decode_rows >= p.prefill_rows and ad.physical_bound_bytes <= 110_000_000_000);
}
