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
const kvc = @import("deepseek_v41_cache.zig");
const dh = @import("deepseek_v41_dspark_head.zig");
const qwen4 = @import("qwen4_exp.zig");
const dsp = @import("deepseek_v41_dspark_serve.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");

const log = std.log.scoped(.dsv41);

const G = ops.MlxOps;
const expert_stream = @import("expert_stream.zig");
const expert_event = @import("expert_event.zig");
const Math = xp.QuantMath(G, xq.Accepted(G));
// The RC routes' rows are the decode-width forwards the experts prove fit one route (never the wide lane).
comptime {
    std.debug.assert(xp.decode_forward_rows == graph.rc_max_rows);
}
/// The expert source: the EXL3 quant's math (C2), the wide (prefill) routed calls on its DIG-X route, the
/// next layer's reads started from the predictor. `A` waits on the host (LOOKAHEAD3, the exact tier);
/// `AGated` builds every wave over event gates (LOOKAHEAD4, the typical tier).
pub const A = arm_mod.ArmWith(G, Math, .{ .prefill = true, .lookahead = true });
pub const AGated = arm_mod.ArmWith(G, Math, .{ .prefill = true, .lookahead = true, .gated = true });

/// The chosen source and the router gates its predictor reads (borrowed from the residents).
fn Tiered(comptime AT: type) type {
    return struct { arm: *AT, gates: []AT.Hook.Gate };
}
/// Built once, by the `expert_event_gates` setting.
pub const Arm = union(enum) { host_waits: Tiered(A), event_gates: Tiered(AGated) };

/// The read-ahead of both tiers (`DSV41_LOOKAHEAD3` / `DSV41_LOOKAHEAD4` `=8:inf:2`): top 8 by the predictor,
/// no threshold, 2 records per call.
pub const lookahead: expert_stream.Lookahead = .{ .k = 8, .tau = std.math.inf(f32), .budget = 2 };
/// A gate whose bytes have not landed by then fails the stream (the lane's watchdog).
const event_watchdog_ms = 2000;
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
    arm: Arm,
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
    /// The inference thread that owns `g.s` (MLX streams are per thread).
    owner: std.Thread.Id = 0,

    /// `config` is the shell's (its bank and token-map paths, the memory baseline); `weights`
    /// the loaded residents (the Engram sidecar joins them here).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, config: *const model_io.ModelConfig, weights: *model_io.Weights, s: mlx.mlx_stream) !*Module {
        const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
        const map = config.engram_token_map_path orelse return error.Dsv41BankDir;
        const layer_major = layerMajor(config) catch |e| {
            log.err("prefill routes refused: {s}", .{@errorName(e)});
            return e;
        };
        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .g = try G.init(gpa, s), .set = undefined, .exl3 = undefined, .arm = undefined, .weights = weights, .engram = undefined, .embed_rows = undefined, .model = undefined, .head = undefined };
        errdefer self.g.deinit();
        self.owner = std.Thread.getCurrentId();
        var diag: arm_mod.Diag = .{};
        var vd0: v41.Diag = .{};
        const c0 = v41.Config.load(gpa, io, dir, &vd0) catch |e| {
            log.err("config refused: {s}", .{vd0.message()});
            return e;
        };
        try self.acceptKernels(gpa, &c0, s, &diag);
        // The box the admission fits: the configured ceiling, else the GPU's working set (the wired limit).
        const ceiling = boxCeiling(config.memory_ceiling_bytes orelse mlx.maxRecommendedWorkingSet(), c0.n_routed_experts);
        errdefer self.dropKernels();
        self.g.clearCache();
        // The allocator cache holds no more than the admission charges for the phase (prefill here).
        _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, prefillCacheLimit(config.numeric_tier orelse .served));
        errdefer setCacheLimit(self.prev_cache_limit);
        self.arm = if (config.expert_event_gates orelse false)
            .{ .event_gates = try self.buildArm(AGated, io, config, weights, s, ceiling, try expert_event.createMetal(), &diag) }
        else
            .{ .host_waits = try self.buildArm(A, io, config, weights, s, ceiling, null, &diag) };
        errdefer self.dropArm();
        var vd: v41.Diag = .{};
        errdefer if (vd.len > 0) log.err("residents refused: {s}", .{vd.message()});
        const c = switch (self.arm) {
            inline else => |t| t.arm.config,
        };
        if (c.engram.n_layers > 0) try loadEngramResidents(gpa, weights, dir);
        self.engram = try eng.RowSource.open(gpa, io, dir, map, &c, &vd);
        errdefer self.engram.deinit();
        self.embed_rows = try dsp.openEmbeddingRows(gpa, io, dir, &c, &vd);
        errdefer self.embed_rows.close();
        var tier = numericTier(config.numeric_tier orelse .served);
        tier.layer_major = layer_major;
        log.info("numeric tier: {t}; prefill layer-major {}, wide feed {}, wide depth {d}, cold rows {d}", .{ config.numeric_tier orelse .served, tier.layer_major, config.expert_wide_feed orelse false, config.expert_wide_depth orelse 1, config.expert_wide_cold_rows orelse 0 });
        self.model = try M.initWith(gpa, &self.g, c, tier, weights, &self.engram, .{ .registry = &self.set.reg });
        errdefer self.model.deinit(&self.g);
        const subset = switch (self.arm) {
            inline else => |t| if (t.arm.draft_subset) |*x| x else null,
        };
        self.head = try H.initWith(gpa, &self.g, c, tier.draftRoutes(), weights, .{ .subset = subset, .registry = &self.set.reg });
        errdefer self.head.deinit(&self.g);
        // The install warm-up (P4.3): every forward width up to the compiled regions' bound traces here,
        // never in a request (the draft block joins once the draft round, P5, serves its depth). Each
        // shape's MLX peak is kept for the bill (C4).
        self.warm_peaks = switch (self.arm) {
            inline else => |t| try dsl.Loop(G).warmFor(&self.g, gpa, self.model, self.head, &t.arm.hook, .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) }, graph.attn_compile_max_rows),
        };
        errdefer gpa.free(self.warm_peaks);
        self.g.clearCache();
        log.info("warm-up: {d} widths, widest peak {d} B above the residents; built residents {d} B", .{ self.warm_peaks.len - 1, std.mem.max(u64, self.warm_peaks), self.model.builtBytes() + self.head.builtBytes() });
        // The bill against the warm-up's measured peak (C4 G7): the widest decode-width wave, the tier's head.
        if (config.dsv41_prefill) |bill| {
            const bt: v41.PrefillBill.Tier = switch (config.numeric_tier orelse .served) {
                .stock => .stock,
                .served => .served,
            };
            const billed = bill.waveBytes(M.scratch_rows, M.scratch_rows, bt) + (if (bt == .stock) bill.head_promotion_bytes else 0);
            const measured = std.mem.max(u64, self.warm_peaks);
            log.info("bill: decode-width wave billed {d} B, warm-up measured {d} B, error {d} B", .{ billed, measured, @as(i64, @intCast(billed)) - @as(i64, @intCast(measured)) });
        }
        return self;
    }

    /// The expert source at the admitted rows, its banks checked against the quant (again at the phase change).
    fn buildArm(self: *Module, comptime AT: type, io: std.Io, config: *const model_io.ModelConfig, weights: *const model_io.Weights, s: mlx.mlx_stream, ceiling: expert_admission.Ceiling, event: ?expert_event.Event, diag: *arm_mod.Diag) !Tiered(AT) {
        const gpa = self.gpa;
        const gates = try routerGates(AT.Hook.Gate, gpa, weights, config.num_hidden_layers);
        errdefer gpa.free(gates);
        var opts = armOptions(config, ceiling, .{ .mlx = s });
        opts.event = if (event) |e| .{ .backend = .{ .metal = e.object }, .watchdog_ms = event_watchdog_ms } else null;
        const arm = AT.initHooked(gpa, io, &self.g, self.exl3, opts, .{ .gates = gates, .event = event, .wide = wideRoute(config) }, diag) catch |e| return refused(e, diag);
        errdefer arm.deinit();
        checkArmBanks(arm, &self.g, self.exl3, diag) catch |e| return refused(e, diag);
        arm.grown_check = .{ .ctx = self.exl3, .check = GrownBanks(AT).check };
        return .{ .arm = arm, .gates = gates };
    }

    fn dropArm(self: *Module) void {
        switch (self.arm) {
            inline else => |t| {
                t.arm.deinit();
                self.gpa.free(t.gates);
            },
        }
    }

    pub fn deinit(self: *Module) void {
        const gpa = self.gpa;
        if (self.state) |*st| st.deinit(&self.g, gpa);
        self.head.deinit(&self.g);
        self.model.deinit(&self.g);
        self.embed_rows.close();
        self.engram.deinit();
        self.dropArm();
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
        // The process's teardown frees the registry off the inference thread, which has stopped launching.
        if (std.Thread.getCurrentId() == self.owner) _ = mlx.mlx_synchronize(self.g.s);
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
        if (phaseChangeDue(ids.len, self.grown())) try self.phaseChange();
        return self.forward(ids);
    }

    /// The prompt fence and the grown slot banks, once (a no-op after): the embedding to its host
    /// rows, the prefill's parked buffers back, the decode cache charge, the banks at the decode rows.
    pub fn phaseChange(self: *Module) !void {
        if (self.grown()) return;
        if (!self.fenced) {
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            self.fenced = true;
        }
        // The prefill's parked buffers go back before the slot banks grow; decode keeps its own charge.
        self.g.clearCache();
        setCacheLimit(envelope.decode_cache_bytes);
        switch (self.arm) {
            inline else => |t| try t.arm.grow(&self.g),
        }
    }

    fn grown(self: *const Module) bool {
        return switch (self.arm) {
            inline else => |t| t.arm.grown,
        };
    }

    fn forward(self: *Module, ids: []const u32) !mlx.mlx_array {
        const g = &self.g;
        const st = &(self.state orelse return error.Dsv41NoRequest);
        switch (self.arm) {
            inline else => |t| return requestForward(G, g, self.model, st, ids, &t.arm.hook),
        }
    }
};

/// Whether `v41.PrefillBill` bills the layer-major pass (`layerMajorBytes`: one chunk's attention
/// side and the routed group's sub-wave per layer, the server's per-request admission reads it).
pub const layer_major_billed = true;

/// The `layer_major_prefill` setting, checked before anything is built. K16 batches each layer's
/// routed call across chunks (the wide lane): the stock tier's prompt forwards are decode-width.
pub fn layerMajor(config: *const model_io.ModelConfig) error{ LayerMajorOnStockTier, LayerMajorNotBilled }!bool {
    if (!(config.layer_major_prefill orelse false)) return false;
    if ((config.numeric_tier orelse .served) == .stock) return error.LayerMajorOnStockTier;
    if (!layer_major_billed) return error.LayerMajorNotBilled;
    return true;
}

/// The wide prefill calls' read schedule from the model settings (off by default).
pub fn wideRoute(config: *const model_io.ModelConfig) xp.Wide {
    return .{ .feed = config.expert_wide_feed orelse false, .depth = config.expert_wide_depth orelse 1, .cold_rows = config.expert_wide_cold_rows orelse 0 };
}

/// The trunk's numerics by construction: `stock` is the exact reference math with every prompt forward
/// decode-width (8 rows: no rounding-class wide lane); `served` is the tier of record (its DIG-X prefill).
pub fn numericTier(t: @import("model_settings.zig").NumericTier) routes.Tier {
    return switch (t) {
        .stock => blk: {
            var s = routes.stock;
            s.prefill_chunk = routes.min_prefill_chunk;
            break :blk s;
        },
        .served => routes.served,
    };
}

/// The admitted modeled peak lands this far under the box's ceiling.
pub const ceiling_stop_bytes: u64 = 2_000_000_000;

/// The box a streamed-expert admission fits under a memory ceiling (the GPU's working set by default):
/// the peak `ceiling_stop_bytes` under it, every layer up to its expert count.
/// The arm's construction options from the shell's config (the admission's inputs): the module builds
/// with them, and a host bill plans the same rows with them (`slot_memory = .host`).
pub fn armOptions(config: *const model_io.ModelConfig, ceiling: expert_admission.Ceiling, slot_memory: expert_stream.SlotMemory) arm_mod.Options {
    return .{
        .model_dir = config.expert_bank_dir.?,
        .envelope = envelope,
        .baseline_bytes = config.memory_baseline_bytes,
        .fixed_rows = config.expert_rows,
        .slot_memory = slot_memory,
        .draft_pruned_bytes = 0,
        .lookahead = lookahead,
        .ceiling = ceiling,
        .wide_depth = config.expert_wide_depth orelse 1,
    };
}

pub fn boxCeiling(ceiling_bytes: u64, n_experts: u32) expert_admission.Ceiling {
    return .ofWorkingSet(ceiling_bytes, ceiling_stop_bytes, n_experts);
}

/// The phase change runs before the first decode-width forward of a prompt, once.
pub fn phaseChangeDue(rows: usize, grown: bool) bool {
    return rows == 1 and !grown;
}

/// One forward of a served request: the model's own chunking, the last row's logits (kept), the
/// hook's settle, one reset.
pub fn requestForward(comptime B: type, g: *B, model: *mdl.Model(B), st: *mdl.Model(B).State, ids: []const u32, hook: anytype) !B.T {
    const r = try model.forward(g, st, ids, .{ .logits = .last }, hook, graph.NoProbe{});
    try mdl.Model(B).fence(g, st, &.{r.logits.?});
    try hook.flush();
    const out = g.keep(r.logits.?);
    g.reset();
    return out;
}

/// The allocator cache the prefill holds: the tier of record's 4 GiB (its CACHE_GIB, charged in the arm's
/// prefill charge), the stock tier the envelope's own.
pub fn prefillCacheLimit(t: @import("model_settings.zig").NumericTier) usize {
    return switch (t) {
        .served => 4 << 30,
        .stock => envelope.prefill_cache_bytes,
    };
}

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
fn checkArmBanks(arm: anytype, g: *G, exl3: *const xq.Accepted(G), diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        exl3.checkBank(g, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @fromBackingInt(@intCast(kind))), kd.message() });
    };
}

/// The phase change's banks, once (`Arm.grown_check`); a refusal is logged by name.
fn GrownBanks(comptime AT: type) type {
    return struct {
        fn check(ctx: *const anyopaque, arm: *AT, g: *G) anyerror!void {
            const exl3: *const xq.Accepted(G) = @ptrCast(@alignCast(ctx));
            var diag: arm_mod.Diag = .{};
            checkArmBanks(arm, g, exl3, &diag) catch |e| {
                log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
                return e;
            };
        }
    };
}

/// `layers.<l>.ffn.gate.{weight,bias}` of every routed layer, refused by name when one is missing.
fn routerGates(comptime Gate: type, gpa: std.mem.Allocator, weights: *const model_io.Weights, n_layers: u32) ![]Gate {
    const gates = try gpa.alloc(Gate, n_layers);
    errdefer gpa.free(gates);
    var buf: [64]u8 = undefined;
    for (gates, 0..) |*gt, l| {
        const w = weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.ffn.gate.weight", .{l}));
        const b = weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.ffn.gate.bias", .{l}));
        if (w == null or b == null) {
            log.err("refused: MissingWeight layers.{d}.ffn.gate", .{l});
            return error.MissingWeight;
        }
        gt.* = .{ .w = w.?, .bias = b.? };
    }
    return gates;
}

test "dsv41 module: each tier's prefill allocator cache is inside what the admission charges the prefill" {
    const charged = envelope.prefill_cache_bytes + arm_mod.pass2_prefill_charge_bytes;
    try std.testing.expect(prefillCacheLimit(.served) <= charged and prefillCacheLimit(.stock) <= charged);
    try std.testing.expectEqual(@as(usize, envelope.prefill_cache_bytes), prefillCacheLimit(.stock));
}

test "dsv41 module: a request's bounded lanes hold its reservation, else the prompt plus the shell's headroom, plus a verify block" {
    try std.testing.expectEqual(@import("transformer.zig").KVCache.RESERVE_GEN_HEADROOM, generation_headroom);
    // 16K prompt, no declared budget: 16384 + 8192 + 8.
    try std.testing.expectEqual(@as(u32, 16384 + 8192 + 8), Module.maxPositions(16384, 0));
    // A reservation (prompt + budget + chunk) is the bound.
    try std.testing.expectEqual(@as(u32, 40000 + 8), Module.maxPositions(32768, 40000));
}

test "dsv41 module: the prefill routes are off by default; layer-major is refused on the stock tier and until billed" {
    var c: model_io.ModelConfig = undefined;
    c.layer_major_prefill = null;
    c.numeric_tier = null;
    c.expert_wide_feed = null;
    c.expert_wide_depth = null;
    c.expert_wide_cold_rows = null;
    try std.testing.expect(!try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{}, wideRoute(&c));
    c.layer_major_prefill = true;
    c.numeric_tier = .stock;
    try std.testing.expectError(error.LayerMajorOnStockTier, layerMajor(&c));
    c.numeric_tier = .served;
    if (layer_major_billed) try std.testing.expect(try layerMajor(&c)) else try std.testing.expectError(error.LayerMajorNotBilled, layerMajor(&c));
    c.expert_wide_feed = true;
    c.expert_wide_depth = 2;
    c.expert_wide_cold_rows = 2;
    try std.testing.expectEqual(xp.Wide{ .feed = true, .depth = 2, .cold_rows = 2 }, wideRoute(&c));
}

test "dsv41 module: the module's construction and forwards analyse (host, nothing runs)" {
    try std.testing.expect(@TypeOf(&Module.init) != void and @TypeOf(&Module.extend) != void);
}

// DSV41_BANK=<bank> [DSV41_MODULE_BASELINE_GB=7.755397656] [DSV41_MODULE_WIRED_GB=3.377741824]
// [DSV41_MODULE_ROWS=<--expert-rows>] [DSV41_MODULE_HEAD=ceiling: the record's pruned draft head]
// [DSV41_MODULE_CEILING_GB=<--memory-ceiling-gb>: the box at that ceiling; unset: the envelope's own]: the module's
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
        .lookahead = lookahead,
        .ceiling = if (std.c.getenv("DSV41_MODULE_CEILING_GB") != null) boxCeiling(try gb("DSV41_MODULE_CEILING_GB", 0), 384) else null,
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
    const box: u64 = if (p.inputs.ceiling) |cl| cl.box_bytes else expert_admission.box_ceiling_bytes;
    const modeled = if (p.plan.peak_fill) |pf| pf.modeled_peak_bytes else ad.physical_bound_bytes;
    std.debug.print("DSV41_MODULE_BOX {{\"box_bytes\": {d}, \"modeled_peak_bytes\": {d}}}\n", .{ box, modeled });
    try std.testing.expect(p.decode_rows >= p.prefill_rows and ad.physical_bound_bytes <= box);
}

/// The bytes one traced forward `[from, to)` holds, as MlxOps frees its waves (the model lane's bank accounting):
/// each outermost wave's nodes, less its nested sub-waves' (released at their reset, their last array kept),
/// plus the widest sub-wave's two largest arrays live at once; the widest such wave plus the nodes outside all.
const WaveBound = struct {
    reset: u64,
    outside: u64,
    widest: u64,

    fn of(g: *const ops.TraceOps, from: usize, to: usize, freed: []const ops.TraceOps.Freed) WaveBound {
        const total = graph.heldBytes(g, from, to).sum;
        var in_waves: u64 = 0;
        var widest: u64 = 0;
        for (freed, 0..) |w, i| {
            if (w.to <= w.from) continue;
            const inner = for (freed, 0..) |v, j| {
                if (j != i and v.from <= w.from and w.to <= v.to and (v.from != w.from or v.to != w.to)) break true;
            } else false;
            if (inner) continue;
            const all = graph.heldBytes(g, w.from, w.to).sum;
            in_waves += all;
            var kept = all;
            var live: u64 = 0;
            for (freed) |r| {
                if (r.from >= w.from and r.to <= w.to and (r.from != w.from or r.to != w.to) and r.to > r.from) {
                    var a: u64 = 0;
                    var b: u64 = 0;
                    var out: u64 = 0;
                    for (r.from..r.to) |k| {
                        const x = graph.heldBytes(g, k, k + 1).sum;
                        if (x == 0) continue;
                        out = x;
                        if (x > a) {
                            b = a;
                            a = x;
                        } else if (x > b) b = x;
                    }
                    kept = kept - graph.heldBytes(g, r.from, r.to).sum + out;
                    live = @max(live, a + b);
                }
            }
            widest = @max(widest, kept + live);
        }
        return .{ .reset = total, .outside = total -| in_waves, .widest = widest };
    }
};

const RandomIds = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260929),
    n_experts: u16,

    fn values(self: *RandomIds) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *RandomIds = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(_: *anyopaque) anyerror!u32 {
        return 0;
    }
};

// DSV41_BANK=<bank> (host, the trace backend): the served prompt forwards on the bank's own config, residents and
// Engram rows, the routed calls through the stock chain and the DIG-X wide lane. Each forward's bytes per wave (the
// widest outermost wave plus what lies outside the waves) fit the prefill bill's wave at its rows and positions.
test "dsv41 module: the prefill bill covers the served prompt forwards' waves on the bank (trace backend)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 module held: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, bank, &vd);
    const bill = v41.PrefillBill.of(&c);
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/" ++ engram_token_map_file, .{bank}), &c, &vd);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    var g = ops.TraceOps.init(a);
    defer g.deinit();
    const TM = mdl.Model(ops.TraceOps);
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const rows = try aa.alloc(u32, c.n_layers);
    @memset(rows, 8);
    var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows });
    defer fsrc.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Wide = xq.DigXPrefill(ops.TraceOps);
    const digx = try aa.alloc(Wide, c.n_layers);
    for (digx) |*d| d.* = try Wide.init(a, &reg, .tier, null);
    defer for (digx) |*d| d.deinit(&g);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, xp.WithPrefillRoutes(ops.TraceOps, TChain, Wide), .{ .prefill = true });
    var ex = try Ex.init(a, &g, &fsrc, .{ .d = TChain.init(.{}, &c), .routes = digx }, &c);
    defer ex.deinit();
    var rid: RandomIds = .{ .n_experts = @intCast(c.n_routed_experts) };
    g.host_values = rid.values();
    const prompt = try aa.alloc(u32, 16384);
    for (prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
    for ([_]struct { name: []const u8, tier: routes.Tier, attn: v41.PrefillBill.Tier }{
        .{ .name = "stock", .tier = routes.stock, .attn = .stock },
        .{ .name = "served", .tier = routes.served, .attn = .served },
    }) |t| {
        const model_ = try TM.initWith(a, &g, c, t.tier, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        // (positions already in the state, rows): the decode lane, the gate's prompt, wide chunks, the 16K prompt.
        for ([_][2]u32{ .{ 0, 8 }, .{ 0, 63 }, .{ 0, 953 }, .{ 1024, 953 }, .{ 0, 16384 } }) |pn| {
            var st = try model_.newState();
            defer st.deinit(&g, a);
            if (pn[0] > 0) {
                const r0 = try model_.forward(&g, &st, prompt[0..pn[0]], .{ .logits = .none }, &ex, graph.NoProbe{});
                try TM.fence(&g, &st, &.{r0.hidden});
                try ex.flush();
                g.reset();
            }
            const n = pn[1];
            const f0 = g.nodes.items.len;
            const w0 = g.freed.items.len;
            const r = try model_.forward(&g, &st, prompt[pn[0]..][0..n], .{ .logits = .last }, &ex, graph.NoProbe{});
            const h = WaveBound.of(&g, f0, g.nodes.items.len, g.freed.items[w0..]);
            try TM.fence(&g, &st, &.{r.logits.?});
            try ex.flush();
            g.reset();
            // The widest wave is a whole chunk's: the model's chunk, reading the whole prompt at its end.
            const chunk = @min(n, bill.chunkRows(pn[0] + n));
            const billed = bill.waveBytes(chunk, pn[0] + n, t.attn);
            std.debug.print("\nDSV41_HELD {{\"tier\": \"{s}\", \"positions\": {d}, \"rows\": {d}, \"outside\": {d}, \"widest\": {d}, \"billed\": {d}}}", .{ t.name, pn[0], n, h.outside, h.widest, billed });
            try std.testing.expect(h.outside + h.widest <= billed);
        }
    }
    // The bill's chunk is the model's.
    for ([_]u64{ 1, 8, 64, 953, 2048, 4096, 16384, 65536, 131072 }) |sq|
        try std.testing.expectEqual(@as(u64, @intCast(kvc.resolvePrefillChunk(&c, sq, null, kvc.default_chunk_target_bytes))), bill.chunkRows(sq));
    inline for (.{ .stock, .served }) |t| std.debug.print("\nDSV41_PREFILL_BILL {{\"tier\": \"{t}\", \"gate_64_32\": {d}, \"cell_16384_1024\": {d}, \"cell_wave\": {d}}}", .{ @as(v41.PrefillBill.Tier, t), bill.bytes(64, 32, t), bill.bytes(16384, 1024, t), bill.waveBytes(bill.chunkRows(16384), 16384, t) });
}

/// The routed hook with a record of each forward's rows (layer 0's routed call), the order the model feeds it.
fn Recorder(comptime Ex: type) type {
    return struct {
        const Self = @This();
        ex: *Ex,
        rows: std.ArrayList(u32) = .empty,
        /// Index into `rows` of the first forward after each grow.
        grown_at: std.ArrayList(usize) = .empty,
        gpa: std.mem.Allocator,

        const Hook = struct {
            r: *Self,
            layer: u32,
            pub fn routed(h: Hook, g: *ops.TraceOps, xf: u32, indices: u32) !u32 {
                if (h.layer == 0) try h.r.rows.append(h.r.gpa, @intCast(g.shapeOf(xf).dim(0)));
                return h.r.ex.at(h.layer).routed(g, xf, indices);
            }
        };
        pub fn at(self: *Self, layer: u32) Hook {
            return .{ .r = self, .layer = layer };
        }
        pub fn flush(self: *Self) !void {
            try self.ex.flush();
        }
        fn grow(self: *Self, g: *ops.TraceOps, rows: []const u32) !void {
            try self.ex.grow(g, rows);
            try self.grown_at.append(self.gpa, self.rows.items.len);
        }
        fn deinit(self: *Self) void {
            self.rows.deinit(self.gpa);
            self.grown_at.deinit(self.gpa);
        }
    };
}

const ScriptedPicks = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260929),
    n_experts: u16,
    picks: []const u32,
    next: usize = 0,

    fn values(self: *ScriptedPicks) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *ScriptedPicks = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        const s: *ScriptedPicks = @ptrCast(@alignCast(ctx));
        defer s.next += 1;
        return s.picks[s.next % s.picks.len];
    }
};

// DSV41_BANK=<bank> [DSV41_SCHEDULE_REF=<ar-ref json>] (host, the trace backend): the served request's schedule
// (mlx-serve's Generator for a whole-prompt arch, generate.zig: the prompt but its last token in one forward, then
// one token per forward; the module's phase change before the first decode-width forward) against the AR harness's
// (`Model.greedy`, prompt forwards of 8 rows). Both feed the same ids and leave the same Engram history; their
// forward shapes differ, which is why the harness's reference is not the served path's.
test "dsv41 module: the served request's forward schedule against the AR harness's (bank, trace backend)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 module schedule: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, bank, &vd);
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/" ++ engram_token_map_file, .{bank}), &c, &vd);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    // The parity prompt and its reference ids (the M3 reference), else a stand-in prompt.
    var prompt: []const u32 = undefined;
    var picks: []const u32 = undefined;
    if (std.c.getenv("DSV41_SCHEDULE_REF")) |p| {
        const Ref = struct { prompt_ids: []const u32, generated_ids: []const u32 };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(p), aa, .limited(16 << 20));
        const ref = try std.json.parseFromSliceLeaky(Ref, aa, text, .{ .ignore_unknown_fields = true });
        prompt = ref.prompt_ids;
        picks = ref.generated_ids;
    } else {
        const pr = try aa.alloc(u32, 64);
        for (pr, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
        prompt = pr;
        picks = &.{ 1, 1, 1528, 9998, 7, 42 };
    }
    const n_new: usize = 6;
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Wide = xq.DigXPrefill(ops.TraceOps);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, xp.WithPrefillRoutes(ops.TraceOps, TChain, Wide), .{ .prefill = true });
    const TM = mdl.Model(ops.TraceOps);
    const Route = enum { harness, served };
    var fed: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&fed) |*f| f.deinit(a);
    var hist: [2][]i64 = undefined;
    var rows: [2][]u32 = undefined;
    var grown: [2][]usize = undefined;
    for ([_]Route{ .harness, .served }, 0..) |route, ri| {
        var g = ops.TraceOps.init(a);
        defer g.deinit();
        var sp: ScriptedPicks = .{ .n_experts = @intCast(c.n_routed_experts), .picks = picks };
        g.host_values = sp.values();
        const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
        const model_ = try TM.initWith(a, &g, c, routes.served, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        const prows = try aa.alloc(u32, c.n_layers);
        @memset(prows, 8);
        const drows = try aa.alloc(u32, c.n_layers);
        @memset(drows, 16);
        var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = prows });
        defer fsrc.deinit();
        const digx = try aa.alloc(Wide, c.n_layers);
        for (digx) |*d| d.* = try Wide.init(a, &reg, .tier, null);
        defer for (digx) |*d| d.deinit(&g);
        var ex = try Ex.init(a, &g, &fsrc, .{ .d = TChain.init(.{}, &c), .routes = digx }, &c);
        defer ex.deinit();
        var rec: Recorder(Ex) = .{ .ex = &ex, .gpa = a };
        defer rec.deinit();
        var st = try model_.newState();
        defer st.deinit(&g, a);
        switch (route) {
            .harness => {
                // The AR harness (deepseek_v41_ar.zig): its stream grown before the prompt, then Model.greedy.
                try rec.grow(&g, drows);
                const out = try aa.alloc(u32, n_new);
                var i: usize = 0;
                while (i < prompt.len) : (i += 8) try fed[ri].appendSlice(a, prompt[i..@min(i + 8, prompt.len)]);
                try model_.greedy(&g, &st, prompt, 8, &rec, out, {});
                try fed[ri].appendSlice(a, out[0 .. n_new - 1]);
            },
            .served => {
                // The Generator: the prompt but its last token (step 0), then one id per forward; the module's
                // phase change before the first decode-width forward.
                var ids: []const u32 = prompt[0 .. prompt.len - 1];
                var step: usize = 0;
                var next: u32 = prompt[prompt.len - 1];
                while (step <= n_new) : (step += 1) {
                    if (step > 0) {
                        if (phaseChangeDue(ids.len, rec.grown_at.items.len > 0)) try rec.grow(&g, drows);
                    }
                    try fed[ri].appendSlice(a, ids);
                    const lg = try requestForward(ops.TraceOps, &g, model_, &st, ids, &rec);
                    if (step > 0) next = try g.hostArgmax(lg);
                    ids = (&next)[0..1];
                    if (fed[ri].items.len >= prompt.len + n_new - 1) break;
                }
            },
        }
        hist[ri] = try aa.dupe(i64, st.hash.?.hist.items);
        rows[ri] = try aa.dupe(u32, rec.rows.items);
        grown[ri] = try aa.dupe(usize, rec.grown_at.items);
    }
    // The lane per forward, by the hook's own rule: a routed call of at most max_route_ids ids is the decode lane.
    var wide: [2]usize = .{ 0, 0 };
    for (rows, 0..) |rs, ri| for (rs, 0..) |r, fi| {
        const lane_wide = r * c.n_experts_per_tok > xp.max_route_ids;
        if (lane_wide) wide[ri] += 1;
        // Every forward of 8 rows or fewer (the last prompt token and every generated id) is the decode lane.
        if (r <= mdl.Model(ops.TraceOps).scratch_rows) try std.testing.expect(!lane_wide);
        _ = fi;
    };
    std.debug.print("\nDSV41_SCHEDULE harness rows {any} wide-lane forwards {d} grown before forward {any}\nDSV41_SCHEDULE served rows {any} wide-lane forwards {d} grown before forward {any}\n", .{ rows[0], wide[0], grown[0], rows[1], wide[1], grown[1] });
    // The harness never takes the wide lane; the served route takes it once, for the prompt but its last token.
    try std.testing.expectEqual(@as(usize, 0), wide[0]);
    try std.testing.expectEqual(@as(usize, 1), wide[1]);
    // Same ids fed, same Engram history: the plumbing feeds the model what the harness does.
    try std.testing.expectEqualSlices(u32, fed[0].items, fed[1].items);
    try std.testing.expectEqualSlices(i64, hist[0], hist[1]);
    // The documented difference: the shapes (8-row prompt forwards vs one prompt forward, the last token at M = 1
    // after the phase change).
    try std.testing.expect(!std.mem.eql(u32, rows[0], rows[1]));
}
