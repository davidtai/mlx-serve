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
const status = @import("status.zig");
const gpu_ceiling = @import("gpu_ceiling.zig");
const bill_mod = @import("deepseek_v41_bill.zig");
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

/// The served tier's DSpark acceptance (the tier of record: typical 0.3 with the greedy correction).
pub const dspark_typical_delta: f32 = 0.3;
const dspark_lane = "dspark typical 0.3";
comptime {
    std.debug.assert(dspark_typical_delta == 0.3); // the lane string states it
}
/// The strategy's settings on the served path: the cell's (draft depth 5, the confidence stop 0.5, the
/// hybrid lookup), the whole prompt in one forward, no internal stop (the shell owns EOS and the budget).
pub const dspark_config: dsl.Config = .{
    .acceptance = .{ .typical = .{ .delta = dspark_typical_delta } },
    .prompt_chunk = dsl.whole_prompt,
    .max_tokens = std.math.maxInt(u32),
};

/// A request's DSpark strategy: the loop over the Module's state and the head's per-request caches.
const Dspark = struct {
    lp: dsl.Loop(G),
    caches: []H.Cache,
};

/// One DSpark round's result (the shell's `DsparkRound` shape, as `deepseek_v4.DsparkRound`).
pub const DsparkRound = struct {
    tokens: []u32,
    accepted: u32,
    next_token: u32,

    pub fn deinit(self: *DsparkRound, a: std.mem.Allocator) void {
        a.free(self.tokens);
        self.tokens = &.{};
    }
};

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
    /// The DSpark strategy's settings (the served tier with a draft head); null: serial decode only.
    dspark_cfg: ?dsl.Config = null,
    /// The request's DSpark strategy, seeded by `prefill` when `dspark_cfg` is set.
    dspark: ?Dspark = null,
    /// The prompt fence ran: the embedding reads its host rows from then on (per process).
    fenced: bool = false,
    /// The native bill at the admitted rows (set by the construction check; the harnesses' phase records read it).
    bill: bill_mod.Bill = undefined,
    /// MLX's allocator cache limit before the module set its own (restored at deinit).
    prev_cache_limit: usize = 0,
    /// The fill's target (the ceiling less upstream's wired margin): each phase's billed total stays under it.
    fill_target: u64 = 0,
    /// The phase change's boundary readings, freed bytes and reclaim time (the receipts carry it).
    phase_change: ?PhaseChangeRecord = null,
    /// The prompt-start reference and the terminal refusal (`PhaseGate`).
    gate: PhaseGate = .{},
    /// The shell's io (the phase change's bounded settle waits on it).
    io: std.Io = undefined,
    /// The prefill routes as built: the trunk's pass and the hook's wide route (with the stream's
    /// windows). The construction log line and the receipts read these, never the settings.
    installed: Installed = .{},
    /// The stream's counters at the request's start; reported once, at its first later forward.
    prompt_stats0: ?expert_stream.Stats = null,
    prompt_tokens: usize = 0,
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
        self.io = io;
        var diag: arm_mod.Diag = .{};
        var vd0: v41.Diag = .{};
        const c0 = v41.Config.load(gpa, io, dir, &vd0) catch |e| {
            log.err("config refused: {s}", .{vd0.message()});
            return e;
        };
        try self.acceptKernels(gpa, &c0, s, &diag);
        // The box the admission fits: upstream's static GPU ceiling (Metal's working set, or its static override:
        // `--memory-ceiling-gb` / MLX_SERVE_GPU_CEILING_MB), unless a harness states its window's ceiling; the
        // fill's target lands upstream's wired margin (`--wired-margin-gib`) under it, and the bill's totals carry
        // the baseline (the preflight's sample of the memory in use before the load, or `--memory-baseline-gb`).
        const ceiling_bytes = config.memory_ceiling_bytes orelse gpu_ceiling.staticGpuMemoryCeiling();
        const target = ceiling_bytes -| gpu_ceiling.wired_limit_margin_bytes;
        const ceiling = boxCeiling(ceiling_bytes, c0.n_routed_experts);
        // The served admission, one kind only (the native bill; the Python envelope planner never runs here):
        // rows filled up to the stop's target, or `--expert-rows R` as the decode rows with the prompt rows
        // the fill's capped at R. Both row counts then reach the arm as native rows.
        var admitted = config.*;
        if (admitted.expert_prefill_rows == null) {
            admitted.memory_ceiling_bytes = ceiling_bytes;
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const nr = try bill_mod.fill(arena.allocator(), io, admitted, fill_prompt_tokens, fill_max_tokens, status.vmBytes().wired, target);
            if (admitted.expert_rows) |forced| {
                admitted.expert_prefill_rows = @min(nr.prefill, forced);
            } else {
                admitted.expert_rows = nr.decode;
                admitted.expert_prefill_rows = nr.prefill;
            }
            log.info("admission: native fill {d} prefill / {d} decode rows per layer (the {d}-token request's bill, baseline {d} B, target {d} B)", .{ admitted.expert_prefill_rows.?, admitted.expert_rows.?, fill_prompt_tokens, admitted.memory_baseline_bytes orelse 0, target });
        }
        errdefer self.dropKernels();
        // The admission at the admitted rows, BEFORE any slot bank or Module resident is allocated
        // (pass3ah refused only after construction, at an 82.7 GiB footprint): the native bill at the
        // box's wired bytes now (nothing of the Module wired yet); a plan that does not fit refuses here.
        {
            var cfg = admitted;
            cfg.memory_ceiling_bytes = ceiling_bytes;
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const b = bill_mod.billAt(arena.allocator(), io, &cfg, fill_prompt_tokens, fill_max_tokens, status.vmBytes().wired) catch |e| {
                log.err("admission refused before construction: {s}", .{@errorName(e)});
                return e;
            };
            self.fill_target = target;
            // Forced rows too: both phases' totals under the target (a baseline-free shell bills the process alone).
            admitPhases(b, self.fill_target) catch |e| {
                log.err("admission refused before construction: {s} (prompt total {d} B, decode total {d} B, target {d} B)", .{ @errorName(e), b.prefillTotal(), b.decodeTotal(), self.fill_target });
                return e;
            };
        }
        self.g.clearCache();
        // The allocator cache holds no more than the admission charges for the phase (prefill here).
        _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, prefillCacheLimit(config.numeric_tier orelse .served));
        errdefer setCacheLimit(self.prev_cache_limit);
        self.arm = if (config.expert_event_gates orelse false)
            .{ .event_gates = try self.buildArm(AGated, io, &admitted, weights, s, ceiling, try expert_event.createMetal(), &diag) }
        else
            .{ .host_waits = try self.buildArm(A, io, &admitted, weights, s, ceiling, null, &diag) };
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
        if (config.prefill_attn) |v| {
            // The core reads K30's selection: only a tier with selected keys can take it.
            if (v and !tier.routes.selected_keys) return error.PrefillAttnNeedsSelectedKeys;
            tier.routes.prefill_attn = v;
        }
        tier.routes.prefill_index = try prefillIndexRoute(config);
        // The HC norms, the combine and the o-projection: a setting overrides the tier's route.
        if (config.prefill_hc) |v| tier.routes.prefill_hc = v;
        if (config.prefill_combine) |v| tier.routes.prefill_combine = v;
        if (config.prefill_host_shared) |v| tier.routes.prefill_host_shared = v;
        if (config.prefill_joinless) |v| tier.routes.prefill_joinless = v;
        if (config.engram_posted) |v| tier.routes.engram_posted = v;
        // The verify-row routes (C23, C27-C29): a setting overrides the tier's route.
        if (config.decode_attn_softmax) |v| tier.routes.rc_attn_softmax = v;
        if (config.decode_index_topk) |v| tier.routes.rc_index_topk = v;
        if (config.decode_smallm) |v| tier.routes.rc_smallm = v;
        if (config.decode_mxfp8_rows) |v| tier.routes.rc_mxfp8_rows = v;
        if (config.prefill_oproj) |v| {
            if (v and !tier.routes.prefill_attn) return error.PrefillOprojNeedsPrefillAttn;
            tier.routes.prefill_oproj = v;
        }
        tier.layer_major = layer_major;
        log.info("numeric tier: {t}", .{config.numeric_tier orelse .served});
        self.model = try M.initWith(gpa, &self.g, c, tier, weights, &self.engram, .{ .registry = &self.set.reg });
        errdefer self.model.deinit(&self.g);
        if (tier.routes.prefill_attn or tier.routes.prefill_index or tier.routes.prefill_hc or tier.routes.prefill_combine or tier.routes.prefill_oproj or tier.routes.prefill_joinless or tier.routes.rc_smallm or tier.routes.rc_mxfp8_rows or tier.routes.rc_index_topk or tier.routes.rc_attn_softmax) try self.checkPrefillRoutes();
        // ENGRAM=prefetch: the poster threads started and their gathers checked against a read past the cache.
        if (tier.routes.engram_posted and tier.layer_major and c.engram.n_layers > 0) {
            // The pass posts slot s + 1 once slot s's layer is taken: the slots run in layer order.
            for (c.engram.layer_ids[0..c.engram.n_layers], 0..) |l, sl| {
                const slot = c.layers[l].engram_slot orelse return error.EngramSlotOrder;
                if (slot != sl or (sl > 0 and l <= c.engram.layer_ids[sl - 1])) return error.EngramSlotOrder;
            }
            try self.engram.enablePosting();
            self.checkEngramPosted(gpa) catch |e| {
                log.err("NATIVE engram posted: the construction self-check against a read past the cache failed: {s}", .{@errorName(e)});
                return e;
            };
            self.model.engram.?.posted = true;
        }
        self.installed = switch (self.arm) {
            inline else => |t| .{ .prefill_unjoined = self.model.tier.routes.prefill_joinless and comptime (@hasDecl(@TypeOf(t.arm.hook).Math, "has_parts") and @TypeOf(t.arm.hook).Math.has_parts), .layer_major = self.model.tier.layer_major, .wide = t.arm.hook.wide_route, .stream_windows = t.arm.stream.wide_depth, .prefill_attn = self.model.tier.routes.prefill_attn, .prefill_index = self.model.tier.routes.prefill_index, .prefill_hc = self.model.tier.routes.prefill_hc, .prefill_combine = self.model.tier.routes.prefill_combine, .prefill_oproj = self.model.tier.routes.prefill_oproj, .prefill_host_shared = self.model.tier.routes.prefill_host_shared, .prefill_joinless = self.model.tier.routes.prefill_joinless, .engram_posted = if (self.model.engram) |en| en.posted else false },
        };
        var line_buf: [384]u8 = undefined;
        log.info("{s}", .{self.installed.line(&line_buf)});
        log.info("{s}", .{self.installed.callSites(&line_buf)});
        self.installed.decode_attn_softmax = self.model.tier.routes.rc_attn_softmax;
        self.installed.decode_index_topk = self.model.tier.routes.rc_index_topk;
        self.installed.decode_smallm = self.model.tier.routes.rc_smallm;
        self.installed.decode_mxfp8_rows = self.model.tier.routes.rc_mxfp8_rows;
        log.info("{s}", .{self.installed.decodeSites(&line_buf)});
        const subset = switch (self.arm) {
            inline else => |t| if (t.arm.draft_subset) |*x| x else null,
        };
        self.head = try H.initWith(gpa, &self.g, c, tier.draftRoutes(), weights, .{ .subset = subset, .registry = &self.set.reg });
        errdefer self.head.deinit(&self.g);
        // The decode lane: DSpark (typical acceptance, the tier of record) on the served tier with a draft head.
        if (self.head.nStages() > 0 and (config.numeric_tier orelse .served) == .served) self.dspark_cfg = dspark_config;
        log.info("NATIVE decode lane installed: {s} (draft block {d})", .{ self.decodeLane(), self.draftBlockSize() });
        // The install warm-up (P4.3): the shapes a request issues trace here, never in a request. With the
        // DSpark strategy those are its own: the verify widths 1..max_rows (8: draft block 5 + the lookup's
        // 2 + 1; seedMain at the same rows) and the 5-row draft block through every stage, so the first
        // round no longer compiles its draft inside timed decode. Widths 9..32 (only a 9-32 token prompt
        // or prompt tail) are left to trace on first use. Without a strategy (serial decode) every width up
        // to the compiled regions' bound, as before. Each shape's MLX peak is kept for the bill (C4).
        const warm_cfg: dsl.Config = self.dspark_cfg orelse .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) };
        const warm_widths: u32 = if (self.dspark_cfg != null) 0 else graph.attn_compile_max_rows;
        self.warm_peaks = switch (self.arm) {
            inline else => |t| try dsl.Loop(G).warmFor(&self.g, gpa, self.model, self.head, &t.arm.hook, warm_cfg, warm_widths),
        };
        errdefer gpa.free(self.warm_peaks);
        // Every warm-up command retired before the clear: their completion handlers hand the buffers they
        // held to the allocator's cache, and a clear ahead of them leaves those cached.
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        log.info("warm-up: {d} widths, widest peak {d} B above the residents; built residents {d} B", .{ self.warm_peaks.len - 1, std.mem.max(u64, self.warm_peaks), self.model.builtBytes() + self.head.builtBytes() });
        log.info("NATIVE warm-up peaks by width 1..{d} then the draft block (0: not warmed), B above the residents: {any}", .{ self.warm_peaks.len - 1, self.warm_peaks });
        // The input embedding moves to its host rows now, not at the phase change: every lookup (the
        // prompt's included) reads the table's rows past the page cache, and the device table is gone
        // from both phases. Checked once: the rows equal the table's, byte for byte.
        if (config.embedding_host_rows orelse true) {
            try self.checkEmbeddingRows(gpa);
            _ = mlx.mlx_synchronize(self.g.s);
            const before = BoundaryMemory.now();
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            // The table's pages can trail its release in the footprint while the driver retires them (the
            // 05:36 / 06:28 cells' constructions: out of MLX's active and cache, 1.32 GB still in the
            // footprint). The phase change's settle waits for the footprint to show the release (bounded)
            // before the construction check reads it.
            const st = settle(LiveReader{ .io = self.io }, before, self.model.embeddingBytes());
            log.info("NATIVE embedding fence: footprint {d} -> {d} B (the table {d} B), settled in {d} ms", .{ before.footprint, st.after.footprint, self.model.embeddingBytes(), st.waited_ms });
            self.fenced = true;
            self.installed.embedding_rows = true;
        }
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
        // The construction check (once, before any request): the native bill at the rows the arm built,
        // against the footprint the module holds now.
        try self.checkConstruction(io, &admitted, ceiling_bytes);
        return self;
    }

    /// The module's native bill at its admitted rows (the standard request's), and the construction
    /// check against it: the footprint after the install (warm-up released, cache cleared) must sit
    /// within `construction_tolerance_bytes` of the bill's construction terms, else the module is
    /// refused by name before any request.
    fn checkConstruction(self: *Module, io: std.Io, admitted: *const model_io.ModelConfig, ceiling_bytes: u64) !void {
        var cfg = admitted.*;
        cfg.memory_ceiling_bytes = ceiling_bytes;
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        // The bill plans through the arm's own inputs: the wired bytes the arm was planned with (a live
        // read here would count this module's own banks and residents as the box's).
        const planned_wired = switch (self.arm) {
            inline else => |t| t.arm.inputs.wired_bytes,
        };
        const b = try bill_mod.billAt(arena.allocator(), io, &cfg, fill_prompt_tokens, fill_max_tokens, planned_wired);
        const rows = switch (self.arm) {
            inline else => |t| arm_mod.NativeRows{ .prefill = t.arm.prefill_rows[0], .decode = t.arm.decode_rows[0] },
        };
        if (b.prefill_rows != rows.prefill or b.decode_rows != rows.decode) {
            log.err("construction check: the bill plans {d} / {d} rows, the arm built {d} / {d}", .{ b.prefill_rows, b.decode_rows, rows.prefill, rows.decode });
            return error.BillRowsMismatch;
        }
        self.bill = b;
        // The footprint the module keeps: every command retired (their completion handlers hand the buffers
        // they held to MLX's cache), then the cache cleared. SERVED9 (pass3an) read it with the fence's
        // table still cached: the fence cleared before its reads retired, and the 2 GiB prefill cache held
        // the 1.32 GB table (+1.33 GB over v7's construction, 0.75 GB over the bill).
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        const measured = status.footprint().now;
        const billed = b.constructionTerms().sum();
        var mlx_active: usize = 0;
        var mlx_cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&mlx_active);
        _ = mlx.mlx_get_cache_memory(&mlx_cache);
        log.info("NATIVE construction check: MLX active {d} B, MLX cache {d} B, host side {d} B (the footprint less both)", .{ mlx_active, mlx_cache, measured -| mlx_active -| mlx_cache });
        log.info("NATIVE construction check: footprint {d} B, billed construction terms {d} B, residual {d} B (tolerance {d} B)", .{ measured, billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(measured)), construction_tolerance_bytes });
        checkConstructionBytes(billed, measured) catch |e| {
            log.err("construction check: the constructed footprint {d} B exceeds the billed construction terms {d} B by more than {d} B", .{ measured, billed, construction_tolerance_bytes });
            return e;
        };
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
        self.dropDspark();
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

    /// The prefill call sites' construction self-checks against the stock chain: once, before the
    /// served lane is used, on the model's own layer weights; a route that does not pass refuses the
    /// Module by name (there is no stock branch inside an installed route).
    fn checkPrefillRoutes(self: *Module) !void {
        const Tr = graph.Trunk(G);
        const c = &self.model.c;
        const n_scratch = @max(@as(usize, 64) * c.n_heads * c.head_dim, @as(usize, 64) * c.n_experts_per_tok * c.hidden_size, @as(usize, 64) * c.hc_mult * c.hidden_size);
        const scratch = try self.gpa.alloc(f32, n_scratch);
        defer self.gpa.free(scratch);
        const m = self.g.mark();
        defer self.g.resetTo(m);
        var checks: [40]Tr.RouteCheck = undefined;
        var n = try Tr.prefillRoutesCheck(&self.g, c, &self.model.tier.routes, &self.model.kx, self.model.layers, scratch, &checks);
        n += try Tr.decodeRoutesCheck(&self.g, c, &self.model.kx, self.model.layers, scratch, checks[n..]);
        // C29's Engram wkv (the model's route): its first slot against the stock qmm at 5 rows.
        if (self.model.engram_m1[0]) |*s| {
            const en = self.model.engram.?;
            const K = self.g.shapeOf(en.w[0].wkv.w).dim(1) * 4;
            var rng = std.Random.DefaultPrng.init(0x5eed_d544);
            const need: usize = @intCast(5 * K);
            if (scratch.len < need) return error.PrefillCheckScratch;
            for (scratch[0..need]) |*v| v.* = (rng.random().float(f32) * 2 - 1);
            const x = try self.g.astype(try self.g.hostArray(std.mem.sliceAsBytes(scratch[0..need]), &.{ 5, K }, .float32), .bfloat16);
            checks[n] = .{ .name = "engram_wkv", .ok = try Tr.checkCloseOf(&self.g, try s.call(&self.g, x), try Tr.qlinear(&self.g, x, en.w[0].wkv), 2e-2) };
            n += 1;
        }
        for (checks[0..n]) |ck| {
            var b: [1]bool = undefined;
            _ = try self.g.hostBool(ck.ok, &b);
            if (!b[0]) {
                log.err("NATIVE prefill route {s}: the construction self-check against the stock chain failed", .{ck.name});
                return error.PrefillRouteSelfCheck;
            }
        }
        log.info("NATIVE prefill routes: {d} construction self-checks against the stock chain passed", .{n});
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
        try self.gate.request();
        self.dropDspark();
        if (self.state) |*st| st.deinit(&self.g, self.gpa);
        self.state = null;
        self.state = try self.model.newStateWith(self.model.boundedKv(maxPositions(ids.len, reserved_tokens)));
        self.prompt_stats0 = self.streamStats();
        self.prompt_tokens = ids.len;
        if (self.dspark_cfg) |cfg| return self.prefillSeeded(ids, cfg);
        return self.forward(ids);
    }

    /// ENGRAM=prefetch's construction check: a fixed span's rows hashed, every Engram slot's posted gather
    /// against a read past the cache, bitwise (`eng.RowSource.checkPosted`).
    fn checkEngramPosted(self: *Module, gpa: std.mem.Allocator) !void {
        const n = 64;
        var ids: [n]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast((i * 7919 + 13) % self.model.c.vocab_size);
        var st: eng.HashState = .{};
        defer st.deinit(gpa);
        const rows = try gpa.alloc(i64, n * self.engram.perToken());
        defer gpa.free(rows);
        try self.engram.advance(gpa, &st, &ids, rows);
        try self.engram.checkPosted(gpa, rows, n);
    }

    /// A few ids' rows through the resident table and through the host rows, compared bitwise.
    fn checkEmbeddingRows(self: *Module, gpa: std.mem.Allocator) !void {
        const g = &self.g;
        const vocab: u32 = self.model.c.vocab_size;
        const dim: u32 = self.model.c.hidden_size;
        const ids = [_]u32{ 0, 1, 7, vocab / 2, vocab - 1 };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const table = switch (self.model.embed) {
            .table => |w| w,
            .rows => return error.EmbeddingRetired,
        };
        const m = g.mark();
        defer g.resetTo(m);
        const from_table = try g.astype(try (M.Embed{ .table = table }).of(g, a, &ids, dim), .float32);
        const from_rows = try g.astype(try (M.Embed{ .rows = &self.embed_rows }).of(g, a, &ids, dim), .float32);
        try g.evalAll(&.{ from_table, from_rows });
        const t = try a.alloc(f32, ids.len * dim);
        const r = try a.alloc(f32, ids.len * dim);
        _ = try g.hostF32(from_table, t);
        _ = try g.hostF32(from_rows, r);
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(t), std.mem.sliceAsBytes(r))) {
            log.err("embedding rows: the host rows differ from the resident table", .{});
            return error.EmbeddingRowsDiffer;
        }
    }

    /// The prompt pass with the DSpark seed (`Loop.prefill`'s: the main taps of every prompt row seed
    /// the draft head, the lookup takes the prompt); the last row's logits, as `forward`'s.
    fn prefillSeeded(self: *Module, ids: []const u32, cfg: dsl.Config) !mlx.mlx_array {
        const caches = try self.gpa.alloc(H.Cache, self.head.nStages());
        for (caches) |*x| x.* = .{};
        self.dspark = .{ .lp = dsl.Loop(G).init(&self.g, self.model, self.head, &self.state.?, caches, cfg), .caches = caches };
        errdefer self.dropDspark();
        switch (self.arm) {
            inline else => |t| return self.dspark.?.lp.prefillLogits(self.gpa, &t.arm.hook, ids),
        }
    }

    fn dropDspark(self: *Module) void {
        if (self.dspark) |*d| {
            d.lp.deinit();
            for (d.caches) |*x| x.deinit(&self.g);
            self.gpa.free(d.caches);
        }
        self.dspark = null;
    }

    /// The draft block this Module serves (the shell's readiness signal): the head's depth under the
    /// strategy's settings, 0 when it decodes serially.
    pub fn draftBlockSize(self: *const Module) u32 {
        const cfg = self.dspark_cfg orelse return 0;
        return dsl.Loop(G).shapesOf(self.head, cfg).k_cap;
    }

    /// The decode lane as installed (the server log and the receipts stamp it).
    pub fn decodeLane(self: *const Module) []const u8 {
        return if (self.dspark_cfg != null) dspark_lane else "serial";
    }

    /// The request's committed length (the Generator mirrors its cache step from it).
    pub fn position(self: *const Module) u64 {
        return if (self.state) |st| st.offset else 0;
    }

    /// The strategy's counters over the request (null: no strategy seeded).
    pub fn dsparkStats(self: *const Module) ?@import("deepseek_v41_dspark.zig").Stats {
        return if (self.dspark) |d| d.lp.stats else null;
    }

    /// One DSpark round at the shell's v2 spec invariant (the state holds the prompt and every
    /// emitted token; `t1`, the next token, is not in it): the head drafts, [t1, drafts] verifies at
    /// the draft rows, typical acceptance (the tier's delta) with the greedy correction decides, the
    /// target keeps [t1, at most `accepted_cap` accepted drafts] (the rejected rows trimmed: the KV
    /// rollback) and the draft windows take them. Returns [t1, the kept drafts] (owned by `a`) and the
    /// next token (the correction; not in the state). It never stops: EOS, stop strings and the token
    /// budget are the caller's. The phase change runs before the first verify. Without a strategy (a
    /// Module that decodes serially) it serves one serial step instead: [t1], its argmax next.
    pub fn dsparkRound(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32) !DsparkRound {
        return self.dsparkRoundLogged(a, t1, accepted_cap, null, {});
    }

    /// `dsparkRound` with the loop's cycle log and a stamper (the cell's receipts; `{}` compiles them out).
    pub fn dsparkRoundLogged(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, cycle_log: ?*dsl.CycleLog, stamp: anytype) !DsparkRound {
        self.reportPrompt();
        if (!self.grown()) try self.phaseChange();
        const d: *Dspark = if (self.dspark) |*x| x else {
            const logits = try self.forward(&.{t1});
            defer _ = mlx.mlx_array_free(logits);
            const next = try self.g.hostArgmax(logits);
            const tokens = try a.alloc(u32, 1);
            tokens[0] = t1;
            return .{ .tokens = tokens, .accepted = 0, .next_token = next };
        };
        switch (self.arm) {
            inline else => |t| {
                const r = try d.lp.round(&t.arm.hook, a, t1, accepted_cap, cycle_log, stamp);
                return .{ .tokens = r.tokens, .accepted = r.accepted, .next_token = r.next_token };
            },
        }
    }

    fn streamStats(self: *Module) expert_stream.Stats {
        return switch (self.arm) {
            inline else => |t| t.arm.stream.stats(),
        };
    }

    /// The prompt pass's reads from the stream's own counters, once per request (end of the phase).
    fn reportPrompt(self: *Module) void {
        const s0 = self.prompt_stats0 orelse return;
        self.prompt_stats0 = null;
        const s1 = self.streamStats();
        log.info("NATIVE prefill stream: {d} prompt tokens, read {d} B in {d} preadv, {d} misses, {d} routes", .{
            self.prompt_tokens, s1.expert_bytes_read - s0.expert_bytes_read, s1.preadv_calls - s0.preadv_calls, s1.expert_cache_misses - s0.expert_cache_misses, s1.route_calls - s0.route_calls,
        });
    }

    /// Positions a request's bounded lanes hold: its reservation (else the prompt plus the shell's
    /// generation headroom), plus one verify block.
    pub fn maxPositions(prompt: usize, reserved_tokens: u64) u32 {
        const budget: u64 = if (reserved_tokens > prompt) reserved_tokens else prompt + generation_headroom;
        return @intCast(budget + mdl.Model(G).scratch_rows);
    }

    /// Later positions of the request: a decode-width forward runs the phase change first, once.
    pub fn extend(self: *Module, ids: []const u32) !mlx.mlx_array {
        try self.gate.request();
        self.reportPrompt();
        if (phaseChangeDue(ids.len, self.grown())) try self.phaseChange();
        // With a strategy the serial rows keep it in step (their main taps into the draft windows,
        // the lookup): the shell's prompt (prefill of all but the last token, then this) seeds as the
        // whole prompt does, and a serial step mid-request leaves the next round valid.
        if (self.dspark) |*d| switch (self.arm) {
            inline else => |t| return d.lp.extendLogits(self.gpa, &t.arm.hook, ids),
        };
        return self.forward(ids);
    }

    /// The reading `reclaimShrink` judges against, taken before the caller frees the grown rows: every
    /// command of the previous request retired first, so nothing it still held is missing from it.
    pub fn boundaryBefore(self: *Module) BoundaryMemory {
        _ = mlx.mlx_synchronize(self.g.s);
        return BoundaryMemory.now();
    }

    /// The served path's return to the prompt phase before a later prompt, after the caller freed the grown
    /// rows (the arm's shrink; `before` read just before it): the MLX cache cleared (the freed rows' buffers
    /// go back to the driver, not into the cache), synchronize, then the same settle and one check as the
    /// phase change on this process's own ledgers (the footprint down by the freed bytes, MLX's cache empty
    /// and active down); a refusal is a typed error and every later request is refused by name. Bill: every
    /// prompt pass then runs at the prompt rows, so max(prompt, decode) holds for every request.
    pub fn reclaimShrink(self: *Module, before: BoundaryMemory, freed_bytes: u64) !void {
        try self.gate.request();
        // Every command that still held the freed rows retires first (its completion handlers hand them to
        // the allocator's cache before the clear, as at the phase change).
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        _ = mlx.mlx_synchronize(self.g.s);
        const st = settle(LiveReader{ .io = self.io }, before, freed_bytes);
        self.phase_change = .{ .kind = "shrink", .before = before, .after = st.after, .freed_bytes = before.cache + freed_bytes, .settle_ms = st.waited_ms };
        checkFreed(before, st.after, freed_bytes) catch |e| return self.refuseBoundary(e);
        self.logPhaseChange();
    }

    /// The phase change, once (a no-op after): the prompt's frees, proven reclaimed, then the grow.
    /// 1. Every GPU command of the prompt retires (synchronize): MLX's completion handlers hand the buffers
    ///    they held back to its allocator, and Metal keeps a released buffer's pages until its command
    ///    buffers complete (v6b: a clear before the handlers ran left 4.1 GB in the footprint into decode).
    /// 2. The frees: the device embedding if it is still there, MLX's buffer cache cleared, the decode cache
    ///    limit set, synchronize.
    /// 3. The settle (read every `phase_change_poll_ms`, at most `phase_change_settle_ms`) and ONE check on
    ///    this process's own ledgers: MLX's cache empty, MLX active and the footprint down by the freed bytes.
    ///    The whole box's pages (the guard's metric, other processes included) are the harness's and the
    ///    guard's to judge: the record carries them.
    /// 4. The grow to the decode rows (the bill admitted both phases at construction).
    /// A refusal is a typed error (upstream's slotFailure reports it; the server stays up) and the Module
    /// refuses every later request by name: no retry grows over what the refused check saw.
    pub fn phaseChange(self: *Module) !void {
        try self.gate.request();
        if (self.grown()) return;
        var marks: [4]VmMark = undefined;
        marks[0] = VmMark.now();
        _ = mlx.mlx_synchronize(self.g.s);
        const before = BoundaryMemory.now();
        var freed_device: u64 = 0;
        if (!self.fenced) {
            freed_device = self.model.embeddingBytes();
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            self.fenced = true;
        }
        marks[1] = VmMark.now();
        self.g.clearCache();
        setCacheLimit(envelope.decode_cache_bytes);
        _ = mlx.mlx_synchronize(self.g.s);
        const st = settle(LiveReader{ .io = self.io }, before, freed_device);
        marks[2] = VmMark.now();
        self.phase_change = .{ .before = before, .after = st.after, .freed_bytes = before.cache + freed_device, .settle_ms = st.waited_ms };
        checkFreed(before, st.after, freed_device) catch |e| return self.refuseBoundary(e);
        switch (self.arm) {
            inline else => |t| try t.arm.grow(&self.g),
        }
        marks[3] = VmMark.now();
        self.phase_change.?.grown = BoundaryMemory.now();
        self.logPhaseChange();
        for (marks, [_][]const u8{ "start", "after the embedding fence", "after the frees (settled)", "after the banks grew" }) |m, name|
            log.info("NATIVE phase change {s}: physical used {d} B, footprint {d} B, outside the footprint {d} B (purgeable {d}, file-backed {d})", .{ name, m.physical, m.footprint, m.physical -| m.footprint, m.purgeable, m.external });
    }

    /// One `NATIVE DSV41_PHASE_CHANGE {json}` line of the record (success or refusal): the server log carries
    /// the settle time too.
    fn logPhaseChange(self: *Module) void {
        const r = self.phase_change orelse return;
        const json = std.json.Stringify.valueAlloc(self.gpa, r, .{}) catch return;
        defer self.gpa.free(json);
        log.info("NATIVE DSV41_PHASE_CHANGE {s}", .{json});
    }

    /// A refused boundary: recorded in the gate (every later request refused by name, PhaseChangeRefused),
    /// logged with its readings, and returned as its typed error, which upstream's slotFailure reports for
    /// the request while the server stays up; a harness fails its run on it.
    fn refuseBoundary(self: *Module, e: anyerror) anyerror {
        self.gate.refuse(e);
        if (self.phase_change) |*r| r.refused = @errorName(e);
        self.logPhaseChange();
        log.err("NATIVE {s} refused: {s}; every later request is refused by name", .{ if (self.phase_change) |r| r.kind else "phase change", @errorName(e) });
        return e;
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

/// The load preflight's requirement (`transformer.archLoadRequirementBytes`): the native bill at the fill's
/// floor rows (`deepseek_v41_bill.loadRequirementBytes`).
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, config: *const model_io.ModelConfig) !u64 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    return bill_mod.loadRequirementBytes(arena.allocator(), io, config.*);
}

/// Whether `v41.PrefillBill` bills the layer-major pass (`layerMajorBytes`: one chunk's attention
/// side and the routed group's sub-wave per layer, the server's per-request admission reads it).
pub const layer_major_billed = true;

/// The `layer_major_prefill` setting, checked before anything is built. K16 batches each layer's
/// routed call across chunks (the wide lane): the stock tier's prompt forwards are decode-width.
pub fn layerMajor(config: *const model_io.ModelConfig) error{ LayerMajorOnStockTier, LayerMajorNotBilled }!bool {
    if (!config.dsv41LayerMajor()) return false;
    if ((config.numeric_tier orelse .served) == .stock) return error.LayerMajorOnStockTier;
    if (!layer_major_billed) return error.LayerMajorNotBilled;
    return true;
}

/// One reading of the box's pages (the guard's physical-used metric) beside this process's footprint.
const VmMark = struct {
    physical: u64,
    footprint: u64,
    purgeable: u64,
    external: u64,

    fn now() VmMark {
        const v = status.vmBytes();
        return .{ .physical = status.physicalUsedBytes(v), .footprint = status.footprint().now, .purgeable = v.purgeable, .external = v.external };
    }
};

/// The prefill routes a module installed (read back from the trunk's tier and the arm's hook and stream).
pub const Installed = struct {
    layer_major: bool = false,
    wide: xp.Wide = .{},
    stream_windows: u8 = 1,
    /// The input embedding reads its host rows from construction (no device table in either phase).
    embedding_rows: bool = false,
    /// JOINLESS reads the DIG-X waves' own outputs (no per-call concatenate + take).
    prefill_unjoined: bool = false,
    /// The prefill attention core (installed and past its construction self-check).
    prefill_attn: bool = false,
    /// The prefill indexer (installed).
    prefill_index: bool = false,
    /// The prefill HC norms, the SMALLK combine, the DENSE16 o-projection (installed).
    prefill_hc: bool = false,
    prefill_combine: bool = false,
    prefill_oproj: bool = false,
    /// PREFILL_HOST shared and JOINLESS (K16's routed group; installed).
    prefill_host_shared: bool = false,
    prefill_joinless: bool = false,
    /// ENGRAM=prefetch: the prompt pass's Engram gathers posted ahead (started and past its self-check).
    engram_posted: bool = false,
    /// The verify-row routes (C23 softmax, C27 select, C28 smallm, C29 mxfp8 rows; installed).
    decode_attn_softmax: bool = false,
    decode_index_topk: bool = false,
    decode_smallm: bool = false,
    decode_mxfp8_rows: bool = false,

    /// The attention call sites' construction line (apart from the ladder routes' line).
    /// The verify-row routes' construction line.
    pub fn decodeSites(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE decode sites installed: softmax {}, select {}, smallm {}, mxfp8 rows {}", .{ self.decode_attn_softmax, self.decode_index_topk, self.decode_smallm, self.decode_mxfp8_rows }) catch buf[0..0];
    }

    pub fn callSites(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE prefill call sites installed: attention core {}, indexer {}, hc norms {}, combine {}, o-projection {}, host shared {}, joinless {}, embedding rows {}, unjoined waves {}, engram posted {}, deferred base calls {}", .{ self.prefill_attn, self.prefill_index, self.prefill_hc, self.prefill_combine, self.prefill_oproj, self.prefill_host_shared, self.prefill_joinless, self.embedding_rows, self.prefill_unjoined, self.engram_posted, self.wide.defer_base }) catch buf[0..0];
    }

    /// The construction log line the gates assert.
    pub fn line(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE prefill routes installed: prefill layer-major {}, wide feed {}, wide depth {d}, stream windows {d}, cold rows {d}, seed {}, hot-first {}", .{
            self.layer_major, self.wide.seed and self.wide.hot_first, self.wide.depth, self.stream_windows, self.wide.cold_rows, self.wide.seed, self.wide.hot_first,
        }) catch buf[0..0];
    }
};

/// The prefill indexer route as the Module builds it (the setting, else the tier's route); the bill
/// reads the same answer. It needs K30's selected keys.
pub fn prefillIndexRoute(config: *const model_io.ModelConfig) !bool {
    const t = numericTier(config.numeric_tier orelse .served);
    const on = config.prefill_index orelse t.routes.prefill_index;
    if (on and !t.routes.selected_keys) return error.PrefillIndexNeedsSelectedKeys;
    return on;
}

/// The wide prefill calls' read schedule from the model settings (the tier's default when unset).
pub fn wideRoute(config: *const model_io.ModelConfig) xp.Wide {
    return .{ .seed = config.dsv41WideSeed(), .hot_first = config.dsv41WideHotFirst(), .depth = config.dsv41WideDepth(), .cold_rows = config.expert_wide_cold_rows orelse 0, .defer_base = config.dsv41WideDeferBase() };
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

/// How far the constructed footprint may sit above the bill's construction terms (the ledger's
/// residual threshold; cell4 measured 0.59 GB UNDER them).
pub const construction_tolerance_bytes: u64 = 250_000_000;

/// MLX's allocator, the process footprint and the box's physical pages (vm_stat's wired + active + inactive
/// + compressor: the guard's own metric, the source it samples) at the phase boundary: existing counters only.
pub const BoundaryMemory = struct {
    active: u64,
    cache: u64,
    footprint: u64,
    physical: u64,

    pub fn now() BoundaryMemory {
        var active: usize = 0;
        var cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        return .{ .active = active, .cache = cache, .footprint = status.footprint().now, .physical = status.physicalUsedBytes(status.vmBytes()) };
    }
};

/// The phase change's record: the readings before the frees (after the prompt's commands retired), after
/// them (settled), after the grow; the bytes freed (the MLX cache cleared + device bytes released); the
/// reclaim time the driver took (the receipts carry it, so the windows learn its behaviour).
pub const PhaseChangeRecord = struct {
    /// "phase change" (the grow after a prompt) or "shrink" (the return to the prompt rows before a later one).
    kind: []const u8 = "phase change",
    before: BoundaryMemory,
    after: BoundaryMemory,
    grown: ?BoundaryMemory = null,
    freed_bytes: u64,
    settle_ms: u32,
    /// The refusal's name, when the phase change refused the grow.
    refused: ?[]const u8 = null,
};

/// The phase change's gate: a refused boundary is kept (every later request refused by name), so no retry
/// grows over what the refused check saw.
pub const PhaseGate = struct {
    refused: ?anyerror = null,

    pub fn request(g: *const PhaseGate) error{PhaseChangeRefused}!void {
        if (g.refused != null) return error.PhaseChangeRefused;
    }

    pub fn refuse(g: *PhaseGate, e: anyerror) void {
        g.refused = e;
    }
};

/// The box's physical pages outside this process's footprint (vm_stat's used less the footprint): recorded
/// for the harness and the guard, never judged by the served path.
pub fn outsideOf(m: BoundaryMemory) u64 {
    return m.physical -| m.footprint;
}

/// The live boundary reader: MLX's counters, the footprint, vm_stat; waits on the shell's io.
const LiveReader = struct {
    io: std.Io,

    fn now(_: LiveReader) BoundaryMemory {
        return BoundaryMemory.now();
    }

    fn sleep(self: LiveReader, ms: u32) void {
        std.Io.sleep(self.io, .fromMilliseconds(ms), .awake) catch {};
    }
};

/// The footprint may sit this far above its expected drop at the boundary (the ledger's page rounding
/// and the host side's own movement).
pub const phase_change_tolerance_bytes: u64 = 250_000_000;
/// The reclaim wait: read every `phase_change_poll_ms`, refuse after `phase_change_settle_ms`.
pub const phase_change_poll_ms: u32 = 250;
pub const phase_change_settle_ms: u32 = 10_000;

fn footprintFreed(before: BoundaryMemory, after: BoundaryMemory, freed_device: u64) bool {
    return after.footprint + before.cache + freed_device <= before.footprint + phase_change_tolerance_bytes;
}

/// After the frees: `reader` read every `phase_change_poll_ms` until this process's footprint shows them
/// (its ledger can trail a release while the driver retires it), at most `phase_change_settle_ms`; the one
/// check then judges the last reading.
pub fn settle(reader: anytype, before: BoundaryMemory, freed_device: u64) struct { after: BoundaryMemory, waited_ms: u32 } {
    var m = reader.now();
    var waited: u32 = 0;
    while (!footprintFreed(before, m, freed_device) and waited < phase_change_settle_ms) {
        reader.sleep(phase_change_poll_ms);
        waited += phase_change_poll_ms;
        m = reader.now();
    }
    return .{ .after = m, .waited_ms = waited };
}

/// The phase boundary's one check, before the grow, on this process's own ledgers: the MLX cache empty (every
/// prompt buffer released, none parked for the grow to miss), MLX active down by the freed device bytes, the
/// footprint down by the cache and those bytes; else a typed error.
pub fn checkFreed(before: BoundaryMemory, after: BoundaryMemory, freed_device: u64) error{ PhaseChangeCacheNotEmpty, PhaseChangeActiveNotFreed, PhaseChangeFootprintNotFreed }!void {
    if (after.cache != 0) return error.PhaseChangeCacheNotEmpty;
    if (after.active + freed_device > before.active) return error.PhaseChangeActiveNotFreed;
    if (!footprintFreed(before, after, freed_device)) return error.PhaseChangeFootprintNotFreed;
}

pub fn checkConstructionBytes(billed: u64, measured: u64) error{ConstructionOverBill}!void {
    if (measured > billed + construction_tolerance_bytes) return error.ConstructionOverBill;
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
        .fixed_rows = if (config.expert_prefill_rows == null) config.expert_rows else null,
        // Rows the native bill filled (both set): the stream's rows. The Module always sets both.
        .native_rows = if (config.expert_prefill_rows) |p| .{ .prefill = p, .decode = config.expert_rows orelse p } else null,
        // `expert_rows` alone (a harness's Python-paired forced-rows admission, never the served Module): the
        // envelope planner runs for its rows and its record.
        .envelope_record = config.expert_prefill_rows == null and config.expert_rows != null,
        // The banks grow at the phase change (two row counts); `phaseChange` proves the prompt's frees
        // complete before the grow, so the growth never meets unreleased buffers (SERVED7).
        .preallocate = false,
        .slot_memory = slot_memory,
        .draft_pruned_bytes = 0,
        .lookahead = lookahead,
        .ceiling = ceiling,
        .wide_depth = config.dsv41WideDepth(),
    };
}

/// The fill's shape and target (the bill module's), re-exported for the module's callers.
pub const FillBill = bill_mod.FillBill;
pub const fillRows = bill_mod.fillRows;
pub const admitPhases = bill_mod.admitPhases;
pub const fill_prompt_tokens = bill_mod.fill_prompt_tokens;
pub const fill_max_tokens = bill_mod.fill_max_tokens;
pub const min_fill_rows = bill_mod.min_fill_rows;

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

/// The allocator cache the prefill holds, which the bill charges at exactly this limit: MLX trims its cache
/// to the limit after every allocation (allocator.cpp malloc: release_cached_buffers(cache - max_pool_size_)),
/// and a free that overshoots it lowers active by as much, so at every footprint peak the cache is at most
/// the limit. The served tier holds 2 GiB, what pass3ak (v6c3) actually held at its prompt peak under 4 GiB:
/// at 1 GiB (pass3am, v7) the prompt read the same 186.0 GB in the same 14.4 s of read-busy time while TTFT
/// rose 37.64 -> 39.14 s, the allocator churning in the prompt pass. The stock tier the envelope's own.
pub fn prefillCacheLimit(t: @import("model_settings.zig").NumericTier) usize {
    return switch (t) {
        .served => 2 << 30,
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

test "dsv41 module: the served tier's prefill routes are on by default, the stock tier's off; a setting overrides; layer-major refused on stock" {
    var c: model_io.ModelConfig = undefined;
    c.layer_major_prefill = null;
    c.numeric_tier = null;
    c.expert_wide_feed = null;
    c.expert_wide_seed = null;
    c.expert_wide_hot_first = null;
    c.expert_wide_depth = null;
    c.expert_wide_cold_rows = null;
    c.expert_wide_defer_base = null;
    try std.testing.expect(try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 2, .defer_base = true }, wideRoute(&c));
    c.expert_wide_hot_first = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .depth = 2, .defer_base = true }, wideRoute(&c));
    c.expert_wide_hot_first = null;
    c.expert_wide_defer_base = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 2 }, wideRoute(&c));
    c.expert_wide_defer_base = null;
    c.numeric_tier = .stock;
    try std.testing.expect(!try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{}, wideRoute(&c));
    c.layer_major_prefill = true;
    try std.testing.expectError(error.LayerMajorOnStockTier, layerMajor(&c));
    c.numeric_tier = .served;
    c.layer_major_prefill = false;
    c.expert_wide_feed = false;
    c.expert_wide_depth = 1;
    c.expert_wide_cold_rows = 2;
    try std.testing.expect(!try layerMajor(&c));
    // Cold rows keep the per-group base calls (the deferred call is off with them).
    try std.testing.expectEqual(xp.Wide{ .cold_rows = 2 }, wideRoute(&c));
}

test "dsv41 module: the module's construction and forwards analyse (host, nothing runs)" {
    try std.testing.expect(@TypeOf(&Module.init) != void and @TypeOf(&Module.extend) != void);
    // The served path's per-request pieces are analysed with the module (their wiring is the served path's).
    const shrink_reclaim: *const fn (*Module, BoundaryMemory, u64) anyerror!void = &Module.reclaimShrink;
    const boundary_before: *const fn (*Module) BoundaryMemory = &Module.boundaryBefore;
    try std.testing.expect(@intFromPtr(boundary_before) != 0);
    try std.testing.expect(@intFromPtr(shrink_reclaim) != 0);
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
        .envelope_record = true,
        .slot_memory = .host,
        .draft_pruned_bytes = if (std.c.getenv("DSV41_MODULE_HEAD") != null) null else 0,
        .lookahead = lookahead,
        .ceiling = if (std.c.getenv("DSV41_MODULE_CEILING_GB") != null) boxCeiling(try gb("DSV41_MODULE_CEILING_GB", 0), 384) else null,
    }, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    const ad = p.plan.?.admission;
    const rec: u64 = 13_315_584;
    std.debug.print("\nDSV41_MODULE_PLAN {{\"prefill_rows\": {d}, \"decode_rows\": {d}, \"slot_bank_prefill_bytes\": {d}, \"slot_bank_decode_bytes\": {d}, \"active_bound_bytes\": {d}, \"physical_bound_bytes\": {d}, \"host_reserve_bytes\": {d}, \"baseline_bytes\": {d}}}\n", .{
        p.prefill_rows, p.decode_rows, (40 * @as(u64, p.prefill_rows) + 48) * rec, (40 * @as(u64, p.decode_rows) + 48) * rec, ad.active_bound_bytes, ad.physical_bound_bytes, ad.host_reserve_bytes, p.inputs.baseline_bytes,
    });
    const box: u64 = if (p.inputs.ceiling) |cl| cl.box_bytes else expert_admission.box_ceiling_bytes;
    const modeled = if (p.plan.?.peak_fill) |pf| pf.modeled_peak_bytes else ad.physical_bound_bytes;
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

test "dsv41 memory: the phase boundary refuses a grow over unreleased buffers, by name (this process's ledgers)" {
    const gb: u64 = 1_000_000_000;
    const emb: u64 = 1_323_827_200;
    // v6b's prompt end after the synchronize: active 85.36, cache 4.63, footprint 91.92 GB.
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000, .physical = 105_848_000_000 };
    const freed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache, .physical = before.physical - before.cache };
    // Released: cache empty, footprint down by the cache.
    try checkFreed(before, freed, 0);
    // The embedding freed at the boundary: active and footprint down by it too.
    try checkFreed(before, .{ .active = before.active - emb, .cache = 0, .footprint = freed.footprint - emb, .physical = freed.physical - emb }, emb);
    // Cache bytes left: refused.
    var left = freed;
    left.cache = 16384;
    try std.testing.expectError(error.PhaseChangeCacheNotEmpty, checkFreed(before, left, 0));
    // v6b as it ran (the cache cleared before the handlers returned their buffers): the footprint 91.07 GB,
    // 4.07 GB above the drop: refused.
    var v6b = freed;
    v6b.footprint = 91_065_000_000;
    try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, v6b, 0));
    // The box's pages are not the served path's to judge: other processes' growth never refuses the grow.
    var busy_box = freed;
    busy_box.physical = before.physical + 5 * gb;
    try checkFreed(before, busy_box, 0);
    // Active not down by the embedding: refused.
    try std.testing.expectError(error.PhaseChangeActiveNotFreed, checkFreed(before, .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache - 2 * gb, .physical = before.physical - before.cache - 2 * gb }, emb));
}

/// A scripted boundary reader: `readings[i]` at the i-th read (the last one repeats), no real sleep.
const FakeReader = struct {
    readings: []const BoundaryMemory,
    i: *usize,
    slept_ms: *u32,

    fn now(self: FakeReader) BoundaryMemory {
        const r = self.readings[@min(self.i.*, self.readings.len - 1)];
        self.i.* += 1;
        return r;
    }

    fn sleep(self: FakeReader, ms: u32) void {
        self.slept_ms.* += ms;
    }
};

test "dsv41 memory: the settle waits for the footprint to show the frees, then the one check judges the last reading" {
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000, .physical = 105_848_000_000 };
    const lagging: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint, .physical = before.physical };
    const freed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache, .physical = before.physical - before.cache };
    // The footprint never shows the frees: the full wait, then refused by name.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{lagging}, .i = &i, .slept_ms = &slept }, before, 0);
        try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
        try std.testing.expectEqual(phase_change_settle_ms, slept);
        try std.testing.expectEqual(@as(usize, phase_change_settle_ms / phase_change_poll_ms + 1), i);
        try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, st.after, 0));
    }
    // The frees show on the third reading: two polls (500 ms), then the check passes.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ lagging, lagging, freed }, .i = &i, .slept_ms = &slept }, before, 0);
        try std.testing.expectEqual(@as(u32, 2 * phase_change_poll_ms), st.waited_ms);
        try checkFreed(before, st.after, 0);
    }
    // Already freed at the first reading: no wait.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{freed}, .i = &i, .slept_ms = &slept }, before, 0);
        try std.testing.expectEqual(@as(u32, 0), st.waited_ms);
        try checkFreed(before, st.after, 0);
    }
}

test "dsv41 memory: a refused boundary refuses every later request by name (no retry grows over it)" {
    var g: PhaseGate = .{};
    try g.request();
    g.refuse(error.PhaseChangeFootprintNotFreed);
    try std.testing.expectError(error.PhaseChangeRefused, g.request());
    try std.testing.expectError(error.PhaseChangeRefused, g.request());
}

test "dsv41 memory: the return to the prompt rows (shrink) is judged like the phase change" {
    // v6b-scale decode state: the grown rows (31 x 40 x 13.3 MB = 16.5 GB) resident, a 0.27 GB decode cache.
    const grown_rows: u64 = 31 * 40 * 13_315_584;
    const before: BoundaryMemory = .{ .active = 101_000_000_000, .cache = 268_000_000, .footprint = 102_900_000_000, .physical = 116_800_000_000 };
    // The rows and the cache released: passes.
    const freed: BoundaryMemory = .{ .active = before.active - grown_rows, .cache = 0, .footprint = before.footprint - before.cache - grown_rows, .physical = before.physical - before.cache - grown_rows };
    try checkFreed(before, freed, grown_rows);
    // The rows' buffers parked in MLX's cache instead of released: refused (the cache is cleared first by design).
    var parked = freed;
    parked.cache = grown_rows;
    try std.testing.expectError(error.PhaseChangeCacheNotEmpty, checkFreed(before, parked, grown_rows));
    // Released by MLX but still in this footprint: waited out, then refused if it never clears.
    var lagging = freed;
    lagging.footprint = before.footprint;
    var i: usize = 0;
    var slept: u32 = 0;
    const st = settle(FakeReader{ .readings = &.{lagging}, .i = &i, .slept_ms = &slept }, before, grown_rows);
    try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
    try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, st.after, grown_rows));
}

test "dsv41 memory: the construction check passes the constructed footprints of record and refuses one over its bill by name" {
    // cell4 (106 / 148 rows): constructed footprint 76.41 GB against 77.00 GB of construction terms.
    try checkConstructionBytes(77_000_000_000, 76_410_000_000);
    // At the tolerance: passes; one byte over: refused.
    try checkConstructionBytes(77_000_000_000, 77_000_000_000 + construction_tolerance_bytes);
    try std.testing.expectError(error.ConstructionOverBill, checkConstructionBytes(77_000_000_000, 77_000_000_001 + construction_tolerance_bytes));
    try std.testing.expect(construction_tolerance_bytes < ceiling_stop_bytes);
}

test "dsv41 module: the envelope admission (the old rule) admits today's 154 decode rows at today's inputs" {
    const f = bill_mod.fill_fixture;
    const ceiling = boxCeiling(f.ceiling, 384);
    const in: expert_admission.Inputs = .{
        .baseline_bytes = f.baseline,
        .wired_bytes = 3_389_000_000,
        .record_bytes = f.record,
        .phase_reserve_bytes = arm_mod.pass2_phase_reserve_bytes,
        .lookahead_staging_bytes = expert_admission.lookaheadCharge(f.record, 2 * lookahead.budget, 16384),
        .host_reserve_bytes = arm_mod.pass2_host_reserve_bytes,
        .prefill_charge_bytes = arm_mod.pass2_prefill_charge_bytes,
        .ceiling = ceiling,
        .draft_pruned_bytes = 0,
    };
    const p = try expert_admission.Admission.plan(envelope, in);
    std.debug.print("old rule: {d} prefill capacity / {d} decode rows\n", .{ p.admission.prefill_capacity, p.admission.decode_rows });
    try std.testing.expectEqual(@as(u32, 154), p.admission.decode_rows);
    try std.testing.expectEqual(@as(u32, 112), p.admission.prefill_capacity);
}

test "dsv41 module: the installed-routes line reads the routes as built, on and off (the gates' assert can fail)" {
    var buf: [192]u8 = undefined;
    const on: Installed = .{ .layer_major = true, .wide = .{ .seed = true, .hot_first = true, .depth = 2 }, .stream_windows = 2 };
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major true, wide feed true, wide depth 2, stream windows 2, cold rows 0, seed true, hot-first true", on.line(&buf));
    const seed_only: Installed = .{ .layer_major = true, .wide = .{ .seed = true, .depth = 2 }, .stream_windows = 2 };
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major true, wide feed false, wide depth 2, stream windows 2, cold rows 0, seed true, hot-first false", seed_only.line(&buf));
    const off: Installed = .{};
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major false, wide feed false, wide depth 1, stream windows 1, cold rows 0, seed false, hot-first false", off.line(&buf));
}
