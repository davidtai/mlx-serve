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
const ar_bill = @import("deepseek_v41_ar.zig");
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
    /// The native bill at the admitted rows (set by the construction check; the harnesses' phase records read it).
    bill: ar_bill.CellBill = undefined,
    /// MLX's allocator cache limit before the module set its own (restored at deinit).
    prev_cache_limit: usize = 0,
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
        var diag: arm_mod.Diag = .{};
        var vd0: v41.Diag = .{};
        const c0 = v41.Config.load(gpa, io, dir, &vd0) catch |e| {
            log.err("config refused: {s}", .{vd0.message()});
            return e;
        };
        try self.acceptKernels(gpa, &c0, s, &diag);
        // The box the admission fits: the configured ceiling, else the GPU's working set (the wired limit).
        const ceiling_bytes = config.memory_ceiling_bytes orelse mlx.maxRecommendedWorkingSet();
        const ceiling = boxCeiling(ceiling_bytes, c0.n_routed_experts);
        // The served admission: rows filled by the native bill (the standard request's) up to the stop's
        // target, unless the shell forced them.
        var admitted = config.*;
        if (admitted.expert_rows == null and admitted.expert_prefill_rows == null and admitted.memory_baseline_bytes != null) {
            admitted.memory_ceiling_bytes = ceiling_bytes;
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const nr = try ar_bill.fillAt(arena.allocator(), io, admitted, fill_prompt_tokens, fill_max_tokens);
            admitted.expert_rows = nr.decode;
            admitted.expert_prefill_rows = nr.prefill;
            log.info("admission: native fill {d} prefill / {d} decode rows per layer (the {d}-token request's bill, baseline {d} B, target {d} B)", .{ nr.prefill, nr.decode, fill_prompt_tokens, admitted.memory_baseline_bytes.?, ceiling_bytes -| ceiling_stop_bytes });
        }
        errdefer self.dropKernels();
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
        if (config.prefill_oproj) |v| {
            if (v and !tier.routes.prefill_attn) return error.PrefillOprojNeedsPrefillAttn;
            tier.routes.prefill_oproj = v;
        }
        tier.layer_major = layer_major;
        log.info("numeric tier: {t}", .{config.numeric_tier orelse .served});
        self.model = try M.initWith(gpa, &self.g, c, tier, weights, &self.engram, .{ .registry = &self.set.reg });
        errdefer self.model.deinit(&self.g);
        if (tier.routes.prefill_attn or tier.routes.prefill_index or tier.routes.prefill_hc or tier.routes.prefill_combine or tier.routes.prefill_oproj or tier.routes.prefill_joinless) try self.checkPrefillRoutes();
        self.installed = switch (self.arm) {
            inline else => |t| .{ .prefill_unjoined = self.model.tier.routes.prefill_joinless and comptime (@hasDecl(@TypeOf(t.arm.hook).Math, "has_parts") and @TypeOf(t.arm.hook).Math.has_parts), .layer_major = self.model.tier.layer_major, .wide = t.arm.hook.wide_route, .stream_windows = t.arm.stream.wide_depth, .prefill_attn = self.model.tier.routes.prefill_attn, .prefill_index = self.model.tier.routes.prefill_index, .prefill_hc = self.model.tier.routes.prefill_hc, .prefill_combine = self.model.tier.routes.prefill_combine, .prefill_oproj = self.model.tier.routes.prefill_oproj, .prefill_host_shared = self.model.tier.routes.prefill_host_shared, .prefill_joinless = self.model.tier.routes.prefill_joinless },
        };
        var line_buf: [192]u8 = undefined;
        log.info("{s}", .{self.installed.line(&line_buf)});
        log.info("{s}", .{self.installed.callSites(&line_buf)});
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
        // The input embedding moves to its host rows now, not at the phase change: every lookup (the
        // prompt's included) reads the table's rows past the page cache, and the device table is gone
        // from both phases. Checked once: the rows equal the table's, byte for byte.
        if (config.embedding_host_rows orelse true) {
            try self.checkEmbeddingRows(gpa);
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
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
        const b = try ar_bill.cellBill(arena.allocator(), io, &cfg, fill_prompt_tokens, fill_max_tokens);
        const rows = switch (self.arm) {
            inline else => |t| arm_mod.NativeRows{ .prefill = t.arm.prefill_rows[0], .decode = t.arm.decode_rows[0] },
        };
        if (b.prefill_rows != rows.prefill or b.decode_rows != rows.decode) {
            log.err("construction check: the bill plans {d} / {d} rows, the arm built {d} / {d}", .{ b.prefill_rows, b.decode_rows, rows.prefill, rows.decode });
            return error.BillRowsMismatch;
        }
        self.bill = b;
        const measured = arm_mod.footprint().now;
        const billed = b.constructionTerms().sum();
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
        var checks: [16]Tr.RouteCheck = undefined;
        const n = try Tr.prefillRoutesCheck(&self.g, c, &self.model.tier.routes, &self.model.kx, self.model.layers, scratch, &checks);
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
        if (self.state) |*st| st.deinit(&self.g, self.gpa);
        self.state = null;
        self.state = try self.model.newStateWith(self.model.boundedKv(maxPositions(ids.len, reserved_tokens)));
        self.prompt_stats0 = self.streamStats();
        self.prompt_tokens = ids.len;
        return self.forward(ids);
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
        self.reportPrompt();
        if (phaseChangeDue(ids.len, self.grown())) try self.phaseChange();
        return self.forward(ids);
    }

    /// The prompt fence and the grown slot banks, once (a no-op after): the embedding to its host
    /// rows, the prefill's parked buffers back, the decode cache charge, the banks at the decode rows.
    pub fn phaseChange(self: *Module) !void {
        if (self.grown()) return;
        // The box's pages and this process's footprint at each step (once per process): what the
        // guard's metric holds beyond the footprint at the frees (the banks are preallocated: grow
        // only flips the phase).
        var marks: [4]VmMark = undefined;
        marks[0] = VmMark.now();
        if (!self.fenced) {
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            self.fenced = true;
        }
        marks[1] = VmMark.now();
        // The prefill's parked buffers go back before the slot banks grow; decode keeps its own charge.
        self.g.clearCache();
        setCacheLimit(envelope.decode_cache_bytes);
        marks[2] = VmMark.now();
        switch (self.arm) {
            inline else => |t| try t.arm.grow(&self.g),
        }
        marks[3] = VmMark.now();
        for (marks, [_][]const u8{ "start", "after the embedding fence", "after the cache release", "after the banks grew" }) |m, name|
            log.info("NATIVE phase change {s}: physical used {d} B, footprint {d} B, outside the footprint {d} B (purgeable {d}, file-backed {d})", .{ name, m.physical, m.footprint, m.physical -| m.footprint, m.purgeable, m.external });
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
        const v = arm_mod.vmBytes();
        return .{ .physical = arm_mod.physicalUsed(v), .footprint = arm_mod.footprint().now, .purgeable = v.purgeable, .external = v.external };
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

    /// The attention call sites' construction line (apart from the ladder routes' line).
    pub fn callSites(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE prefill call sites installed: attention core {}, indexer {}, hc norms {}, combine {}, o-projection {}, host shared {}, joinless {}, embedding rows {}, unjoined waves {}", .{ self.prefill_attn, self.prefill_index, self.prefill_hc, self.prefill_combine, self.prefill_oproj, self.prefill_host_shared, self.prefill_joinless, self.embedding_rows, self.prefill_unjoined }) catch buf[0..0];
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
    return .{ .seed = config.dsv41WideSeed(), .hot_first = config.dsv41WideHotFirst(), .depth = config.dsv41WideDepth(), .cold_rows = config.expert_wide_cold_rows orelse 0 };
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
        // Rows the caller's native bill filled (both set): the stream's rows, the envelope's record only.
        .native_rows = if (config.expert_prefill_rows) |p| .{ .prefill = p, .decode = config.expert_rows orelse p } else null,
        // No growth transient: the banks at their decode rows from construction (one row count).
        .preallocate = true,
        .slot_memory = slot_memory,
        .draft_pruned_bytes = 0,
        .lookahead = lookahead,
        .ceiling = ceiling,
        .wide_depth = config.dsv41WideDepth(),
    };
}

/// A native bill in the fill's shape: each phase's billed bytes (the box baseline included) without its
/// persistent slot rows, and one row on every routed layer (layers x the record); a phase's total at
/// r rows is `fixed + r * per_row`.
pub const FillBill = struct { prefill_fixed: u64, decode_fixed: u64, per_row: u64 };

/// The native admission's fill, the bill of record since SERVED7 (no grow): ONE row count for both
/// phases, so the slot banks are allocated once, at construction, and the phase change adds none
/// (`expert_stream.Stream.grow` allocates only rows above the capacity). There is no free-then-grow at
/// the phase change, so the guard's physical metric has no transition term to carry (SERVED7: 7.2 GiB
/// outside the footprint at the grow). The most rows whose billed total in BOTH phases stays within
/// `ceiling_stop_bytes` of the ceiling (<= the layer's experts); refused by name under `min_fill_rows`.
pub fn fillRows(b: FillBill, ceiling_bytes: u64, n_experts: u32) error{NativeBillDoesNotFit}!arm_mod.NativeRows {
    const target = ceiling_bytes -| ceiling_stop_bytes;
    const most = struct {
        fn f(fixed: u64, t: u64, per_row: u64) u64 {
            return if (fixed >= t) 0 else (t - fixed) / per_row;
        }
    }.f;
    const rows = @min(@min(most(b.prefill_fixed, target, b.per_row), most(b.decode_fixed, target, b.per_row)), n_experts);
    if (rows < min_fill_rows) return error.NativeBillDoesNotFit;
    return .{ .prefill = @intCast(rows), .decode = @intCast(rows) };
}

/// The request the served admission's fill bills: the standard 16K cell's prompt and token cap (a longer
/// request is admitted, or refused by name, by the server's per-request prefill bill at its time).
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;

/// The fewest rows per layer the fill admits (the envelope admission's prefill floor).
pub const min_fill_rows = 16;

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

test "dsv41 module: the served tier's prefill routes are on by default, the stock tier's off; a setting overrides; layer-major refused on stock" {
    var c: model_io.ModelConfig = undefined;
    c.layer_major_prefill = null;
    c.numeric_tier = null;
    c.expert_wide_feed = null;
    c.expert_wide_seed = null;
    c.expert_wide_hot_first = null;
    c.expert_wide_depth = null;
    c.expert_wide_cold_rows = null;
    try std.testing.expect(try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 2 }, wideRoute(&c));
    c.expert_wide_hot_first = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .depth = 2 }, wideRoute(&c));
    c.expert_wide_hot_first = null;
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
    try std.testing.expectEqual(xp.Wide{ .cold_rows = 2 }, wideRoute(&c));
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

/// The fastest cell at the full admission (served-cell-typical-fastest-20260929-172908): the guard's
/// baseline and the cell bill's phase totals at the envelope's 112 / 154 rows (decimal GB, 3 places).
const fill_fixture = struct {
    const record: u64 = 13_315_584;
    const per_row: u64 = 40 * record;
    const baseline: u64 = 13_408_305_152;
    const prefill_total: u64 = 114_365_000_000;
    const decode_total: u64 = 115_140_000_000;
    const ceiling: u64 = 120_259_000_000;

    fn at(base: u64) FillBill {
        return .{ .prefill_fixed = prefill_total - 112 * per_row - baseline + base, .decode_fixed = decode_total - 154 * per_row - baseline + base, .per_row = per_row };
    }
};

test "dsv41 memory: the native fill is one row count for both phases, at the binding phase's target within one row" {
    const f = fill_fixture;
    const target = f.ceiling - ceiling_stop_bytes;
    for ([_]u64{ 9_000_000_000, 11_000_000_000, 13_400_000_000, f.baseline }) |base| {
        const b = f.at(base);
        const r = try fillRows(b, f.ceiling, 384);
        try std.testing.expectEqual(r.prefill, r.decode);
        // Both phases fit; the binding one (the larger fixed terms) takes no row more.
        try std.testing.expect(b.decode_fixed + r.decode * b.per_row <= target and b.prefill_fixed + r.prefill * b.per_row <= target);
        try std.testing.expect(@max(b.prefill_fixed, b.decode_fixed) + (r.decode + 1) * b.per_row > target);
        std.debug.print("native fill at baseline {d:.1} GB: {d} prefill / {d} decode rows per layer (target {d:.2} GB)\n", .{ @as(f64, @floatFromInt(base)) / 1e9, r.prefill, r.decode, @as(f64, @floatFromInt(target)) / 1e9 });
    }
    // The measured baselines' rows (non-file ~9 GB, 11 GB, and the credited 13.4 GB).
    // (The two-count fill admitted 127 / 168 here and grew 41 rows at the phase change.)
    try std.testing.expectEqual(arm_mod.NativeRows{ .prefill = 127, .decode = 127 }, try fillRows(f.at(9_000_000_000), f.ceiling, 384));
    // Capped at the layer's experts; refused by name when not even the floor fits.
    const cap = try fillRows(.{ .prefill_fixed = 0, .decode_fixed = 0, .per_row = 100_000_000 }, f.ceiling, 384);
    try std.testing.expectEqual(@as(u32, 384), cap.decode);
    try std.testing.expectError(error.NativeBillDoesNotFit, fillRows(.{ .prefill_fixed = target - 10 * f.per_row, .decode_fixed = 0, .per_row = f.per_row }, f.ceiling, 384));
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
    const f = fill_fixture;
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
