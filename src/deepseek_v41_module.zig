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
const xo = @import("exl3_kernel_ops.zig");
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
const Chain = xp.EagerChain(G, xp.MlxGemv);
/// The expert source: the op chain around the kernels' EXL3 decode GEMV, the wide (prefill) routed calls on
/// the kernels' DIG-X route, the next layer's reads started from the predictor. `A` waits on the host
/// (LOOKAHEAD3, the exact tier); `AGated` builds every wave over event gates (LOOKAHEAD4, the typical tier).
pub const A = arm_mod.ArmWith(G, Chain, .{ .prefill = xo.DigXPrefill(G), .lookahead = true });
pub const AGated = arm_mod.ArmWith(G, Chain, .{ .prefill = xo.DigXPrefill(G), .lookahead = true, .gated = true });

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

/// Beside the model's shards: the Engram token map the converter exports.
pub const engram_token_map_file = "engram-token-map.u32";

/// The admission's calibration; its MLX allocator cache charges are the limits the module sets per phase.
const envelope = expert_admission.Envelope.dsv41_pass2;

pub const Module = struct {
    gpa: std.mem.Allocator,
    g: G,
    kernels: *xo.Accepted(G),
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
        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .g = try G.init(gpa, s), .kernels = undefined, .arm = undefined, .weights = weights, .engram = undefined, .embed_rows = undefined, .model = undefined, .head = undefined };
        errdefer self.g.deinit();
        self.owner = std.Thread.getCurrentId();
        var diag: arm_mod.Diag = .{};
        self.kernels = acceptKernels(gpa, &self.g, .{ .device = .{ .stream = s } }, &diag) catch |e| return refused(e, &diag);
        errdefer self.dropKernels();
        _ = mlx.mlx_clear_cache();
        // The allocator cache holds no more than the admission charges for the phase (prefill here).
        _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, envelope.prefill_cache_bytes);
        errdefer setCacheLimit(self.prev_cache_limit);
        self.arm = if (config.expert_event_gates orelse false)
            .{ .event_gates = try self.buildArm(AGated, io, config, weights, s, try expert_event.createMetal(), &diag) }
        else
            .{ .host_waits = try self.buildArm(A, io, config, weights, s, null, &diag) };
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
        self.model = try M.init(gpa, &self.g, c, routes.served, weights, &self.engram);
        errdefer self.model.deinit(&self.g);
        const subset = switch (self.arm) {
            inline else => |t| if (t.arm.draft_subset) |*x| x else null,
        };
        self.head = try H.initWith(gpa, &self.g, c, routes.served.draftRoutes(), weights, .{ .subset = subset });
        errdefer self.head.deinit(&self.g);
        // The install warm-up (P4.3): the served tier's compiled regions trace here, never in a request.
        // Decode forwards only until the draft round (P5) serves its depth; each shape's MLX peak is the bill's.
        var peak: [1]u64 = undefined;
        switch (self.arm) {
            inline else => |t| try dsl.Loop(G).warmFor(&self.g, gpa, self.model, self.head, &t.arm.hook, .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) }, &peak),
        }
        _ = mlx.mlx_clear_cache();
        log.info("warm-up: decode forward peak {d} B above the residents; built residents {d} B (W97)", .{ peak[0], self.model.builtBytes() + self.head.builtBytes() });
        return self;
    }

    /// The expert source at the admitted rows, its banks checked against the kernels (again at the phase change).
    fn buildArm(self: *Module, comptime AT: type, io: std.Io, config: *const model_io.ModelConfig, weights: *const model_io.Weights, s: mlx.mlx_stream, event: ?expert_event.Event, diag: *arm_mod.Diag) !Tiered(AT) {
        const gpa = self.gpa;
        const gates = try routerGates(AT.Hook.Gate, gpa, weights, config.num_hidden_layers);
        errdefer gpa.free(gates);
        const arm = AT.initHooked(gpa, io, &self.g, self.kernels.gemvRoute(xp.MlxGemv), .{
            .model_dir = config.expert_bank_dir.?,
            .envelope = envelope,
            .baseline_bytes = config.memory_baseline_bytes,
            .fixed_rows = config.expert_rows,
            .slot_memory = .{ .mlx = s },
            .prefill = .{ .reg = &self.kernels.reg },
            .draft_pruned_bytes = 0,
            .lookahead = lookahead,
            .event = if (event) |e| .{ .backend = .{ .metal = e.object }, .watchdog_ms = event_watchdog_ms } else null,
        }, .{ .gates = gates, .event = event }, diag) catch |e| return refused(e, diag);
        errdefer arm.deinit();
        checkArmBanks(arm, &self.g, &self.kernels.reg, diag) catch |e| return refused(e, diag);
        arm.grown_check = .{ .ctx = &self.kernels.reg, .check = GrownBanks(AT).check };
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
        self.dropKernels();
        self.g.deinit();
        setCacheLimit(self.prev_cache_limit);
        gpa.destroy(self);
    }

    /// The kernels go after the last launch drained.
    fn dropKernels(self: *Module) void {
        // The process's teardown frees the registry off the inference thread, which has stopped launching.
        if (std.Thread.getCurrentId() == self.owner) _ = mlx.mlx_synchronize(self.g.s);
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
        switch (self.arm) {
            inline else => |t| if (ids.len == 1 and !t.arm.grown) {
                if (!self.fenced) {
                    try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
                    self.fenced = true;
                }
                // The prefill's parked buffers go back before the slot banks grow; decode keeps its own charge.
                _ = mlx.mlx_clear_cache();
                setCacheLimit(envelope.decode_cache_bytes);
                try t.arm.grow(&self.g);
            },
        }
        return self.forward(ids);
    }

    fn forward(self: *Module, ids: []const u32) !mlx.mlx_array {
        const g = &self.g;
        const st = &(self.state orelse return error.Dsv41NoRequest);
        switch (self.arm) {
            inline else => |t| {
                const r = try self.model.forward(g, st, ids, .{ .logits = .last }, &t.arm.hook, graph.NoProbe{});
                try M.fence(g, st, &.{r.logits.?});
                try t.arm.hook.flush();
                const out = g.keep(r.logits.?);
                g.reset();
                return out;
            },
        }
    }
};

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
fn checkArmBanks(arm: anytype, g: *G, reg: *const xk.Registry, diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        checkBank(g, reg, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @fromBackingInt(@intCast(kind))), kd.message() });
    };
}

/// The phase change's banks, once (`Arm.grown_check`); a refusal is logged by name.
fn GrownBanks(comptime AT: type) type {
    return struct {
        fn check(ctx: *const anyopaque, arm: *AT, g: *G) anyerror!void {
            const reg: *const xk.Registry = @ptrCast(@alignCast(ctx));
            var diag: arm_mod.Diag = .{};
            checkArmBanks(arm, g, reg, &diag) catch |e| {
                log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
                return e;
            };
        }
    };
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
        .lookahead = lookahead,
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

/// The bytes one traced forward `[from, to)` holds under its waves: every node outside the outermost waves plus the
/// widest outermost wave (its inner waves counted as live at once: an upper bound).
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
            const b = graph.heldBytes(g, w.from, w.to).sum;
            in_waves += b;
            widest = @max(widest, b);
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
    const model_ = try TM.init(a, &g, c, try routes.parse(&.{}, null), &lookup, &src);
    defer model_.deinit(&g);
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const rows = try aa.alloc(u32, c.n_layers);
    @memset(rows, 8);
    var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows });
    defer fsrc.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, TChain, .{ .prefill = xo.DigXPrefill(ops.TraceOps) });
    var ex = try Ex.initWith(a, &g, &fsrc, TChain.init(.{}, &c), &c, .{ .prefill = .{ .reg = &reg } });
    defer ex.deinit();
    var rid: RandomIds = .{ .n_experts = @intCast(c.n_routed_experts) };
    g.host_values = rid.values();
    const prompt = try aa.alloc(u32, 2048);
    for (prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
    // (positions already in the state, rows of the measured forward): the decode lane, the gate's prompt, wide chunks.
    for ([_][2]u32{ .{ 0, 8 }, .{ 0, 63 }, .{ 0, 256 }, .{ 0, 953 }, .{ 0, 2048 }, .{ 1024, 256 }, .{ 1024, 953 } }) |pn| {
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
        const billed = bill.waveBytes(n, pn[0] + n);
        std.debug.print("\nDSV41_HELD {{\"positions\": {d}, \"rows\": {d}, \"one_reset\": {d}, \"waves\": {d}, \"billed\": {d}}}", .{ pn[0], n, h.reset, h.outside + h.widest, billed });
        try std.testing.expect(h.outside + h.widest <= billed);
    }
    // The bill's chunk is the model's.
    for ([_]u64{ 1, 8, 64, 953, 2048, 4096, 16384, 65536, 131072 }) |sq|
        try std.testing.expectEqual(@as(u64, @intCast(kvc.resolvePrefillChunk(&c, sq, null, kvc.default_chunk_target_bytes))), bill.chunkRows(sq));
    std.debug.print("\nDSV41_PREFILL_BILL {{\"kv_pos_bytes\": {d}, \"gate_64_32\": {d}, \"cell_16384_1024\": {d}, \"cell_wave\": {d}}}\n", .{ bill.kv_pos_bytes, bill.bytes(64, 32), bill.bytes(16384, 1024), bill.waveBytes(bill.chunkRows(16384), 16384) });
}
