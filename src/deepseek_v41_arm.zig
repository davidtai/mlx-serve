//! The native deepseek_v41 arm: its construction from the model directory and
//! the bench cell over it. Construction is config (`deepseek_v41.Config`) ->
//! expert bank -> memory admission (`Admission.plan` over the box envelope)
//! -> expert stream at the admitted rows -> the model's routed-expert hook
//! over the stream. Every refusal is a named error at construction; nothing
//! here runs per token.
//!
//! The decode loop is a seam, bound at compile time: `decode.prefill(arm, g,
//! prompt) !u32` (the primary token), `decode.cycle(arm, g, a, out) !bool`
//! (appends the cycle's tokens; true when done), `decode.stats() Stats`. The
//! DSpark loop binds it; until then `StandIn` does: seeded routes through the
//! same hook with no model math, so construction, the reads and the receipt
//! can be exercised and measured. The server refuses the arch while its
//! decode binding is the stand-in (`serving_decode`).
//!
//! Threads: mlx-serve builds and runs a model on the scheduler's inference
//! thread, the only MLX caller (model load, every forward, growth, unload);
//! connection, sampler, idle-evict, disk-writer and LAN threads never touch a
//! model. The stream's read pool threads only pread / memcpy into slot memory
//! and signal events. `Stream.grow` allocates slot memory and refuses any
//! thread but the one that built the stream, so the Python tier's growth
//! overlap (extension banks built on a helper thread) cannot be ported by
//! calling it from another thread.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("expert_io.zig");
const expert_stream = @import("expert_stream.zig");
const expert_admission = @import("expert_admission.zig");

pub const DecodeBinding = enum { stand_in, dspark };
/// The decode loop the server would generate with; the DSpark loop sets it.
pub const serving_decode: DecodeBinding = .stand_in;

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn refuse(diag: *Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    const s = std.fmt.bufPrint(&diag.buf, fmt, args) catch diag.buf[0..];
    diag.len = s.len;
    return err;
}

/// The tier's composite on pass 2 (pipeline 6 x 64 MiB + wide 1536 MiB + layer
/// compile 64 MiB + member charge 64 MiB; host; the prefill members' cache
/// charge), the admission inputs until the native stack measures its own.
pub const pass2_phase_reserve_bytes: u64 = expert_admission.base_pipeline_bytes + 1536 * (1 << 20) + 2 * 67_108_864;
pub const pass2_host_reserve_bytes: u64 = 408_944_640;
pub const pass2_prefill_charge_bytes: u64 = 5_703_196_672;

pub const Options = struct {
    /// Absolute: config.json, the resident shards and the expert bank.
    model_dir: []const u8,
    implemented: expert_bank.Implemented = expert_bank.dsv41,
    envelope: expert_admission.Envelope = .dsv41_pass2,
    /// The box's non-file baseline the guard measured (its
    /// MTPLX_DSV41_BOX_BASELINE_GB); the admission refuses to guess it.
    baseline_bytes: ?u64,
    /// Wired bytes at construction; null reads them now.
    wired_bytes: ?u64 = null,
    fixed_rows: ?u32 = null,
    allocation: expert_admission.Allocation = .prefill_excess,
    phase_reserve_bytes: u64 = pass2_phase_reserve_bytes,
    host_reserve_bytes: u64 = pass2_host_reserve_bytes,
    prefill_charge_bytes: u64 = pass2_prefill_charge_bytes,
    peak_fill: ?expert_admission.PeakFill = .{},
    rowsx: ?expert_admission.Rowsx = null,
    slot_memory: expert_stream.SlotMemory,
    lookahead: ?expert_stream.Lookahead = null,
    event: ?expert_stream.Event = null,
    pool: expert_io.Options = .{ .tickets = 1024 },
};

/// The arm's construction up to the admitted rows: config, bank, plan. No
/// slot memory yet (the caller owns `bank`).
pub const Planned = struct {
    config: v41.Config,
    bank: expert_bank.Bank,
    inputs: expert_admission.Inputs,
    plan: expert_admission.Plan,
    /// Per layer, before and after the phase change.
    prefill_rows: u32,
    decode_rows: u32,
};

pub fn planRows(a: std.mem.Allocator, io: std.Io, opt: Options, diag: *Diag) !Planned {
    var cdiag: v41.Diag = .{};
    const c = v41.Config.load(a, io, opt.model_dir, &cdiag) catch |e| return refuse(diag, e, "config: {s}", .{cdiag.message()});
    const im = opt.implemented;
    if (c.hidden_size != im.hidden or c.moe_intermediate_size != im.inter or c.n_routed_experts != im.n_experts or c.n_layers != im.n_layers)
        return refuse(diag, error.ConfigBankMismatch, "config: hidden {d}, inter {d}, {d} experts, {d} layers; the bank lane decodes {d}, {d}, {d}, {d}", .{
            c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers, im.hidden, im.inter, im.n_experts, im.n_layers,
        });
    const baseline = opt.baseline_bytes orelse return refuse(diag, error.BaselineMissing, "admission: no measured box baseline", .{});
    var bdiag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, opt.model_dir, im, &bdiag) catch |e| return refuse(diag, e, "bank: {s}", .{bdiag.message()});
    errdefer bank.deinit();
    var record: u64 = 0;
    for (bank.layers) |l| record = @max(record, l.logical_bytes);
    const inputs: expert_admission.Inputs = .{
        .baseline_bytes = baseline,
        .wired_bytes = opt.wired_bytes orelse wiredBytes(),
        .record_bytes = record,
        .fixed_rows = opt.fixed_rows,
        .allocation = opt.allocation,
        .phase_reserve_bytes = opt.phase_reserve_bytes,
        .lookahead_staging_bytes = if (opt.lookahead) |la| expert_admission.lookaheadCharge(record, 2 * la.budget, std.heap.pageSize()) else 0,
        .host_reserve_bytes = opt.host_reserve_bytes,
        .prefill_charge_bytes = opt.prefill_charge_bytes,
        .peak_fill = opt.peak_fill,
        .rowsx = opt.rowsx,
    };
    const plan_ = expert_admission.Admission.plan(opt.envelope, inputs) catch |e| return refuse(diag, e, "admission: {s}", .{@errorName(e)});
    // The stream holds what the admitted prefill bank bound holds (the
    // Python engine resolves its own plan within it); a layer never
    // holds more rows than it has experts.
    const n_experts = bank.n_experts;
    const prefill = @min(plan_.admission.prefill_capacity, n_experts);
    const decode = @min(plan_.admission.decode_rows, n_experts);
    if (prefill > decode) return refuse(diag, error.PrefillAboveDecode, "admission: prefill capacity {d} exceeds the decode rows {d}", .{ prefill, decode });
    return .{ .config = c, .bank = bank, .inputs = inputs, .plan = plan_, .prefill_rows = prefill, .decode_rows = decode };
}

/// The arm over graph backend `G` (`MlxOps` serving, `TraceOps` host tests)
/// with routed-expert math `M` (`M.init(math_arg, *const Config)`).
pub fn Arm(comptime G: type, comptime M: type) type {
    return struct {
        const Self = @This();
        pub const Backend = G;
        pub const Hook = xp.Experts(G, xp.StreamSource, M);

        a: std.mem.Allocator,
        /// Borrowed from `Options.model_dir`.
        model_dir: []const u8,
        config: v41.Config,
        bank: expert_bank.Bank,
        inputs: expert_admission.Inputs,
        plan: expert_admission.Plan,
        /// Per layer: the stream's rows before and after the phase change.
        prefill_rows: []u32,
        decode_rows: []u32,
        stream: *expert_stream.Stream,
        source: xp.StreamSource,
        hook: Hook,
        grown: bool = false,
        /// Run over the banks the phase change binds, before the arm counts as
        /// grown (the binding's kernels layout check; set once, at construction).
        /// A refusal leaves the arm ungrown: every later request is refused too.
        grown_check: ?GrownCheck = null,

        pub const GrownCheck = struct {
            ctx: *const anyopaque,
            check: *const fn (ctx: *const anyopaque, arm: *Self, g: *G) anyerror!void,
        };

        pub fn init(a: std.mem.Allocator, io: std.Io, g: *G, math_arg: anytype, opt: Options, diag: *Diag) !*Self {
            const self = try a.create(Self);
            errdefer a.destroy(self);
            var p = try planRows(a, io, opt, diag);
            errdefer p.bank.deinit();
            const c = p.config;
            const prefill_rows = try a.alloc(u32, c.n_layers);
            errdefer a.free(prefill_rows);
            @memset(prefill_rows, p.prefill_rows);
            const decode_rows = try a.alloc(u32, c.n_layers);
            errdefer a.free(decode_rows);
            @memset(decode_rows, p.decode_rows);
            self.* = .{
                .a = a,
                .model_dir = opt.model_dir,
                .config = c,
                .bank = p.bank,
                .inputs = p.inputs,
                .plan = p.plan,
                .prefill_rows = prefill_rows,
                .decode_rows = decode_rows,
                .stream = undefined,
                .source = undefined,
                .hook = undefined,
            };
            self.stream = expert_stream.Stream.init(a, &self.bank, .{
                .rows = prefill_rows,
                .slot_memory = opt.slot_memory,
                .lookahead = opt.lookahead,
                .event = opt.event,
                .pool = opt.pool,
            }) catch |e| return refuse(diag, e, "stream: {s}", .{@errorName(e)});
            errdefer self.stream.deinit();
            self.source = xp.StreamSource.init(self.stream);
            self.hook = Hook.init(a, g, &self.source, M.init(math_arg, &self.config), &self.config) catch |e|
                return refuse(diag, e, "routed-expert hook: {s}", .{@errorName(e)});
            return self;
        }

        pub fn deinit(self: *Self) void {
            const a = self.a;
            self.hook.deinit();
            self.stream.deinit();
            self.bank.deinit();
            a.free(self.prefill_rows);
            a.free(self.decode_rows);
            a.destroy(self);
        }

        pub fn admissionRecord(self: *const Self) AdmissionRecord {
            return AdmissionRecord.of(self.inputs, self.plan, self.prefill_rows[0], self.decode_rows[0], self.config.n_layers);
        }

        /// The one phase change, at the admitted decode rows.
        pub fn grow(self: *Self, g: *G) !void {
            try self.hook.grow(g, self.decode_rows);
            if (self.grown_check) |c| try c.check(c.ctx, self, g);
            self.grown = true;
        }
    };
}

/// The admission as the pass-2 receipts' MTP_BOUND `growth_admission` names it
/// (plus the stream's own rows per layer).
pub const AdmissionRecord = struct {
    const Phases = struct { growth: u64, seed: u64, prime: u64, decode: u64 };
    const Summary = struct {
        decode_slots_per_layer: u32,
        prefill_slots_per_layer: u32,
        tcq3_prefill_max_slots_per_layer: u32,
        tcq3_final_slot_storage_bytes: u64,
        post_prefill_phase_active_bounds: Phases,
        binding_phase: []const u8,
        binding_phase_active_bytes: u64,
        binding_phase_physical_bytes: u64,
        prefill_active_bound_bytes: u64,
        prefill_physical_bound_bytes: u64,
        physical_bound_bytes: u64,
        allocator_limit_bytes: u64,
        active_plus_cache_bytes: u64,
        wired_plus_active_plus_cache_bytes: u64,
    };
    const PeakFill = struct {
        target_physical_bytes: u64,
        total_phase_credit_bytes: u64,
        modeled_peak_physical_bytes: u64,
        control: Summary,
        filled: Summary,
    };
    const Rowsx = struct {
        first_capacity_uncredited: u32,
        first_capacity_credited: u32,
        added_rows: u32,
        added_bytes: u64,
        credit_bytes: u64,
        mlx_side_credit_bytes: u64,
        modeled_peak_uncredited_bytes: u64,
        modeled_peak_credited_bytes: u64,
        target_bytes: u64,
        allocator_margin_bytes: i64,
        wired_margin_bytes: i64,
        box_ceiling_margin_modeled_bytes: i64,
        baseline_bytes: u64,
        transition_start_active_bound_bytes: u64,
        transition_start_credit_restored_bytes: u64,
    };

    baseline_bytes: u64,
    wired_before_bytes: u64,
    decode_slots_per_layer: u32,
    prefill_slots_per_layer: u32,
    stream_prefill_rows_per_layer: u32,
    stream_decode_rows_per_layer: u32,
    growth_payload_bytes: u64,
    capacity_search_ceiling: u32,
    tcq3_fixed_capacity: ?u32,
    tcq3_matched_prefill_capacity: ?u32,
    tcq3_prefill_max_slots_per_layer: u32,
    tcq3_slot_bytes: u64,
    tcq3_final_slot_storage_bytes: u64,
    tcq3_prefill_slot_storage_bound_bytes: u64,
    transition_start_active_bound_bytes: u64,
    retirement_entry_active_bound_bytes: u64,
    tcq3_post_prefill_phase_active_bounds: Phases,
    steady_decode_active_bound_bytes: u64,
    resize_active_bound_bytes: u64,
    seed_active_bound_bytes: u64,
    prime_active_bound_bytes: u64,
    prefill_active_bound_bytes: u64,
    prefill_physical_bound_bytes: u64,
    physical_bound_bytes: u64,
    active_bound_bytes: u64,
    host_reserve_bytes: u64,
    allocator_limit_bytes: u64,
    prefill_cache_allowance_bytes: u64,
    decode_cache_allowance_bytes: u64,
    embedding_post_prefill_credit_bytes: u64,
    tail_transition_active_credit_bytes: u64,
    tcq3_additional_active_reserve_bytes: u64,
    tcq3_additional_host_reserve_bytes: u64,
    tcq3_allocation: []const u8,
    tcq3_io_staging_bytes: u64,
    tcq3_embedding_rows: bool,
    tcq3_tail_rows: ?u32,
    tcq3_peak_fill: ?PeakFill,
    q3_rowsx: ?Rowsx,

    fn phases(p: expert_admission.Phases) Phases {
        return .{ .growth = p.growth, .seed = p.seed, .prime = p.prime, .decode = p.decode };
    }

    fn summary(s: expert_admission.Summary) Summary {
        return .{
            .decode_slots_per_layer = s.decode_rows,
            .prefill_slots_per_layer = s.prefill_rows,
            .tcq3_prefill_max_slots_per_layer = s.prefill_max_rows,
            .tcq3_final_slot_storage_bytes = s.final_bank_bytes,
            .post_prefill_phase_active_bounds = phases(s.phases),
            .binding_phase = @tagName(s.binding),
            .binding_phase_active_bytes = s.binding_active_bytes,
            .binding_phase_physical_bytes = s.binding_physical_bytes,
            .prefill_active_bound_bytes = s.prefill_active_bytes,
            .prefill_physical_bound_bytes = s.prefill_physical_bytes,
            .physical_bound_bytes = s.physical_bound_bytes,
            .allocator_limit_bytes = s.allocator_limit_bytes,
            .active_plus_cache_bytes = s.active_plus_cache_bytes,
            .wired_plus_active_plus_cache_bytes = s.wired_plus_active_plus_cache_bytes,
        };
    }

    pub fn of(in: expert_admission.Inputs, plan: expert_admission.Plan, prefill_rows: u32, decode_rows: u32, n_layers: u32) AdmissionRecord {
        const a = plan.admission;
        return .{
            .baseline_bytes = a.baseline_bytes,
            .wired_before_bytes = a.wired_bytes,
            .decode_slots_per_layer = a.decode_rows,
            .prefill_slots_per_layer = a.prefill_rows,
            .stream_prefill_rows_per_layer = prefill_rows,
            .stream_decode_rows_per_layer = decode_rows,
            .growth_payload_bytes = @as(u64, decode_rows - prefill_rows) * n_layers * in.record_bytes,
            .capacity_search_ceiling = a.search_ceiling,
            .tcq3_fixed_capacity = in.fixed_rows,
            .tcq3_matched_prefill_capacity = in.matched_prefill_rows,
            .tcq3_prefill_max_slots_per_layer = a.prefill_max_rows,
            .tcq3_slot_bytes = in.record_bytes,
            .tcq3_final_slot_storage_bytes = a.final_bank_bytes,
            .tcq3_prefill_slot_storage_bound_bytes = a.prefill_bank_bound_bytes,
            .transition_start_active_bound_bytes = a.transition_start_bytes,
            .retirement_entry_active_bound_bytes = a.retirement_entry_bytes,
            .tcq3_post_prefill_phase_active_bounds = phases(a.phases),
            .steady_decode_active_bound_bytes = a.steady_bytes,
            .resize_active_bound_bytes = a.resize_bytes,
            .seed_active_bound_bytes = a.phases.seed,
            .prime_active_bound_bytes = a.phases.prime,
            .prefill_active_bound_bytes = a.prefill_active_bytes,
            .prefill_physical_bound_bytes = a.prefill_physical_bytes,
            .physical_bound_bytes = a.physical_bound_bytes,
            .active_bound_bytes = a.active_bound_bytes,
            .host_reserve_bytes = a.host_reserve_bytes,
            .allocator_limit_bytes = a.allocator_limit_bytes,
            .prefill_cache_allowance_bytes = a.prefill_cache_bytes,
            .decode_cache_allowance_bytes = a.decode_cache_bytes,
            .embedding_post_prefill_credit_bytes = a.embedding_credit_bytes,
            .tail_transition_active_credit_bytes = a.tail_credit_bytes,
            .tcq3_additional_active_reserve_bytes = in.phase_reserve_bytes + (if (in.rowsx == null) in.lookahead_staging_bytes else 0),
            .tcq3_additional_host_reserve_bytes = in.host_reserve_bytes,
            .tcq3_allocation = @tagName(in.allocation),
            .tcq3_io_staging_bytes = if (in.io_layout == .gate_up) 36 << 20 else 32 << 20,
            .tcq3_embedding_rows = in.embedding_rows,
            .tcq3_tail_rows = in.tail_rows,
            .tcq3_peak_fill = if (plan.peak_fill) |pf| .{
                .target_physical_bytes = in.peak_fill.?.target_bytes,
                .total_phase_credit_bytes = pf.total_credit_bytes,
                .modeled_peak_physical_bytes = pf.modeled_peak_bytes,
                .control = summary(pf.control),
                .filled = summary(pf.filled),
            } else null,
            .q3_rowsx = if (plan.rowsx) |r| .{
                .first_capacity_uncredited = r.uncredited_rows,
                .first_capacity_credited = r.credited_rows,
                .added_rows = r.added_rows,
                .added_bytes = r.added_bytes,
                .credit_bytes = r.credit_bytes,
                .mlx_side_credit_bytes = r.mlx_side_credit_bytes,
                .modeled_peak_uncredited_bytes = r.modeled_peak_uncredited_bytes,
                .modeled_peak_credited_bytes = r.modeled_peak_credited_bytes,
                .target_bytes = r.target_bytes,
                .allocator_margin_bytes = r.allocator_margin_bytes,
                .wired_margin_bytes = r.wired_margin_bytes,
                .box_ceiling_margin_modeled_bytes = r.box_margin_bytes,
                .baseline_bytes = r.baseline_bytes,
                .transition_start_active_bound_bytes = r.restored_gate_bytes,
                .transition_start_credit_restored_bytes = r.credit_bytes,
            } else null,
        };
    }
};

/// Routed-expert math that computes nothing: zeros of the math's output
/// shapes (the stand-in decode; the kernels bind `EagerChain`'s GEMV).
pub fn StandInMath(comptime G: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        hidden: c_int,
        inter: c_int,

        pub fn init(_: void, c: *const v41.Config) Self {
            return .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, _: T, _: xp.ProjOf(T), _: xp.ProjOf(T)) !T {
            return g.zeros(&.{ g.shapeOf(x).dim(0), self.inter }, .float32);
        }

        pub fn down(self: *const Self, g: *G, h: T, _: T, _: xp.ProjOf(T)) !T {
            return g.zeros(&.{ g.shapeOf(h).dim(0), self.hidden }, .float32);
        }
    };
}

/// What a served request asks of the decode loop (`decode.begin`).
pub const DecodeConfig = struct {
    /// DSpark draft depth; a cycle verifies depth + 1 rows.
    depth: u32,
    /// null = greedy acceptance (the exact tier); else typical at this delta.
    typical_delta: ?f32,
    seed: u64 = 0,
};

/// Counters of the decode loop, as the receipt's `stats` names them.
pub const Stats = struct {
    cycles: u32 = 0,
    verify_calls: u32 = 0,
    generated_tokens: u32 = 0,
};

fn mix(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// The decode seam's stand-in: every forward routes seeded ids (k distinct
/// per row) through each layer's hook with zero activations, evaluates the
/// outputs, then flushes the hook, as a forward of the DSpark loop would.
/// A cycle is one verify forward of `rows` rows emitting 1..rows tokens.
/// Its token ids are a seeded sequence; they mean nothing.
pub fn StandIn(comptime A: type) type {
    const G = A.Backend;
    return struct {
        const Self = @This();
        seed: u64,
        rows: u32,
        max_cycles: u32,
        forwards: u64 = 0,
        st: Stats = .{},
        ids: [xp.max_route_ids]u16 = undefined,

        pub fn init(seed: u64, rows: u32, max_cycles: u32) Self {
            return .{ .seed = seed, .rows = rows, .max_cycles = max_cycles };
        }

        /// A served request: `rows` verify rows per cycle (depth + 1), no cycle
        /// cap (the server stops on stop ids and max_tokens), counters reset.
        pub fn begin(self: *Self, cfg: DecodeConfig) error{}!void {
            const forwards = self.forwards;
            self.* = .{ .seed = cfg.seed, .rows = cfg.depth + 1, .max_cycles = std.math.maxInt(u32), .forwards = forwards };
        }

        /// The trace backend reads the routing barrier's ids from here.
        pub fn bind(self: *Self, g: *G) void {
            if (G == ops.TraceOps) g.host_values = .{ .ctx = self, .ids = hostIds, .argmax = hostArgmax };
        }

        fn hostIds(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.ids[0..out.len]);
        }

        fn hostArgmax(_: *anyopaque) anyerror!u32 {
            return error.StandInHasNoLogits;
        }

        fn fillIds(self: *Self, layer: usize, n: u32, n_experts: u32, k: u32) void {
            for (0..n) |r| {
                const row = self.ids[r * k ..][0..k];
                var h = mix(self.seed ^ mix(self.forwards) ^ mix(layer << 8 | r));
                var j: usize = 0;
                while (j < k) {
                    h = mix(h);
                    const e: u16 = @intCast(h % n_experts);
                    if (std.mem.indexOfScalar(u16, row[0..j], e) == null) {
                        row[j] = e;
                        j += 1;
                    }
                }
            }
        }

        fn forward(self: *Self, arm: *A, g: *G, n: u32) !void {
            const c = &arm.config;
            const k = c.n_experts_per_tok;
            if (n * k > xp.max_route_ids) return error.StandInRowsTooWide;
            var outs: [v41.max_layers]G.T = undefined;
            for (0..c.n_layers) |l| {
                self.fillIds(l, n, c.n_routed_experts, k);
                var ids32: [xp.max_route_ids]u32 = undefined;
                for (ids32[0 .. n * k], self.ids[0 .. n * k]) |*o, e| o.* = e;
                const xf = try g.zeros(&.{ @intCast(n), @intCast(c.hidden_size) }, .bfloat16);
                const indices = try g.hostArray(std.mem.sliceAsBytes(ids32[0 .. n * k]), &.{ @intCast(n), @intCast(k) }, .uint32);
                outs[l] = try arm.hook.at(@intCast(l)).routed(g, xf, indices);
            }
            try g.evalAll(outs[0..c.n_layers]);
            try arm.hook.flush();
            g.reset();
            self.forwards += 1;
        }

        fn token(self: *const Self, i: u64, vocab: u32) u32 {
            return @intCast(mix(self.seed ^ 0x70C3 ^ mix(i)) % vocab);
        }

        /// The prompt in forwards of at most `rows` rows; the primary token.
        pub fn prefill(self: *Self, arm: *A, g: *G, prompt: []const u32) !u32 {
            var i: usize = 0;
            while (i < prompt.len) : (i += self.rows) try self.forward(arm, g, @intCast(@min(self.rows, prompt.len - i)));
            self.st.generated_tokens = 1;
            return self.token(0, arm.config.vocab_size);
        }

        pub fn cycle(self: *Self, arm: *A, g: *G, a: std.mem.Allocator, out: *std.ArrayList(u32)) !bool {
            try self.forward(arm, g, self.rows);
            const emit: u32 = @intCast(1 + mix(self.seed ^ mix(self.st.cycles + 1)) % self.rows);
            for (0..emit) |_| {
                try out.append(a, self.token(self.st.generated_tokens, arm.config.vocab_size));
                self.st.generated_tokens += 1;
            }
            self.st.cycles += 1;
            self.st.verify_calls += 1;
            return self.st.cycles >= self.max_cycles;
        }

        pub fn stats(self: *const Self) Stats {
            return self.st;
        }
    };
}

// ── Memory probes (Mach) ──

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;

/// task_vm_info through the rev3 ledger block (<mach/task_info.h> order).
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    rest: [16]u64,
    phys_footprint: u64,
    min_address: u64,
    max_address: u64,
    ledger_phys_footprint_peak: i64,
    ledger_rest: [20]i64,
};

pub const Footprint = struct { now: u64, peak: u64 };

/// The process phys_footprint (Metal included) and its lifetime peak.
pub fn footprint() Footprint {
    if (comptime !builtin.os.tag.isDarwin()) return .{ .now = 0, .peak = 0 };
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 22, @ptrCast(&info), &count) != 0) return .{ .now = 0, .peak = 0 };
    const full = count * @sizeOf(i32) >= @offsetOf(TaskVmInfo, "ledger_rest");
    return .{ .now = info.phys_footprint, .peak = if (full) @intCast(@max(info.ledger_phys_footprint_peak, 0)) else info.phys_footprint };
}

/// vm_stat's "Pages wired down", in bytes.
pub fn wiredBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var stats: [40]i32 = @splat(0);
    var count: u32 = stats.len;
    if (host_statistics64(mach_host_self(), 4, &stats, &count) != 0) return 0;
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;
    const wire_count: u32 = @bitCast(stats[3]);
    return @as(u64, wire_count) * page;
}

// ── Tests ──

const testing = std.testing;

/// A mini deepseek_v41 directory for host tests: the model lane's mini config
/// (5 layers, hidden 64, inter 32, 4 experts, top-2) over a synthetic bank of
/// that geometry.
pub const TestModel = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    root_buf: [512]u8 = undefined,
    root: []const u8 = "",

    pub const implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = 4, .n_layers = 5 };

    pub fn create(with_bank: bool) !*TestModel {
        const a = testing.allocator;
        const self = try a.create(TestModel);
        errdefer a.destroy(self);
        self.* = .{ .tmp = std.testing.tmpDir(.{}), .image = &.{} };
        errdefer self.tmp.cleanup();
        if (with_bank) self.image = try expert_bank.writeSynth(a, &self.tmp, .{ .n_experts = 4, .k = &.{ 3, 3, 3, 3, 3 } });
        errdefer a.free(self.image);
        const cfg = try v41.testConfigJson(a, .mini);
        defer a.free(cfg);
        try self.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = cfg });
        self.root = try expert_bank.tmpRoot(&self.tmp, &self.root_buf);
        return self;
    }

    pub fn destroy(self: *TestModel) void {
        testing.allocator.free(self.image);
        self.tmp.cleanup();
        testing.allocator.destroy(self);
    }

    pub fn options(self: *const TestModel) Options {
        return .{
            .model_dir = self.root,
            .implemented = implemented,
            .baseline_bytes = 7_200_000_000,
            .wired_bytes = 3_300_000_000,
            // A 2,880 B record under the causal allocator's room leaves no
            // predecessor row budget small enough: the uniform allocation.
            .allocation = .uniform,
            .slot_memory = .host,
            .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 },
        };
    }
};

const TraceArm = Arm(ops.TraceOps, StandInMath(ops.TraceOps));

test "dsv41 arm: a synthetic model builds at its admitted rows with the routed-expert hook bound" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    const arm = TraceArm.init(testing.allocator, std.testing.io, &g, {}, tm.options(), &diag) catch |e| {
        std.debug.print("dsv41 arm: {s}\n", .{diag.message()});
        return e;
    };
    defer arm.deinit();
    // The plan is Admission.plan's; a layer never holds more rows than its 4 experts.
    const want = try expert_admission.Admission.plan(.dsv41_pass2, arm.inputs);
    try testing.expectEqual(want.admission.decode_rows, arm.plan.admission.decode_rows);
    try testing.expectEqual(@as(u64, 2880), arm.inputs.record_bytes);
    for (arm.prefill_rows, arm.decode_rows) |p, d| {
        try testing.expectEqual(@as(u32, 4), p);
        try testing.expectEqual(@as(u32, 4), d);
    }
    for (0..5) |l| try testing.expectEqual(@as(u32, 4), arm.source.bankRows(@intCast(l), .base));
    for (arm.hook.banks) |b| {
        try testing.expect(b[@intFromEnum(xp.BankKind.base)] != null);
        try testing.expect(b[@intFromEnum(xp.BankKind.transient)] != null);
    }
    const rec = arm.admissionRecord();
    try testing.expectEqual(arm.plan.admission.decode_rows, rec.decode_slots_per_layer);
    try testing.expectEqual(@as(u32, 4), rec.stream_decode_rows_per_layer);
    try testing.expect(rec.tcq3_peak_fill != null and rec.q3_rowsx == null);
    try arm.grow(&g);
    try testing.expect(arm.grown);
}

// DSV41_BANK=<the 3.0 bank dir>: the planning half on the real bank (CPU; no slot memory).
test "dsv41 arm: the real bank plans a pass-2 receipt's rows and bounds" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var diag: Diag = .{};
    // pass2-host-fast-exact-exl3: its baseline, wired, forced rows and the lookahead lane.
    var p = planRows(testing.allocator, std.testing.io, .{
        .model_dir = dir,
        .baseline_bytes = 7_755_397_656,
        .wired_bytes = 3_377_741_824,
        .fixed_rows = 147,
        .lookahead = .{},
        .slot_memory = .host,
    }, &diag) catch |e| {
        std.debug.print("dsv41 arm: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    try testing.expectEqual(@as(u64, 13_315_584), p.inputs.record_bytes);
    try testing.expectEqual(@as(u64, 54_460_416), p.inputs.lookahead_staging_bytes);
    const adm = p.plan.admission;
    try testing.expectEqual(@as(u32, 147), adm.decode_rows);
    try testing.expectEqual(@as(u32, 80), adm.prefill_rows);
    try testing.expectEqual(@as(u32, 113), p.prefill_rows);
    try testing.expectEqual(@as(u32, 147), p.decode_rows);
    try testing.expectEqual(@as(u64, 75_741_338_100), adm.transition_start_bytes);
    try testing.expectEqual(@as(u64, 108_921_111_644), adm.physical_bound_bytes);
    try testing.expectEqual(@as(u64, 78_934_781_952), adm.final_bank_bytes);
    std.debug.print("dsv41 arm on the real bank: {d} prefill / {d} decode rows per layer, slot banks {d} B, modeled peak {d} B\n", .{
        p.prefill_rows, p.decode_rows, adm.final_bank_bytes, p.plan.peak_fill.?.modeled_peak_bytes,
    });
}

test "dsv41 arm: every construction refusal is named" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    const bare = try TestModel.create(false);
    defer bare.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const a = testing.allocator;
    const io = std.testing.io;
    var diag: Diag = .{};
    var o = tm.options();
    o.model_dir = "relative/model";
    try testing.expectError(error.ConfigMissing, TraceArm.init(a, io, &g, {}, o, &diag));
    o = tm.options();
    o.implemented.hidden = 128;
    try testing.expectError(error.ConfigBankMismatch, TraceArm.init(a, io, &g, {}, o, &diag));
    o = tm.options();
    o.baseline_bytes = null;
    try testing.expectError(error.BaselineMissing, TraceArm.init(a, io, &g, {}, o, &diag));
    try testing.expectError(error.ManifestMissing, TraceArm.init(a, io, &g, {}, bare.options(), &diag));
    o = tm.options();
    o.baseline_bytes = 60_000_000_000;
    try testing.expectError(error.PrefillDoesNotFit, TraceArm.init(a, io, &g, {}, o, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "PrefillDoesNotFit") != null);
    o = tm.options();
    o.fixed_rows = 83;
    try testing.expectError(error.InvalidFixedRows, TraceArm.init(a, io, &g, {}, o, &diag));
}

test "dsv41 arm: the stand-in routes every layer of every forward through the hook and flushes" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    const arm = try TraceArm.init(testing.allocator, std.testing.io, &g, {}, tm.options(), &diag);
    defer arm.deinit();
    var d = StandIn(TraceArm).init(7, 3, 2);
    d.bind(&g);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(testing.allocator);
    // 7 prompt tokens in forwards of 3 rows: 3 forwards.
    const primary = try d.prefill(arm, &g, &.{ 1, 2, 3, 4, 5, 6, 7 });
    try testing.expect(primary < arm.config.vocab_size);
    try testing.expectEqual(@as(u64, 3 * 5), arm.stream.stats().route_calls);
    try arm.grow(&g);
    try testing.expect(!try d.cycle(arm, &g, testing.allocator, &out));
    try testing.expect(try d.cycle(arm, &g, testing.allocator, &out));
    const st = d.stats();
    try testing.expectEqual(@as(u32, 2), st.cycles);
    try testing.expectEqual(@as(u32, 1 + @as(u32, @intCast(out.items.len))), st.generated_tokens);
    try testing.expect(out.items.len >= 2 and out.items.len <= 6);
    const ss = arm.stream.stats();
    try testing.expectEqual(@as(u64, 5 * 5), ss.route_calls);
    // 4 rows hold all 4 experts: every miss is a first sight, read once into a persistent row.
    try testing.expect(ss.persistent_loads >= 5 and ss.persistent_loads <= 5 * 4);
    try testing.expectEqual(@as(u64, 0), ss.transient_loads);
    try testing.expectEqual(ss.persistent_loads, ss.expert_cache_misses);
    try testing.expectEqual(ss.persistent_loads * 2880, ss.expert_bytes_read);
}
