//! The DeepSeek-V4.1 streamed-expert module's native memory bill (NATIVE): each phase's terms, the fill that
//! takes slot rows up to a box's target, the load requirement upstream's preflight bills, and the per-phase
//! memory record. Production code, imported by the module and the harnesses alike; no environment reads (the
//! harnesses pass their window's numbers explicitly).

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const mdl = @import("deepseek_v41_model.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_stream = @import("expert_stream.zig");
const exl3 = @import("exl3_quant.zig");
const engram = @import("deepseek_v41_engram.zig");
const status = @import("status.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const module = @import("deepseek_v41_module.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_admission = @import("expert_admission.zig");
const graph = @import("deepseek_v41_graph.zig");

const log = std.log.scoped(.dsv41);

/// ASSUMPTION the bill rests on: the step creates no page cache. The guard credits only the file cache present
/// at its start and does not count speculative pages until the kernel ages them into inactive, so page cache
/// the step creates is unbilled memory that can land at any later allocation (v6c2: 15 GB of it from
/// construction, 7.7 GB aged in at the grow). The harnesses assert it (`checkPageCache`), and each
/// phase record carries `file_cache_created_bytes` and `box_speculative_bytes`.
///
/// Every prompt pass is billed at the prompt rows: the first one before the phase change grows the banks,
/// every later one after the served path returned them to the prompt rows (the arm's shrink, proven by
/// `Module.reclaimShrink` before the prompt allocates), so max(prompt total, decode total) bounds every request.
///
/// The cell's memory bill (decimal bytes), each term by construction from the bank's headers, the
/// admission the module builds with (`Module.armOptions` at the same config) and the arch's prefill
/// bill (`v41.PrefillBill`, its wave pinned by the served 16K trace test): the prompt phase and the
/// decode phase over the box baseline. `processBound` is what the child may hold above the baseline.
pub const Bill = struct {
    baseline: u64,
    /// The slot banks' geometry: routed layers, the transient rows (the prompt's: max_route_ids x wide depth; decode's:
    /// `transientDecodeRows` at the release route the Module installs, `module.transientRelease`), the layer's experts
    /// (the rows' cap).
    layers: u32 = 0,
    transient_rows: u64 = 0,
    transient_decode_rows: u64 = 0,
    /// The variant this bill was built at, and the prompt wave the tight variant bills (the conservative arm's judge
    /// compares the measured prompt transient against it; equal to `prefill_wave` without the model's fence).
    variant: BillVariant = .conservative,
    prefill_wave_tight: u64 = 0,
    n_experts: u32 = 0,
    prefill_rows: u32,
    decode_rows: u32,
    /// (layers x rows + the transient bank's rows: one max_route_ids window per wide read in flight) x the
    /// bank's record.
    slot_prefill: u64,
    slot_decode: u64,
    lookahead_staging: u64,
    /// Every resident tensor the index names (trunk, head, embedding, the DSpark head); the
    /// embedding leaves the device at the prompt fence (decode phase).
    residents: u64,
    embedding: u64,
    /// The Engram sidecar's residents and its row caches (host).
    engram: u64,
    /// The prompt pass's transient: K16's layer-major wave + the wide lane's routed-output copy
    /// (`PrefillBill.layerMajorBilledBytes`), or the chunk-major widest wave x 5 / 4.
    prefill_wave: u64,
    /// The request's bounded KV for prompt + max_tokens + a block, per phase: every lane at its cap, the window ring
    /// at its widest in the phase (`v41.PrefillBill.kvPromptBytes`, `kvDecodeBytes`).
    kv: u64,
    kv_decode: u64,
    /// The served tier's prefill allocator cache (4 GiB, D5) and the decode charge.
    prefill_cache: u64,
    decode_cache: u64,
    /// A verify forward's wave (8 rows) with its index chain over every position, and the draft block's.
    decode_wave: u64,
    draft_wave: u64,
    /// The admission's host reserve (pools, tables, the token map, the process).
    host_reserve: u64,
    /// The wide read schedule's depth window (the admission's `wide_window_bytes`: process lifetime).
    wide_window: u64 = 0,
    /// The process overhead no term above names (`unbilled_process_overhead_bytes`), in the prompt phase.
    unbilled_overhead: u64 = unbilled_process_overhead_bytes,
    /// The input embedding reads its host rows from construction (`embedding_host_rows`, default on): the
    /// device table is freed after the install warm-up, so no phase holds it.
    embedding_host_rows: bool = false,
    /// What the prompt pass leaves alive through decode beyond the KV: the DSpark seed's retained state, as
    /// the loop states it (`dsl.seedRetainedBytes`; today a view of the whole prompt's main taps and each draft
    /// stage's window a view of its whole-prompt main KV, 1.11 GB at 16K; v6b measured +1.30 GB persistent
    /// after the prompt). Decode phase only (inside the prompt wave's kept state during the pass).
    prompt_state: u64 = 0,
    /// ENGRAM=prefetch's posted gathers (`engramPostedBytes`: one Engram slot's ids and records, host), prompt
    /// phase only; 0 when the route is off.
    engram_posted: u64 = 0,

    pub fn prefillTotal(b: Bill) u64 {
        return b.baseline + b.prefillTerms().sum();
    }

    pub fn decodeTotal(b: Bill) u64 {
        return b.baseline + b.decodeTerms().sum();
    }

    /// The prompt phase's process terms (the prompt pass's peak: every term live at once).
    pub fn prefillTerms(b: Bill) PhaseTerms {
        return .{ .slot_banks = b.slot_prefill, .lookahead_staging = b.lookahead_staging, .residents = if (b.embedding_host_rows) b.residents - b.embedding else b.residents, .engram = b.engram, .waves = b.prefill_wave, .kv = b.kv, .mlx_cache = b.prefill_cache, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .unbilled_overhead = b.unbilled_overhead, .engram_posted = b.engram_posted };
    }

    /// The decode phase's process terms (the embedding off at the fence; the larger of the verify and draft waves:
    /// a round drafts, then verifies, so the two never hold their transients at once).
    pub fn decodeTerms(b: Bill) PhaseTerms {
        return .{ .slot_banks = b.slot_decode, .lookahead_staging = b.lookahead_staging, .residents = b.residents - b.embedding, .engram = b.engram, .waves = @max(b.decode_wave, b.draft_wave), .kv = b.kv_decode, .mlx_cache = b.decode_cache, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .prompt_state = b.prompt_state };
    }

    /// What the constructed module holds before any request (after the install warm-up released its
    /// buffers and the allocator cache): the prompt phase's persistent terms, no wave, no KV, no cache.
    pub fn constructionTerms(b: Bill) PhaseTerms {
        var t = b.prefillTerms();
        t.waves = 0;
        t.kv = 0;
        t.mlx_cache = 0;
        t.engram_posted = 0;
        return t;
    }

    pub fn processBound(b: Bill) u64 {
        return @max(b.prefillTotal(), b.decodeTotal()) - b.baseline;
    }
};

/// One phase's billed process terms (decimal bytes; the box baseline apart), as the receipt records them.
pub const PhaseTerms = struct {
    slot_banks: u64 = 0,
    lookahead_staging: u64 = 0,
    residents: u64 = 0,
    engram: u64 = 0,
    /// The prompt wave (prompt phase) or the verify + draft waves (decode phase).
    waves: u64 = 0,
    kv: u64 = 0,
    mlx_cache: u64 = 0,
    host_reserve: u64 = 0,
    wide_window: u64 = 0,
    unbilled_overhead: u64 = 0,
    /// The retained prompt state (decode phase).
    prompt_state: u64 = 0,
    /// ENGRAM=prefetch's posted gathers (prompt phase).
    engram_posted: u64 = 0,

    pub fn sum(t: PhaseTerms) u64 {
        var n: u64 = 0;
        inline for (@typeInfo(PhaseTerms).@"struct".field_names) |name| n += @field(t, name);
        return n;
    }
};

/// One phase boundary's memory record (NATIVE; probes at the four boundaries only: module constructed,
/// end of the prompt pass, after the phase change, end of decode): the phase's billed terms, the
/// process ledgers (the guard's footprint and its split), MLX's allocator (active, cache, the peak since
/// the previous boundary), the box's pages as the guard reads them, and billed minus measured.
pub const PhaseMemory = struct {
    phase: []const u8,
    billed: PhaseTerms,
    billed_process_bytes: u64,
    process: status.ProcessMemory,
    mlx_active_bytes: u64,
    mlx_cache_bytes: u64,
    mlx_peak_bytes: u64,
    box_physical_used_bytes: u64,
    box_file_backed_bytes: u64,
    /// Speculative (read-ahead) pages: not in the guard's used count until the kernel ages them into
    /// inactive, which it can do at any later allocation (the v6c2 / SERVED kills: 7.7 GB at the grow).
    box_speculative_bytes: u64 = 0,
    /// The page cache the step created: file-backed pages now less at the step's vm start (the bill assumes 0).
    file_cache_created_bytes: i64 = 0,
    /// The phase's billed process bytes less its measured footprint high-water mark (negative: over the bill).
    residual_bytes: i64,
    /// Billed MLX-device terms (slots, residents, Engram residents, waves, KV) less MLX's peak over the phase.
    mlx_residual_bytes: i64,
    /// The phase change's boundary only: how long the driver took to reclaim the frees before the grow.
    settle_ms: ?u32 = null,
};

/// The boundary's record from what the kernel and MLX already track (no new counter): reads the
/// ledgers and MLX's allocator, then restarts both high-water marks for the next phase.
/// `engram_host_bytes`: the part of the billed Engram term that is host memory (0 since the host side is billed
/// as measured: the Engram term holds its device residents only).
pub fn phaseMemory(phase: []const u8, billed: PhaseTerms, engram_host_bytes: u64, file_backed_start: u64) PhaseMemory {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    const pm = status.processMemory();
    const v = status.vmBytes();
    _ = mlx.mlx_reset_peak_memory();
    status.startFootprintInterval();
    return recordOf(phase, billed, pm, active, cache, peak, status.physicalUsedBytes(v), v.external, v.speculative, file_backed_start, engram_host_bytes);
}

/// `phaseMemory`'s arithmetic (host-testable): the MLX-device share of the bill is every term but the
/// host ones (lookahead staging, host reserve, the wide window's host records, the overhead, the
/// Engram row caches) and the allocator cache.
pub fn recordOf(phase: []const u8, billed: PhaseTerms, pm: status.ProcessMemory, active: u64, cache: u64, peak: u64, physical: u64, file_backed: u64, speculative: u64, file_backed_start: u64, engram_host_bytes: u64) PhaseMemory {
    const process = billed.sum();
    const measured = @max(pm.footprint_interval_peak, pm.footprint);
    const device = billed.slot_banks + billed.residents + (billed.engram -| engram_host_bytes) + billed.waves + billed.kv;
    return .{
        .phase = phase,
        .billed = billed,
        .billed_process_bytes = process,
        .process = pm,
        .mlx_active_bytes = active,
        .mlx_cache_bytes = cache,
        .mlx_peak_bytes = @max(peak, active),
        .box_physical_used_bytes = physical,
        .box_file_backed_bytes = file_backed,
        .box_speculative_bytes = speculative,
        .file_cache_created_bytes = @as(i64, @intCast(file_backed)) - @as(i64, @intCast(file_backed_start)),
        .residual_bytes = @as(i64, @intCast(process)) - @as(i64, @intCast(measured)),
        .mlx_residual_bytes = @as(i64, @intCast(device)) - @as(i64, @intCast(@max(peak, active))),
    };
}

pub fn printPhaseMemory(a: std.mem.Allocator, r: PhaseMemory) void {
    const json = std.json.Stringify.valueAlloc(a, r, .{}) catch return;
    std.debug.print("NATIVE DSV41_PHASE_MEMORY {s}\n", .{json});
}

/// The measured process overhead the named terms do not cover, PROMPT PHASE ONLY: calibrated from the
/// served cells' peak phys_footprint over their own bill's bound (fastest 20260929-152450: 78.294 vs 77.657
/// GB = 0.637; standard 20260929-153540: 77.139 vs 76.591 = 0.548), the larger, rounded up. Those peaks were
/// decode peaks, and what they measured there is now attributed: the retained prompt state (`prompt_state`,
/// 1.11 GB at 16K; pass3ak's decode: MLX 1.46 GB above the constructed module after the prompt, the host side
/// 0.38-0.59 GB against 1.26 billed with this term), so the decode phase no longer carries it (pass3ak's decode
/// residual was +1.30 GB with both). The prompt phase keeps it: its host side measured 1.64-1.87 GB against
/// 1.26 billed without it.
pub const unbilled_process_overhead_bytes: u64 = 640_000_000;

/// The process's host side (its footprint less MLX's active and cache: the read pool and its staging, the
/// lookahead staging, the Engram row caches, the tables, the process itself), billed as measured with a
/// 0.3 GB margin in place of the named host terms (lookahead staging, row caches, host reserve, the wide
/// window's second transient window, the prompt phase's unattributed overhead: 1.90 GB together). pass3am (v7,
/// served-cell-typical-fastest-20260930-065643): 0.31 GB after construction; host and cache together 0.59 GB
/// at the prompt pass's footprint peak (MLX 104.90, footprint 105.49 GB); 0.50-0.59 GB in decode; 1.93 GB at the
/// prompt's end, after its waves were freed (footprint 96.1 GB, far under the peak). MLX active equals the
/// device terms without the wide window at every boundary, so that window is no device memory either.
pub const measured_host_side_bytes: u64 = 900_000_000;

/// The bill's variant (DSV41_BILL_VARIANT=conservative|tight, read where the bill is built, at construction): `tight`
/// bills the main taps' chunk fences (one live stream in the K16 routed group) when the model declares them
/// (`deepseek_v41_model.main_taps_in_chunk_fence`); `conservative` (the default) keeps the four streams. SERVED16 runs a
/// tight arm only after the conservative arm's measured prompt transient sits a gigabyte under the tight wave.
pub const BillVariant = enum { conservative, tight };

pub fn billVariant() error{BillVariantUnknown}!BillVariant {
    return parseBillVariant(if (std.c.getenv("DSV41_BILL_VARIANT")) |v| std.mem.span(v) else null);
}

pub fn parseBillVariant(v: ?[]const u8) error{BillVariantUnknown}!BillVariant {
    const s = v orelse return .conservative;
    return std.meta.stringToEnum(BillVariant, s) orelse error.BillVariantUnknown;
}

/// Whether this tree's model evaluates the main taps in their chunk fences (ee80e40's declaration).
pub const model_taps_fenced: bool = blk: {
    if (!@hasDecl(mdl, "main_taps_in_chunk_fence")) break :blk false;
    break :blk mdl.main_taps_in_chunk_fence;
};

/// Whether this tree's model also releases the second stream SERVED16 measured live at the routed group's peak with the
/// fence (the holder; the model lane declares it when released).
pub const model_group_one_stream: bool = blk: {
    if (!@hasDecl(mdl, "routed_group_one_stream")) break :blk false;
    break :blk mdl.routed_group_one_stream;
};

/// The K16 routed group's live hc-width streams the tight variant bills: four without the fence, two with it (SERVED16's
/// measured drop), one once the holder is declared released.
pub fn tightGroupStreams(fenced: bool, one_stream: bool) u64 {
    if (!fenced) return 4;
    return if (one_stream) 1 else 2;
}

/// Decode's own staging rows beside window 0 after the release (`expert_stream.decode_staging_rows`, declared with the
/// release; 0 without the declaration).
pub const stream_decode_staging_rows: u64 = blk: {
    if (!@hasDecl(expert_stream, "decode_staging_rows")) break :blk 0;
    break :blk expert_stream.decode_staging_rows;
};

/// Decode's transient rows: once the phase change releases the prompt's windows (the whole scratch freed, then window 0
/// reallocated: decode's calls take at most max_route_ids ids), window 0 plus decode's staging rows; else every window
/// the prompt's wide reads allocated. `releases` is the route the Module installs (`module.transientRelease`: the
/// stream's capability and the request's setting over the default, off since SERVED17), which `billAt` resolves from
/// the same overrides the Module builds with, so one binary bills both arms.
pub fn transientDecodeRows(wide_depth: u8, releases: bool, staging_rows: u64) u64 {
    return if (releases) xp.max_route_ids + staging_rows else @as(u64, wide_depth) * xp.max_route_ids;
}

/// The bill at `config`'s rows (both set: the native rows; `expert_rows` alone: the Python-paired forced-rows
/// admission a harness asks for) for a request of `prompt_tokens` + `max_tokens`. `wired_bytes` pins the wired
/// bytes `planRows` reads (null: now). A bill taken after construction passes the wired bytes the arm was
/// planned with (`arm.inputs.wired_bytes`), never a live read: the module's own banks are wired by then.
pub fn billAt(a: std.mem.Allocator, io: std.Io, config: *const model.ModelConfig, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
    var vd: v41.Diag = .{};
    errdefer if (vd.len > 0) log.err("bill: {s}", .{vd.message()});
    const c = try v41.Config.load(a, io, dir, &vd);
    // The box: the caller's ceiling, passed explicitly (the Module's own, a harness's window ceiling, the load
    // preflight's upstream static ceiling): no hidden global, and a host-side bill never queries the device.
    const ceiling = module.boxCeiling(ceiling_bytes, c.n_routed_experts);
    var diag: arm_mod.Diag = .{};
    var opts = module.armOptions(config, ceiling, .host);
    if (wired_bytes) |w| opts.wired_bytes = w;
    var p = arm_mod.planRows(a, io, opts, &diag) catch |e| {
        log.err("bill: refused: {s}", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    defer if (p.draft_subset) |*x| x.deinit();
    const rec = p.inputs.record_bytes;
    // The stream's transient bank, allocated whole at construction: one window of max_route_ids rows per wide
    // read the stream holds in flight (the arm's `.transient_rows = wide_depth x max_route_ids`). Billing one
    // window left 48 records (0.64 GB) of device memory unbilled after c47001e folded the second window's named
    // term into the host side; 9b's construction hid it behind ~0.64 GB of draft-head residents that load at the
    // first draft block, and SERVED10b's draft-block warm-up showed it (MLX active +642,935,748 B).
    const transient: u64 = @as(u64, opts.wide_depth) * xp.max_route_ids;
    // Decode's: the release route the Module installs from these overrides (`Module.init` sets the stream's
    // `transient_release` from the same resolver after `armOptions`).
    const transient_decode = transientDecodeRows(opts.wide_depth, module.transientRelease(ov), stream_decode_staging_rows);
    var ck = try v41.Checkpoint.openIndexed(a, io, dir, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    const epath = try std.fmt.allocPrint(a, "{s}/engram/engram-residents.safetensors", .{dir});
    var eck = try v41.Checkpoint.openFile(a, epath, &vd);
    defer eck.deinit();
    const em = try v41.WeightMap.build(a, try v41.engramSpec(a, &c), &eck, &vd);
    // JOINLESS (the served default): the routed group's joined input is the minimal copy's bound (`joinedBytes`).
    const joinless = ov.prefill_joinless orelse module.numericTier(.served).routes.prefill_joinless;
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids };
    const variant = try billVariant();
    const bill = v41.PrefillBill.of(&c).withIndexLaunch(try module.prefillIndexRoute(config, ov)).withJoinless(if (joinless) shape else null).withGroupStreams(if (variant == .tight) tightGroupStreams(model_taps_fenced, model_group_one_stream) else 4);
    const positions = prompt_tokens + max_tokens + mdl.Model(ops.MlxOps).scratch_rows;
    const rows: u64 = mdl.Model(ops.MlxOps).scratch_rows;
    // A verify forward: the fixed wave at 8 rows plus its index chain over every position (two arrays live).
    const decode_wave = bill.waveBytes(rows, rows, .served) + v41.PrefillBill.chain_copies * rows * bill.index_heads * positions * 4;
    return .{
        // Unset (a shell without a box baseline): the process terms alone.
        .baseline = config.memory_baseline_bytes orelse 0,
        .layers = c.n_layers,
        .transient_rows = transient,
        .transient_decode_rows = transient_decode,
        .n_experts = c.n_routed_experts,
        .prefill_rows = p.prefill_rows,
        .decode_rows = p.decode_rows,
        .slot_prefill = (@as(u64, c.n_layers) * p.prefill_rows + transient) * rec,
        .slot_decode = (@as(u64, c.n_layers) * p.decode_rows + transient_decode) * rec,
        // The host side is billed as measured (`measured_host_side_bytes`, in host_reserve).
        .lookahead_staging = 0,
        .residents = m.totalBytes() - droppedResidentBytes(&m, headRoute(ov)) + builtResidentBytes(&c, headRoute(ov)),
        .embedding = m.bytes_by_module[@backingInt(v41.Module.embed)],
        .engram = em.totalBytes(),
        // K16 (the layer-major route) bills its own wave (every chunk's kept state + one sub-wave). With
        // JOINLESS (the served default) the combine reads the DIG-X waves' own outputs: no wide-lane copy, and
        // the wave alone covers the pass, its routed group's joined input at the minimal copy's bound
        // (`PrefillBill.joinedBytes`: 63 / 86 of the routed rows at 16K, the most outputs a call can make);
        // without it, the wide lane's routed-output copy. The chunk-major wave keeps its x 5/4 margin.
        .prefill_wave = promptWave(bill, config.dsv41LayerMajor(), joinless, prompt_tokens),
        .variant = variant,
        .prefill_wave_tight = promptWave(bill.withGroupStreams(tightGroupStreams(model_taps_fenced, model_group_one_stream)), config.dsv41LayerMajor(), joinless, prompt_tokens),
        .kv = bill.kvPromptBytes(prompt_tokens, positions),
        .kv_decode = bill.kvDecodeBytes(prompt_tokens, positions),
        .prefill_cache = module.prefillCacheLimit(.served),
        .decode_cache = expert_admission.Envelope.dsv41_pass2.decode_cache_bytes,
        .decode_wave = decode_wave,
        .draft_wave = decode_wave,
        .host_reserve = measured_host_side_bytes,
        .wide_window = 0,
        .unbilled_overhead = 0,
        .embedding_host_rows = config.embedding_host_rows orelse true,
        .prompt_state = dsl.seedRetainedBytes(&c, prompt_tokens),
        .engram_posted = if (engramPostedRoute(config, ov, &c)) engramPostedBytes(c.engram, prompt_tokens) else 0,
    };
}

/// The prompt pass's billed transient: K16's layer-major wave (JOINLESS: the wave alone; else with the wide lane's
/// routed-output copy), or the chunk-major widest wave x 5 / 4.
fn promptWave(bill: v41.PrefillBill, layer_major: bool, joinless: bool, prompt_tokens: u64) u64 {
    if (!layer_major) return bill.waveBytes(bill.chunkRows(prompt_tokens), prompt_tokens, .served) / 4 * 5;
    return if (joinless) bill.layerMajorWaveBytes(prompt_tokens, .served) else bill.layerMajorBilledBytes(prompt_tokens, .served);
}

/// The head codec the request's model builds: the override's, else the served tier's (`RouteOverrides.head_mode`).
fn headRoute(ov: module.RouteOverrides) graph.Routes.Head {
    return ov.head_mode orelse module.numericTier(.served).routes.head;
}

/// Device bytes the model builds at construction beyond the checkpoint's residents (`Model.builtBytes`, computed before
/// construction from the same formulas): HEAD_MODE mxfp8's codes and scales (vocab x hidden x 33 / 32), and W97's dense
/// f32 wo_a per layer when the served tier routes it (off today).
pub fn builtResidentBytes(c: *const v41.Config, head: graph.Routes.Head) u64 {
    var n: u64 = 0;
    if (module.numericTier(.served).routes.wo_a_f32) n += @as(u64, c.n_layers) * graph.woaDenseBytes(c);
    if (head == .mxfp8) n += @as(u64, c.vocab_size) * c.hidden_size * 33 / 32;
    return n;
}

/// Checkpoint residents the Module drops once the model is built (`Model.droppedBytes`): the dense bf16 head under
/// HEAD_MODE mxfp8, its bytes as the resident map holds them (`head.weight`).
pub fn droppedResidentBytes(m: *const v41.WeightMap, head: graph.Routes.Head) u64 {
    return if (head == .mxfp8) m.bytes_by_module[@backingInt(v41.Module.head)] else 0;
}

/// ENGRAM=prefetch (the served tier's `engram_posted` route, dsv41-engram-prefetch b198dbd): the K16 prompt pass
/// posts each Engram layer slot's gathers ahead of the layer that reads them and holds one slot's at a time
/// (released after that slot's layer, before the next slot's are posted): every prompt position's hashed row ids
/// (i64) and records (the mxfp8 codes and their E8M0 scales, `eng.Bank.record_bytes`). Host memory (the row
/// source's allocator), prompt phase only. The two poster threads' stacks (256 KB each) are not billed.
pub fn engramPostedBytes(e: v41.Engram, prompt_tokens: u64) u64 {
    const record: u64 = @as(u64, e.head_dim) + e.head_dim / 32;
    return prompt_tokens * e.hashCols() * (record + @sizeOf(i64));
}

/// Whether `config`'s prompt pass posts its Engram gathers: the K16 pass over a bank with Engram layers, the
/// route set (the harness's override, else the served tier's).
fn engramPostedRoute(config: *const model.ModelConfig, ov: module.RouteOverrides, c: *const v41.Config) bool {
    if (!config.dsv41LayerMajor() or c.engram.n_layers == 0) return false;
    return ov.engram_posted orelse module.numericTier(.served).routes.engram_posted;
}

/// The fill for `config`'s routes: the bill at the floor rows (both phases' rows-free totals by
/// construction; no admission of another kind), then `fillRows` up to `target` (the caller's box: the served
/// Module's is the GPU ceiling less upstream's wired margin, a harness's the guard's ceiling less its stop).
/// `wired_bytes` as `billAt`.
pub fn fill(a: std.mem.Allocator, io: std.Io, config: model.ModelConfig, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, target: u64, ov: module.RouteOverrides) !arm_mod.NativeRows {
    const b0 = try billAtFloor(a, io, config, prompt_tokens, max_tokens, wired_bytes, ceiling_bytes, ov);
    return fillRows(fillBillOf(b0), target, b0.n_experts);
}

/// A bill in the fill's shape: its phases' totals less their slot rows, and one row on every routed layer.
pub fn fillBillOf(b: Bill) FillBill {
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows);
    const per_row = @as(u64, b.layers) * rec;
    return .{
        .prefill_fixed = b.prefillTotal() - b.prefill_rows * per_row,
        .decode_fixed = b.decodeTotal() - b.decode_rows * per_row,
        .per_row = per_row,
    };
}

/// The bill at the fill's floor rows (`min_fill_rows` in both phases).
fn billAtFloor(a: std.mem.Allocator, io: std.Io, config: model.ModelConfig, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    var c = config;
    c.expert_rows = min_fill_rows;
    c.expert_prefill_rows = min_fill_rows;
    return billAt(a, io, &c, prompt_tokens, max_tokens, wired_bytes, ceiling_bytes, ov);
}

/// What the module needs free to load at all, for upstream's load preflight (`scheduler.loadRequirementBytes`
/// is fed this in place of the shards' disk bytes): the process bound of the standard request's bill at the
/// fill's floor rows. The fill then takes rows up to the box's target; below the floor it refuses by name.
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, config: model.ModelConfig, ceiling_bytes: u64) !u64 {
    var c = config;
    c.memory_baseline_bytes = 0;
    // The server's load preflight: the served routes, no harness override.
    const b = try billAtFloor(a, io, c, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
    return b.processBound();
}

/// A native bill in the fill's shape: each phase's billed bytes (the box baseline included) without its
/// persistent slot rows, and one row on every routed layer (layers x the record); a phase's total at
/// r rows is `fixed + r * per_row`.
pub const FillBill = struct { prefill_fixed: u64, decode_fixed: u64, per_row: u64 };

/// The native admission's fill: the most decode rows and the most prompt rows (prompt <= decode <= the
/// layer's experts) whose phase totals each stay under `target` (the caller's box: the ceiling less its margin). The slot banks
/// hold the prompt rows through the prompt pass and grow to the decode rows at the phase change, which
/// grows only after the prompt's frees are proven complete (`Module.phaseChange`), so the process bound is
/// max(prompt total, decode total) with no transition term. Refused by name under `min_fill_rows`.
pub fn fillRows(b: FillBill, target: u64, n_experts: u32) error{NativeBillDoesNotFit}!arm_mod.NativeRows {
    const most = struct {
        fn f(fixed: u64, t: u64, per_row: u64) u64 {
            return if (fixed >= t) 0 else (t - fixed) / per_row;
        }
    }.f;
    const decode = @min(most(b.decode_fixed, target, b.per_row), n_experts);
    const prefill = @min(most(b.prefill_fixed, target, b.per_row), decode);
    if (prefill < min_fill_rows) return error.NativeBillDoesNotFit;
    return .{ .prefill = @intCast(prefill), .decode = @intCast(decode) };
}

/// The request the served admission's fill bills: the standard 16K cell's prompt and token cap (a longer
/// request is admitted, or refused by name, by the server's per-request prefill bill at its time).
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;

/// The fewest rows per layer the fill admits (the envelope admission's prefill floor).
pub const min_fill_rows = 16;


/// Both phases' billed totals within the fill's target (the fill guarantees it; forced rows are checked
/// here), once, before construction: the grow at the phase change is then admitted by construction.
pub fn admitPhases(b: Bill, target: u64) error{ PromptOverTarget, DecodeOverTarget }!void {
    if (b.prefillTotal() > target) return error.PromptOverTarget;
    if (b.decodeTotal() > target) return error.DecodeOverTarget;
}


// ── Tests ──

const testing = std.testing;

/// cell4's bill (served-cell-typical-fastest-20260929-195452: 106 / 148 rows, the 8.716 GB non-file
/// baseline), term by term in bytes as the bill built it on the bank then (a test fixture).
pub fn cell4Bill() Bill {
    const rec: u64 = 13_315_584;
    return .{
        .baseline = 8_716_419_072,
        .layers = 40,
        .transient_rows = 48,
        .transient_decode_rows = 48,
        .n_experts = 384,
        .prefill_rows = 106,
        .decode_rows = 148,
        .slot_prefill = (40 * 106 + 48) * rec,
        .slot_decode = (40 * 148 + 48) * rec,
        .lookahead_staging = 54_460_416,
        .residents = 17_680_000_000,
        .embedding = 1_323_827_200,
        .engram = 480_000_000,
        .prefill_wave = 17_995_900_000,
        .kv = 160_000_000,
        .kv_decode = 160_000_000,
        .prefill_cache = 4_294_967_296,
        .decode_cache = 270_000_000,
        .decode_wave = 350_000_000,
        .draft_wave = 350_000_000,
        .host_reserve = 408_944_640,
        .wide_window = 48 * rec,
    };
}

test "dsv41 memory: a phase's total is the baseline plus its terms; the construction terms drop the wave, the KV and the cache" {
    const b = cell4Bill();
    try testing.expectEqual(b.baseline + b.slot_prefill + b.lookahead_staging + b.residents + b.engram + b.prefill_wave + b.kv + b.prefill_cache + b.host_reserve + b.unbilled_overhead + b.wide_window, b.prefillTotal());
    // The decode phase carries no unbilled overhead (what it covered there is the retained prompt state).
    try testing.expectEqual(b.baseline + b.slot_decode + b.lookahead_staging + b.residents - b.embedding + b.engram + b.kv_decode + @max(b.decode_wave, b.draft_wave) + b.decode_cache + b.host_reserve + b.wide_window + b.prompt_state, b.decodeTotal());
    try testing.expectEqual(@as(u64, 0), b.decodeTerms().unbilled_overhead);
    const c = b.constructionTerms();
    try testing.expectEqual(b.prefillTerms().sum() - b.prefill_wave - b.kv - b.prefill_cache, c.sum());
    // cell4's constructed footprint (76.41 GB) sits under its construction terms (77.00 GB).
    try testing.expect(c.sum() > 76_410_000_000 and c.sum() < 77_100_000_000);
}

test "dsv41 memory: with the embedding on its host rows no phase bills the device table, and the one-count fill gains 2 rows" {
    var dev = cell4Bill();
    dev.embedding_host_rows = false;
    var host = dev;
    host.embedding_host_rows = true;
    try testing.expectEqual(dev.prefillTotal() - dev.embedding, host.prefillTotal());
    try testing.expectEqual(dev.decodeTotal(), host.decodeTotal());
    try testing.expectEqual(dev.constructionTerms().sum() - dev.embedding, host.constructionTerms().sum());
    const per_row = @as(u64, dev.layers) * 13_315_584;
    const at = struct {
        fn f(b: Bill, base: u64, pr: u64) FillBill {
            return .{ .prefill_fixed = b.prefillTotal() - b.baseline + base - @as(u64, b.prefill_rows) * pr, .decode_fixed = b.decodeTotal() - b.baseline + base - @as(u64, b.decode_rows) * pr, .per_row = pr };
        }
    }.f;
    const r_dev = try fillRows(at(dev, 9_200_000_000, per_row), 120_259_084_288 - module.ceiling_stop_bytes, 384);
    const r_host = try fillRows(at(host, 9_200_000_000, per_row), 120_259_084_288 - module.ceiling_stop_bytes, 384);
    try testing.expectEqual(r_dev.prefill + 2, r_host.prefill);
}

test "dsv41 memory: ENGRAM=prefetch's posted gathers are one slot's ids and records, billed in the prompt phase only" {
    // The 3.0 bank's Engram geometry: 24 columns (n-gram orders 2..4 x 8 heads), 256 code bytes + 8 scale bytes.
    const e: v41.Engram = .{ .n_layers = 2, .max_ngram_size = 4, .n_heads = 8, .head_dim = 256 };
    try testing.expectEqual(@as(u32, 24), e.hashCols());
    // 16,384 positions x 24 columns x (264 record + 8 id) bytes: 107 MB at the fill's request.
    try testing.expectEqual(@as(u64, 106_954_752), engramPostedBytes(e, fill_prompt_tokens));
    const off = cell4Bill();
    var on = off;
    on.engram_posted = engramPostedBytes(e, fill_prompt_tokens);
    try testing.expectEqual(off.prefillTotal() + on.engram_posted, on.prefillTotal());
    try testing.expectEqual(off.decodeTotal(), on.decodeTotal());
    try testing.expectEqual(off.constructionTerms().sum(), on.constructionTerms().sum());
    try testing.expectEqual(on.engram_posted, on.prefillTerms().engram_posted);
    // The fill's shape carries it in the prompt phase alone.
    try testing.expectEqual(fillBillOf(off).prefill_fixed + on.engram_posted, fillBillOf(on).prefill_fixed);
    try testing.expectEqual(fillBillOf(off).decode_fixed, fillBillOf(on).decode_fixed);
}

test "dsv41 memory: the phase record's residuals: billed less the interval peak, billed device terms less MLX's peak" {
    const b = cell4Bill();
    // cell4's prompt boundary: footprint 83.03 GB now; MLX active 76.56, peak 91.35 GB.
    const pm: status.ProcessMemory = .{ .footprint = 83_030_000_000, .footprint_interval_peak = 97_000_000_000, .footprint_lifetime_peak = 97_000_000_000 };
    const r = recordOf("prompt pass", b.prefillTerms(), pm, 76_560_000_000, 5_000_000_000, 91_350_000_000, 110_000_000_000, 3_000_000_000, 850_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // The page cache the step created (file-backed now less at its vm start) and the speculative pages, recorded.
    try testing.expectEqual(@as(i64, 3_000_000_000 - 4_870_000_000), r.file_cache_created_bytes);
    try testing.expectEqual(@as(u64, 850_000_000), r.box_speculative_bytes);
    try testing.expectEqual(b.prefillTotal() - b.baseline, r.billed_process_bytes);
    try testing.expectEqual(@as(i64, @intCast(r.billed_process_bytes)) - 97_000_000_000, r.residual_bytes);
    const device = b.slot_prefill + b.residents + (b.engram - engram.row_cache_host_bytes) + b.prefill_wave + b.kv;
    try testing.expectEqual(@as(i64, @intCast(device)) - 91_350_000_000, r.mlx_residual_bytes);
    // A footprint above its (stale) interval peak counts as the measurement; a peak below active reads active.
    const late: status.ProcessMemory = .{ .footprint = 99_000_000_000, .footprint_interval_peak = 0 };
    const r2 = recordOf("decode", b.decodeTerms(), late, 97_000_000_000, 0, 0, 0, 19_950_000_000, 15_940_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // v6c2's construction: 15.08 GB of page cache created, 15.94 GB of it speculative.
    try testing.expectEqual(@as(i64, 15_080_000_000), r2.file_cache_created_bytes);
    try testing.expectEqual(@as(i64, @intCast(b.decodeTotal() - b.baseline)) - 99_000_000_000, r2.residual_bytes);
    try testing.expectEqual(@as(u64, 97_000_000_000), r2.mlx_peak_bytes);
    // The record serialises for the receipt.
    const json = try std.json.Stringify.valueAlloc(testing.allocator, r, .{});
    defer testing.allocator.free(json);
    try testing.expect(std.mem.indexOf(u8, json, "\"footprint_interval_peak\":97000000000") != null);
}

// DSV41_BANK=<bank> (host): the fill and the bill that checks it agree at the same inputs (v6 211422's:
// non-file 8.5487616 GB, box 119.259 GB, wired 3.380 GB); a bill re-read with the constructed module's wired
// bytes (live, +85 GB) refused v6 through the envelope planner, which the served path no longer runs.
test "dsv41 memory: the fill and its admission agree at the same inputs (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 8_548_761_600;
    const ceiling_bytes: u64 = 119_259_000_000;
    const wired: u64 = 3_380_379_648;
    const nr = try fill(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, wired, ceiling_bytes, ceiling_bytes - module.ceiling_stop_bytes, .{});
    try testing.expect(nr.prefill <= nr.decode);
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const b = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, wired, ceiling_bytes, .{});
    try testing.expectEqual(nr.prefill, b.prefill_rows);
    try testing.expectEqual(nr.decode, b.decode_rows);
    try testing.expect(b.prefillTotal() <= ceiling_bytes - module.ceiling_stop_bytes);
    try testing.expect(b.decodeTotal() <= ceiling_bytes - module.ceiling_stop_bytes);
    // The prompt phase charges the served tier's cache limit exactly (the limit it sets).
    try testing.expectEqual(@as(u64, module.prefillCacheLimit(.served)), b.prefill_cache);
    try testing.expectEqual(@as(u64, 2 << 30), b.prefill_cache);
    // The host side billed as measured, the named host terms folded into it.
    try testing.expectEqual(measured_host_side_bytes, b.host_reserve);
    try testing.expectEqual(@as(u64, 0), b.lookahead_staging + b.wide_window + b.unbilled_overhead);
    std.debug.print("\nfill and admission at v6's inputs: {d} / {d} rows, prompt total {d} B\n", .{ nr.prefill, nr.decode, b.prefillTotal() });
    // v6's failure mode is gone by construction: without the envelope planner the native bill does not read
    // the wired bytes at all (the constructed module's own +85 GB changes nothing).
    const b_live = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, wired + 85_000_000_000, ceiling_bytes, .{});
    try testing.expectEqual(b.prefillTotal(), b_live.prefillTotal());
    try testing.expectEqual(b.decodeTotal(), b_live.decodeTotal());
}

// DSV41_BANK=<bank> (host): this tree's rows at the served windows' inputs (box 120.259 GB less the guard's 2.0 GB
// stop; baselines 9.2 GB and pass3an's 9.55 GB), with the seed's copies as the retained prompt state (ac2121c:
// 847,872 B) and every transient window billed (b4473fa; P1's v1b third window: one row less than depth 2's
// 137 / 167 at 9.2 GB), the served KV lanes by owner per phase (G7 57409c7: one prompt row at 9.2 GB with the posted
// gathers on, one at 9.55 GB off), the Engram posted gathers off and on (the served tier's route).
test "dsv41 memory: this tree's fill rows at the windows' inputs, ENGRAM=prefetch's posted gathers off and on (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    try testing.expectEqual(@as(u64, 106_954_752), posted);
    const Want = struct { base: u64, off: arm_mod.NativeRows, on: arm_mod.NativeRows };
    // Wide depth 5 (P1c, 240 transient rows), the minimal copy's bound at the most outputs a call can make (63 / 86
    // of the routed rows at 16K), and the frontier as rings (3ebd8a7: -0.191 GB in the prompt, -0.211 GB in decode;
    // before it 8.99 GB 135 / 164 off, 134 / 164 on; 9.20 GB 134 / 163 both; 9.55 GB 134 / 163 off, 133 / 163 on).
    const every_window = [_]Want{
        .{ .base = 8_990_000_000, .off = .{ .prefill = 135, .decode = 164 }, .on = .{ .prefill = 135, .decode = 164 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 135, .decode = 164 }, .on = .{ .prefill = 134, .decode = 164 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 134, .decode = 163 }, .on = .{ .prefill = 134, .decode = 163 } },
    };
    // With the transient release installed (SERVED16 for every request; since SERVED17 the route,
    // DSV41_CELL_TRANSIENT_RELEASE), decode bills window 0 only: +5 decode rows at each baseline.
    const window_0 = [_]Want{
        .{ .base = 8_990_000_000, .off = .{ .prefill = 135, .decode = 169 }, .on = .{ .prefill = 135, .decode = 169 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 135, .decode = 169 }, .on = .{ .prefill = 134, .decode = 169 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 134, .decode = 168 }, .on = .{ .prefill = 134, .decode = 168 } },
    };
    // The release route as the Module resolves it: the default (off), then each override.
    for ([_]?bool{ null, false, true }) |route| {
        const ov: module.RouteOverrides = .{ .transient_release = route };
        for (if (module.transientRelease(ov)) window_0 else every_window) |w| {
            config.memory_baseline_bytes = w.base;
            var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, ov);
            // This tree's own route decision: off without the route's declarations, the served tier's with them.
            try testing.expectEqual(if (engramPostedRoute(&config, ov, &c)) posted else 0, b0.engram_posted);
            b0.engram_posted = 0;
            const off = try fillRows(fillBillOf(b0), target, b0.n_experts);
            b0.engram_posted = posted;
            const on = try fillRows(fillBillOf(b0), target, b0.n_experts);
            const name = if (route) |r| (if (r) "on" else "off") else "default";
            std.debug.print("\nrows at baseline {d:.2} GB (target {d:.3} GB, transient release {s}, {d} decode transient rows): posted gathers off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, @as(f64, @floatFromInt(target)) / 1e9, name, b0.transient_decode_rows, off.prefill, off.decode, on.prefill, on.decode });
            try testing.expectEqual(w.off, off);
            try testing.expectEqual(w.on, on);
        }
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): the bounded KV by owner at the fill's request (16,384 + 1,024 + one verify block of
// positions). SERVED11 (full-length frontier lanes) held 351,152,128 B after the prompt, the lanes, ring and frontier
// of 57409c7's bill within 0.42 MB. Since 3ebd8a7 the frontier of each ratio-2 kv source (layers 2, 8, 14) is two rings
// of window 2, so it is billed as rings, per phase.
test "dsv41 memory: the bounded KV lanes by owner, per phase, at the fill's request (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const pb = v41.PrefillBill.of(&c);
    const positions = fill_prompt_tokens + fill_max_tokens + mdl.Model(ops.MlxOps).scratch_rows;
    // Compressed 89,235,456 + index 22,308,864 (the four kv sources).
    try testing.expectEqual(@as(u64, 111_544_320), pb.laneBytes(positions));
    // The window ring: 2,160 rows over the prompt (both slots at 953 + 127), 518 at decode's first step (310 + 208).
    try testing.expectEqual(@as(u64, 174_735_360), pb.ringPromptBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 41_904_128), pb.ringDecodeBytes(fill_prompt_tokens));
    // The frontier rings (window 2, 2,048 B a row, two a source, three sources): 1,908 rows a ring over the prompt
    // (2 x (953 + 1)), 266 at decode's first step (184 + 82), against 214,106,112 B of full-length lanes.
    try testing.expectEqual(@as(u64, 954), v41.PrefillBill.ringBase(2) + 872);
    try testing.expectEqual(@as(u64, 23_445_504), pb.frontierPromptBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 3_268_608), pb.frontierDecodeBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 309_725_184), pb.kvPromptBytes(fill_prompt_tokens, positions));
    try testing.expectEqual(@as(u64, 156_717_056), pb.kvDecodeBytes(fill_prompt_tokens, positions));
    // At the phase change: the lanes, the window ring's last chunk (310 rows) and the frontier's (184 a ring).
    const at_change = pb.laneBytes(positions) + pb.ring_row_bytes * 310 + 3 * 2 * 2048 * 184;
    try testing.expectEqual(@as(u64, 138_883_072), at_change);
    // The bill carries them per phase.
    var config = try model.parseConfig(testing.io, a, bank_dir);
    // Option B: the ceiling is the bill's argument, not a config field.
    config.memory_baseline_bytes = 9_200_000_000;
    const b = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, 120_259_084_288, .{});
    try testing.expectEqual(pb.kvPromptBytes(fill_prompt_tokens, positions), b.kv);
    try testing.expectEqual(pb.kvDecodeBytes(fill_prompt_tokens, positions), b.kv_decode);
}

// DSV41_BANK=<bank> (host): the bill's transient rows are the arm's allocation. The stream allocates its transient
// bank whole at construction, one max_route_ids window per wide read in flight (Arm.init: `.transient_rows =
// wide_depth x max_route_ids`; the admission's record of the rows past the first window is `wideWindowBytes`). On
// the served tier (wide depth 5, P1c) that is 240 rows: 192 records more than the one window billed until c47001e.
test "dsv41 memory: the bill's transient rows are the arm's allocation, every window (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    // Option B: the ceiling is the harness's argument, not a config field.
    const ceiling: u64 = 120_259_084_288;
    config.memory_baseline_bytes = 9_200_000_000;
    config.expert_rows = min_fill_rows;
    config.expert_prefill_rows = min_fill_rows;
    const b = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{});
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const opts = module.armOptions(&config, module.boxCeiling(ceiling, c.n_routed_experts), .host);
    try testing.expectEqual(@as(u8, 5), opts.wide_depth);
    try testing.expectEqual(@as(u64, opts.wide_depth) * xp.max_route_ids, b.transient_rows);
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows);
    try testing.expectEqual(@as(u64, 13_315_584), rec);
    try testing.expectEqual(xp.max_route_ids * rec + arm_mod.wideWindowBytes(opts.wide_depth, rec), b.transient_rows * rec);
    try testing.expectEqual((@as(u64, b.layers) * b.prefill_rows + @as(u64, opts.wide_depth) * xp.max_route_ids) * rec, b.slot_prefill);
    // The windows past the first: 4 x 48 records, 2,556,592,128 B (the second, 639,148,032 B, was the 10b
    // construction's unbilled MLX active less ~3.8 MB; each later one is as large).
    try testing.expectEqual(@as(u64, 2_556_592_128), arm_mod.wideWindowBytes(opts.wide_depth, rec));
    // Decode's transient rows follow the route the Module installs (`module.transientRelease`): the default (off)
    // keeps every window; through billAt, the override off bills 240 rows and on bills window 0 (48, no staging rows),
    // 2,556,592,128 B apart; the prompt's transient rows are the same on both routes.
    try testing.expectEqual(transientDecodeRows(opts.wide_depth, module.transientRelease(.{}), stream_decode_staging_rows), b.transient_decode_rows);
    try testing.expect(!module.transientRelease(.{}));
    const route_off = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{ .transient_release = false });
    const route_on = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{ .transient_release = true });
    try testing.expectEqual(@as(u64, 240), route_off.transient_decode_rows);
    try testing.expectEqual(@as(u64, 48), route_on.transient_decode_rows);
    try testing.expectEqual(@as(u64, 2_556_592_128), route_off.slot_decode - route_on.slot_decode);
    try testing.expectEqual(route_off.transient_rows, route_on.transient_rows);
    try testing.expectEqual(route_off.slot_prefill, route_on.slot_prefill);
    try testing.expectEqual(route_off.prefillTotal(), route_on.prefillTotal());
    try testing.expectEqual(@as(u64, 240), transientDecodeRows(5, false, 0));
    try testing.expectEqual(@as(u64, 48), transientDecodeRows(5, true, 0));
    // Decode's staging rows ride window 0 once declared (the release's commit declares 0).
    try testing.expectEqual(@as(u64, 56), transientDecodeRows(5, true, 8));
    try testing.expectEqual(@as(u64, 2_556_592_128), (transientDecodeRows(5, false, 0) - transientDecodeRows(5, true, 0)) * rec);
    try testing.expectEqual((@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows) * rec, b.slot_decode);
}

// DSV41_BANK=<bank> (host): the bill's variants at the windows' baselines. Conservative (the default) bills the K16
// routed group's four hc-width streams; tight bills two once the model declares its main taps fenced (ee80e40:
// `main_taps_in_chunk_fence`; SERVED16 measured the second stream still live), one once it declares the holder released
// (`routed_group_one_stream`). The tight rows are computed at two streams, whether or not this tree declares the fence.
test "dsv41 memory: the bill's variants, conservative and tight, at the windows' baselines (bank)" {
    try testing.expectEqual(BillVariant.conservative, try parseBillVariant(null));
    try testing.expectEqual(BillVariant.tight, try parseBillVariant("tight"));
    try testing.expectError(error.BillVariantUnknown, parseBillVariant("loose"));
    try testing.expectEqual(@as(u64, 4), tightGroupStreams(false, false));
    try testing.expectEqual(@as(u64, 2), tightGroupStreams(true, false));
    try testing.expectEqual(@as(u64, 1), tightGroupStreams(true, true));
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids };
    const fenced = v41.PrefillBill.of(&c).withIndexLaunch(try module.prefillIndexRoute(&config, .{})).withJoinless(shape).withGroupStreams(tightGroupStreams(true, false));
    const Want = struct { base: u64, conservative: arm_mod.NativeRows, tight: arm_mod.NativeRows };
    for ([_]Want{
        // The default route (the transient release off: every window through decode, SERVED17's arm 1 and -tight); the
        // fence at two streams (-2.68 GB) adds 5 prompt rows.
        .{ .base = 8_990_000_000, .conservative = .{ .prefill = 135, .decode = 164 }, .tight = .{ .prefill = 140, .decode = 164 } },
        .{ .base = 9_200_000_000, .conservative = .{ .prefill = 134, .decode = 164 }, .tight = .{ .prefill = 139, .decode = 164 } },
        .{ .base = 9_550_000_000, .conservative = .{ .prefill = 134, .decode = 163 }, .tight = .{ .prefill = 139, .decode = 163 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        b0.engram_posted = posted;
        const cons = try fillRows(fillBillOf(b0), target, b0.n_experts);
        b0.prefill_wave = fenced.layerMajorWaveBytes(fill_prompt_tokens, .served);
        const tight = try fillRows(fillBillOf(b0), target, b0.n_experts);
        std.debug.print("\nbill variants at baseline {d:.2} GB (posted gathers on): conservative {d} / {d}, tight {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, cons.prefill, cons.decode, tight.prefill, tight.decode });
        try testing.expectEqual(w.conservative, cons);
        try testing.expectEqual(w.tight, tight);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): HEAD_MODE mxfp8 (cell arm 5) bills the head it runs: the dense bf16 head the Module drops
// after construction (1,323,827,200 B) out of the residents, its codes and scales (682,598,400 B) in, net -641,228,800 B.
// Arm 5's own fill therefore sits about a row a phase above the bf16 arm's. Both at the default route (the transient
// release off: every window through decode), as SERVED17's -mxfp8head arm runs.
test "dsv41 memory: HEAD_MODE mxfp8 bills its codes, not the dense head it drops (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, bank_dir, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    try testing.expectEqual(@as(u64, 1_323_827_200), droppedResidentBytes(&m, .mxfp8));
    try testing.expectEqual(@as(u64, c.vocab_size) * c.hidden_size * 2, droppedResidentBytes(&m, .mxfp8));
    try testing.expectEqual(@as(u64, 0), droppedResidentBytes(&m, .bf16));
    try testing.expectEqual(@as(u64, 682_598_400), builtResidentBytes(&c, .mxfp8) - builtResidentBytes(&c, .bf16));
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, bf16: arm_mod.NativeRows, mxfp8: arm_mod.NativeRows };
    for ([_]Want{
        .{ .base = 8_990_000_000, .bf16 = .{ .prefill = 135, .decode = 164 }, .mxfp8 = .{ .prefill = 136, .decode = 165 } },
        .{ .base = 9_200_000_000, .bf16 = .{ .prefill = 134, .decode = 164 }, .mxfp8 = .{ .prefill = 136, .decode = 165 } },
        .{ .base = 9_550_000_000, .bf16 = .{ .prefill = 134, .decode = 163 }, .mxfp8 = .{ .prefill = 135, .decode = 164 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b1 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        var b5 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .head_mode = .mxfp8 });
        try testing.expectEqual(@as(i64, -641_228_800), @as(i64, @intCast(b5.residents)) - @as(i64, @intCast(b1.residents)));
        b1.engram_posted = posted;
        b5.engram_posted = posted;
        const r1 = try fillRows(fillBillOf(b1), target, b1.n_experts);
        const r5 = try fillRows(fillBillOf(b5), target, b5.n_experts);
        std.debug.print("\nhead modes at baseline {d:.2} GB (posted gathers on): bf16 {d} / {d}, mxfp8 {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, r1.prefill, r1.decode, r5.prefill, r5.decode });
        try testing.expectEqual(w.bf16, r1);
        try testing.expectEqual(w.mxfp8, r5);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): SERVED17's four arms, the bill's variant (the prompt's wave: four group streams or the
// fence's two) against the transient release (decode's transient rows: every window, 240, or window 0, 48), the
// release through the route's override both ways (the cell's DSV41_CELL_TRANSIENT_RELEASE). The variant moves only
// prompt rows, the release only decode rows.
test "dsv41 memory: the four arms, variant by release, at the windows' baselines (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids };
    const pb = v41.PrefillBill.of(&c).withIndexLaunch(try module.prefillIndexRoute(&config, .{})).withJoinless(shape);
    const Rows = arm_mod.NativeRows;
    const Want = struct { base: u64, cons_off: Rows, cons_on: Rows, tight_off: Rows, tight_on: Rows };
    for ([_]Want{
        .{ .base = 8_990_000_000, .cons_off = .{ .prefill = 135, .decode = 164 }, .cons_on = .{ .prefill = 135, .decode = 169 }, .tight_off = .{ .prefill = 140, .decode = 164 }, .tight_on = .{ .prefill = 140, .decode = 169 } },
        .{ .base = 9_200_000_000, .cons_off = .{ .prefill = 134, .decode = 164 }, .cons_on = .{ .prefill = 134, .decode = 169 }, .tight_off = .{ .prefill = 139, .decode = 164 }, .tight_on = .{ .prefill = 139, .decode = 169 } },
        .{ .base = 9_550_000_000, .cons_off = .{ .prefill = 134, .decode = 163 }, .cons_on = .{ .prefill = 134, .decode = 168 }, .tight_off = .{ .prefill = 139, .decode = 163 }, .tight_on = .{ .prefill = 139, .decode = 168 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        const by_route = [2]Bill{
            try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = false }),
            try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = true }),
        };
        try testing.expectEqual(@as(u64, 240), by_route[0].transient_decode_rows);
        try testing.expectEqual(@as(u64, 48), by_route[1].transient_decode_rows);
        var got: [4]Rows = undefined;
        for ([_]u64{ 4, 2 }, 0..) |streams, vi| for (by_route, 0..) |br, ri| {
            var b = br;
            b.engram_posted = posted;
            b.prefill_wave = pb.withGroupStreams(streams).layerMajorWaveBytes(fill_prompt_tokens, .served);
            got[vi * 2 + ri] = try fillRows(fillBillOf(b), target, b.n_experts);
        };
        std.debug.print("\nfour arms at baseline {d:.2} GB (posted gathers on): conservative release off {d} / {d}, on {d} / {d}; tight off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, got[0].prefill, got[0].decode, got[1].prefill, got[1].decode, got[2].prefill, got[2].decode, got[3].prefill, got[3].decode });
        try testing.expectEqual(w.cons_off, got[0]);
        try testing.expectEqual(w.cons_on, got[1]);
        try testing.expectEqual(w.tight_off, got[2]);
        try testing.expectEqual(w.tight_on, got[3]);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): the decode rows the phase change's window release returns (SERVED16). Decode keeps window 0
// of the transient bank (48 rows) and gives windows 1..4 back (192 records, 2,556,592,128 B at depth 5); the prompt
// phase is unchanged. The rows are the release route's (`.transient_release = true`; the default is off).
test "dsv41 memory: the decode rows the PhaseGate's window release returns (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, off: arm_mod.NativeRows, on: arm_mod.NativeRows };
    for ([_]Want{
        // Without the release (this tree's fill): 164 / 164 / 163 decode rows; with it, +5 at each baseline.
        .{ .base = 8_990_000_000, .off = .{ .prefill = 135, .decode = 169 }, .on = .{ .prefill = 135, .decode = 169 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 135, .decode = 169 }, .on = .{ .prefill = 134, .decode = 169 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 134, .decode = 168 }, .on = .{ .prefill = 134, .decode = 168 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = true });
        try testing.expectEqual(@as(u64, 48), b0.transient_decode_rows);
        b0.engram_posted = 0;
        const off = try fillRows(fillBillOf(b0), target, b0.n_experts);
        b0.engram_posted = posted;
        const on = try fillRows(fillBillOf(b0), target, b0.n_experts);
        std.debug.print("\nwindow release: rows at baseline {d:.2} GB: posted gathers off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, off.prefill, off.decode, on.prefill, on.decode });
        try testing.expectEqual(w.off, off);
        try testing.expectEqual(w.on, on);
    }
    std.debug.print("\n", .{});
}

/// The fastest cell at the full admission (served-cell-typical-fastest-20260929-172908): the guard's
/// baseline and the cell bill's phase totals at the envelope's 112 / 154 rows (decimal GB, 3 places).
pub const fill_fixture = struct {
    pub const record: u64 = 13_315_584;
    pub const per_row: u64 = 40 * record;
    pub const baseline: u64 = 13_408_305_152;
    pub const prefill_total: u64 = 114_365_000_000;
    pub const decode_total: u64 = 115_140_000_000;
    pub const ceiling: u64 = 120_259_000_000;

    pub fn at(base: u64) FillBill {
        return .{ .prefill_fixed = prefill_total - 112 * per_row - baseline + base, .decode_fixed = decode_total - 154 * per_row - baseline + base, .per_row = per_row };
    }
};

test "dsv41 memory: the native fill takes two row counts, each phase at its target within one row" {
    const f = fill_fixture;
    const target = f.ceiling - module.ceiling_stop_bytes;
    for ([_]u64{ 9_000_000_000, 11_000_000_000, 13_400_000_000, f.baseline }) |base| {
        const b = f.at(base);
        const r = try fillRows(b, target, 384);
        try std.testing.expect(r.prefill <= r.decode);
        try std.testing.expect(b.decode_fixed + r.decode * b.per_row <= target and b.decode_fixed + (r.decode + 1) * b.per_row > target);
        try std.testing.expect(b.prefill_fixed + r.prefill * b.per_row <= target and b.prefill_fixed + (r.prefill + 1) * b.per_row > target);
        std.debug.print("native fill at baseline {d:.1} GB: {d} prefill / {d} decode rows per layer (target {d:.2} GB)\n", .{ @as(f64, @floatFromInt(base)) / 1e9, r.prefill, r.decode, @as(f64, @floatFromInt(target)) / 1e9 });
    }
    try std.testing.expectEqual(arm_mod.NativeRows{ .prefill = 127, .decode = 168 }, try fillRows(f.at(9_000_000_000), target, 384));
    // Capped at the layer's experts; refused by name when not even the floor fits.
    const cap = try fillRows(.{ .prefill_fixed = 0, .decode_fixed = 0, .per_row = 100_000_000 }, target, 384);
    try std.testing.expectEqual(@as(u32, 384), cap.decode);
    try std.testing.expectError(error.NativeBillDoesNotFit, fillRows(.{ .prefill_fixed = target - 10 * f.per_row, .decode_fixed = 0, .per_row = f.per_row }, target, 384));
}

test "dsv41 memory: the grow is refused when the two-count decode total exceeds the fill's target" {
    var b = cell4Bill();
    const target: u64 = 118_259_084_288;
    try admitPhases(b, target);
    // Decode rows forced past the target (148 -> 170 rows: +11.72 GB).
    b.decode_rows = 170;
    b.slot_decode = (40 * 170 + 48) * 13_315_584;
    try std.testing.expect(b.decodeTotal() > target);
    try std.testing.expectError(error.DecodeOverTarget, admitPhases(b, target));
    // The prompt phase over it is refused first.
    b.prefill_wave += 20_000_000_000;
    try std.testing.expectError(error.PromptOverTarget, admitPhases(b, target));
}


// DSV41_BANK=<bank> (host): what upstream's load preflight bills for the module (in place of the shards' disk
// bytes): the standard request's process bound at the fill's floor rows, well above the shards' bytes and
// well under a full fill's bound.
test "dsv41 memory: the load preflight's requirement is the bill at the fill's floor rows (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config = try model.parseConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const need = try loadRequirementBytes(a, testing.io, config, ceiling_bytes);
    var floor = config;
    floor.memory_baseline_bytes = 0;
    floor.expert_rows = min_fill_rows;
    floor.expert_prefill_rows = min_fill_rows;
    const b = try billAt(a, testing.io, &floor, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
    try testing.expectEqual(b.processBound(), need);
    // Residents (17.7 GB) + the prompt wave (14.4 GB) + the floor's slot rows: tens of GB, never the bank's 204 GB.
    try testing.expect(need > 30_000_000_000 and need < 60_000_000_000);
    std.debug.print("\nload preflight requirement: {d} B at {d} rows\n", .{ need, min_fill_rows });
}
