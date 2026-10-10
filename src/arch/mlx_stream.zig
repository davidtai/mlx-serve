//! The archs of mlx-stream (`lib/mlx-stream`) that stream their routed experts from an `experts.bin` bank: the plugin
//! owns each arch, its kernels and the expert streamer, and loads its own weights; the host parses the config,
//! renders the template, samples and schedules. A pack is the plugin's when `ModelConfig.plugin_dir` is set; the arch
//! is the one of the plugin's archs that claims its config.json.

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const server = @import("../server.zig");
const status = @import("../status.zig");
const model_settings = @import("../model_settings.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const plugin = @import("mlx_stream");
const sdk = plugin.sdk;

pub const built = true;

/// The plugin's arch namespaces, in registration order (`archs`, or the one `arch` of a single-arch plugin).
const arch_decls = if (@hasDecl(plugin, "archs")) plugin.archs else .{plugin.arch};

/// One erased table per arch.
const tables: [arch_decls.len]sdk.Arch = blk: {
    var t: [arch_decls.len]sdk.Arch = undefined;
    for (arch_decls, 0..) |T, i| t[i] = sdk.Arch.of(T);
    break :blk t;
};

/// The arch that claims `p`: the highest priority, the first registered on a tie; null when none claims it.
fn pick(p: *const sdk.ConfigPeek) ?*const sdk.Arch {
    var best: ?*const sdk.Arch = null;
    var best_priority: u8 = 0;
    for (&tables) |*t| {
        const priority = t.claims(p) orelse continue;
        if (@intFromEnum(priority) <= best_priority) continue;
        best = t;
        best_priority = @intFromEnum(priority);
    }
    return best;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    arch: sdk.ArchInstance,
    /// The trunk the module binds (it keeps references into the map).
    weights: sdk.Weights,
    /// The prompts this load billed (`billedContext`).
    billed_context: u64,
    /// The request `begin` started: its prompt pass's shape, then its decode handover.
    request: ?sdk.RequestShape = null,
    /// The next forward is the request's prompt pass (the generator hands it the whole remaining prompt).
    prompt_due: bool = false,
    handover_due: bool = false,
    /// The draft lane is armed for this request (`arm`), under this sampling.
    drafting: bool = false,
    sampling: sdk.SamplingParams = .{},
};

const Parsed = struct { vt: *const sdk.Arch, cfg: *anyopaque };

/// The claiming arch's config of the pack, with the model's settings applied. Refused by name when no arch of the
/// plugin claims the pack.
fn parse(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !Parsed {
    const dir = config.plugin_dir.?;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", arena.allocator(), .limited(16 << 20));
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), dir, text);
    const vt = pick(&peek) orelse {
        log.err("mlx-stream: no arch of this build serves model_type {s} ({s})\n", .{ config.model_type, dir });
        return error.NoMlxStreamArch;
    };
    var diag: sdk.Diag = .{};
    const cfg = vt.parse(gpa, &peek, &diag) catch |e| {
        log.err("mlx-stream: {s}\n", .{diag.message()});
        return e;
    };
    errdefer vt.free_config(gpa, cfg);
    try applySettings(arena.allocator(), io, vt, cfg, config);
    return .{ .vt = vt, .cfg = cfg };
}

/// The model's `model-settings.json` entry (`.null` without one), then the host's manual context, which wins
/// (`server.manualContext`: the entry's `ctx_size`, else `--ctx-size`).
fn applySettings(arena: std.mem.Allocator, io: std.Io, vt: *const sdk.Arch, cfg: *anyopaque, config: *const ModelConfig) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const settings = model_settings.load(arena, io, model_settings.defaultPath(&buf));
    vt.apply_settings(cfg, settings.entry(config.plugin_dir.?) orelse .null);
    const context = server.manualContext(config);
    if (context > 0) vt.apply_settings(cfg, try std.json.parseFromSliceLeaky(std.json.Value, arena, try std.fmt.allocPrint(arena, "{{\"ctx_size\": {d}}}", .{context}), .{}));
}

/// Sampled before the weights load: the memory already in use, which the plugin's rows fill on top of, and the
/// host's wired-limit margin.
fn facts() sdk.LoadFacts {
    return .{ .memory_baseline_bytes = status.getTotalMemBytes() -| status.getAvailableMemBytes(), .wired_margin_bytes = server.wired_limit_margin_bytes };
}

/// The prompts a load bills: the model's context, else the plugin's standard request.
fn billedContext(config: *const ModelConfig) u64 {
    const context = server.manualContext(config);
    return if (context > 0) context else plugin.default_context;
}

/// What a load holds under the GPU ceiling: the preflight's bill.
pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, config: *const ModelConfig) !u64 {
    const f = facts();
    const p = try parse(gpa, io, config);
    defer p.vt.free_config(gpa, p.cfg);
    return p.vt.load_bytes(gpa, io, p.cfg, &f, server.staticGpuMemoryCeiling());
}

/// Positions a request may span: the prompts the load bills and the generation past them.
pub fn contextLength(config: *const ModelConfig) u32 {
    return @intCast(billedContext(config) + plugin.generation_headroom);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, config: *const ModelConfig) !*Model {
    const p = try parse(gpa, io, config);
    errdefer p.vt.free_config(gpa, p.cfg);
    if (p.vt.claim_process) |claim| try claim();
    errdefer if (p.vt.release_process) |release| release();
    const f = facts();
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{
        .gpa = gpa,
        .arch = .{ .vt = p.vt, .cfg = p.cfg, .module = undefined },
        .weights = try sdk.loader.dir(io, gpa, config.plugin_dir.?, .{ .nocache = p.vt.caps.residents_past_page_cache }),
        .billed_context = billedContext(config),
    };
    errdefer m.weights.deinit();
    const load: sdk.LoadCtx = .{ .gpa = gpa, .io = io, .stream = s, .weights = &m.weights, .loader = &sdk.loader, .facts = f, .ceiling = server.staticGpuMemoryCeiling() };
    m.arch.module = try p.vt.init(&load, p.cfg);
    log.info("[mlx-stream] {s}\n", .{p.vt.name});
    return m;
}

pub fn close(m: *Model) void {
    const vt = m.arch.vt;
    vt.deinit(m.arch.module);
    m.weights.deinit();
    vt.free_config(m.gpa, m.arch.cfg);
    if (vt.release_process) |release| release();
    m.gpa.destroy(m);
}

/// A request's start: how many leading positions of `prompt` the module's kept state already holds (the rest, at
/// least the last token, runs through `forward`), and the shape its prompt pass bills. A prompt past the context
/// the load billed is refused before any work (a 400).
pub fn begin(m: *Model, prompt: []const u32, max_tokens: u32, context: u64) !u64 {
    if (prompt.len > m.billed_context) {
        log.warn("[mlx-stream] a {d}-token prompt is over the {d} tokens this load billed (raise the model's ctx_size)\n", .{ prompt.len, m.billed_context });
        return error.PrefillDoesNotFit;
    }
    m.request = .{ .prompt_tokens = prompt.len, .max_tokens = max_tokens, .host_context = context };
    m.prompt_due = true;
    m.handover_due = true;
    m.drafting = false;
    const restore = m.arch.vt.restore_prefix orelse return 0;
    return restore(m.arch.module, prompt[0 .. prompt.len -| 1]);
}

/// The request's prompt pass first, decode after it: the last row's logits, f32 `[1, 1, vocab]`. The decode
/// handover runs before the first decode forward.
pub fn forward(m: *Model, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const logits = if (m.prompt_due) blk: {
        m.prompt_due = false;
        break :blk try m.arch.vt.prefill(m.arch.module, ids, m.request.?);
    } else blk: {
        try handover(m);
        break :blk try m.arch.vt.step(m.arch.module, ids);
    };
    defer _ = mlx.mlx_array_free(logits);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, logits, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, f, &[_]c_int{ 1, 1, @intCast(mlx.mlx_array_size(f)) }, 3, s));
    return out;
}

/// The request's end (the scheduler's finish, on the inference thread): the arch's `request_end`, if it has one.
pub fn end(m: *Model) void {
    // The `lib/mlx-stream` pin predates the hook.
    if (@hasField(sdk.Arch, "request_end")) if (m.arch.vt.request_end) |f| f(m.arch.module);
}

fn handover(m: *Model) !void {
    if (!m.handover_due) return;
    m.handover_due = false;
    const req = m.request orelse return;
    const h = m.arch.vt.handover orelse return;
    try h(m.arch.module, .{ .prompt_tokens = @intCast(req.prompt_tokens), .reserved_tokens = req.prompt_tokens + req.max_tokens, .native_draft = m.drafting });
}

pub fn position(m: *const Model) u64 {
    return m.arch.vt.position(m.arch.module);
}

/// The arch's draft lane, null when it has none.
fn lane(m: *const Model) ?*const sdk.DraftLane {
    return switch (m.arch.vt.spec) {
        .none => null,
        .draft_lane => |*l| l,
    };
}

/// The draft block, 0 when the arch has no draft lane or the pack ships no stages.
pub fn blockSize(m: *const Model) u32 {
    const l = lane(m) orelse return 0;
    return l.block_size(m.arch.module);
}

pub const SamplingParams = sdk.SamplingParams;

/// Arms the draft lane for a request with nothing that shapes its logits (null: serial). The plugin samples a sampled
/// request itself, from `sampling`, which every `round` of the request receives.
pub fn arm(m: *Model, sampling: ?SamplingParams) bool {
    const sp = sampling orelse {
        m.drafting = false;
        return false;
    };
    m.sampling = sp;
    m.drafting = blockSize(m) > 0 and lane(m).?.arm(m.arch.module, .{ .greedy = sp.greedy(), .clean = true, .sampling = sp }) != .off;
    return m.drafting;
}

/// One draft round from `t1`: the committed tokens (`t1` first) and the next round's token.
pub fn round(m: *Model, gpa: std.mem.Allocator, t1: u32, accepted_cap: u32) !sdk.DraftRound {
    try handover(m);
    const l = lane(m) orelse return error.NoDraftLane;
    return l.round(m.arch.module, gpa, t1, accepted_cap, m.sampling);
}

const testing = std.testing;

test "mlx_stream: every arch of the plugin claims its own model_type, and nothing claims another" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    for (&tables) |*t| {
        const text = try std.fmt.allocPrint(arena.allocator(), "{{\"model_type\": \"{s}\"}}", .{t.name});
        const vt = pick(&try sdk.ConfigPeek.parse(arena.allocator(), "/m", text)) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(t.name, vt.name);
    }
    try testing.expect(pick(&try sdk.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\": \"llama\"}")) == null);
    try testing.expect(pick(&try sdk.ConfigPeek.parse(arena.allocator(), "/m", "{}")) == null);
}
