//! The trunk's weights a pack's quant formats serve (the weight-encoding seam, `abi/quant.zig`), bound once at load.
//! The claimant of `trunk_quant` loads every trunk linear stored in neither of MLX's encodings (`load_tensor`) and runs
//! it from `qmatmul` (`matmul`); the claimant of `expert_quant` loads each MoE layer's routed banks (`load_bank`) and
//! runs the layer's SwiGLU (`swiglu`). A claimant is `sdk/quant/formats.zig`'s, over the built-in formats and the
//! installed quant plugins: the built-in EXL3 table is Sushi's (banks only), and an installed EXL3 plugin outranks it on
//! the blocks it claims. Also the numerics (MLX_SERVE_BIT_EXACT) the trunk's reordering paths read.

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const host_abi = @import("../../../abi/host.zig");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const model = @import("../../../sdk/config/model.zig");
const formats = @import("../../../sdk/quant/formats.zig");
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const MoeLayerWeights = @import("../../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const Trunk = @import("../trunk.zig").Trunk;

/// The arithmetic the trunk's own paths run in. `exact`: no path that reorders a sum (the reference). `reordered`: the
/// paths that do.
pub const Numerics = enum { exact, reordered };

/// MLX_SERVE_BIT_EXACT's numerics (`raw` its value): `0` reordered, any other value exact; unset or empty, reordered on
/// the profile (the configuration the reordered paths were measured on) and exact elsewhere.
pub fn numericsFor(profile: bool, raw: ?[]const u8) Numerics {
    const v = raw orelse "";
    if (v.len == 0) return if (profile) .reordered else .exact;
    return if (v[0] == '0') .reordered else .exact;
}

/// What the load hands the formats beside the map: the pack's config text, its format blocks and the host table.
pub const Pack = struct {
    config_json: []const u8 = "{}",
    /// `trunk_quant` and `expert_quant` as JSON; null where the pack has none.
    trunk_quant: ?[]const u8 = null,
    expert_quant: ?[]const u8 = null,
    host: ?*const host_abi.Host = null,
    model_dir: []const u8 = "",
};

pub const Formats = struct {
    /// The format of the trunk's linears and of the routed banks; null where they are the trunk's own.
    trunk: ?*const abi.QuantTable = null,
    experts: ?*const abi.QuantTable = null,
    /// The linears `trunk` loaded, by their `.weight` array in the map.
    linears: std.AutoHashMapUnmanaged(usize, abi.Weight) = .empty,
    /// The banks `experts` loaded (gate, up, down), by the gate bank's array in the map.
    banks: std.AutoHashMapUnmanaged(usize, [3]abi.Weight) = .empty,
    numerics: Numerics = .exact,
    /// A Qwen3.8 model on an EXL3 pack: the configuration the reordered numerics default on.
    profile: bool = false,
    /// The bytes the trunk format keeps resident for this model (`armTrunkResident`; 0: per-call decode).
    resident_bytes: u64 = 0,
    /// The resident linears were rotated into the standard basis at load (`MLX_SERVE_EXL3_TRUNK=bf16`).
    unrotated: bool = false,

    pub fn deinit(self: *Formats, a: std.mem.Allocator) void {
        if (self.trunk) |t| {
            // The kept matrices first: they are keyed by the weights' handles.
            if (self.resident_bytes > 0) if (t.resident) |keep| {
                var held: u64 = 0;
                _ = keep(.{}, 0, .{}, &held);
            };
            var it = self.linears.valueIterator();
            while (it.next()) |w| t.release(w);
        }
        if (self.experts) |t| {
            var it = self.banks.valueIterator();
            while (it.next()) |b| for (b) |*w| t.release(w);
        }
        self.linears.deinit(a);
        self.banks.deinit(a);
        self.* = .{};
    }

    /// The trunk format's weight for the map's `.weight` array `w`; null for a linear the trunk runs itself.
    pub fn linearOf(self: *const Formats, w: mlx.mlx_array) ?*const abi.Weight {
        if (self.linears.count() == 0) return null;
        return self.linears.getPtr(@intFromPtr(w.ctx));
    }

    /// The expert format's banks of the MoE block `mw`; null for banks the trunk runs itself.
    pub fn banksOf(self: *const Formats, mw: *const MoeMlpWeights) ?*const [3]abi.Weight {
        if (self.banks.count() == 0) return null;
        return self.banks.getPtr(@intFromPtr(mw.switch_gate_w.ctx));
    }

    /// `x @ W^T` through the trunk format.
    pub fn matmul(self: *const Formats, w: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        const t = self.trunk orelse return error.TrunkFormatMissing;
        var out: abi.Array = .{};
        const v = t.matmul(w, .{ .ctx = x.ctx }, .{ .ctx = s.ctx }, &out);
        if (!v.isOk()) {
            log.err("[qwen4] {s}: a trunk linear's matmul refused ({s})\n", .{ t.name.slice(), v.why.slice() });
            return error.TrunkFormatMatmulRefused;
        }
        return .{ .ctx = out.ctx };
    }

    /// `x` times the joined matrix of an un-rotated group, `members` its linears' `.weight` arrays in the group's order:
    /// [..., the sum of their widths]. Null where the trunk is not un-rotated, the format joined them otherwise, or the
    /// call is a decode width: the caller runs each linear's `qmatmul`.
    pub fn joined(self: *const Formats, members: []const mlx.mlx_array, x: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
        if (!self.unrotated) return null;
        const f = (self.trunk orelse return null).linear_joined orelse return null;
        var ws: [unrot_max_join]abi.Weight = undefined;
        if (members.len < 2 or members.len > ws.len) return null;
        for (members, ws[0..members.len]) |m, *w| w.* = (self.linearOf(m) orelse return null).*;
        var out: abi.Array = .{};
        if (!f(.of(ws[0..members.len]), .{ .ctx = x.ctx }, .{ .ctx = s.ctx }, &out).isOk()) return null;
        return .{ .ctx = out.ctx };
    }

    /// The routed SwiGLU of `x` [B, S, D] through banks `b`'s experts `inds` [B, S, K], weighted by `scores`.
    pub fn swiglu(self: *const Formats, b: *const [3]abi.Weight, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, opts: abi.Swiglu, s: mlx.mlx_stream) !mlx.mlx_array {
        const t = self.experts orelse return error.ExpertFormatMissing;
        var out: abi.Array = .{};
        const v = t.swiglu.?(&b[0], &b[1], &b[2], .{ .ctx = x.ctx }, .{ .ctx = inds.ctx }, .{ .ctx = scores.ctx }, &opts, .{ .ctx = s.ctx }, &out);
        if (!v.isOk()) {
            log.err("[qwen4] {s}: a routed call refused ({s}, {d} rows)\n", .{ t.name.slice(), v.why.slice(), rowsOf(x) });
            return error.ExpertFormatSwigluRefused;
        }
        return .{ .ctx = out.ctx };
    }
};

/// The reordered numerics: a path that reorders a sum may run.
pub fn reordered(self: *const Trunk) bool {
    return self.formats.numerics == .reordered;
}

/// MLX_SERVE_EXL3_TRUNK: how the trunk format's linears serve prefill rows. `exl3`: decoded per call. `resident`: each
/// decoded matrix kept from its first prefill call (the same launch writes the same bytes once: exl3's bytes). `bf16`:
/// resident and rotated into the standard basis once at load, one rounding of each weight to the activation dtype (its
/// own roundings, so an exact mode keeps the rotated matrices).
pub const TrunkMode = enum { exl3, resident, bf16 };

/// MLX_SERVE_EXL3_TRUNK's mode from its value: unset or empty is `resident`, `0` is exl3; null for a value that names no
/// mode.
pub fn trunkModeFor(raw: ?[]const u8) ?TrunkMode {
    const v = raw orelse return .resident;
    if (v.len == 0) return .resident;
    if (std.mem.eql(u8, v, "exl3") or std.mem.eql(u8, v, "0")) return .exl3;
    if (std.mem.eql(u8, v, "resident")) return .resident;
    if (std.mem.eql(u8, v, "bf16")) return .bf16;
    return null;
}

fn trunkMode() TrunkMode {
    const raw: ?[]const u8 = if (env.span(.MLX_SERVE_EXL3_TRUNK)) |v| v else null;
    return trunkModeFor(raw) orelse {
        log.warn("[qwen4] MLX_SERVE_EXL3_TRUNK={s} names no mode (exl3, resident, bf16): resident\n", .{raw.?});
        return .resident;
    };
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1024.0 * 1024.0 * 1024.0);
}

/// The MTP head's linears' `.weight` arrays (0 where absent): fc_embedding, fc_hidden, its attention's q, k, v, o and
/// indexer, its shared expert's gate, up and down.
fn mtpLinears(self: *const Trunk) [10]usize {
    var keys: [10]usize = @splat(0);
    const m = if (self.qwen4_mtp) |*h| h else return keys;
    keys[0] = @intFromPtr(m.fc_emb_w.ctx);
    keys[1] = @intFromPtr(m.fc_hid_w.ctx);
    switch (m.layer.attn) {
        .full => |*fa| for ([_]mlx.mlx_array{ fa.q_w, fa.k_w, fa.v_w, fa.o_w, fa.idx_qk_w }, keys[2..7]) |w, *k| {
            k.* = @intFromPtr(w.ctx);
        },
        .linear => {},
    }
    switch (m.layer.mlp) {
        .moe => |*mw| for ([_]mlx.mlx_array{ mw.shared_gate_w, mw.shared_up_w, mw.shared_down_w }, keys[7..10]) |w, *k| {
            k.* = @intFromPtr(w.ctx);
        },
        .dense => {},
    }
    return keys;
}

/// Whether the resident set takes the linear whose `.weight` is `w`: one the trunk format loaded, not the lm_head (one row
/// per prefill), not the MTP head's on a load that serves no head.
fn residentAdmits(self: *const Trunk, w: mlx.mlx_array, head: *const [10]usize, mtp_head: bool) bool {
    if (w.ctx == null or w.ctx == self.lm_head_w.ctx or self.formats.linearOf(w) == null) return false;
    return mtp_head or std.mem.indexOfScalar(usize, head, @intFromPtr(w.ctx)) == null;
}

/// The resident trunk's arm, from the engine's load once everything the load keeps is resident and before the warm-ups
/// and the hot-cache plan, which then bill it: the trunk format keeps every linear `residentAdmits`, within `room` bytes
/// (the GPU ceiling less the active bytes, the context's KV and a prefill chunk's reserve); `mtp_head` whether the head
/// serves this load. A decline keeps the per-call decode; `bf16` then rotates the set (`armTrunkUnrot`).
pub fn armTrunkResident(self: *Trunk, room: u64, mtp_head: bool) void {
    const f = &self.formats;
    const t = f.trunk orelse return;
    const mode = trunkMode();
    if (mode == .exl3) return;
    const keep = t.resident orelse {
        log.info("[trunk-resident] declined: {s} keeps no decoded matrices; each call decodes its own\n", .{t.name.slice()});
        return;
    };
    var ws: std.ArrayList(abi.Weight) = .empty;
    defer ws.deinit(self.allocator);
    const head = mtpLinears(self);
    var it = f.linears.iterator();
    while (it.next()) |e| {
        if (!residentAdmits(self, .{ .ctx = @ptrFromInt(e.key_ptr.*) }, &head, mtp_head)) continue;
        ws.append(self.allocator, e.value_ptr.*) catch {
            log.info("[trunk-resident] declined: no host memory for the list\n", .{});
            return;
        };
    }
    if (ws.items.len == 0) return;
    var bytes: u64 = 0;
    const v = keep(.of(ws.items), room, .{ .ctx = self.s.ctx }, &bytes);
    if (!v.isOk()) {
        log.info("[trunk-resident] declined: {d} linears, {d:.2} GB, in {d:.2} GB of room ({s}); each call decodes its own\n", .{ ws.items.len, gib(bytes), gib(room), v.why.slice() });
        return;
    }
    f.resident_bytes = bytes;
    log.info("[trunk-resident] engaged: {d} linears, {d:.2} GB, each decoded matrix kept at its first prefill call\n", .{ ws.items.len, gib(bytes) });
    if (mode == .bf16) armTrunkUnrot(self, mtp_head);
}

/// Linears the bf16 trunk joins into one matrix at most (q | k | v | indexer).
pub const unrot_max_join = 4;

pub const UnrotClass = enum { gdn, qsa, shared, fc };

/// The bf16 trunk's groups as the format takes them: `ws` in groups of `sizes[g]` linears that read one x, in the joined
/// matrix's column order; `keys` the members' `.weight` arrays (a linear is grouped once).
const UnrotGroups = struct {
    ws: std.ArrayList(abi.Weight) = .empty,
    keys: std.ArrayList(usize) = .empty,
    sizes: std.ArrayList(u8) = .empty,
    classes: std.ArrayList(UnrotClass) = .empty,

    fn deinit(g: *UnrotGroups, a: std.mem.Allocator) void {
        g.ws.deinit(a);
        g.keys.deinit(a);
        g.sizes.deinit(a);
        g.classes.deinit(a);
    }

    fn seen(g: *const UnrotGroups, w: mlx.mlx_array) bool {
        return std.mem.indexOfScalar(usize, g.keys.items, @intFromPtr(w.ctx)) != null;
    }

    /// `ls` as one group when `join` and every member is admitted and new, else each admitted new member alone.
    fn add(g: *UnrotGroups, t: *const Trunk, a: std.mem.Allocator, ls: []const mlx.mlx_array, class: UnrotClass, join: bool, head: *const [10]usize, mtp_head: bool) !void {
        var all = true;
        for (ls) |w| if (!residentAdmits(t, w, head, mtp_head) or g.seen(w)) {
            all = false;
        };
        if (join and ls.len > 1 and all) {
            for (ls) |w| try g.member(t, a, w);
            try g.sizes.append(a, @intCast(ls.len));
            try g.classes.append(a, class);
            return;
        }
        for (ls) |w| if (residentAdmits(t, w, head, mtp_head) and !g.seen(w)) {
            try g.member(t, a, w);
            try g.sizes.append(a, 1);
            try g.classes.append(a, class);
        };
    }

    fn member(g: *UnrotGroups, t: *const Trunk, a: std.mem.Allocator, w: mlx.mlx_array) !void {
        try g.ws.append(a, t.formats.linearOf(w).?.*);
        try g.keys.append(a, @intFromPtr(w.ctx));
    }

    /// Per GDN layer in_proj_qkv | in_proj_z and out_proj; per QSA layer q | k | v | indexer and o_proj; per MoE layer the
    /// shared expert's gate | up (a gated one) and down.
    fn layer(g: *UnrotGroups, t: *const Trunk, a: std.mem.Allocator, lw: *const MoeLayerWeights, head: *const [10]usize, mtp_head: bool) !void {
        switch (lw.attn) {
            .linear => |*la| if (!la.combined_proj) {
                try g.add(t, a, &.{ la.qkv_w, la.z_w }, .gdn, true, head, mtp_head);
                try g.add(t, a, &.{la.out_w}, .gdn, false, head, mtp_head);
            },
            .full => |*fa| {
                const qkvi = [_]mlx.mlx_array{ fa.q_w, fa.k_w, fa.v_w, fa.idx_qk_w };
                try g.add(t, a, if (fa.idx_qk_w.ctx != null) &qkvi else qkvi[0..3], .qsa, true, head, mtp_head);
                try g.add(t, a, &.{fa.o_w}, .qsa, false, head, mtp_head);
            },
        }
        switch (lw.mlp) {
            .moe => |*mw| {
                try g.add(t, a, &.{ mw.shared_gate_w, mw.shared_up_w }, .shared, !mw.shared_ungated, head, mtp_head);
                try g.add(t, a, &.{mw.shared_down_w}, .shared, false, head, mtp_head);
            },
            .dense => {},
        }
    }
};

fn unrotGroupsOf(self: *const Trunk, g: *UnrotGroups, head: *const [10]usize, mtp_head: bool) !void {
    for (self.moe_layers) |*lw| try g.layer(self, self.allocator, lw, head, mtp_head);
    if (self.qwen4_mtp) |*m| {
        try g.layer(self, self.allocator, &m.layer, head, mtp_head);
        for ([_]mlx.mlx_array{ m.fc_emb_w, m.fc_hid_w }) |w| try g.add(self, self.allocator, &.{w}, .fc, false, head, mtp_head);
    }
}

/// The bf16 trunk, once the resident set is armed and before any matrix is kept: the groups (`UnrotGroups.layer` over
/// the trunk's layers and the MTP head's, its fc_embedding and fc_hidden alone) rotated by the format once, in the
/// activation dtype. A group the format cannot rotate keeps its rotated path; a refusal keeps them all.
fn armTrunkUnrot(self: *Trunk, mtp_head: bool) void {
    const f = &self.formats;
    const t = f.trunk.?;
    if (f.numerics == .exact) {
        log.info("[trunk-bf16] declined: the exact numerics keep the rotated matrices\n", .{});
        return;
    }
    const rotate = t.unrotate orelse {
        log.info("[trunk-bf16] declined: {s} rotates no matrices; the resident trunk stays rotated\n", .{t.name.slice()});
        return;
    };
    const a = self.allocator;
    var g: UnrotGroups = .{};
    defer g.deinit(a);
    const head = mtpLinears(self);
    unrotGroupsOf(self, &g, &head, mtp_head) catch {
        log.info("[trunk-bf16] declined: no host memory for the groups; the resident trunk stays rotated\n", .{});
        return;
    };
    const took = a.alloc(bool, g.sizes.items.len) catch {
        log.info("[trunk-bf16] declined: no host memory for the groups; the resident trunk stays rotated\n", .{});
        return;
    };
    defer a.free(took);
    @memset(took, false);
    const v = rotate(.of(g.ws.items), .of(g.sizes.items), @intCast(@intFromEnum(self.actDtype())), .{ .ctx = self.s.ctx }, took.ptr);
    if (!v.isOk()) {
        log.info("[trunk-bf16] declined: {s} refused the rotation ({s}); the resident trunk stays rotated\n", .{ t.name.slice(), v.why.slice() });
        return;
    }
    var modules: usize = 0;
    var kept: usize = 0;
    var joins: [4]u32 = @splat(0);
    for (g.sizes.items, g.classes.items, took) |n, c, ok| {
        if (!ok) {
            kept += 1;
            continue;
        }
        modules += n;
        if (n > 1) joins[@intFromEnum(c)] += 1;
    }
    f.unrotated = modules > 0;
    log.info("[trunk-bf16] engaged: {d} linears in {s} (joined: gdn {d}, qsa {d}, shared {d}; {d} groups kept rotated)\n", .{ modules, @tagName(self.actDtype()), joins[0], joins[1], joins[2], kept });
}

/// One load's formats while they are open: `begin` claims and opens them and loads the trunk's linears, `banks` loads
/// one MoE layer's routed banks, `end` closes them (the weights outlive it, in `formats`). Stays at its address while
/// open: the load contexts point into it.
pub const Loading = struct {
    formats: Formats = .{},
    a: std.mem.Allocator,
    src: abi.Source,
    trunk_ctx: ?abi.LoadCtx = null,
    experts_ctx: ?abi.LoadCtx = null,

    /// Refused by name: a format block nobody claims, a format that refuses it, a linear or bank it does not serve.
    pub fn begin(self: *Loading, a: std.mem.Allocator, config: *const model.ModelConfig, weights: *const model.Weights, pack: Pack, s: mlx.mlx_stream) !void {
        self.* = .{ .a = a, .src = .{ .ctx = @constCast(weights), .get = tensorOf } };
        errdefer self.abort();
        const f = &self.formats;
        f.profile = config.exl3 != null or config.trunk_format;
        const raw: ?[]const u8 = if (env.span(.MLX_SERVE_BIT_EXACT)) |v| v else null;
        f.numerics = numericsFor(f.profile, raw);
        if (config.trunk_format) {
            const json = pack.trunk_quant orelse return error.TrunkFormatBlockMissing;
            const t = try claimed(json, pack.config_json, "trunk_quant");
            try self.open(&self.trunk_ctx, t, json, pack, s, "trunk_quant");
            f.trunk = t;
            const n = try self.loadLinears(t, weights);
            if (n == 0) {
                log.err("[qwen4] the pack names a trunk_quant format, and its weights hold no linear outside MLX's encodings\n", .{});
                return error.TrunkFormatHasNoLinears;
            }
            log.info("[qwen4] trunk_quant: {d} linears on {s}\n", .{ n, t.name.slice() });
        }
        if (config.exl3 != null) {
            const json = pack.expert_quant orelse return error.ExpertFormatBlockMissing;
            const t = try claimed(json, pack.config_json, "expert_quant");
            if (t.swiglu == null) {
                log.err("[qwen4] {s} runs routed banks per projection only; the trunk runs a layer's banks as one SwiGLU\n", .{t.name.slice()});
                return error.ExpertFormatHasNoSwiglu;
            }
            try self.open(&self.experts_ctx, t, json, pack, s, "expert_quant");
            f.experts = t;
        }
        const set = raw != null and raw.?.len > 0;
        if (f.profile or set) log.info("[qwen4] EXL3 numerics: {s} ({s}); trunk linears on {s}, routed experts on {s}\n", .{
            @tagName(f.numerics),
            if (set) "set" else "the EXL3 pack's default",
            if (f.trunk) |t| t.name.slice() else "the trunk's kernels",
            if (f.experts) |t| t.name.slice() else "their affine banks",
        });
    }

    /// Layer `layer`'s banks under `<prefix>.layers.<layer>.mlp.switch_mlp` through the expert format (none: nothing);
    /// `hidden` / `inter` pin their widths (0: unchecked).
    pub fn banks(self: *Loading, prefix: []const u8, layer: u32, mw: *const MoeMlpWeights, hidden: u32, inter: u32) !void {
        const t = self.formats.experts orelse return;
        var out: [3]abi.Weight = @splat(.{});
        errdefer for (&out) |*w| t.release(w);
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, inter }, &out) |proj, in_dim, *w| {
            var nb: [192]u8 = undefined;
            const base = try std.fmt.bufPrint(&nb, "{s}.layers.{d}.mlp.switch_mlp.{s}", .{ prefix, layer, proj });
            const v = t.load_bank(&self.experts_ctx.?, abi.Str.of(base), in_dim, w);
            if (!v.isOk()) {
                log.err("[qwen4] {s}: {s} does not serve the bank ({s})\n", .{ base, t.name.slice(), v.why.slice() });
                return error.ExpertBankUnserved;
            }
        }
        const slot = try self.formats.banks.getOrPut(self.a, @intFromPtr(mw.switch_gate_w.ctx));
        if (slot.found_existing) {
            for (&out) |*w| t.release(w);
            return;
        }
        slot.value_ptr.* = out;
    }

    /// Closes the open formats; their weights stay in `formats`.
    pub fn end(self: *Loading) void {
        if (self.trunk_ctx) |*c| self.formats.trunk.?.close(c);
        if (self.experts_ctx) |*c| self.formats.experts.?.close(c);
        self.trunk_ctx = null;
        self.experts_ctx = null;
    }

    fn abort(self: *Loading) void {
        self.end();
        self.formats.deinit(self.a);
    }

    fn open(self: *Loading, slot: *?abi.LoadCtx, t: *const abi.QuantTable, block_json: []const u8, pack: Pack, s: mlx.mlx_stream, comptime key: []const u8) !void {
        slot.* = .{
            .model_dir = abi.Str.of(pack.model_dir),
            .block_json = abi.Str.of(block_json),
            .config_json = abi.Str.of(pack.config_json),
            .source = &self.src,
            .stream = .{ .ctx = s.ctx },
            .host = pack.host,
        };
        const v = t.open(&slot.*.?);
        if (v.isOk()) return;
        slot.* = null;
        log.err("[qwen4] {s} refuses the pack's " ++ key ++ " block ({s})\n", .{ t.name.slice(), v.why.slice() });
        return error.QuantFormatRefused;
    }

    /// Every `.weight` in neither of MLX's encodings, through the trunk format; the count.
    fn loadLinears(self: *Loading, t: *const abi.QuantTable, weights: *const model.Weights) !u32 {
        var n: u32 = 0;
        var it = weights.map.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or mlxEncoded(e.value_ptr.*)) continue;
            var w: abi.Weight = .{};
            const v = t.load_tensor(&self.trunk_ctx.?, abi.Str.of(name[0 .. name.len - ".weight".len]), 0, &w);
            if (!v.isOk()) {
                log.err("[qwen4] {s}: {s} does not serve it ({s})\n", .{ name, t.name.slice(), v.why.slice() });
                return error.TrunkLinearUnserved;
            }
            const slot = self.formats.linears.getOrPut(self.a, @intFromPtr(e.value_ptr.ctx)) catch |err| {
                t.release(&w);
                return err;
            };
            // Two names over one array (a tied table) are one linear.
            if (slot.found_existing) {
                t.release(&w);
                continue;
            }
            slot.value_ptr.* = w;
            n += 1;
        }
        return n;
    }
};

/// The claimant of the pack's `key` block, refused by name when nobody claims it.
fn claimed(block_json: []const u8, config_json: []const u8, comptime key: []const u8) !*const abi.QuantTable {
    const got = formats.claimant(block_json, config_json) catch {
        log.err("[qwen4] two formats claim the pack's " ++ key ++ " block alike\n", .{});
        return error.QuantFormatAmbiguous;
    };
    return got orelse {
        log.err("[qwen4] no quant format claims the pack's " ++ key ++ " block {s}; a fully EXL3 pack's trunk needs an installed EXL3 quant plugin (`mlx-serve plugin install`)\n", .{block_json});
        return error.QuantFormatUnclaimed;
    };
}

/// MLX's own encodings of a `.weight`: a float table, or the uint32 words of MLX's quantized modes.
fn mlxEncoded(w: mlx.mlx_array) bool {
    return switch (mlx.mlx_array_dtype(w)) {
        .float16, .bfloat16, .float32, .uint32 => true,
        else => false,
    };
}

fn tensorOf(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Array {
    const w: *const model.Weights = @ptrCast(@alignCast(ctx.?));
    return .{ .ctx = (w.get(name.slice()) orelse return .{}).ctx };
}

fn rowsOf(x: mlx.mlx_array) usize {
    const sh = mlx.getShape(x);
    var rows: usize = 1;
    for (sh, 0..) |d, i| {
        if (i + 1 < sh.len) rows *= @intCast(d);
    }
    return rows;
}

// ── tests ──

const testing = std.testing;
const outside = @import("../../../sdk/quant/outside.zig");
const exl3_format = @import("../../../sdk/quant/exl3/format.zig");

/// A quant plugin's table as the trunk reads it: it claims a `"fake"` block, loads any linear but one named `bad` (a
/// new handle on the map's array) and returns `x` from its matmul.
const Fake = struct {
    fn claims(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
        return if (std.mem.indexOf(u8, block_json.slice(), "\"fake\"") != null) .by_marker else .no;
    }
    fn open(_: *abi.LoadCtx) callconv(.c) abi.Verdict {
        return .ok;
    }
    fn close(_: *abi.LoadCtx) callconv(.c) void {}
    fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, _: u32, out: *abi.Weight) callconv(.c) abi.Verdict {
        if (std.mem.endsWith(u8, name.slice(), "bad")) return abi.Verdict.refuse(.unsupported, "FakeLayoutUnserved");
        var nb: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "{s}.weight", .{name.slice()}) catch return abi.Verdict.refuse(.failed, "FakeName");
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, .{ .ctx = ctx.source.tensor(key).ctx }) != 0) return abi.Verdict.refuse(.failed, "FakeShare");
        out.* = .{ .w = .{ .ctx = h.ctx } };
        return .ok;
    }
    fn release(w: *abi.Weight) callconv(.c) void {
        if (w.w.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = w.w.ctx });
        w.* = .{};
    }
    fn matmul(_: *const abi.Weight, x: abi.Array, _: abi.Stream, out: *abi.Array) callconv(.c) abi.Verdict {
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, .{ .ctx = x.ctx }) != 0) return abi.Verdict.refuse(.failed, "FakeShare");
        out.* = .{ .ctx = h.ctx };
        return .ok;
    }
    const table: abi.QuantTable = .{ .name = abi.Str.of("fake"), .claims = &claims, .validate = undefined, .bill = undefined, .open = &open, .close = &close, .load_tensor = &loadTensor, .load_bank = undefined, .release = &release, .matmul = &matmul, .moe = undefined };
};

fn putZeros(w: *model.Weights, name: []const u8, dt: mlx.mlx_dtype, shape: []const c_int) !void {
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, mlx.mlx_default_cpu_stream_new()));
    try w.put(name, a);
}

test "qwen4_exp formats: the numerics are 1 exact and 0 reordered on any model, the profile's default unset" {
    for ([_]bool{ false, true }) |profile| {
        try testing.expectEqual(Numerics.exact, numericsFor(profile, "1"));
        try testing.expectEqual(Numerics.reordered, numericsFor(profile, "0"));
        try testing.expectEqual(Numerics.exact, numericsFor(profile, "yes"));
    }
    try testing.expectEqual(Numerics.reordered, numericsFor(true, null));
    try testing.expectEqual(Numerics.reordered, numericsFor(true, ""));
    try testing.expectEqual(Numerics.exact, numericsFor(false, null));
}

test "qwen4_exp formats: a trunk format loads the linears outside MLX's encodings, and qmatmul runs them through it" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    try putZeros(&w, "m.q.weight", .uint16, &.{ 8, 8, 32 });
    try putZeros(&w, "m.q.scales", .float16, &.{128});
    try putZeros(&w, "m.o.weight", .bfloat16, &.{ 128, 128 });
    try putZeros(&w, "m.k.weight", .uint32, &.{ 128, 16 });
    const saved = outside.tables.get().list;
    defer outside.tables.get().list = saved;
    outside.tables.get().list = &.{&Fake.table};
    const s = mlx.mlx_default_cpu_stream_new();
    const config: model.ModelConfig = .{ .trunk_format = true };
    var ld: Loading = undefined;
    try ld.begin(testing.allocator, &config, &w, .{ .trunk_quant = "{\"format\":\"fake\"}" }, s);
    ld.end();
    var f = ld.formats;
    defer f.deinit(testing.allocator);
    try testing.expectEqual(&Fake.table, f.trunk.?);
    try testing.expectEqual(@as(u32, 1), f.linears.count());
    try testing.expect(f.linearOf(w.get("m.o.weight").?) == null and f.linearOf(w.get("m.k.weight").?) == null);
    var t: Trunk = undefined;
    t.s = s;
    t.formats = f;
    const x = w.get("m.o.weight").?;
    const y = try t.qmatmul(x, w.get("m.q.weight").?, w.get("m.q.scales").?, .{});
    defer _ = mlx.mlx_array_free(y);
    try testing.expectEqualSlices(c_int, mlx.getShape(x), mlx.getShape(y));
}

test "qwen4_exp formats: an unclaimed trunk block, a linear its format refuses and a pack without the block are refused by name" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    try putZeros(&w, "m.q.weight", .uint16, &.{ 8, 8, 32 });
    const s = mlx.mlx_default_cpu_stream_new();
    const config: model.ModelConfig = .{ .trunk_format = true };
    var ld: Loading = undefined;
    const saved = outside.tables.get().list;
    defer outside.tables.get().list = saved;
    outside.tables.get().list = &.{};
    try testing.expectError(error.QuantFormatUnclaimed, ld.begin(testing.allocator, &config, &w, .{ .trunk_quant = "{\"format\":\"fake\"}" }, s));
    try testing.expectError(error.TrunkFormatBlockMissing, ld.begin(testing.allocator, &config, &w, .{}, s));
    outside.tables.get().list = &.{&Fake.table};
    try putZeros(&w, "m.bad.weight", .uint16, &.{ 8, 8, 32 });
    try testing.expectError(error.TrunkLinearUnserved, ld.begin(testing.allocator, &config, &w, .{ .trunk_quant = "{\"format\":\"fake\"}" }, s));
    var none = model.Weights.init(testing.allocator);
    defer none.deinit();
    try testing.expectError(error.TrunkFormatHasNoLinears, ld.begin(testing.allocator, &config, &none, .{ .trunk_quant = "{\"format\":\"fake\"}" }, s));
}

test "qwen4_exp formats: the built-in Sushi table serves an EXL3 pack's banks, in the numerics the switch sets" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    const saved = outside.tables.get().list;
    defer outside.tables.get().list = saved;
    outside.tables.get().list = &.{};
    const s = mlx.mlx_default_cpu_stream_new();
    const config: model.ModelConfig = .{ .exl3 = .{ .rate = .{ .n = 32 }, .codebook = .mul1 } };
    const pack: Pack = .{ .expert_quant = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul1\"}" };
    for ([_]struct { raw: ?[*:0]const u8, want: Numerics }{ .{ .raw = null, .want = .reordered }, .{ .raw = "1", .want = .exact } }) |c| {
        const o = env.override(.MLX_SERVE_BIT_EXACT, c.raw);
        defer o.restore();
        var ld: Loading = undefined;
        try ld.begin(testing.allocator, &config, &w, pack, s);
        ld.end();
        var f = ld.formats;
        defer f.deinit(testing.allocator);
        try testing.expectEqual(&exl3_format.table, f.experts.?);
        try testing.expect(f.trunk == null);
        try testing.expectEqual(c.want, f.numerics);
    }
    const plain: model.ModelConfig = .{};
    var ld: Loading = undefined;
    try ld.begin(testing.allocator, &plain, &w, .{}, s);
    ld.end();
    try testing.expect(ld.formats.experts == null and ld.formats.numerics == .exact and !ld.formats.profile);
}

test "qwen4_exp formats: MLX_SERVE_EXL3_TRUNK names exl3, resident or bf16; unset or empty is resident, 0 is exl3" {
    try testing.expectEqual(TrunkMode.resident, trunkModeFor(null).?);
    try testing.expectEqual(TrunkMode.resident, trunkModeFor("").?);
    try testing.expectEqual(TrunkMode.resident, trunkModeFor("resident").?);
    try testing.expectEqual(TrunkMode.exl3, trunkModeFor("exl3").?);
    try testing.expectEqual(TrunkMode.exl3, trunkModeFor("0").?);
    try testing.expectEqual(TrunkMode.bf16, trunkModeFor("bf16").?);
    try testing.expect(trunkModeFor("1") == null and trunkModeFor("on") == null);
}

/// `Fake` with the resident entries: each weight takes `per` bytes; `linear_joined` records the members it was handed.
const Keep = struct {
    const per: u64 = 1024;
    var last: usize = 0;
    var dropped = false;
    var members: [4]?*anyopaque = @splat(null);

    fn resident(ws: abi.Span(abi.Weight), room: u64, _: abi.Stream, bytes: *u64) callconv(.c) abi.Verdict {
        last = ws.len;
        bytes.* = ws.len * per;
        if (ws.len == 0) dropped = true;
        if (room != 0 and bytes.* > room) return abi.Verdict.refuse(.too_large, "FakeRoom");
        return .ok;
    }
    fn joined(ws: abi.Span(abi.Weight), x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) abi.Verdict {
        for (ws.slice(), members[0..ws.len]) |w, *m| m.* = w.w.ctx;
        return Fake.matmul(&ws.slice()[0], x, s, out);
    }
    const table: abi.QuantTable = blk: {
        var t = Fake.table;
        t.resident = &resident;
        t.linear_joined = &joined;
        break :blk t;
    };
};

fn keepTrunk(w: *model.Weights, s: mlx.mlx_stream) !Trunk {
    const config: model.ModelConfig = .{ .trunk_format = true };
    var ld: Loading = undefined;
    try ld.begin(testing.allocator, &config, w, .{ .trunk_quant = "{\"format\":\"fake\"}" }, s);
    ld.end();
    var t: Trunk = undefined;
    t.formats = ld.formats;
    t.allocator = testing.allocator;
    t.s = s;
    t.lm_head_w = w.get("lm_head.weight").?;
    t.qwen4_mtp = null;
    t.moe_layers = &.{};
    return t;
}

test "qwen4_exp formats: the resident trunk keeps the trunk format's linears but the lm_head, within the room; exl3 and a refusal keep none" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    for ([_][]const u8{ "m.q.weight", "m.o.weight", "lm_head.weight" }) |name| try putZeros(&w, name, .uint16, &.{ 8, 8, 32 });
    const saved = outside.tables.get().list;
    defer outside.tables.get().list = saved;
    outside.tables.get().list = &.{&Keep.table};
    const s = mlx.mlx_default_cpu_stream_new();
    var t = try keepTrunk(&w, s);
    defer t.formats.deinit(testing.allocator);
    {
        const o = env.override(.MLX_SERVE_EXL3_TRUNK, "exl3");
        defer o.restore();
        armTrunkResident(&t, 0, true);
        try testing.expectEqual(@as(u64, 0), t.formats.resident_bytes);
    }
    const o = env.override(.MLX_SERVE_EXL3_TRUNK, null);
    defer o.restore();
    armTrunkResident(&t, Keep.per, true);
    try testing.expectEqual(@as(u64, 0), t.formats.resident_bytes);
    armTrunkResident(&t, 0, true);
    try testing.expectEqual(@as(usize, 2), Keep.last);
    try testing.expectEqual(2 * Keep.per, t.formats.resident_bytes);
    try testing.expect(!t.formats.unrotated);
    Keep.dropped = false;
    var f = t.formats;
    t.formats = .{};
    f.deinit(testing.allocator);
    try testing.expect(Keep.dropped);
}

test "qwen4_exp formats: a joined product goes through the format only on an un-rotated trunk, its members in order" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    for ([_][]const u8{ "m.q.weight", "m.k.weight", "lm_head.weight" }) |name| try putZeros(&w, name, .uint16, &.{ 8, 8, 32 });
    try putZeros(&w, "m.v.weight", .bfloat16, &.{ 128, 128 });
    const saved = outside.tables.get().list;
    defer outside.tables.get().list = saved;
    outside.tables.get().list = &.{&Keep.table};
    const s = mlx.mlx_default_cpu_stream_new();
    var t = try keepTrunk(&w, s);
    defer t.formats.deinit(testing.allocator);
    const q = w.get("m.q.weight").?;
    const k = w.get("m.k.weight").?;
    const x = w.get("m.v.weight").?;
    try testing.expect(try t.formats.joined(&.{ q, k }, x, s) == null);
    t.formats.unrotated = true;
    // A member the format did not load, or one member alone, is no join.
    try testing.expect(try t.formats.joined(&.{ q, x }, x, s) == null);
    try testing.expect(try t.formats.joined(&.{q}, x, s) == null);
    const y = (try t.formats.joined(&.{ k, q }, x, s)) orelse return error.TestJoinMissing;
    defer _ = mlx.mlx_array_free(y);
    try testing.expectEqual(t.formats.linearOf(k).?.w.ctx, Keep.members[0]);
    try testing.expectEqual(t.formats.linearOf(q).?.w.ctx, Keep.members[1]);
}
