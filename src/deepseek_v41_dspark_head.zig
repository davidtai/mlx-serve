//! The DSpark draft head (Python `deepseek_v41_dspark.DSparkHead`; K33 draft
//! compile when the routes set `draft_rows`). Three stages under `mtp.{0,1,2}`, each a V4.1
//! decoder block whose attention is a pure sliding window over the backbone's
//! committed main hiddens (`Cache`, seeded by `seedMain`) plus the block's own
//! draft rows, and whose MoE is its own 128-expert top-3 mxfp4 switch, resident
//! (`gather_qmm`). Stage 0 projects the target-layer hiddens (`main_proj`,
//! `main_norm`); the last stage's head autoregresses the markov bias over the
//! block and scores each draft (`confidence_head`).

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const mdl = @import("deepseek_v41_model.zig");

/// Bytes of one DSpark head expert (gate, up and down, mxfp4 with one e8m0
/// scale per 32 weights): 18,800,640 on the bank of record, the stack of
/// record's per-expert saving (`MTP_PRUNED_BYTES / 201`).
pub fn expertBytes(c: *const v41.Config) u64 {
    return 3 * @as(u64, c.moe_intermediate_size) * c.hidden_size * 17 / 32;
}

/// Resident bytes a subset of the head's experts leaves out (0: the full head).
pub fn prunedBytes(c: *const v41.Config, subset: ?*const Subset) u64 {
    const s = subset orelse return 0;
    return s.pruned() * expertBytes(c);
}

/// Which head a run drafted with (receipts): every expert resident, or the
/// compact head of a pinned subset.
pub const Identity = union(enum) {
    full,
    compact: [32]u8,

    pub fn text(self: Identity, buf: *[72]u8) []const u8 {
        return switch (self) {
            .full => "full",
            .compact => |sha| std.fmt.bufPrint(buf, "compact:{s}", .{&std.fmt.bytesToHex(sha, .lower)}) catch unreachable,
        };
    }
};

pub fn Head(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const T = G.T;
        const Tr = graph.Trunk(G);
        const Q = graph.Q(T);

        /// A stage's routed experts stacked `[E, out, in]` (the `SwitchGLU`
        /// banks): gate = w1, up = w3, down = w2.
        pub const Experts = struct { w1: Q, w3: Q, w2: Q };
        /// `lut`: with a subset, each routed id's slot in the compact banks.
        pub const Stage = struct { w: graph.LayerW(T), experts: Experts, lut: ?T = null };

        /// Construction choices (default: the full head, every expert resident).
        pub const Options = struct {
            /// Keep only these experts per stage (a pinned subset file; the
            /// stack of record's compact head is one, a trace-derived ceiling).
            subset: ?*const Subset = null,
        };

        /// `DSparkStageCache`: the last `window` post-RoPE main-KV rows of a
        /// stage and the count of main tokens seen.
        pub const Cache = struct {
            window: ?T = null,
            offset: u32 = 0,

            pub fn deinit(self: *Cache, g: *G) void {
                if (self.window) |w| g.release(w);
                self.* = .{};
            }

            /// `append_main`: keep the last `size` rows, advance the offset.
            fn appendMain(self: *Cache, g: *G, main_kv: T, size: u32) !void {
                var all = if (self.window) |w| try g.concat(&.{ w, main_kv }, 1) else main_kv;
                const s = g.shapeOf(all);
                const rows = s.dim(1);
                if (rows > size) all = try g.slice(all, &.{ 0, rows - @as(c_int, @intCast(size)), 0 }, s.slice(), &.{ 1, 1, 1 });
                const kept = g.keep(all);
                if (self.window) |w| g.release(w);
                self.window = kept;
                self.offset += @intCast(g.shapeOf(main_kv).dim(1));
            }
        };

        pub const Draft = struct {
            /// The block's drafted ids `[1, block_size]` (uint32).
            ids: T,
            /// The draft logits `[1, block_size, vocab]` f32.
            logits: T,
            /// `confidence_head` scores `[1, block_size]` f32 (before the sigmoid).
            conf: T,
        };

        /// K33 (`MTPLX_DSV41_DRAFT_COMPILE`): at rows <= `draft_rows` the stages
        /// replay compiled regions (the backbone's K22 / K4 ones, the draft's
        /// main-KV, markov step and confidence), byte-identical to the eager
        /// bodies; 0 runs the eager bodies. The regions are built at `init`.
        pub const DraftKv = struct {
            pub const region: ops.Region = .draft_kv;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            /// in main_x, cos, sin, kv_norm, wkv (words, scales): `rope(rmsnorm(wkv(m)))`.
            pub fn run(g: *G, c: *const Ctx, in: []const T, out: []T) !void {
                const kv = try Tr.rmsnorm(g, try g.qmm(in[0], in[4], in[5], .mxfp8), in[3], c.rms_norm_eps);
                out[0] = try Tr.ropeLast(g, kv, .{ .cos = in[1], .sin = in[2] }, false);
            }
        };

        pub const MarkovStep = struct {
            pub const region: ops.Region = .markov_step;
            pub const Ctx = v41.Config;
            pub const n_out = 3;
            /// in token, base row, markov embed, markov head: out (logits row, embed, argmax).
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                const me = try Tr.embed(g, in[2], in[0]);
                const li = try g.add(in[1], try Tr.linear(g, me, in[3]));
                out[0] = li;
                out[1] = me;
                out[2] = try g.argmax(li, -1);
            }
        };

        pub const Confidence = struct {
            pub const region: ops.Region = .confidence;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            /// in hidden, markov embed, proj: `(concat(h, me).f32 @ w.f32.T)` without its last axis.
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                const h = try g.astype(try g.concat(&.{ in[0], in[1] }, -1), .float32);
                const conf = try g.matmul(h, try g.transpose(try g.astype(in[2], .float32)));
                const s = g.shapeOf(conf);
                out[0] = try g.reshape(conf, s.slice()[0 .. s.n - 1]);
            }
        };

        gpa: std.mem.Allocator,
        c: v41.Config,
        /// The stages' MoE shape: `n_routed_experts` / `n_experts_per_tok` of the DSpark head.
        mc: v41.Config,
        /// The head codec shared with the target (`Routes.head`: f32, bf16 or mxfp8).
        rt: graph.Routes,
        /// The stages' routes: K22 / K4 at rows <= `rt.draft_rows` when K33 is on.
        stage_rt: graph.Routes,
        stages: []Stage,
        main_proj: Q,
        main_norm: T,
        norm: T,
        markov_embed: T,
        markov_head: T,
        conf_proj: T,
        inv_swa: T,
        owned: std.ArrayList(T) = .empty,
        /// Host bytes of one block's input lookup (its ids, or its embedding rows
        /// once the target's table is on the host), allocated once.
        block_scratch: []u8 = &.{},
        identity: Identity = .full,
        /// Resident bytes the subset left out (0 for the full head).
        pruned_bytes: u64 = 0,

        /// Binds the residents once and stacks each stage's experts into its
        /// switch banks. `rt` carries the head codec the draft head shares with the target.
        pub fn init(gpa: std.mem.Allocator, g: *G, c: v41.Config, rt: graph.Routes, lookup: anytype) !*Self {
            return initWith(gpa, g, c, rt, lookup, .{});
        }

        /// `init` with a subset: each stage stacks only its kept experts (in
        /// ascending order: slot `i` is kept expert `i`) and maps every routed id
        /// through its `lut` (an id outside the subset takes slot 0, as the
        /// stack of record's compact head does); a dropping lookup forgets
        /// every per-expert array, so a left-out expert is never read.
        pub fn initWith(gpa: std.mem.Allocator, g: *G, c: v41.Config, rt: graph.Routes, lookup: anytype, opts: Options) !*Self {
            const ds = c.dspark;
            if (ds.n_stages == 0 or ds.block_size == 0) return error.NoDsparkHead;
            if (opts.subset) |sub| if (sub.n_experts != ds.n_routed_experts or sub.selected.len != ds.n_stages) return error.SubsetGeometry;
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            var mc = c;
            mc.n_routed_experts = ds.n_routed_experts;
            mc.n_experts_per_tok = ds.n_experts_per_tok;
            const stage_rt: graph.Routes = if (rt.draft_rows > 0) .{ .attn_rows = rt.draft_rows, .hc_rows = rt.draft_rows } else .{};
            self.* = .{ .gpa = gpa, .c = c, .mc = mc, .rt = rt, .stage_rt = stage_rt, .stages = &.{}, .main_proj = undefined, .main_norm = undefined, .norm = undefined, .markov_embed = undefined, .markov_head = undefined, .conf_proj = undefined, .inv_swa = undefined };
            errdefer self.deinitOwned(g);
            self.stages = try gpa.alloc(Stage, ds.n_stages);
            errdefer gpa.free(self.stages);
            const M = mdl.Model(G);
            var b: [160]u8 = undefined;
            if (ds.n_routed_experts > 512) return error.TooManyExperts;
            for (self.stages, 0..) |*st, s| {
                st.* = .{ .w = try M.bindBlock(lookup, "mtp", c.layers[c.n_layers + s], @intCast(s)), .experts = undefined };
                // W97 on the draft's attention too (DSparkAttention inherits `_o_lora_dense_weight`).
                if (rt.wo_a_f32) st.w.wo_a_dense = try self.own(g, try Tr.woaDenseF32(g, &self.c, st.w.wo_a));
                const first_owned = self.owned.items.len;
                var all: [512]u16 = undefined;
                for (all[0..ds.n_routed_experts], 0..) |*e, i| e.* = @intCast(i);
                const kept: []const u16 = if (opts.subset) |sub| sub.selected[s] else all[0..ds.n_routed_experts];
                inline for (.{ "w1", "w3", "w2" }) |name| {
                    var ws: [512]T = undefined;
                    var ss: [512]T = undefined;
                    for (kept, 0..) |e, i| {
                        ws[i] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        ss[i] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    }
                    const n = kept.len;
                    @field(st.experts, name) = .{ .w = try self.own(g, try g.stack(ws[0..n], 0)), .s = try self.own(g, try g.stack(ss[0..n], 0)), .mode = .mxfp4 };
                    // The stacks hold the kept arrays until they evaluate; a lookup that can
                    // forget them frees each stage's inputs once its stacks are built (and a
                    // left-out expert's before anything read it).
                    if (comptime canDrop(@TypeOf(lookup))) for (0..ds.n_routed_experts) |e| {
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    };
                }
                if (opts.subset) |sub| {
                    var l: [512]i32 = undefined;
                    sub.lut(s, l[0..ds.n_routed_experts]);
                    st.lut = try self.own(g, try g.hostArray(std.mem.sliceAsBytes(l[0..ds.n_routed_experts]), &.{@intCast(ds.n_routed_experts)}, .int32));
                }
                try g.evalAll(self.owned.items[first_owned..]);
            }
            if (opts.subset) |sub| {
                self.identity = .{ .compact = sub.sha256 };
                self.pruned_bytes = prunedBytes(&self.c, sub);
            }
            const last = ds.n_stages - 1;
            self.main_proj = .{ .w = try need(lookup, "mtp.0.main_proj.weight"), .s = try need(lookup, "mtp.0.main_proj.scales"), .mode = .mxfp8 };
            self.main_norm = try need(lookup, "mtp.0.main_norm.weight");
            self.norm = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.norm.weight", .{last}));
            self.markov_embed = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.embed.weight", .{last}));
            self.markov_head = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.head.weight", .{last}));
            self.conf_proj = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.confidence_head.proj.weight", .{last}));
            self.inv_swa = try self.own(g, try Tr.swaInvFreq(g, &self.c));
            self.block_scratch = try gpa.alloc(u8, @max(ds.block_size * @sizeOf(i32), @as(usize, ds.block_size) * c.hidden_size * 2) + 16);
            errdefer gpa.free(self.block_scratch);
            if (rt.draft_rows > 0) {
                try Tr.prepareRegions(g, &self.mc, &self.stage_rt, false);
                inline for (.{ DraftKv, MarkovStep, Confidence }) |B| try g.prepareTape(B, &self.mc);
            }
            try g.evalAll(self.owned.items);
            return self;
        }

        fn canDrop(comptime L: type) bool {
            return switch (@typeInfo(L)) {
                .pointer => |p| @hasDecl(p.child, "drop"),
                else => false,
            };
        }

        fn need(lookup: anytype, name: []const u8) !T {
            return lookup.get(name) orelse error.MissingWeight;
        }

        fn own(self: *Self, g: *G, x: T) !T {
            const k = g.keep(x);
            try self.owned.append(self.gpa, k);
            return k;
        }

        fn deinitOwned(self: *Self, g: *G) void {
            for (self.owned.items) |x| g.release(x);
            self.owned.deinit(self.gpa);
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.deinitOwned(g);
            self.gpa.free(self.block_scratch);
            self.gpa.free(self.stages);
            self.gpa.destroy(self);
        }

        pub fn nStages(self: *const Self) usize {
            return self.stages.len;
        }

        /// Device bytes the head builds beyond the checkpoint's residents (the
        /// bill's resident term): W97's dense f32 wo_a per stage. Its expert
        /// stacks replace the per-expert arrays they drop (no net bytes).
        pub fn builtBytes(self: *const Self) u64 {
            return if (self.rt.wo_a_f32) @as(u64, self.stages.len) * graph.woaDenseBytes(&self.c) else 0;
        }

        pub fn blockSize(self: *const Self) u32 {
            return self.c.dspark.block_size;
        }

        /// `main_x = main_norm(main_proj(main_hidden))` (stage 0).
        fn mainProject(self: *const Self, g: *G, main_hidden: T) !T {
            return Tr.rmsnorm(g, try Tr.qlinear(g, main_hidden, self.main_proj), self.main_norm, self.c.rms_norm_eps);
        }

        /// A stage's main KV: `rope(rmsnorm(wkv(main_x)))` at the main positions `[offset, offset + S)`.
        fn mainKv(self: *const Self, g: *G, st: *const Stage, main_x: T, offset: u32) !T {
            const S = g.shapeOf(main_x).dim(1);
            const pos = try g.arange(@floatFromInt(offset), @floatFromInt(offset + @as(u32, @intCast(S))), 1, .int32);
            const cs = try Tr.cosSin(g, self.inv_swa, pos);
            return Tr.ropeLast(g, try Tr.rmsnorm(g, try Tr.qlinear(g, main_x, st.w.wkv), st.w.kv_norm, self.c.rms_norm_eps), cs, false);
        }

        /// `seed_main`: every stage appends the committed rows' main KV to its window.
        pub fn seedMain(self: *const Self, g: *G, main_hidden: T, caches: []Cache) !void {
            const main_x = try self.mainProject(g, main_hidden);
            for (self.stages, caches) |*st, *cache| {
                try cache.appendMain(g, try self.mainKv(g, st, main_x, cache.offset), self.c.window);
            }
        }

        /// `DSparkAttention.__call__` (draft): the draft queries over the window,
        /// this cycle's main KV (not appended) and the block's own draft KV.
        fn attention(self: *const Self, g: *G, st: *const Stage, x: T, main_x: T, cache: *const Cache) !T {
            const c = &self.c;
            const w = &st.w;
            const S = g.shapeOf(main_x).dim(1);
            const main_kv = if (g.shapeOf(main_x).dim(0) * S <= @as(c_int, @intCast(self.rt.draft_rows))) blk: {
                const mpos = try g.arange(@floatFromInt(cache.offset), @floatFromInt(cache.offset + @as(u32, @intCast(S))), 1, .int32);
                const mcs = try Tr.cosSin(g, self.inv_swa, mpos);
                var o: [1]T = undefined;
                try g.tape(DraftKv, &self.mc, &.{ main_x, mcs.cos, mcs.sin, w.kv_norm, w.wkv.w, w.wkv.s }, &o);
                break :blk o[0];
            } else try self.mainKv(g, st, main_x, cache.offset);
            var win = if (cache.window) |wd| try g.concat(&.{ wd, main_kv }, 1) else main_kv;
            const ws = g.shapeOf(win);
            const wr = ws.dim(1);
            const size: c_int = @intCast(c.window);
            if (wr > size) win = try g.slice(win, &.{ 0, wr - size, 0 }, ws.slice(), &.{ 1, 1, 1 });
            const wp = g.shapeOf(win).dim(1);
            const sx = g.shapeOf(x);
            const b = sx.d[0];
            const t = sx.d[1];
            const base = cache.offset + @as(u32, @intCast(g.shapeOf(main_x).dim(1)));
            const dpos = try g.arange(@floatFromInt(base), @floatFromInt(base + @as(u32, @intCast(t))), 1, .int32);
            const cs = try Tr.cosSin(g, self.inv_swa, dpos);
            const compiled = b * t <= @as(c_int, @intCast(self.rt.draft_rows));
            var q: T = undefined;
            var kv: T = undefined;
            if (compiled) {
                var o3: [3]T = undefined;
                try g.tape(Tr.QkvPrep, &self.mc, &.{ x, cs.cos, cs.sin, w.q_norm, w.kv_norm, w.wq_a.w, w.wq_a.s, w.wq_b.w, w.wq_b.s, w.wkv.w, w.wkv.s }, &o3);
                q = o3[0];
                kv = o3[2];
            } else {
                const qr = try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wq_a), w.q_norm, c.rms_norm_eps);
                q = try Tr.ropeLast(g, try g.reshape(try Tr.qlinear(g, qr, w.wq_b), &.{ b, t, @intCast(c.n_heads), @intCast(c.head_dim) }), cs, false);
                kv = try Tr.ropeLast(g, try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wkv), w.kv_norm, c.rms_norm_eps), cs, false);
            }
            const keys = try g.concat(&.{ win, kv }, 1);
            const attend = try g.ones(&.{ b, t, wp + t }, .bool_);
            const o = try Tr.sparseAttend(g, c, w, q, keys, attend);
            const w_ol = try Tr.woaDense(g, c, w);
            if (compiled) {
                var o1: [1]T = undefined;
                try g.tape(Tr.OutPrep, &self.mc, &.{ o, cs.cos, cs.sin, w_ol, w.wo_b.w, w.wo_b.s }, &o1);
                return o1[0];
            }
            return Tr.outProj(g, c, o, cs, w_ol, w.wo_b, false);
        }

        /// The stage's resident `SwitchGLU` with `ClampedSwiGLU` (mlx_lm arg order:
        /// the activation gets (up, gate)); unsorted below 64 routed ids.
        pub const Resident = struct {
            ex: *const Experts,
            limit: f64,
            /// The compact banks' slot of every routed id (a subset head).
            lut: ?T = null,

            pub fn at(self: Resident, _: u32) Resident {
                return self;
            }

            pub fn routed(self: Resident, g: *G, xf: T, routed_ids: T) !T {
                const s = g.shapeOf(xf);
                const si = g.shapeOf(routed_ids);
                if (si.numel() >= 64) return error.SortedSwitchNotPorted;
                // `_CompactMTPExpertSwitch.__call__`: `mapped = mx.take(LUT, indices)`.
                const indices = if (self.lut) |l| try g.take(l, routed_ids, 0) else routed_ids;
                const x = try g.reshape(xf, &.{ s.d[0], 1, 1, s.d[1] });
                var up = try g.gatherQmm(x, self.ex.w3.w, self.ex.w3.s, indices, .mxfp4);
                var gate = try g.gatherQmm(x, self.ex.w1.w, self.ex.w1.s, indices, .mxfp4);
                if (self.limit > 0) {
                    const dt = g.dtypeOf(up);
                    up = try g.clip(up, try g.scalar(-self.limit, dt), try g.scalar(self.limit, dt));
                    gate = try g.minimum(gate, try g.scalar(self.limit, g.dtypeOf(gate)));
                }
                const h = try g.mul(try g.silu(gate), up);
                const y = try g.gatherQmm(h, self.ex.w2.w, self.ex.w2.s, indices, .mxfp4);
                return g.reshape(y, &.{ si.d[0], si.d[1], s.d[1] });
            }
        };

        /// `DSparkBlock.__call__` (draft): HC attention prep, the draft
        /// attention, HC ffn prep, the stage MoE, HC post.
        fn stage(self: *const Self, g: *G, st: *const Stage, h: T, pre_mix: T, main_x: T, cache: *const Cache) !Tr.Out {
            const c = &self.c;
            const w = &st.w;
            const sh = g.shapeOf(h);
            const use = sh.d[0] * sh.d[1] <= @as(c_int, @intCast(self.stage_rt.hc_rows));
            var a: [4]T = undefined;
            if (use) {
                try g.tape(Tr.HcAttnPrep, &self.mc, &.{ h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm }, &a);
            } else a = try Tr.hcAttnPrep(g, c, .{}, h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
            const ao = try self.attention(g, st, a[0], main_x, cache);
            var f: [5]T = undefined;
            if (use) {
                try g.tape(Tr.HcFfnPrep, &self.mc, &.{ ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm }, &f);
            } else f = try Tr.hcFfnPrep(g, c, .{}, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
            const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &self.stage_rt, .{}, w, f[0], Resident{ .ex = &st.experts, .limit = c.swiglu_limit, .lut = st.lut });
            if (use) {
                var o: [1]T = undefined;
                try g.tape(Tr.HcPost, &self.mc, &.{ mo, f[1], f[2], f[3] }, &o);
                return .{ .h = o[0], .pre_mix = f[4] };
            }
            return .{ .h = try Tr.hcPost(g, mo, f[1], f[2], f[3]), .pre_mix = f[4] };
        }

        /// `draft_block`: embed `[primary, noise, ...]`, the stages (threading
        /// `main_x`), then `forward_head`: the base logits, the markov
        /// autoregression (greedy) and the confidence scores.
        pub fn draftBlock(self: *const Self, g: *G, main_hidden: T, primary: u32, caches: []const Cache, embed: mdl.Model(G).Embed, head_w: Tr.HeadW) !Draft {
            const c = &self.c;
            const ds = c.dspark;
            const bs: c_int = @intCast(ds.block_size);
            const main_x = try self.mainProject(g, main_hidden);
            var ids: [64]i32 = undefined;
            if (ds.block_size > ids.len) return error.BlockTooWide;
            ids[0] = @intCast(primary);
            for (ids[1..ds.block_size]) |*d| d.* = @intCast(ds.noise_token_id);
            var block_ids: [64]u32 = undefined;
            for (block_ids[0..ds.block_size], ids[0..ds.block_size]) |*d, v| d.* = @intCast(v);
            var fba = std.heap.FixedBufferAllocator.init(self.block_scratch);
            const e = try Tr.expandEmbedding(g, c, try embed.of(g, fba.allocator(), block_ids[0..ds.block_size], c.hidden_size));
            var cur: Tr.Out = e;
            // One wave per stage, as the trunk's layers (`Tr.Carry`).
            var carry: Tr.Carry = .{};
            errdefer carry.release(g);
            for (self.stages, caches) |*st, *cache| {
                const wave = g.mark();
                cur = try self.stage(g, st, cur.h, cur.pre_mix, main_x, cache);
                carry.persist(g, &cur.h, &cur.pre_mix, null);
                g.resetTo(wave);
            }
            // forward_head
            const x = try Tr.hcPre(g, cur.h, cur.pre_mix);
            carry.release(g);
            const hn = try Tr.rmsnorm(g, x, self.norm, c.rms_norm_eps);
            const base = try Tr.head(g, &self.rt, hn, head_w);
            const vocab = g.shapeOf(base).dim(-1);
            var prev = try g.hostArray(std.mem.sliceAsBytes(ids[0..1]), &.{1}, .int32);
            var outs: [64]T = undefined;
            var logit_cols: [64]T = undefined;
            var embeds: [64]T = undefined;
            const n: usize = ds.block_size;
            const compiled = bs <= @as(c_int, @intCast(self.rt.draft_rows));
            for (0..n) |i| {
                const row = try g.reshape(try g.slice(base, &.{ 0, @intCast(i), 0 }, &.{ 1, @intCast(i + 1), vocab }, &.{ 1, 1, 1 }), &.{ 1, vocab });
                if (compiled) {
                    var o3: [3]T = undefined;
                    try g.tape(MarkovStep, &self.mc, &.{ prev, row, self.markov_embed, self.markov_head }, &o3);
                    logit_cols[i] = o3[0];
                    embeds[i] = o3[1];
                    prev = o3[2];
                } else {
                    const me = try Tr.embed(g, self.markov_embed, prev);
                    const li = try g.add(row, try Tr.linear(g, me, self.markov_head));
                    logit_cols[i] = li;
                    embeds[i] = me;
                    prev = try g.argmax(li, -1);
                }
                outs[i] = prev;
            }
            const conf = if (compiled) blk: {
                // The markov steps' own embeds (each step's gather, not a second one).
                var o: [1]T = undefined;
                try g.tape(Confidence, &self.mc, &.{ x, try g.stack(embeds[0..n], 1), self.conf_proj }, &o);
                break :blk o[0];
            } else blk: {
                const markov = try g.stack(embeds[0..n], 1);
                const hcat = try g.astype(try g.concat(&.{ x, markov }, -1), .float32);
                break :blk try g.matmul(hcat, try g.transpose(try g.astype(self.conf_proj, .float32)));
            };
            return .{
                .ids = try g.stack(outs[0..n], 1),
                .logits = try g.stack(logit_cols[0..n], 1),
                .conf = try g.reshape(conf, &.{ 1, bs }),
            };
        }
    };
}

// ── The compact head's data: a pinned subset of each stage's experts ──
//
// The stack of record's compact DSpark head keeps some of each stage's experts
// resident and maps every routed id to its compact slot (an id outside the
// subset takes slot 0: `_CompactMTPExpertSwitch`, tcq_runner/packed/
// run_full.py:89-93, 107-109). The subset is data, loaded and pinned by its
// sha256 (default: none, the full head).
//
// File (`mlx-serve-expert-subset-v1`): `{"format", "kind", "n_experts",
// "selected": [[ids of stage 0], ...], ...}`; any other field documents it.

pub const subset_format = "mlx-serve-expert-subset-v1";

pub const SubsetError = error{ SubsetFile, SubsetNotPinned, SubsetFormat, SubsetIds, OutOfMemory };

/// Where a subset file is and the sha256 (hex) it must have.
pub const SubsetPin = struct { path: []const u8, sha256: []const u8 };

pub const Subset = struct {
    arena: std.heap.ArenaAllocator,
    sha256: [32]u8,
    kind: []const u8,
    n_experts: u32,
    /// Per block, the kept experts in ascending order; slot `i` of the
    /// block's compact bank is expert `selected[block][i]`.
    selected: []const []const u16,

    /// The file at `pin.path`, refused unless its sha256 is `pin.sha256`
    /// (hex) and every block's ids are ascending, unique, below `n_experts`
    /// and at least one.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, pin: SubsetPin, diag: ?*SubsetDiag) SubsetError!Subset {
        var self: Subset = .{ .arena = std.heap.ArenaAllocator.init(gpa), .sha256 = undefined, .kind = "", .n_experts = 0, .selected = &.{} };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        const text = std.Io.Dir.cwd().readFileAlloc(io, pin.path, a, .limited(16 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.SubsetFile, "{s}: {t}", .{ pin.path, e }),
        };
        std.crypto.hash.sha2.Sha256.hash(text, &self.sha256, .{});
        const hex = std.fmt.bytesToHex(self.sha256, .lower);
        if (!std.ascii.eqlIgnoreCase(&hex, pin.sha256))
            return refuse(diag, error.SubsetNotPinned, "{s}: sha256 {s}, pinned {s}", .{ pin.path, &hex, pin.sha256 });
        const Json = struct { format: []const u8, kind: []const u8 = "", n_experts: u32, selected: []const []const u16 };
        const j = std.json.parseFromSliceLeaky(Json, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.SubsetFormat, "{s}: {t}", .{ pin.path, e }),
        };
        if (!std.mem.eql(u8, j.format, subset_format)) return refuse(diag, error.SubsetFormat, "{s}: format \"{s}\", not {s}", .{ pin.path, j.format, subset_format });
        if (j.n_experts == 0 or j.selected.len == 0) return refuse(diag, error.SubsetFormat, "{s}: no blocks or no experts", .{pin.path});
        for (j.selected, 0..) |ids, b| {
            if (ids.len == 0) return refuse(diag, error.SubsetIds, "{s}: block {d} keeps no expert", .{ pin.path, b });
            for (ids, 0..) |e, i| {
                if (e >= j.n_experts) return refuse(diag, error.SubsetIds, "{s}: block {d} keeps expert {d} of {d}", .{ pin.path, b, e, j.n_experts });
                if (i > 0 and ids[i - 1] >= e) return refuse(diag, error.SubsetIds, "{s}: block {d} is not ascending at {d}", .{ pin.path, b, i });
            }
        }
        self.kind = j.kind;
        self.n_experts = j.n_experts;
        self.selected = j.selected;
        return self;
    }

    pub fn deinit(self: *Subset) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Block `block`'s slot for every routed id `0..n_experts`: its position
    /// in the kept list, 0 for an id outside it.
    pub fn lut(self: *const Subset, block: usize, out: []i32) void {
        std.debug.assert(out.len == self.n_experts);
        @memset(out, 0);
        for (self.selected[block], 0..) |e, i| out[e] = @intCast(i);
    }

    /// Experts the subset leaves out, over every block.
    pub fn pruned(self: *const Subset) u64 {
        var n: u64 = 0;
        for (self.selected) |ids| n += self.n_experts - ids.len;
        return n;
    }

    pub fn shaHex(self: *const Subset) [64]u8 {
        return std.fmt.bytesToHex(self.sha256, .lower);
    }
};

/// A refusal's message (the caller names the subset in its own diag).
pub const SubsetDiag = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const SubsetDiag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn refuse(diag: ?*SubsetDiag, err: SubsetError, comptime fmt: []const u8, args: anytype) SubsetError {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

// ── Tests (host) ──

const testing = std.testing;

fn writeSubsetFile(tmp: *std.testing.TmpDir, name: []const u8, text: []const u8, path_buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = text });
    var root: [512]u8 = undefined;
    return std.fmt.bufPrint(path_buf, "{s}/{s}", .{ root[0..try tmp.dir.realPath(testing.io, &root)], name });
}

fn subsetSha(text: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "dsv41 dspark head: a pinned subset file loads; every routed id maps to its compact slot, the rest to slot 0" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const text =
        \\{"format": "mlx-serve-expert-subset-v1", "kind": "test", "n_experts": 8, "note": "any field documents",
        \\ "selected": [[1, 4, 6], [0, 7]]}
    ;
    var pb: [700]u8 = undefined;
    const path = try writeSubsetFile(&tmp, "s.json", text, &pb);
    const sha = subsetSha(text);
    var s = try Subset.load(testing.allocator, testing.io, .{ .path = path, .sha256 = &sha }, null);
    defer s.deinit();
    try testing.expectEqualStrings("test", s.kind);
    try testing.expectEqual(@as(u64, 5 + 6), s.pruned());
    try testing.expectEqualStrings(&sha, &s.shaHex());
    var l: [8]i32 = undefined;
    s.lut(0, &l);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 1, 0, 2, 0 }, &l);
    s.lut(1, &l);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 0, 0, 1 }, &l);
}

test "dsv41 dspark head: an unpinned, malformed or unordered subset file is refused by name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pb: [700]u8 = undefined;
    var diag: SubsetDiag = .{};
    const good = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[1, 4]]}";
    const path = try writeSubsetFile(&tmp, "g.json", good, &pb);
    const zero: [64]u8 = @splat('0');
    try testing.expectError(error.SubsetNotPinned, Subset.load(testing.allocator, testing.io, .{ .path = path, .sha256 = &zero }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "pinned") != null);
    const cases = [_]struct { text: []const u8, err: SubsetError }{
        .{ .text = "{\"format\": \"other\", \"n_experts\": 8, \"selected\": [[1]]}", .err = error.SubsetFormat },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[4, 1]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[1, 1]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[8]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8}", .err = error.SubsetFormat },
    };
    for (cases, 0..) |cs, i| {
        var nb: [16]u8 = undefined;
        var pb2: [700]u8 = undefined;
        const p = try writeSubsetFile(&tmp, try std.fmt.bufPrint(&nb, "c{d}.json", .{i}), cs.text, &pb2);
        const sha = subsetSha(cs.text);
        try testing.expectError(cs.err, Subset.load(testing.allocator, testing.io, .{ .path = p, .sha256 = &sha }, null));
    }
    try testing.expectError(error.SubsetFile, Subset.load(testing.allocator, testing.io, .{ .path = "/nonexistent/s.json", .sha256 = &zero }, null));
}
