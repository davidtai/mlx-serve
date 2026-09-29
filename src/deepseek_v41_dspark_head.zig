//! The DSpark draft head (Python `deepseek_v41_dspark.DSparkHead`, the eager
//! path: draft compile K33 off). Three stages under `mtp.{0,1,2}`, each a V4.1
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

pub fn Head(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const T = G.T;
        const Tr = graph.Trunk(G);
        const Q = graph.Q(T);

        /// A stage's routed experts stacked `[E, out, in]` (the `SwitchGLU`
        /// banks): gate = w1, up = w3, down = w2.
        pub const Experts = struct { w1: Q, w3: Q, w2: Q };
        pub const Stage = struct { w: graph.LayerW(T), experts: Experts };

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

        /// The stages run the eager bodies (K33 draft compile is not ported): no trunk lever applies.
        const stage_routes: graph.Routes = .{};

        gpa: std.mem.Allocator,
        c: v41.Config,
        /// The stages' MoE shape: `n_routed_experts` / `n_experts_per_tok` of the DSpark head.
        mc: v41.Config,
        /// The head codec shared with the target (`Routes.head`: f32, bf16 or mxfp8).
        rt: graph.Routes,
        stages: []Stage,
        main_proj: Q,
        main_norm: T,
        norm: T,
        markov_embed: T,
        markov_head: T,
        conf_proj: T,
        inv_swa: T,
        owned: std.ArrayList(T) = .empty,

        /// Binds the residents once and stacks each stage's experts into its
        /// switch banks. `rt` carries the head codec the draft head shares with the target.
        pub fn init(gpa: std.mem.Allocator, g: *G, c: v41.Config, rt: graph.Routes, lookup: anytype) !*Self {
            const ds = c.dspark;
            if (ds.n_stages == 0 or ds.block_size == 0) return error.NoDsparkHead;
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            var mc = c;
            mc.n_routed_experts = ds.n_routed_experts;
            mc.n_experts_per_tok = ds.n_experts_per_tok;
            self.* = .{ .gpa = gpa, .c = c, .mc = mc, .rt = rt, .stages = &.{}, .main_proj = undefined, .main_norm = undefined, .norm = undefined, .markov_embed = undefined, .markov_head = undefined, .conf_proj = undefined, .inv_swa = undefined };
            errdefer self.deinitOwned(g);
            self.stages = try gpa.alloc(Stage, ds.n_stages);
            errdefer gpa.free(self.stages);
            const M = mdl.Model(G);
            var b: [160]u8 = undefined;
            for (self.stages, 0..) |*st, s| {
                st.w = try M.bindBlock(lookup, "mtp", c.layers[c.n_layers + s], @intCast(s));
                const first_owned = self.owned.items.len;
                inline for (.{ "w1", "w3", "w2" }) |name| {
                    var ws: [512]T = undefined;
                    var ss: [512]T = undefined;
                    if (ds.n_routed_experts > ws.len) return error.TooManyExperts;
                    for (0..ds.n_routed_experts) |e| {
                        ws[e] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        ss[e] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    }
                    const n = ds.n_routed_experts;
                    @field(st.experts, name) = .{ .w = try self.own(g, try g.stack(ws[0..n], 0)), .s = try self.own(g, try g.stack(ss[0..n], 0)), .mode = .mxfp4 };
                    // The stacks hold the per-expert arrays until they evaluate; a lookup that
                    // can forget them frees each stage's inputs once its stacks are built.
                    if (comptime canDrop(@TypeOf(lookup))) for (0..n) |e| {
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    };
                }
                try g.evalAll(self.owned.items[first_owned..]);
            }
            const last = ds.n_stages - 1;
            self.main_proj = .{ .w = try need(lookup, "mtp.0.main_proj.weight"), .s = try need(lookup, "mtp.0.main_proj.scales"), .mode = .mxfp8 };
            self.main_norm = try need(lookup, "mtp.0.main_norm.weight");
            self.norm = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.norm.weight", .{last}));
            self.markov_embed = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.embed.weight", .{last}));
            self.markov_head = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.head.weight", .{last}));
            self.conf_proj = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.confidence_head.proj.weight", .{last}));
            self.inv_swa = try self.own(g, try Tr.swaInvFreq(g, &self.c));
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
            self.gpa.free(self.stages);
            self.gpa.destroy(self);
        }

        pub fn nStages(self: *const Self) usize {
            return self.stages.len;
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
            const main_kv = try self.mainKv(g, st, main_x, cache.offset);
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
            const qr = try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wq_a), w.q_norm, c.rms_norm_eps);
            const q = try Tr.ropeLast(g, try g.reshape(try Tr.qlinear(g, qr, w.wq_b), &.{ b, t, @intCast(c.n_heads), @intCast(c.head_dim) }), cs, false);
            const kv = try Tr.ropeLast(g, try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wkv), w.kv_norm, c.rms_norm_eps), cs, false);
            const keys = try g.concat(&.{ win, kv }, 1);
            const attend = try g.ones(&.{ b, t, wp + t }, .bool_);
            const o = try Tr.sparseAttend(g, c, w, q, keys, attend);
            return Tr.outProj(g, c, o, cs, try Tr.woaDense(g, c, w), w.wo_b, false);
        }

        /// The stage's resident `SwitchGLU` with `ClampedSwiGLU` (mlx_lm arg order:
        /// the activation gets (up, gate)); unsorted below 64 routed ids.
        pub const Resident = struct {
            ex: *const Experts,
            limit: f64,

            pub fn at(self: Resident, _: u32) Resident {
                return self;
            }

            pub fn routed(self: Resident, g: *G, xf: T, indices: T) !T {
                const s = g.shapeOf(xf);
                const si = g.shapeOf(indices);
                if (si.numel() >= 64) return error.SortedSwitchNotPorted;
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
            const a = try Tr.hcAttnPrep(g, c, h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
            const ao = try self.attention(g, st, a[0], main_x, cache);
            const f = try Tr.hcFfnPrep(g, c, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
            const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &stage_routes, w, f[0], Resident{ .ex = &st.experts, .limit = c.swiglu_limit });
            return .{ .h = try Tr.hcPost(g, mo, f[1], f[2], f[3]), .pre_mix = f[4] };
        }

        /// `draft_block`: embed `[primary, noise, ...]`, the stages (threading
        /// `main_x`), then `forward_head`: the base logits, the markov
        /// autoregression (greedy) and the confidence scores.
        pub fn draftBlock(self: *const Self, g: *G, main_hidden: T, primary: u32, caches: []const Cache, embed_w: T, head_w: Tr.HeadW) !Draft {
            const c = &self.c;
            const ds = c.dspark;
            const bs: c_int = @intCast(ds.block_size);
            const main_x = try self.mainProject(g, main_hidden);
            var ids: [64]i32 = undefined;
            if (ds.block_size > ids.len) return error.BlockTooWide;
            ids[0] = @intCast(primary);
            for (ids[1..ds.block_size]) |*d| d.* = @intCast(ds.noise_token_id);
            const tok = try g.hostArray(std.mem.sliceAsBytes(ids[0..ds.block_size]), &.{ 1, bs }, .int32);
            const e = try Tr.expandEmbedding(g, c, try Tr.embed(g, embed_w, tok));
            var cur: Tr.Out = e;
            for (self.stages, caches) |*st, *cache| cur = try self.stage(g, st, cur.h, cur.pre_mix, main_x, cache);
            // forward_head
            const x = try Tr.hcPre(g, cur.h, cur.pre_mix);
            const hn = try Tr.rmsnorm(g, x, self.norm, c.rms_norm_eps);
            const base = try Tr.head(g, &self.rt, hn, head_w);
            const vocab = g.shapeOf(base).dim(-1);
            var prev = try g.hostArray(std.mem.sliceAsBytes(ids[0..1]), &.{1}, .int32);
            var outs: [64]T = undefined;
            var logit_cols: [64]T = undefined;
            var embeds: [64]T = undefined;
            for (0..ds.block_size) |i| {
                const me = try Tr.embed(g, self.markov_embed, prev);
                const bias = try Tr.linear(g, me, self.markov_head);
                const row = try g.reshape(try g.slice(base, &.{ 0, @intCast(i), 0 }, &.{ 1, @intCast(i + 1), vocab }, &.{ 1, 1, 1 }), &.{ 1, vocab });
                const li = try g.add(row, bias);
                logit_cols[i] = li;
                embeds[i] = me;
                prev = try g.argmax(li, -1);
                outs[i] = prev;
            }
            const n: usize = ds.block_size;
            const markov = try g.stack(embeds[0..n], 1);
            const hcat = try g.astype(try g.concat(&.{ x, markov }, -1), .float32);
            const conf = try g.matmul(hcat, try g.transpose(try g.astype(self.conf_proj, .float32)));
            return .{
                .ids = try g.stack(outs[0..n], 1),
                .logits = try g.stack(logit_cols[0..n], 1),
                .conf = try g.reshape(conf, &.{ 1, bs }),
            };
        }
    };
}
