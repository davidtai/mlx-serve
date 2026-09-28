//! DeepSeek-V4.1 trunk modules as graph builders over an op
//! backend (`deepseek_v41_ops.zig`). Each function transliterates our Python
//! stock eager path (`mtplx/models/deepseek_v41.py`, `_moe`, `_cache`, every
//! MTPLX_DSV41_* lever off) op for op: same ops, order, dtypes and the same
//! `mx.compile` regions, so the Python runtime stays a bitwise oracle.
//! The routed experts are a caller-supplied `routed` source (the expert
//! streamer when served; a stand-in or the Python dump's output in parity
//! runs); everything else is here.

const std = @import("std");
const model = @import("model.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");

const Dtype = ops.Dtype;

pub fn Q(comptime T: type) type {
    return struct { w: T, s: T, mode: model.QuantMode = .mxfp8 };
}

/// One trunk layer's residents (handles owned by the weights map).
pub fn LayerW(comptime T: type) type {
    return struct {
        attn_norm: T,
        ffn_norm: T,
        hc_attn_fn: T,
        hc_attn_base: T,
        hc_attn_scale: T,
        hc_ffn_fn: T,
        hc_ffn_base: T,
        hc_ffn_scale: T,
        attn_sink: T,
        q_norm: T,
        kv_norm: T,
        wq_a: Q(T),
        wq_b: Q(T),
        wkv: Q(T),
        wo_a: Q(T),
        wo_b: Q(T),
        /// kv sources: the gated pooling compressor (wgate only when ratio > 1).
        comp: ?struct { wkv: T, wgate: ?T, norm: T } = null,
        /// kv sources: the index-key projection.
        idx_k: ?struct { wk: T, k_norm: T } = null,
        /// index sources: the indexer queries and head weights.
        idx_q: ?struct { wq_b: Q(T), weights_proj: T } = null,
        gate_w: T,
        gate_bias: T,
        sh_w1: Q(T),
        sh_w2: Q(T),
        sh_w3: Q(T),
    };
}

pub fn EngramW(comptime T: type) type {
    return struct { wkv: Q(T), q_weight: T, k_weight: T };
}

/// Per-layer attention state (Python `LayerAttentionCache`, full-history
/// backing): the post-RoPE window rows, the compressed KV + index keys of a
/// kv source, and the compressor frontier. Arrays are `keep` references.
pub fn LayerCache(comptime T: type) type {
    return struct {
        offset: u32 = 0,
        window: ?T = null,
        compress_kv: ?T = null,
        index_k: ?T = null,
        raw_kv: ?T = null,
        raw_score: ?T = null,
        n_fed: u32 = 0,
    };
}

/// What a source layer hands down the stack within ONE forward (Python
/// `SharedAttentionRuntime`); borrowed references, dropped with the forward.
pub fn Shared(comptime T: type) type {
    return struct {
        compress_kv: ?T = null,
        index_k: ?T = null,
        topk_mask: ?T = null,
        candidates: ?T = null,
    };
}

/// A probe that records nothing (serving).
pub const NoProbe = struct {
    pub fn put(_: NoProbe, _: []const u8, _: anytype) !void {}
};

pub fn Trunk(comptime G: type) type {
    return struct {
        pub const T = G.T;
        pub const W = LayerW(T);
        pub const Cache = LayerCache(T);
        pub const Share = Shared(T);
        pub const Mixes = struct { pre: T, post: T, comb: T };
        pub const CosSin = struct { cos: T, sin: T };
        pub const Out = struct { h: T, pre_mix: T };

        /// A weak Python float against `like` (MLX `to_array(v, like.dtype)`).
        fn sf(g: *G, v: f64, like: T) !T {
            const d = g.dtypeOf(like);
            return g.scalar(v, if (ops.isFloat(d)) d else .float32);
        }

        fn sliceLast(g: *G, x: T, lo: c_int, hi: c_int) !T {
            const s = g.shapeOf(x);
            var start: [ops.max_dims]c_int = @splat(0);
            var stop: [ops.max_dims]c_int = undefined;
            const strides: [ops.max_dims]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            start[s.n - 1] = lo;
            stop[s.n - 1] = hi;
            return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        fn sliceAxis1(g: *G, x: T, lo: c_int, hi: c_int) !T {
            const s = g.shapeOf(x);
            var start: [ops.max_dims]c_int = @splat(0);
            var stop: [ops.max_dims]c_int = undefined;
            const strides: [ops.max_dims]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            start[1] = lo;
            stop[1] = hi;
            return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        /// `x[..., i]`.
        fn lastIndex(g: *G, x: T, i: c_int) !T {
            const s = g.shapeOf(x);
            return g.reshape(try sliceLast(g, x, i, i + 1), s.d[0 .. s.n - 1]);
        }

        /// `x[i]` of a 1-D array: a 0-d view.
        fn index0(g: *G, x: T, i: c_int) !T {
            return g.reshape(try sliceLast(g, x, i, i + 1), &.{});
        }

        /// `_grow`: append along the sequence axis; the cache keeps its own reference.
        fn grow(g: *G, slot: *?T, new: T) !void {
            const next = if (slot.*) |old| try g.concat(&.{ old, new }, 1) else new;
            const kept = g.keep(next);
            if (slot.*) |old| g.release(old);
            slot.* = kept;
        }

        /// Python `_rmsnorm`.
        pub fn rmsnorm(g: *G, x: T, w: T, eps: f64) !T {
            const dt = g.dtypeOf(x);
            var xf = try g.astype(x, .float32);
            const v = try g.mean(try g.square(xf), -1, true);
            xf = try g.mul(xf, try g.rsqrt(try g.add(v, try sf(g, eps, v))));
            return g.astype(try g.mul(try g.astype(w, .float32), xf), dt);
        }

        /// A dense `nn.Linear` without bias: `x @ W.T`.
        fn linear(g: *G, x: T, w: T) !T {
            return g.matmul(x, try g.transpose(w));
        }

        fn qlinear(g: *G, x: T, q: Q(T)) !T {
            return g.qmm(x, q.w, q.s, q.mode);
        }

        /// Python `_swa_inv_freq`: `1.0 / (rope_theta ** (arange(0, rd, 2) / rd))`.
        pub fn swaInvFreq(g: *G, c: *const v41.Config) !T {
            const rd: f64 = @floatFromInt(c.rope_head_dim);
            const e = try g.div(try g.arange(0, rd, 2, .float32), try g.scalar(rd, .float32));
            const p = try g.power(try g.scalar(c.rope_theta, .float32), e);
            return g.div(try g.scalar(1.0, .float32), p);
        }

        /// Python `_compress_inv_freq` -> V4 `_yarn_inv_freq` at compress_rope_theta.
        pub fn yarnInvFreq(g: *G, c: *const v41.Config) !T {
            const dim: f64 = @floatFromInt(c.rope_head_dim);
            const base = c.compress_rope_theta;
            const e = try g.div(try g.arange(0, dim, 2, .float32), try g.scalar(dim, .float32));
            var freqs = try g.div(try g.scalar(1.0, .float32), try g.power(try g.scalar(base, .float32), e));
            const y = c.yarn;
            const orig: f64 = @floatFromInt(y.original_seq_len);
            const lr = yarnRamp(dim, base, orig, y.beta_fast, y.beta_slow);
            const ramp0 = try g.div(try g.sub(try g.arange(0, dim / 2, 1, .float32), try g.scalar(lr.low, .float32)), try g.scalar(lr.high - lr.low, .float32));
            const ramp = try g.clip(ramp0, try g.scalar(0.0, .float32), try g.scalar(1.0, .float32));
            const smooth = try g.sub(try g.scalar(1.0, .float32), ramp);
            const a = try g.mul(try g.div(freqs, try g.scalar(y.factor, .float32)), try g.sub(try g.scalar(1, .float32), smooth));
            freqs = try g.add(a, try g.mul(freqs, smooth));
            return freqs;
        }

        /// Python `_cos_sin(inv_freq, positions)`.
        pub fn cosSin(g: *G, inv_freq: T, positions: T) !CosSin {
            const ang = try g.mul(try g.expandDims(try g.astype(positions, .float32), 1), try g.expandDims(inv_freq, 0));
            return .{ .cos = try g.cos(ang), .sin = try g.sin(ang) };
        }

        /// V4 `_apply_interleaved_rope`: adjacent pairs rotated in f32, stored at x's dtype.
        pub fn interleavedRope(g: *G, x: T, cs: T, sn: T) !T {
            const shape = g.shapeOf(x);
            const dt = g.dtypeOf(x);
            var pairs = shape;
            pairs.d[shape.n - 1] = @divExact(shape.d[shape.n - 1], 2);
            pairs.d[shape.n] = 2;
            pairs.n += 1;
            const xp = try g.reshape(x, pairs.slice());
            const x0 = try lastIndex(g, xp, 0);
            const x1 = try lastIndex(g, xp, 1);
            const r0 = try g.sub(try g.mul(x0, cs), try g.mul(x1, sn));
            const r1 = try g.add(try g.mul(x0, sn), try g.mul(x1, cs));
            const out = try g.stack(&.{ r0, r1 }, -1);
            return g.astype(try g.reshape(out, shape.slice()), dt);
        }

        /// Python `_rope_last`: RoPE the last `2 * half` dims; `inverse` conjugates.
        pub fn ropeLast(g: *G, x: T, cs: CosSin, inverse: bool) !T {
            const shape = g.shapeOf(x);
            const csh = g.shapeOf(cs.cos);
            const half = csh.dim(-1);
            const rd = half * 2;
            const D = shape.dim(-1);
            const lead = try sliceLast(g, x, 0, D - rd);
            const tail = try sliceLast(g, x, D - rd, D);
            const extra: usize = shape.n - 3;
            var bs: [ops.max_dims]c_int = @splat(1);
            bs[0] = csh.dim(0);
            bs[extra + 1] = half;
            const c = try g.reshape(cs.cos, bs[0 .. extra + 2]);
            var s = try g.reshape(cs.sin, bs[0 .. extra + 2]);
            if (inverse) s = try g.neg(s);
            const roped = try interleavedRope(g, tail, c, s);
            if (D - rd == 0) return roped;
            return g.concat(&.{ lead, roped }, -1);
        }

        /// V4 `_sinkhorn_ops`: row softmax, then alternating column / row normalisation.
        pub fn sinkhorn(g: *G, comb: T, iters: u32, eps: f64) !T {
            var cb = try g.softmax(comb, -1);
            cb = try g.add(cb, try sf(g, eps, cb));
            cb = try g.div(cb, try g.add(try g.sum(cb, -2, true), try sf(g, eps, cb)));
            for (1..iters) |_| {
                cb = try g.div(cb, try g.add(try g.sum(cb, -1, true), try sf(g, eps, cb)));
                cb = try g.div(cb, try g.add(try g.sum(cb, -2, true), try sf(g, eps, cb)));
            }
            return cb;
        }

        /// `DecoderLayer._mixes` + `hc_split_sinkhorn`: the pre / post / Sinkhorn comb mixes.
        pub fn hcMixes(g: *G, c: *const v41.Config, x: T, fnw: T, base: T, scale: T) !Mixes {
            const hc: c_int = @intCast(c.hc_mult);
            const xf = try g.astype(x, .float32);
            const sh = g.shapeOf(xf);
            var fs = sh;
            fs.n -= 1;
            fs.d[fs.n - 1] = hc * sh.dim(-1);
            const flat = try g.reshape(xf, fs.slice());
            const rs = try g.rsqrt(try g.add(try g.mean(try g.square(flat), -1, true), try g.scalar(c.rms_norm_eps, .float32)));
            const mixes = try g.mul(try g.matmul(flat, try g.transpose(try g.astype(fnw, .float32))), rs);
            const total: c_int = @intCast(c.hcMix());
            const eps = c.hc_eps;
            const pre_in = try g.add(try g.mul(try sliceLast(g, mixes, 0, hc), try index0(g, scale, 0)), try sliceLast(g, base, 0, hc));
            const pre = try g.add(try g.sigmoid(pre_in), try sf(g, eps, pre_in));
            const post_in = try g.add(try g.mul(try sliceLast(g, mixes, hc, 2 * hc), try index0(g, scale, 1)), try sliceLast(g, base, hc, 2 * hc));
            const post = try g.mul(try g.scalar(2.0, .float32), try g.sigmoid(post_in));
            var comb = try g.add(try g.mul(try sliceLast(g, mixes, 2 * hc, total), try index0(g, scale, 2)), try sliceLast(g, base, 2 * hc, total));
            const ms = g.shapeOf(comb);
            var cshape = ms;
            cshape.d[ms.n - 1] = hc;
            cshape.d[ms.n] = hc;
            cshape.n += 1;
            comb = try g.reshape(comb, cshape.slice());
            return .{ .pre = pre, .post = post, .comb = try sinkhorn(g, comb, c.hc_sinkhorn_iters, eps) };
        }

        /// `DecoderLayer._hc_pre`: collapse the hc copies with the threaded pre mix.
        pub fn hcPre(g: *G, x: T, pre_mix: T) !T {
            const y = try g.sum(try g.mul(try g.expandDims(pre_mix, -1), try g.astype(x, .float32)), 2, false);
            return g.astype(y, g.dtypeOf(x));
        }

        /// V4 `_hc_post_impl`: `post * x + sum_j comb[j, k] * residual[j]`, at x's dtype.
        pub fn hcPost(g: *G, x: T, residual: T, post: T, comb: T) !T {
            const dt = g.dtypeOf(x);
            const xf = try g.astype(x, .float32);
            const rf = try g.astype(residual, .float32);
            const term = try g.mul(try g.expandDims(post, -1), try g.expandDims(xf, -2));
            const mixed = try g.einsum("...jk,...jd->...kd", &.{ comb, rf });
            return g.astype(try g.add(term, mixed), dt);
        }

        /// `_forward_span` prologue: the embedding rows expanded to hc copies
        /// (a broadcast view) and the identity pre mix.
        pub fn expandEmbedding(g: *G, c: *const v41.Config, rows: T) !Out {
            const s = g.shapeOf(rows);
            const hc: c_int = @intCast(c.hc_mult);
            const h = try g.broadcastTo(try g.expandDims(rows, 2), &.{ s.d[0], s.d[1], hc, s.d[2] });
            const pm = try g.concat(&.{ try g.ones(&.{ s.d[0], s.d[1], 1 }, .float32), try g.zeros(&.{ s.d[0], s.d[1], hc - 1 }, .float32) }, -1);
            return .{ .h = h, .pre_mix = try g.astype(pm, .float32) };
        }

        /// `nn.Embedding`: `weight[ids]`.
        pub fn embed(g: *G, weight: T, ids: T) !T {
            return g.take(weight, ids, 0);
        }

        /// `_forward_span` epilogue: collapse with the last pre mix, then the final norm.
        pub fn finalNorm(g: *G, c: *const v41.Config, h: T, pre_mix: T, norm_w: T) !T {
            const y = try g.sum(try g.mul(try g.expandDims(pre_mix, -1), try g.astype(h, .float32)), 2, false);
            return rmsnorm(g, try g.astype(y, g.dtypeOf(h)), norm_w, c.rms_norm_eps);
        }

        /// `Model._apply_head` default codec: `head(x.astype(f32))`.
        pub fn head(g: *G, x: T, head_w: T) !T {
            return linear(g, try g.astype(x, .float32), head_w);
        }

        /// `Attention._window_attend` (full-history backing, drop offset 0).
        fn windowAttend(g: *G, c: *const v41.Config, positions: T, t_len: c_int, b: c_int, s: c_int) !T {
            const wpos = try g.add(try g.scalar(0, .int32), try g.arange(0, @floatFromInt(t_len), 1, .int32));
            const qp = try g.expandDims(positions, 1);
            const wp = try g.expandDims(wpos, 0);
            const inside = try g.logicalAnd(try g.lessEqual(wp, qp), try g.greater(wp, try g.sub(qp, try g.scalar(@floatFromInt(c.window), .int32))));
            return g.broadcastTo(try g.expandDims(inside, 0), &.{ b, s, t_len });
        }

        /// `_topk_rows`: the k highest per row, ties to the lowest index.
        fn topkRows(g: *G, score: T, k: c_int) !T {
            const n = g.shapeOf(score).dim(-1);
            if (k >= n) return g.greater(score, try sf(g, -std.math.inf(f64), score));
            // ranked[..., k-1] of the descending sort == sorted[..., n-k].
            const thr = try sliceLast(g, try g.sort(score, -1), n - k, n - k + 1);
            const gt = try g.greater(score, thr);
            const eq = try g.equal(score, thr);
            const n_gt = try g.sum(try g.astype(gt, .int32), -1, true);
            const tie_rank = try g.sub(try g.cumsum(try g.astype(eq, .int32), -1), try g.scalar(1, .int32));
            const quota = try g.sub(try g.scalar(@floatFromInt(k), .int32), n_gt);
            return g.logicalOr(gt, try g.logicalAnd(eq, try g.less(tie_rank, quota)));
        }

        /// `_select_candidate_blocks`: the best `topk_blocks` blocks of compressed
        /// positions per query, the query's newest block pinned in.
        fn candidateBlocks(g: *G, c: *const v41.Config, logits: T, compress_lens: T) !T {
            const sh = g.shapeOf(logits);
            const b = sh.d[0];
            const s = sh.d[1];
            const width = sh.d[2];
            const bs: c_int = @intCast(c.candidate_block_size);
            const pad = @mod(-width, bs);
            var l = logits;
            if (pad != 0) {
                const ninf = try sf(g, -std.math.inf(f64), l);
                l = try g.concat(&.{ l, try g.full(&.{ b, s, pad }, ninf, g.dtypeOf(l)) }, -1);
            }
            const nb = @divExact(width + pad, bs);
            const scores0 = try g.max(try g.reshape(l, &.{ b, s, nb, bs }), -1, false);
            const last = try g.floorDiv(try g.sub(compress_lens, try g.scalar(1, .int32)), try g.scalar(@floatFromInt(bs), .int32));
            const blocks = try g.expandDims(try g.expandDims(try g.arange(0, @floatFromInt(nb), 1, .int32), 0), 0);
            const pin = try g.equal(blocks, try g.expandDims(try g.expandDims(last, 0), -1));
            const scores = try g.where(pin, try g.scalar(std.math.inf(f64), g.dtypeOf(scores0)), scores0);
            const kb = @min(@as(c_int, @intCast(c.candidate_topk_blocks)), nb);
            var keep = try topkRows(g, scores, kb);
            keep = try g.logicalAnd(keep, try g.greater(scores, try sf(g, -std.math.inf(f64), scores)));
            return sliceLast(g, try g.repeat(keep, bs, -1), 0, width);
        }

        /// `Indexer.keys`: wk -> k_norm -> RoPE tail of a pre-RoPE latent.
        fn indexerKeys(g: *G, c: *const v41.Config, w: *const W, latent: T, cs: CosSin) !T {
            const k = try rmsnorm(g, try linear(g, latent, w.idx_k.?.wk), w.idx_k.?.k_norm, c.rms_norm_eps);
            return ropeLast(g, k, cs, false);
        }

        const Selection = struct { mask: T, cand: ?T };

        /// `Indexer.select`: score the compressed rows, keep the top `index_topk`.
        fn indexerSelect(g: *G, p: anytype, c: *const v41.Config, w: *const W, x: T, qr: T, index_k: T, cs: CosSin, compress_lens: T, n_comp: c_int, candidates: ?T, set_candidates: bool) !Selection {
            const sh = g.shapeOf(x);
            const iq = w.idx_q.?;
            const H: c_int = @intCast(c.index_n_heads);
            const D: c_int = @intCast(c.index_head_dim);
            var q = try g.reshape(try qlinear(g, qr, iq.wq_b), &.{ sh.d[0], sh.d[1], H, D });
            q = try ropeLast(g, q, cs, false);
            const softmax_scale = std.math.pow(f64, @floatFromInt(c.index_head_dim), -0.5);
            const wts0 = try linear(g, x, iq.weights_proj);
            const wts = try g.mul(wts0, try sf(g, softmax_scale * std.math.pow(f64, @floatFromInt(c.index_n_heads), -0.5), wts0));
            var score = try g.einsum("bshd,btd->bsht", &.{ try g.astype(q, .float32), try g.astype(index_k, .float32) });
            score = try g.mul(try g.maximum(score, try sf(g, 0.0, score)), try g.expandDims(try g.astype(wts, .float32), -1));
            score = try g.sum(score, 2, false);
            const ar = try g.expandDims(try g.expandDims(try g.arange(0, @floatFromInt(n_comp), 1, .int32), 0), 0);
            const reach = try g.less(ar, try g.expandDims(try g.expandDims(compress_lens, 0), -1));
            score = try g.where(reach, score, try sf(g, -std.math.inf(f64), score));
            var cand: ?T = null;
            if (set_candidates) {
                cand = try candidateBlocks(g, c, score, compress_lens);
            } else if (candidates) |cm| {
                score = try g.where(cm, score, try sf(g, -std.math.inf(f64), score));
            }
            try p.put("attn.index_score", score);
            const k = @min(@as(c_int, @intCast(c.index_topk)), n_comp);
            const mask = try g.logicalAnd(try topkRows(g, score, k), reach);
            return .{ .mask = mask, .cand = cand };
        }

        /// `Compressor.pool` + `CompressorState.push`: the normed, pre-RoPE latents
        /// of the groups this call completes (null when none completed).
        fn compressorPool(g: *G, c: *const v41.Config, li: v41.LayerInfo, w: *const W, x: T, cache: *Cache) !?T {
            const comp = w.comp.?;
            if (li.ratio == 1) return try rmsnorm(g, try linear(g, x, comp.wkv), comp.norm, c.rms_norm_eps);
            const xf = try g.astype(x, .float32);
            const kv = try linear(g, xf, comp.wkv);
            const score = try linear(g, xf, comp.wgate.?);
            const s: u32 = @intCast(g.shapeOf(kv).dim(1));
            const n_before = cache.n_fed;
            try grow(g, &cache.raw_kv, kv);
            try grow(g, &cache.raw_score, score);
            cache.n_fed += s;
            const r: u32 = li.ratio;
            const g_before = n_before / r;
            const g_after = cache.n_fed / r;
            if (g_after == g_before) return null;
            const lo: c_int = @intCast(g_before * r);
            const hi: c_int = @intCast(g_after * r);
            const sh = g.shapeOf(kv);
            const grp: [4]c_int = .{ sh.d[0], @intCast(g_after - g_before), @intCast(r), sh.d[2] };
            const grp_kv = try g.reshape(try sliceAxis1(g, cache.raw_kv.?, lo, hi), &grp);
            const grp_sc = try g.reshape(try sliceAxis1(g, cache.raw_score.?, lo, hi), &grp);
            const pooled = try g.sum(try g.mul(grp_kv, try g.softmax(grp_sc, 2)), 2, false);
            return try rmsnorm(g, pooled, comp.norm, c.rms_norm_eps);
        }

        /// `Attention._publish_compressed`: pool, RoPE at group positions, index
        /// keys, append, publish to the forward's shared runtime.
        fn publishCompressed(g: *G, p: anytype, c: *const v41.Config, li: v41.LayerInfo, w: *const W, inv_freq: T, x: T, cache: *Cache, shared: *Share) !void {
            if (try compressorPool(g, c, li, w, x, cache)) |lat| {
                const n_prev: c_int = if (cache.compress_kv) |ck| g.shapeOf(ck).dim(1) else 0;
                const n_new = g.shapeOf(lat).dim(1);
                const gpos = try g.mul(try g.arange(@floatFromInt(n_prev), @floatFromInt(n_prev + n_new), 1, .int32), try g.scalar(@floatFromInt(li.ratio), .int32));
                const gcs = try cosSin(g, inv_freq, gpos);
                const cnew = try ropeLast(g, lat, gcs, false);
                const inew = try indexerKeys(g, c, w, lat, gcs);
                try p.put("attn.comp_latent", lat);
                try p.put("attn.compress_new", cnew);
                try p.put("attn.index_new", inew);
                try grow(g, &cache.compress_kv, cnew);
                try grow(g, &cache.index_k, inew);
            }
            shared.compress_kv = cache.compress_kv;
            shared.index_k = cache.index_k;
        }

        const Compressed = struct { kv: T, mask: T };

        /// `Attention._compressed`: the CSA2 mode dispatch.
        fn compressed(g: *G, p: anytype, c: *const v41.Config, li: v41.LayerInfo, w: *const W, inv_freq: T, x: T, qr: T, positions: T, cs: CosSin, cache: *Cache, shared: *Share) !?Compressed {
            if (li.kv_source) try publishCompressed(g, p, c, li, w, inv_freq, x, cache, shared);
            const ckv = shared.compress_kv orelse return null;
            const n_comp = g.shapeOf(ckv).dim(1);
            var mask: T = undefined;
            if (li.index_source) {
                const lens = try g.floorDiv(try g.add(positions, try g.scalar(1, .int32)), try g.scalar(@floatFromInt(li.ratio), .int32));
                const cand = if (li.candidate_source) null else shared.candidates;
                const sel = try indexerSelect(g, p, c, w, x, qr, shared.index_k.?, cs, lens, n_comp, cand, li.candidate_source);
                shared.topk_mask = sel.mask;
                if (li.candidate_source) shared.candidates = sel.cand;
                mask = sel.mask;
            } else {
                // The config refuses a reuse layer with no index source before it.
                mask = shared.topk_mask.?;
            }
            try p.put("attn.topk_mask", mask);
            return .{ .kv = ckv, .mask = mask };
        }

        /// `Attention._sparse_attend_oneshot` (f32 score path): one softmax over
        /// window + compressed rows with the per-head value-0 sink column.
        fn sparseAttend(g: *G, c: *const v41.Config, w: *const W, q: T, kv: T, attend: T) !T {
            const qs = g.shapeOf(q);
            const H: c_int = qs.d[2];
            const tk = g.shapeOf(kv).dim(1);
            const scale = std.math.pow(f64, @floatFromInt(c.head_dim), -0.5);
            var scores = try g.einsum("bshd,btd->bsht", &.{ try g.astype(q, .float32), try g.astype(kv, .float32) });
            scores = try g.mul(scores, try sf(g, scale, scores));
            scores = try g.where(try g.expandDims(attend, 2), scores, try sf(g, -std.math.inf(f64), scores));
            const sink = try g.reshape(try g.astype(w.attn_sink, .float32), &.{ 1, 1, H, 1 });
            const sink_b = try g.broadcastTo(sink, &.{ qs.d[0], qs.d[1], H, 1 });
            const full = try g.concat(&.{ scores, sink_b }, -1);
            const wts = try sliceLast(g, try g.softmax(full, -1), 0, tk);
            return g.einsum("bsht,btd->bshd", &.{ wts, try g.astype(kv, .float32) });
        }

        /// `Attention._attend`, stock eager path (masked-full attention, no
        /// compile tapes, no fused projections).
        pub fn attention(g: *G, p: anytype, c: *const v41.Config, li: v41.LayerInfo, w: *const W, inv_freq: T, x: T, positions: T, cache: *Cache, shared: *Share) !T {
            const sh = g.shapeOf(x);
            const b = sh.d[0];
            const s = sh.d[1];
            const H: c_int = @intCast(c.n_heads);
            const hd: c_int = @intCast(c.head_dim);
            const cs = try cosSin(g, inv_freq, positions);
            const qr = try rmsnorm(g, try qlinear(g, x, w.wq_a), w.q_norm, c.rms_norm_eps);
            const q = try ropeLast(g, try g.reshape(try qlinear(g, qr, w.wq_b), &.{ b, s, H, hd }), cs, false);
            const kv_new = try ropeLast(g, try rmsnorm(g, try qlinear(g, x, w.wkv), w.kv_norm, c.rms_norm_eps), cs, false);
            try p.put("attn.qr", qr);
            try p.put("attn.q", q);
            try p.put("attn.kv_new", kv_new);
            try grow(g, &cache.window, kv_new);
            const window = cache.window.?;
            var attend = try windowAttend(g, c, positions, g.shapeOf(window).dim(1), b, s);
            var kv = window;
            if (li.ratio > 0) {
                if (try compressed(g, p, c, li, w, inv_freq, x, qr, positions, cs, cache, shared)) |comp| {
                    kv = try g.concat(&.{ window, comp.kv }, 1);
                    attend = try g.concat(&.{ attend, comp.mask }, -1);
                }
            }
            const o0 = try sparseAttend(g, c, w, q, kv, attend);
            try p.put("attn.o", o0);
            const G_: c_int = @intCast(c.o_groups);
            const R: c_int = @intCast(c.o_lora_rank);
            const o1 = try g.reshape(try ropeLast(g, o0, cs, true), &.{ b, s, G_, -1 });
            const woa = try g.reshape(try g.dequantize(w.wo_a.w, w.wo_a.s, w.wo_a.mode), &.{ G_, R, -1 });
            const o2 = try g.einsum("bsgd,grd->bsgr", &.{ try g.astype(o1, .float32), try g.astype(woa, .float32) });
            const out = try qlinear(g, try g.reshape(o2, &.{ b, s, -1 }), w.wo_b);
            try p.put("attn.out", out);
            return out;
        }

        pub const Route = struct { weights: T, indices: T };

        /// `Gate.__call__`: sqrtsoftplus scores, noaux_tc biased selection,
        /// unbiased normalised weights x route scale.
        pub fn router(g: *G, p: anytype, c: *const v41.Config, w: *const W, xf: T) !Route {
            const logits = try g.div(try linear(g, try g.astype(xf, .float32), try g.astype(w.gate_w, .float32)), try g.scalar(1.0, .float32));
            const scores = try g.sqrt(try g.softplus(logits));
            const biased = try g.add(scores, w.gate_bias);
            const k: c_int = @intCast(c.n_experts_per_tok);
            const part = try sliceLast(g, try g.argpartition(try g.neg(biased), k - 1, -1), 0, k);
            const order = try g.argsort(try g.neg(try g.takeAlongAxis(biased, part, -1)), -1);
            const indices = try g.astype(try g.takeAlongAxis(part, order, -1), .int32);
            var weights = try g.takeAlongAxis(scores, indices, -1);
            if (c.norm_topk_prob and k > 1) {
                weights = try g.div(weights, try g.add(try g.sum(weights, -1, true), try g.scalar(1e-20, .float32)));
            }
            weights = try g.mul(weights, try g.scalar(c.routed_scaling_factor, .float32));
            try p.put("gate.logits", logits);
            try p.put("gate.scores", scores);
            try p.put("gate.indices", indices);
            try p.put("gate.weights", weights);
            return .{ .weights = weights, .indices = indices };
        }

        /// `Expert.__call__` (the shared expert): clamped SwiGLU in f32.
        pub fn sharedExpert(g: *G, c: *const v41.Config, w: *const W, x: T) !T {
            const dt = g.dtypeOf(x);
            var gate = try g.astype(try qlinear(g, x, w.sh_w1), .float32);
            var up = try g.astype(try qlinear(g, x, w.sh_w3), .float32);
            if (c.swiglu_limit > 0) {
                up = try g.clip(up, try g.scalar(-c.swiglu_limit, .float32), try g.scalar(c.swiglu_limit, .float32));
                gate = try g.minimum(gate, try g.scalar(c.swiglu_limit, .float32));
            }
            const h = try g.mul(try g.silu(gate), up);
            return qlinear(g, try g.astype(h, dt), w.sh_w2);
        }

        /// `MoE.__call__`: gate, routed experts (`routed.routed(g, xf, indices)`
        /// returns the unweighted `[n, k, dim]` outputs), shared expert, f32 combine.
        pub fn moe(g: *G, p: anytype, c: *const v41.Config, w: *const W, x: T, routed: anytype) !T {
            const sh = g.shapeOf(x);
            const dim: c_int = @intCast(c.hidden_size);
            const xf = try g.reshape(x, &.{ -1, dim });
            const r = try router(g, p, c, w, xf);
            const ro = try routed.routed(g, xf, r.indices);
            try p.put("moe.routed", ro);
            const shared = try g.astype(try sharedExpert(g, c, w, xf), .float32);
            try p.put("moe.shared", shared);
            const y = try g.add(try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(r.weights, -1)), -2, false), shared);
            return g.reshape(try g.astype(y, g.dtypeOf(x)), sh.slice());
        }

        /// `DecoderLayer.__call__` (eager): attention and MoE, each inside a
        /// Hyper-Connection pre / post, the pre mix threaded across sublayers.
        pub fn layer(g: *G, p: anytype, c: *const v41.Config, li: v41.LayerInfo, w: *const W, inv_freq: T, h: T, pre_mix: T, positions: T, cache: *Cache, shared: *Share, routed: anytype) !Out {
            const am = try hcMixes(g, c, h, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale);
            try p.put("attn.pre", am.pre);
            try p.put("attn.post", am.post);
            try p.put("attn.comb", am.comb);
            const ax = try rmsnorm(g, try hcPre(g, h, pre_mix), w.attn_norm, c.rms_norm_eps);
            try p.put("attn.x", ax);
            const ao = try attention(g, p, c, li, w, inv_freq, ax, positions, cache, shared);
            const h1 = try hcPost(g, ao, h, am.post, am.comb);
            try p.put("hc1.h", h1);
            const fm = try hcMixes(g, c, h1, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale);
            try p.put("ffn.pre", fm.pre);
            try p.put("ffn.post", fm.post);
            try p.put("ffn.comb", fm.comb);
            const fx = try rmsnorm(g, try hcPre(g, h1, am.pre), w.ffn_norm, c.rms_norm_eps);
            try p.put("ffn.x", fx);
            const mo = try moe(g, p, c, w, fx, routed);
            try p.put("moe.y", mo);
            const out = try hcPost(g, mo, h1, fm.post, fm.comb);
            try p.put("out.h", out);
            try p.put("out.pre_mix", fm.pre);
            return .{ .h = out, .pre_mix = fm.pre };
        }

        /// `EngramV41.__call__` after the row fetch: `rows` are the dequantized
        /// `[B, L, cols, head_dim]` bank rows; returns `h + gate * value`.
        pub fn engramApply(g: *G, c: *const v41.Config, w: EngramW(T), hidden: T, rows: T) !T {
            const hs = g.shapeOf(hidden);
            const B = hs.d[0];
            const L = hs.d[1];
            const hc: c_int = @intCast(c.hc_mult);
            const dim: c_int = @intCast(c.hidden_size);
            const kv = try qlinear(g, try g.reshape(rows, &.{ B, L, -1 }), w.wkv);
            const split = hc * dim;
            const key = try g.reshape(try g.astype(try sliceLast(g, kv, 0, split), .float32), &.{ B, L, hc, dim });
            const value = try g.astype(try sliceLast(g, kv, split, split + dim), .float32);
            const hf = try g.astype(hidden, .float32);
            const weight = try g.astype(try g.mul(w.q_weight, w.k_weight), .float32);
            const eps = c.rms_norm_eps;
            const m1 = try g.mean(try g.mul(hf, hf), -1, false);
            const m2 = try g.mean(try g.mul(key, key), -1, false);
            const rstd = try g.mul(try g.rsqrt(try g.add(m1, try sf(g, eps, m1))), try g.rsqrt(try g.add(m2, try sf(g, eps, m2))));
            const prod = try g.mul(try g.mul(hf, weight), key);
            const dot0 = try g.mul(try g.sum(prod, -1, false), rstd);
            const dot = try g.mul(dot0, try sf(g, std.math.pow(f64, @floatFromInt(c.hidden_size), -0.5), dot0));
            const mag = try g.sqrt(try g.maximum(try g.abs(dot), try sf(g, 1e-6, dot)));
            const signed = try g.where(try g.less(dot, try sf(g, 0, dot)), try g.neg(mag), mag);
            const gate = try g.sigmoid(signed);
            const contribution = try g.mul(try g.expandDims(gate, -1), try g.expandDims(value, 2));
            return g.astype(try g.add(hf, contribution), g.dtypeOf(hidden));
        }
    };
}

/// The parity dumps' routed-expert stand-in (never served): expert e's
/// output is `x * (1 + e / n)` per row, f32, the op sequence of the Python
/// `StandInSwitch` (`x.astype(f32)[:, None, :] * scale[indices][..., None]`),
/// so a chained run needs no injected expert outputs.
pub fn StandIn(comptime G: type) type {
    return struct {
        scale: G.T,

        /// `1 + e / n` rounded per f32 op, as numpy builds the Python table.
        pub fn table(buf: []f32) void {
            const n: f32 = @floatFromInt(buf.len);
            for (buf, 0..) |*v, e| v.* = 1.0 + @as(f32, @floatFromInt(e)) / n;
        }

        pub fn routed(self: @This(), g: *G, xf: G.T, indices: G.T) !G.T {
            const xs = try g.expandDims(try g.astype(xf, .float32), 1);
            return g.mul(xs, try g.expandDims(try g.take(self.scale, indices, 0), -1));
        }
    };
}

pub const Ramp = struct { low: f64, high: f64 };

/// The YaRN correction range (`_yarn_inv_freq`, Python floats): floor / ceil
/// of the correction dims, `high += 0.001` when they meet.
pub fn yarnRamp(dim: f64, base: f64, orig: f64, beta_fast: f64, beta_slow: f64) Ramp {
    const corr = struct {
        fn f(d: f64, b: f64, o: f64, rot: f64) f64 {
            return d * @log(o / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(corr(dim, base, orig, beta_fast)), 0);
    var high = @min(@ceil(corr(dim, base, orig, beta_slow)), dim - 1);
    if (low == high) high += 0.001;
    return .{ .low = low, .high = high };
}

// ── tests (host-only: TraceOps records shapes, dtypes and ops; no MLX array) ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const Tr = Trunk(TraceOps);
const Shape = ops.Shape;

/// Records the named stages a trunk function hands its probe.
const TraceProbe = struct {
    a: std.mem.Allocator,
    names: std.ArrayList([]const u8) = .empty,
    nodes: std.ArrayList(u32) = .empty,

    fn deinit(self: *TraceProbe) void {
        self.names.deinit(self.a);
        self.nodes.deinit(self.a);
    }

    pub fn put(self: *TraceProbe, name: []const u8, x: u32) !void {
        try self.names.append(self.a, name);
        try self.nodes.append(self.a, x);
    }

    /// The latest node recorded under `name`.
    fn get(self: *const TraceProbe, name: []const u8) ?u32 {
        var i = self.names.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.names.items[i], name)) return self.nodes.items[i];
        }
        return null;
    }
};

/// The deterministic routed stand-in's shape: unweighted `[n, k, dim]` f32.
const TraceRouted = struct {
    pub fn routed(_: TraceRouted, g: *TraceOps, xf: u32, indices: u32) !u32 {
        return g.input(&.{ g.shapeOf(xf).dim(0), g.shapeOf(indices).dim(1), g.shapeOf(xf).dim(1) }, .float32);
    }
};

fn qIn(g: *TraceOps, out: u64, in: u64, mode: model.QuantMode) !Q(u32) {
    const o: c_int = @intCast(out);
    const i: c_int = @intCast(in);
    const bits: c_int = @intCast(ops.quantBits(mode));
    return .{ .w = try g.input(&.{ o, @divExact(i * bits, 32) }, .uint32), .s = try g.input(&.{ o, @divExact(i, 32) }, .uint8), .mode = mode };
}

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// A layer's residents as trace inputs, dtypes as the checkpoint stores them.
fn traceLayerW(g: *TraceOps, c: *const v41.Config, li: v41.LayerInfo) !LayerW(u32) {
    const H: u64 = c.hidden_size;
    const hd: u64 = c.head_dim;
    const mix = ci(c.hcMix());
    var w: LayerW(u32) = .{
        .attn_norm = try g.input(&.{ci(H)}, .bfloat16),
        .ffn_norm = try g.input(&.{ci(H)}, .bfloat16),
        .hc_attn_fn = try g.input(&.{ mix, ci(c.hc_mult * H) }, .float32),
        .hc_attn_base = try g.input(&.{mix}, .float32),
        .hc_attn_scale = try g.input(&.{3}, .float32),
        .hc_ffn_fn = try g.input(&.{ mix, ci(c.hc_mult * H) }, .float32),
        .hc_ffn_base = try g.input(&.{mix}, .float32),
        .hc_ffn_scale = try g.input(&.{3}, .float32),
        .attn_sink = try g.input(&.{ci(c.n_heads)}, .float32),
        .q_norm = try g.input(&.{ci(c.q_lora_rank)}, .bfloat16),
        .kv_norm = try g.input(&.{ci(hd)}, .bfloat16),
        .wq_a = try qIn(g, c.q_lora_rank, H, .mxfp8),
        .wq_b = try qIn(g, c.n_heads * hd, c.q_lora_rank, .mxfp8),
        .wkv = try qIn(g, hd, H, .mxfp8),
        .wo_a = try qIn(g, c.o_groups * c.o_lora_rank, c.n_heads * hd / c.o_groups, .mxfp8),
        .wo_b = try qIn(g, H, c.o_groups * c.o_lora_rank, .mxfp8),
        .gate_w = try g.input(&.{ ci(c.n_routed_experts), ci(H) }, .bfloat16),
        .gate_bias = try g.input(&.{ci(c.n_routed_experts)}, .float32),
        .sh_w1 = try qIn(g, c.moe_intermediate_size, H, .mxfp8),
        .sh_w2 = try qIn(g, H, c.moe_intermediate_size, .mxfp8),
        .sh_w3 = try qIn(g, c.moe_intermediate_size, H, .mxfp8),
    };
    if (li.kv_source) {
        w.comp = .{
            .wkv = try g.input(&.{ ci(hd), ci(H) }, .bfloat16),
            .wgate = if (li.ratio > 1) try g.input(&.{ ci(hd), ci(H) }, .bfloat16) else null,
            .norm = try g.input(&.{ci(hd)}, .bfloat16),
        };
        w.idx_k = .{ .wk = try g.input(&.{ ci(c.index_head_dim), ci(hd) }, .bfloat16), .k_norm = try g.input(&.{ci(c.index_head_dim)}, .bfloat16) };
    }
    if (li.index_source) {
        w.idx_q = .{ .wq_b = try qIn(g, c.index_n_heads * c.index_head_dim, c.q_lora_rank, .mxfp8), .weights_proj = try g.input(&.{ ci(c.index_n_heads), ci(H) }, .bfloat16) };
    }
    return w;
}

fn realConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

fn miniConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .mini);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

fn expectShape(g: *TraceOps, x: u32, want: []const c_int, dt: Dtype) !void {
    const got = g.shapeOf(x);
    if (!got.eql(Shape.of(want)) or g.dtypeOf(x) != dt) {
        std.debug.print("shape {any} {s}, want {any} {s}\n", .{ got.slice(), @tagName(g.dtypeOf(x)), want, @tagName(dt) });
        return error.TestUnexpectedResult;
    }
}

fn expectStage(g: *TraceOps, p: *const TraceProbe, name: []const u8, want: []const c_int, dt: Dtype) !void {
    const x = p.get(name) orelse {
        std.debug.print("stage {s} not recorded\n", .{name});
        return error.TestUnexpectedResult;
    };
    expectShape(g, x, want, dt) catch |e| {
        std.debug.print("  at stage {s}\n", .{name});
        return e;
    };
}

test "dsv41 graph: rmsnorm traces _rmsnorm's op sequence and keeps the input dtype" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const x = try g.input(&.{ 1, 3, 5120 }, .bfloat16);
    const w = try g.input(&.{5120}, .bfloat16);
    const mark = g.nodes.items.len;
    const y = try Tr.rmsnorm(&g, x, w, 1e-20);
    try expectShape(&g, y, &.{ 1, 3, 5120 }, .bfloat16);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    const O = ops.Op;
    try testing.expectEqualSlices(O, &.{ .astype, .square, .mean, .scalar, .add, .rsqrt, .mul, .astype, .mul, .astype }, seq);
    // An f32 input has no casts of its own (MLX's same-dtype astype is the input).
    const xf = try g.input(&.{ 1, 3, 5120 }, .float32);
    const mark2 = g.nodes.items.len;
    _ = try Tr.rmsnorm(&g, xf, w, 1e-20);
    const seq2 = try g.opsSince(testing.allocator, mark2);
    defer testing.allocator.free(seq2);
    try testing.expectEqualSlices(O, &.{ .square, .mean, .scalar, .add, .rsqrt, .mul, .astype, .mul }, seq2);
}

test "dsv41 graph: HC mixes split pre / post / a Sinkhorn comb with 1 + 1 + 2 x 19 normalisations" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const h = try g.input(&.{ 1, 3, 4, 5120 }, .bfloat16);
    const w = try traceLayerW(&g, &c, c.layers[0]);
    const mark = g.nodes.items.len;
    const m = try Tr.hcMixes(&g, &c, h, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale);
    try expectShape(&g, m.pre, &.{ 1, 3, 4 }, .float32);
    try expectShape(&g, m.post, &.{ 1, 3, 4 }, .float32);
    try expectShape(&g, m.comb, &.{ 1, 3, 4, 4 }, .float32);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    var divs: usize = 0;
    var softmaxes: usize = 0;
    for (seq) |o| switch (o) {
        .div => divs += 1,
        .softmax => softmaxes += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), softmaxes);
    try testing.expectEqual(@as(usize, 1 + 2 * 19), divs);
    const x = try Tr.hcPre(&g, h, try g.input(&.{ 1, 3, 4 }, .float32));
    try expectShape(&g, x, &.{ 1, 3, 5120 }, .bfloat16);
    const post = try Tr.hcPost(&g, try g.input(&.{ 1, 3, 5120 }, .float32), h, m.post, m.comb);
    try expectShape(&g, post, &.{ 1, 3, 4, 5120 }, .float32);
}

test "dsv41 graph: the real layer-0 (SWA) forward turns the bf16 residual f32 at its attention" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.swaInvFreq(&g, &c);
    try expectShape(&g, inv, &.{32}, .float32);
    const rows = try g.input(&.{ 1, 5, 5120 }, .bfloat16);
    const e = try Tr.expandEmbedding(&g, &c, rows);
    try expectShape(&g, e.h, &.{ 1, 5, 4, 5120 }, .bfloat16);
    try expectShape(&g, e.pre_mix, &.{ 1, 5, 4 }, .float32);
    var cache: Tr.Cache = .{};
    var shared: Tr.Share = .{};
    const pos = try g.arange(0, 5, 1, .int32);
    const out = try Tr.layer(&g, &p, &c, li, &w, inv, e.h, e.pre_mix, pos, &cache, &shared, TraceRouted{});
    try expectStage(&g, &p, "attn.x", &.{ 1, 5, 5120 }, .bfloat16);
    try expectStage(&g, &p, "attn.qr", &.{ 1, 5, 1280 }, .bfloat16);
    try expectStage(&g, &p, "attn.q", &.{ 1, 5, 64, 512 }, .bfloat16);
    try expectStage(&g, &p, "attn.kv_new", &.{ 1, 5, 512 }, .bfloat16);
    try expectStage(&g, &p, "attn.o", &.{ 1, 5, 64, 512 }, .float32);
    // o-LoRA in f32 feeds wo_b an f32 input: the attention output (and from here
    // the residual stream) is f32, as in the Python stock path.
    try expectStage(&g, &p, "attn.out", &.{ 1, 5, 5120 }, .float32);
    try expectStage(&g, &p, "hc1.h", &.{ 1, 5, 4, 5120 }, .float32);
    try expectStage(&g, &p, "gate.indices", &.{ 5, 6 }, .int32);
    try expectStage(&g, &p, "gate.weights", &.{ 5, 6 }, .float32);
    try expectStage(&g, &p, "moe.shared", &.{ 5, 5120 }, .float32);
    try expectStage(&g, &p, "moe.y", &.{ 1, 5, 5120 }, .float32);
    try expectShape(&g, out.h, &.{ 1, 5, 4, 5120 }, .float32);
    try expectShape(&g, out.pre_mix, &.{ 1, 5, 4 }, .float32);
    try testing.expect(p.get("attn.topk_mask") == null); // SWA: no compressed branch
    try expectShape(&g, cache.window.?, &.{ 1, 5, 512 }, .bfloat16);
    const fin = try Tr.finalNorm(&g, &c, out.h, out.pre_mix, try g.input(&.{5120}, .bfloat16));
    try expectShape(&g, fin, &.{ 1, 5, 5120 }, .float32);
    const logits = try Tr.head(&g, fin, try g.input(&.{ 4096, 5120 }, .bfloat16));
    try expectShape(&g, logits, &.{ 1, 5, 4096 }, .float32);
}

test "dsv41 graph: layer 2 (Full, ratio 2) pools a group every 2 tokens across prefill and decode" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[2];
    try testing.expectEqual(v41.LayerMode.full, li.mode);
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.yarnInvFreq(&g, &c);
    try expectShape(&g, inv, &.{32}, .float32);
    var cache: Tr.Cache = .{};
    // prefill 5 tokens, then decode 2 single tokens
    const steps = [_]struct { s: c_int, rows: c_int, comp: c_int, fresh: bool }{
        .{ .s = 5, .rows = 5, .comp = 2, .fresh = true },
        .{ .s = 1, .rows = 6, .comp = 3, .fresh = true },
        .{ .s = 1, .rows = 7, .comp = 3, .fresh = false },
    };
    var pos0: c_int = 0;
    for (steps) |st| {
        var shared: Tr.Share = .{};
        p.names.clearRetainingCapacity();
        p.nodes.clearRetainingCapacity();
        const x = try g.input(&.{ 1, st.s, 5120 }, .float32);
        const pos = try g.arange(@floatFromInt(pos0), @floatFromInt(pos0 + st.s), 1, .int32);
        const out = try Tr.attention(&g, &p, &c, li, &w, inv, x, pos, &cache, &shared);
        try expectShape(&g, out, &.{ 1, st.s, 5120 }, .float32);
        try expectShape(&g, cache.window.?, &.{ 1, st.rows, 512 }, .float32);
        try expectShape(&g, cache.compress_kv.?, &.{ 1, st.comp, 512 }, .float32);
        try expectShape(&g, cache.index_k.?, &.{ 1, st.comp, 128 }, .float32);
        try testing.expectEqual(p.get("attn.compress_new") != null, st.fresh);
        try expectStage(&g, &p, "attn.index_score", &.{ 1, st.s, st.comp }, .float32);
        try expectStage(&g, &p, "attn.topk_mask", &.{ 1, st.s, st.comp }, .bool_);
        try testing.expectEqual(shared.topk_mask.?, p.get("attn.topk_mask").?);
        try testing.expect(shared.candidates == null); // layer 2 is not the candidate source
        pos0 += st.s;
    }
    try testing.expectEqual(@as(u32, 7), cache.n_fed);
    try expectShape(&g, cache.raw_kv.?, &.{ 1, 7, 512 }, .float32);
}

test "dsv41 graph: reuse, candidate and reindex layers share one forward's runtime (mini geometry)" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try miniConfig();
    var caches: [5]Tr.Cache = @splat(.{});
    var shared: Tr.Share = .{};
    const s: c_int = 9;
    const pos = try g.arange(0, 9, 1, .int32);
    const inv_c = try Tr.yarnInvFreq(&g, &c);
    for (1..5) |l| {
        const li = c.layers[l];
        const w = try traceLayerW(&g, &c, li);
        const x = try g.input(&.{ 1, s, ci(c.hidden_size) }, .float32);
        p.names.clearRetainingCapacity();
        p.nodes.clearRetainingCapacity();
        _ = try Tr.attention(&g, &p, &c, li, &w, inv_c, x, pos, &caches[l], &shared);
        switch (l) {
            // Full, ratio 2: 4 groups from 9 tokens.
            1 => try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 4 }, .bool_),
            // Reuse: reads layer 1's selection, computes none.
            2 => {
                try testing.expect(p.get("attn.index_score") == null);
                try testing.expect(caches[2].compress_kv == null);
                try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 4 }, .bool_);
            },
            // Full, ratio 1, the candidate source: sets the block mask.
            3 => {
                try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 9 }, .bool_);
                try expectShape(&g, shared.candidates.?, &.{ 1, 9, 9 }, .bool_);
            },
            // Reindex: its own queries over layer 3's keys, masked by the candidates.
            4 => {
                try expectStage(&g, &p, "attn.index_score", &.{ 1, 9, 9 }, .float32);
                try testing.expect(caches[4].compress_kv == null);
            },
            else => unreachable,
        }
    }
}

test "dsv41 graph: router, shared expert and Engram apply keep the Python dtypes" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const w = try traceLayerW(&g, &c, c.layers[3]);
    const xf = try g.input(&.{ 5, 5120 }, .float32);
    const r = try Tr.router(&g, &p, &c, &w, xf);
    try expectShape(&g, r.indices, &.{ 5, 6 }, .int32);
    try expectShape(&g, r.weights, &.{ 5, 6 }, .float32);
    try expectStage(&g, &p, "gate.scores", &.{ 5, 384 }, .float32);
    const sh = try Tr.sharedExpert(&g, &c, &w, xf);
    try expectShape(&g, sh, &.{ 5, 5120 }, .float32);
    const ew: EngramW(u32) = .{ .wkv = try qIn(&g, 5120 * 5, 24 * 256, .mxfp8), .q_weight = try g.input(&.{ 4, 5120 }, .float32), .k_weight = try g.input(&.{ 4, 5120 }, .float32) };
    const hid = try g.input(&.{ 1, 3, 4, 5120 }, .bfloat16);
    const rows = try g.input(&.{ 1, 3, 24, 256 }, .bfloat16);
    const e = try Tr.engramApply(&g, &c, ew, hid, rows);
    try expectShape(&g, e, &.{ 1, 3, 4, 5120 }, .bfloat16);
}

test "dsv41 graph: host constants round as the Python floats do" {
    // Goldens from CPython (struct.pack('<f', v)).
    const f32bits = struct {
        fn b(v: f64) u32 {
            return @bitCast(@as(f32, @floatCast(v)));
        }
    }.b;
    try testing.expectEqual(@as(u32, 0x3d3504f3), f32bits(std.math.pow(f64, 512, -0.5)));
    try testing.expectEqual(@as(u32, 0x3c800000), f32bits(std.math.pow(f64, 128, -0.5) * std.math.pow(f64, 32, -0.5)));
    try testing.expectEqual(@as(u32, 0x1e3ce508), f32bits(1e-20));
    try testing.expectEqual(@as(u32, 0x358637bd), f32bits(1e-6));
    // `_yarn_inv_freq` correction range on the real geometry (dim 64, base 160000).
    try testing.expectEqual(Ramp{ .low = 15, .high = 25 }, yarnRamp(64, 160000, 65536, 32, 1));
}

test "dsv41 graph: the routed stand-in keeps the Python table and shapes" {
    var buf: [384]f32 = undefined;
    StandIn(TraceOps).table(&buf);
    try testing.expectEqual(@as(f32, 1.0), buf[0]);
    // Goldens from numpy: 1.0 + np.arange(384, dtype=np.float32) / 384.
    try testing.expectEqual(@as(u32, 0x3F805555), @as(u32, @bitCast(buf[1])));
    try testing.expectEqual(@as(u32, 0x3FFFAAAA), @as(u32, @bitCast(buf[383])));
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const si: StandIn(TraceOps) = .{ .scale = try g.input(&.{384}, .float32) };
    const out = try si.routed(&g, try g.input(&.{ 5, 5120 }, .float32), try g.input(&.{ 5, 6 }, .int32));
    try expectShape(&g, out, &.{ 5, 6, 5120 }, .float32);
}
