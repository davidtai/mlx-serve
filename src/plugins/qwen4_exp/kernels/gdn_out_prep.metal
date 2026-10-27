// SPDX-License-Identifier: Apache-2.0
// The GDN tail on a trunk format's linears: in_proj_z's finish, the norm-gate and out_proj's prepare in one launch,
// xp = prepare(normgate(y, finish(zq))). One simdgroup per (row, value head = one 128-column block), NSG heads per
// threadgroup. The finish and the prepare are the format's (the quant table's `linear_finish` / `linear_prepared`
// contract): per 128-column block the Hadamard butterfly below (v[j] holds block element L + 32 j in lane L), scaled by
// 1/sqrt(128), f16 roundings where the format rounds. The finish leaves z's channel L + 32 j in lane L, the prepare's
// layout too; the norm's sum of squares runs in kernels/gdn/gdn_normgate.metal's layout (lane L over channels 4L..4L+3,
// then simd_sum), and the normalized value of any channel is elementwise once `inv` is known, so each lane reads y again
// at its own channels. The gated value is the norm-gate's (`norm_w * T(y * inv)`, the sigmoid by the table or the
// formula, the swish or sigmoid gate).
constexpr float HAD_SCALE = 0.08838834764831845f;
constexpr uint VD = uint(HV) * uint(DV);
const uint L = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.y;
const uint head = threadgroup_position_in_grid.z * uint(NSG) + simdgroup_index_in_threadgroup;
// in_proj_z's finish of block `head`: zf[j] is z at channel head * 128 + L + 32 j
float v[4];
for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(zq[row * VD + head * 128u + L + 32u * j])));
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
T zf[4];
for (uint j = 0; j < 4u; ++j) { const uint o = head * 128u + L + 32u * j; zf[j] = T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
// the norm-gate's RMS over the head: lane L sums channels 4L..4L+3
uint base = (row * uint(HV) + head) * uint(DV) + L * 4;
float xs[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    xs[i] = float(y[base + i]);
    sumsq += xs[i] * xs[i];
}
sumsq = simd_sum(sumsq);
float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
// the gated value of channel c = L + 32 j, then out_proj's prepare of it
const uint hbase = (row * uint(HV) + head) * uint(DV);
for (uint j = 0; j < 4u; ++j) {
    const uint c = L + 32u * j;
    const T normed = norm_w[c] * T(float(y[hbase + c]) * inv);
    const T zv = zf[j];
    T sig;
    if constexpr (TAB) sig = sigtab[as_type<ushort>(zv)];
    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); sig = zv < T(0) ? sy : T(1) - sy; }
    // swish gate: silu(z) * normed; sigmoid gate: normed * sigmoid(z) (qwen4_exp)
    const T gated = SWISH ? (zv * sig) * normed : normed * sig;
    v[j] = float(half(float(gated))) * float(suh[head * 128u + c]);
}
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
for (uint j = 0; j < 4u; ++j) xp[row * VD + head * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
