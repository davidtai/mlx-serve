  {
    // MLX's gemv (gemv.h GEMVKernel::run, BM 1 BN 8 SM 1 SN 32 TN 4) for one output row: simdgroup gs, its lanes
    const int gs = int(tid / 32u) - 1;
    float gres = 0.0f;
    int gbn = (32 * gs + int(lane)) * 4;
    const device bfloat16_t* gmat = gx + (size_t)row * (size_t)GK;
    float gcoef[4];
    bfloat16_t ginter[4];
    for (int i = 0; i < GK / 1024; ++i) {
      for (int tn = 0; tn < 4; tn++) gcoef[tn] = static_cast<float>(gw[gbn + tn]);
      for (int tn = 0; tn < 4; tn++) ginter[tn] = static_cast<bfloat16_t>(gmat[gbn + tn]);
      for (int tn = 0; tn < 4; tn++) gres += ginter[tn] * gcoef[tn];
      gbn += 1024;
    }
    if (GK % 1024 > 0) {
      if (gbn + 4 <= GK) {
        for (int tn = 0; tn < 4; tn++) gcoef[tn] = static_cast<float>(gw[gbn + tn]);
      } else {
        for (int tn = 0; tn < 4; tn++) gcoef[tn] = gbn + tn < GK ? static_cast<float>(gw[gbn + tn]) : float(0);
      }
      if (gbn + 4 <= GK) {
        for (int tn = 0; tn < 4; tn++) ginter[tn] = static_cast<bfloat16_t>(gmat[gbn + tn]);
      } else {
        for (int tn = 0; tn < 4; tn++) ginter[tn] = gbn + tn < GK ? static_cast<bfloat16_t>(gmat[gbn + tn]) : bfloat16_t(0);
      }
      for (int tn = 0; tn < 4; tn++) gres += ginter[tn] * gcoef[tn];
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) gres += simd_shuffle_down(gres, sn);
    if (lane == 0) gp[gs] = gres;
  }
