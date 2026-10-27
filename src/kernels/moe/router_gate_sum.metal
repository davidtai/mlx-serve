    float gt = gp[0];
    for (int sgn = 1; sgn < 8; sgn++) gt += gp[sgn];
    glogit[row] = static_cast<bfloat16_t>(gt);
