// Frontier-restricted check update shared by the CUDA kernel (fr_kernels.cu) and the OptiX
// raygen (fr_programs.cu), so both variants compute bit-identical messages.
//
// Frozen bits keep their bit-to-check messages for the whole beam round; their contribution to
// check c is summarised once per round as (fmin1, fmin2, farg, fneg). Each iteration only the
// frontier edges fe[0..k) of the check are read; c2v is written for them only. min1/min2 follow
// bp_kernel.cu (strict '<'), so the result does not depend on the order of fe.
#pragma once

__device__ __forceinline__ void fr_check(const float* V, float* C, unsigned char syn, float fmin1, float fmin2,
                                         int farg, unsigned char fneg, const int* fe, int k)
{
    float min1 = fmin1, min2 = fmin2;
    int arg = farg, neg = syn + fneg;
    for (int i = 0; i < k; ++i) {
        const float v = V[fe[i]];
        neg += (v <= 0);
        const float a = fabsf(v);
        if (a < min1) { min2 = min1; min1 = a; arg = fe[i]; }
        else if (a < min2) { min2 = a; }
    }
    for (int i = 0; i < k; ++i) {
        const int e = fe[i];
        const float mag = (e == arg) ? min2 : min1;
        C[e] = ((neg + (V[e] <= 0)) & 1) ? -mag : mag;
    }
}
