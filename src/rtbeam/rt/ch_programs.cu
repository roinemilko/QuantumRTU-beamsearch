// OptiX programs for min-sum BP where the RT core computes the check-node minimums.
//
// Scene (one GAS per batch of runs, rebuilt or refitted by the host): the message on edge e of
// run r is a triangle lying in the plane z = |v2c[r][e]|, centred on (x = check(e), y = r).
// So every (check, run) pair owns a "tube" along +z whose triangles sit at the magnitudes of its
// incoming messages; edges of fixed bits sit at z = CH_FAR, beyond the ray's tmax.
//
// Check update of (c, r): ray 1 from (c, r, -CH_ORIGIN) along +z with any-hit disabled. The RT
// core's closest-hit search returns the smallest magnitude and its edge (min1, argmin) without
// running any shader per hit. Ray 2 starts just before min1 and ignores argmin (one any-hit
// filter), so its closest hit is min2 (equal to min1 on ties, as in the CUDA kernel). Values
// are read back from memory by edge id, so min1/min2 are the exact fp32 messages.
// Sign parity and the syndrome test of the previous iteration come from one byte per edge
// ("flags", CSR order) written by the CUDA bit kernel: bit 0 = message <= 0, bit 1 = decision.

#include <optix.h>

#include "ch_params.h"

#define CH_FAR 1.0e30f
#define CH_TMAX 1.0e29f
#define CH_ORIGIN 0.0009765625f  // 2^-10
#define CH_NONE 0xffffffffu

extern "C" {
__constant__ ChParams params;
}

// payload 0: primitive index (CH_NONE on miss); 1: hit distance bits; 2: primitive to ignore.
extern "C" __global__ void __closesthit__min()
{
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __anyhit__skip()
{
    if (optixGetPrimitiveIndex() == optixGetPayload_2()) optixIgnoreIntersection();
}

extern "C" __global__ void __miss__none() { optixSetPayload_0(CH_NONE); }

static __forceinline__ __device__ unsigned closest(float x, float y, float tmin, unsigned skip, unsigned flags, float* t)
{
    unsigned prim = CH_NONE, tbits = 0, ignore = skip;
    optixTrace(static_cast<OptixTraversableHandle>(params.gas), make_float3(x, y, -CH_ORIGIN), make_float3(0.0f, 0.0f, 1.0f),
               tmin, CH_TMAX, 0.0f, OptixVisibilityMask(255), flags, 0, 1, 0, prim, tbits, ignore);
    *t = __uint_as_float(tbits);
    return prim;
}

static __device__ void check_minimums(int c, size_t r)
{
    const int m = params.m, E = params.num_edges;
    const float x = static_cast<float>(c), y = static_cast<float>(r);
    float t1, t2;
    const unsigned p1 = closest(x, y, 0.0f, CH_NONE, OPTIX_RAY_FLAG_DISABLE_ANYHIT, &t1);
    unsigned p2 = CH_NONE;
    if (p1 != CH_NONE) p2 = closest(x, y, nextafterf(t1, 0.0f), p1, OPTIX_RAY_FLAG_NONE, &t2);
    const float* V = reinterpret_cast<const float*>(params.v2c);
    const size_t base = r * static_cast<size_t>(E);
    const size_t s = r * m + c;
    const int a1 = p1 == CH_NONE ? -1 : static_cast<int>(p1 - base);
    reinterpret_cast<float*>(params.min1)[s] = a1 < 0 ? 3.402823466e38f : fabsf(V[base + a1]);
    reinterpret_cast<float*>(params.min2)[s] = p2 == CH_NONE ? 3.402823466e38f : fabsf(V[p2]);
    reinterpret_cast<int*>(params.arg)[s] = a1;
    if (params.hit_t) reinterpret_cast<float*>(params.hit_t)[s] = p1 == CH_NONE ? -1.0f : t1;
}

extern "C" __global__ void __raygen__check()
{
    const uint3 idx = optixGetLaunchIndex();
    const int c = static_cast<int>(idx.x);
    const size_t r = idx.y;
    const int mode = params.mode;
    if (mode == CH_MIN_ONLY) { check_minimums(c, r); return; }
    if (mode == CH_BRANCH) {  // scene: bit b of run r at depth |LS[r][b]| in tube (0, r)
        float t;
        const unsigned p = closest(0.0f, static_cast<float>(r), 0.0f, CH_NONE, OPTIX_RAY_FLAG_DISABLE_ANYHIT, &t);
        reinterpret_cast<int*>(params.arg)[r] = p == CH_NONE ? -1 : static_cast<int>(p - r * static_cast<size_t>(params.n));
        return;
    }
    if (reinterpret_cast<const int*>(params.done_at)[r]) return;
    const int* row_ptr = reinterpret_cast<const int*>(params.row_ptr);
    const unsigned char* F = reinterpret_cast<const unsigned char*>(params.flags) + r * static_cast<size_t>(params.num_edges);
    const unsigned char syn = reinterpret_cast<const unsigned char*>(params.syn_all)[r * params.m + c];
    unsigned acc = 0;
    for (int k = row_ptr[c]; k < row_ptr[c + 1]; ++k) acc ^= F[k];
    if ((mode == CH_SYNDROME || params.it > 1) && (syn ^ ((acc >> 1) & 1u))) reinterpret_cast<int*>(params.unsat)[r] = 1;
    if (mode != CH_CHECK) return;
    reinterpret_cast<unsigned char*>(params.neg)[r * params.m + c] = static_cast<unsigned char>((syn + (acc & 1u)) & 1u);
    check_minimums(c, r);
}
