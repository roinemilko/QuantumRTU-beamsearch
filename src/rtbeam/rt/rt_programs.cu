// OptiX programs for min-sum BP with the Tanner graph as geometry (Task B baseline mapping).
//
// Scene: edge e = (check c, bit v) is one triangle around (x = v, y = c, z = 0) lying in the
// plane x + y = v + c, so it is hit both by the row ray of check c (direction +x) and by the
// column ray of bit v (direction +y). Neighbour discovery is the RT core's BVH traversal; the
// any-hit program only records the edge id (primitive index). Raygen does the arithmetic with
// exactly the operations of the CUDA kernel (bp_kernel.cu):
//   - check side: min1/min2/sign parity are independent of hit order (ties give equal values);
//   - bit side: hits are sorted by edge id so the prefix/suffix sums run in CSC order.
// Hence REAL=double reproduces IonQ bit for bit, REAL=float reproduces the CUDA float kernel.
// Launch grid: x = check or bit, y = run (shot or beam child).

#include <optix.h>

#include "rt_params.h"

#ifndef REAL
#define REAL double
#endif
#ifndef REAL_MAX
#define REAL_MAX 1.7976931348623157e308
#endif
#ifndef MAX_HITS
#define MAX_HITS 64
#endif

extern "C" {
__constant__ RtParams params;
}

struct Hits {
    int count;
    int edges[MAX_HITS];
};

static __forceinline__ __device__ void trace(float3 origin, float3 direction, float tmax, Hits* hits)
{
    const unsigned long long ptr = reinterpret_cast<unsigned long long>(hits);
    unsigned int p0 = static_cast<unsigned int>(ptr >> 32), p1 = static_cast<unsigned int>(ptr);
    hits->count = 0;
    optixTrace(static_cast<OptixTraversableHandle>(params.gas), origin, direction, 0.0f, tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);
}

extern "C" __global__ void __anyhit__edge()
{
    const unsigned long long ptr = (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    Hits* hits = reinterpret_cast<Hits*>(ptr);
    if (hits->count < MAX_HITS) hits->edges[hits->count] = static_cast<int>(optixGetPrimitiveIndex());
    hits->count++;
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__none() {}

static __device__ void row_ray(int c, size_t r)
{
    const int m = params.m, n = params.n;
    Hits hits;
    trace(make_float3(-1.0f, static_cast<float>(c), 0.0f), make_float3(1.0f, 0.0f, 0.0f), n + 1.0f, &hits);
    int* out = reinterpret_cast<int*>(params.unsat);
    if (params.mode == RT_COUNT_ROWS) { out[c] = hits.count; return; }

    const int* edge_bit = reinterpret_cast<const int*>(params.edge_bit);
    const unsigned char* fr = reinterpret_cast<const unsigned char*>(params.free_all) + r * n;
    const unsigned char* syn = reinterpret_cast<const unsigned char*>(params.syn_all) + r * m;
    const unsigned char* D = reinterpret_cast<const unsigned char*>(params.dec) + r * n;
    const REAL* V = reinterpret_cast<const REAL*>(params.v2c) + r * params.num_edges;
    REAL* C = reinterpret_cast<REAL*>(params.c2v) + r * params.num_edges;

    if (params.mode == RT_SYNDROME || params.it > 1) {  // syndrome test of the previous iteration
        int parity = syn[c];
        for (int k = 0; k < hits.count; ++k) {
            const int b = edge_bit[hits.edges[k]];
            parity ^= (fr[b] & D[b]);
        }
        if (parity) out[r] = 1;
    }
    if (params.mode != RT_CHECK) return;

    REAL min1 = REAL_MAX, min2 = REAL_MAX;
    int arg = -1, neg = syn[c];
    for (int k = 0; k < hits.count; ++k) {
        const int e = hits.edges[k];
        if (!fr[edge_bit[e]]) continue;
        const REAL v = V[e];
        neg += (v <= 0);
        const REAL a = fabs(v);
        if (a < min1) { min2 = min1; min1 = a; arg = e; }
        else if (a < min2) { min2 = a; }
    }
    for (int k = 0; k < hits.count; ++k) {
        const int e = hits.edges[k];
        if (!fr[edge_bit[e]]) continue;
        const REAL mag = (e == arg) ? min2 : min1;
        C[e] = ((neg + (V[e] <= 0)) & 1) ? -mag : mag;
    }
}

static __device__ void column_ray(int b, size_t r)
{
    const int n = params.n;
    const unsigned char* fr = reinterpret_cast<const unsigned char*>(params.free_all) + r * n;
    if (params.mode == RT_BIT && !fr[b]) return;
    Hits hits;
    trace(make_float3(static_cast<float>(b), -1.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), params.m + 1.0f, &hits);
    if (params.mode == RT_COUNT_COLS) { reinterpret_cast<int*>(params.unsat)[b] = hits.count; return; }

    for (int i = 1; i < hits.count; ++i) {  // hit order is unspecified: sort into CSC order
        const int e = hits.edges[i];
        int j = i - 1;
        for (; j >= 0 && hits.edges[j] > e; --j) hits.edges[j + 1] = hits.edges[j];
        hits.edges[j + 1] = e;
    }
    REAL* V = reinterpret_cast<REAL*>(params.v2c) + r * params.num_edges;
    const REAL* C = reinterpret_cast<const REAL*>(params.c2v) + r * params.num_edges;
    REAL t = reinterpret_cast<const REAL*>(params.prior)[b];
    for (int k = 0; k < hits.count; ++k) { const int e = hits.edges[k]; V[e] = t; t += C[e]; }
    reinterpret_cast<REAL*>(params.ls)[r * n + b] += t;
    reinterpret_cast<unsigned char*>(params.dec)[r * n + b] = (t <= 0);
    REAL s = 0;
    for (int k = hits.count - 1; k >= 0; --k) { const int e = hits.edges[k]; V[e] += s; s += C[e]; }
}

extern "C" __global__ void __raygen__bp()
{
    const uint3 idx = optixGetLaunchIndex();
    const size_t r = idx.y;
    const int mode = params.mode;
    if (mode <= RT_SYNDROME && reinterpret_cast<const int*>(params.done_at)[r]) return;
    if (mode == RT_BIT || mode == RT_COUNT_COLS) column_ray(static_cast<int>(idx.x), r);
    else row_ray(static_cast<int>(idx.x), r);
}
