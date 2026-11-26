// OptiX programs of the S3 frontier-scene experiment: the scene holds one triangle per frontier
// edge only (rebuilt once per beam round, run r in the z-slab at 2r). A row ray per live
// (run, check) finds that check's frontier edges by BVH traversal; any-hit records their edge
// ids; raygen then runs the same frontier check update as the CUDA variant (fr_common.h).

#include <optix.h>

#include "fr_common.h"
#include "fr_params.h"

#define FR_MAX_HITS 64

extern "C" {
__constant__ FrParams params;
}

struct Hits {
    int count;
    int edges[FR_MAX_HITS];
};

extern "C" __global__ void __anyhit__collect()
{
    const unsigned long long ptr = (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    Hits* hits = reinterpret_cast<Hits*>(ptr);
    if (hits->count < FR_MAX_HITS)
        hits->edges[hits->count++] = reinterpret_cast<const int*>(params.prim_edge)[optixGetPrimitiveIndex()];
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__none() {}

extern "C" __global__ void __raygen__rows()
{
    const int i = static_cast<int>(optixGetLaunchIndex().x);
    if (i >= params.live_count) return;
    const size_t r = reinterpret_cast<const int*>(params.live_run)[i];
    const int c = reinterpret_cast<const int*>(params.live_check)[i];
    Hits hits;
    hits.count = 0;
    const unsigned long long ptr = reinterpret_cast<unsigned long long>(&hits);
    unsigned p0 = static_cast<unsigned>(ptr >> 32), p1 = static_cast<unsigned>(ptr);
    optixTrace(static_cast<OptixTraversableHandle>(params.gas), make_float3(-1.0f, static_cast<float>(c), 2.0f * r),
               make_float3(1.0f, 0.0f, 0.0f), 0.0f, 1.0e7f, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);
    const size_t s = r * params.m + c;
    const size_t E = params.num_edges;
    fr_check(reinterpret_cast<const float*>(params.v2c) + r * E, reinterpret_cast<float*>(params.c2v) + r * E,
             reinterpret_cast<const unsigned char*>(params.syn)[s], reinterpret_cast<const float*>(params.fz_min1)[s],
             reinterpret_cast<const float*>(params.fz_min2)[s], reinterpret_cast<const int*>(params.fz_arg)[s],
             reinterpret_cast<const unsigned char*>(params.fz_neg)[s], hits.edges, hits.count);
}
