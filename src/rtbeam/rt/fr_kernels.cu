// CUDA side of the S3 frontier-scene experiment (float32, all bits free).
#include "fr_common.h"

#define FLT_BIG 3.402823466e38f

// Per (run, check): min1/min2/argmin/parity over the frozen (non-frontier) edges of the row.
extern "C" __global__ void fr_frozen(int m, int E, int runs, const int* row_ptr, const int* row_edges,
                                     const unsigned char* frontier, const float* v2c, float* min1, float* min2,
                                     int* arg, unsigned char* neg)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)runs * m) return;
    const size_t r = i / m;
    const int c = (int)(i % m);
    const float* V = v2c + r * E;
    const unsigned char* F = frontier + r * E;
    float a1 = FLT_BIG, a2 = FLT_BIG;
    int ar = -1, ng = 0;
    for (int k = row_ptr[c]; k < row_ptr[c + 1]; ++k) {
        const int e = row_edges[k];
        if (F[e]) continue;
        const float v = V[e];
        ng += (v <= 0);
        const float a = fabsf(v);
        if (a < a1) { a2 = a1; a1 = a; ar = e; }
        else if (a < a2) { a2 = a; }
    }
    min1[i] = a1; min2[i] = a2; arg[i] = ar; neg[i] = (unsigned char)(ng & 1);
}

// CUDA variant of the frontier check pass: one thread per live (run, check), compacted edge list.
extern "C" __global__ void fr_check_cuda(int count, int m, int E, const int* live_run, const int* live_check,
                                         const int* fe_ptr, const int* fe_edges, const float* v2c, float* c2v,
                                         const unsigned char* syn, const float* fz_min1, const float* fz_min2,
                                         const int* fz_arg, const unsigned char* fz_neg)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t r = live_run[i];
    const int c = live_check[i];
    const size_t s = r * m + c;
    fr_check(v2c + r * E, c2v + r * E, syn[s], fz_min1[s], fz_min2[s], fz_arg[s], fz_neg[s], fe_edges + fe_ptr[i],
             fe_ptr[i + 1] - fe_ptr[i]);
}

// Bit pass over frontier bits (shared by both variants): same arithmetic as bp_kernel.cu's bit_phase.
extern "C" __global__ void fr_bit(int count, int n, int E, const int* fb_run, const int* fb_bit, const int* col_ptr,
                                  const float* prior, float* v2c, const float* c2v)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t r = fb_run[i];
    const int b = fb_bit[i];
    float* V = v2c + r * E;
    const float* C = c2v + r * E;
    const int k0 = col_ptr[b], k1 = col_ptr[b + 1];
    float t = prior[b];
    for (int e = k0; e < k1; ++e) { V[e] = t; t += C[e]; }
    float s = 0;
    for (int e = k1 - 1; e >= k0; --e) { V[e] += s; s += C[e]; }
}

// RT variant: one Task-B triangle per frontier edge, run r in the z-slab around z = 2r.
extern "C" __global__ void fr_geometry(int count, const int* fe_run, const int* fe_edges, const int* edge_bit,
                                       const int* edge_check, float* verts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int e = fe_edges[i];
    const float v = (float)edge_bit[e], c = (float)edge_check[e], z = 2.0f * (float)fe_run[i];
    float* t = verts + 9 * (size_t)i;
    t[0] = v - 0.25f; t[1] = c + 0.25f; t[2] = z - 0.25f;
    t[3] = v + 0.25f; t[4] = c - 0.25f; t[5] = z - 0.25f;
    t[6] = v;         t[7] = c;         t[8] = z + 0.4f;
}
