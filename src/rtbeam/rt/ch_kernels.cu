// CUDA side of the closest-hit min-sum backend (fp32 only): geometry/flag initialisation and
// the bit-node update. Arithmetic and its order equal bp_kernel.cu's bit_phase, so with the
// same min1/min2/argmin/parity per check the messages are bit-identical to the CUDA kernel.
// Vertex buffer: 9 floats per triangle, triangle (r, e) at row r * E + e.

#define CH_FAR 1.0e30f
#define CH_HALF 0.3f
#define FLT_BIG 3.402823466e38f
#define MAX_DEG 16

__device__ __forceinline__ float depth(float v) { const float a = fabsf(v); return a < FLT_BIG ? a : CH_FAR; }

__device__ __forceinline__ void set_depth(float* tri, float z) { tri[2] = z; tri[5] = z; tri[8] = z; }

// x/y of the triangles of runs first..last-1 (constant per GAS slot) and z = CH_FAR.
extern "C" __global__ void ch_layout(int E, int first, int last, const int* edge_check, float* verts)
{
    const size_t i = (size_t)first * E + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)last * E) return;
    const float x = (float)edge_check[i % E], y = (float)(i / E);
    float* t = verts + 9 * i;
    t[0] = x - CH_HALF; t[1] = y - CH_HALF;
    t[3] = x + CH_HALF; t[4] = y - CH_HALF;
    t[6] = x;           t[7] = y + CH_HALF;
    set_depth(t, CH_FAR);
}

// Depth and flags from the starting messages; fixed bits' edges are absent (far, flags 0).
extern "C" __global__ void ch_init(int E, int n, int runs, const int* edge_bit, const int* csr_pos,
                                   const unsigned char* free_all, const float* v2c, float* verts, unsigned char* flags)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)runs * E) return;
    const size_t r = i / E;
    const int e = (int)(i % E);
    const int fr = free_all[r * n + edge_bit[e]];
    const float v = v2c[i];
    set_depth(verts + 9 * i, fr ? depth(v) : CH_FAR);
    flags[r * E + csr_pos[e]] = fr ? (unsigned char)(v <= 0) : 0;
}

// Bit-reliability scene for branch selection: bit b of run r at (0, r, |LS|) if free and
// branchable, else far; runs >= `runs` (unused capacity) far.
extern "C" __global__ void ch_branch_scene(int n, int runs, int capacity, const unsigned char* free_all,
                                           const unsigned char* branchable, const float* ls, float* verts)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)capacity * n) return;
    const size_t r = i / n;
    const int b = (int)(i % n);
    const float y = (float)r;
    float* t = verts + 9 * i;
    t[0] = -CH_HALF; t[1] = y - CH_HALF;
    t[3] = CH_HALF;  t[4] = y - CH_HALF;
    t[6] = 0.0f;     t[7] = y + CH_HALF;
    set_depth(t, (r < (size_t)runs && free_all[i] && branchable[b]) ? depth(ls[i]) : CH_FAR);
}

extern "C" __global__ void ch_bit(int m, int n, int E, int runs, const int* col_ptr, const int* edge_check,
                                  const int* csr_pos, const float* prior, const unsigned char* free_all,
                                  const int* done_at, const float* min1, const float* min2, const int* arg,
                                  const unsigned char* neg, float* v2c, float* ls, unsigned char* dec,
                                  float* verts, unsigned char* flags)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)runs * n) return;
    const size_t r = i / n;
    const int b = (int)(i % n);
    if (done_at[r] || !free_all[i]) return;
    const int k0 = col_ptr[b], k1 = col_ptr[b + 1];
    float* V = v2c + r * E;
    float C[MAX_DEG];
    for (int e = k0; e < k1; ++e) {
        const size_t s = r * m + edge_check[e];
        const float mag = (e == arg[s]) ? min2[s] : min1[s];
        C[e - k0] = ((neg[s] + (V[e] <= 0)) & 1) ? -mag : mag;
    }
    float t = prior[b];
    for (int e = k0; e < k1; ++e) { V[e] = t; t += C[e - k0]; }
    ls[i] += t;
    const unsigned char d = (t <= 0);
    dec[i] = d;
    float s = 0;
    for (int e = k1 - 1; e >= k0; --e) { V[e] += s; s += C[e - k0]; }
    for (int e = k0; e < k1; ++e) {
        set_depth(verts + 9 * (r * E + e), depth(V[e]));
        flags[r * E + csr_pos[e]] = (unsigned char)((V[e] <= 0) | (d << 1));
    }
}
