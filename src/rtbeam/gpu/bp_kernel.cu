// Masked flooding min-sum BP: one thread block per independent run (a shot or a beam child).
// All iterations run inside the kernel; block-wide barriers separate the check, bit and
// syndrome phases. With REAL=double and --fmad=false the arithmetic and its order match
// IonQ's beam_search.hpp exactly:
//   - bit-to-check message <= 0 counts as negative;
//   - exclusive min over the other free edges, starting from REAL_MAX, strict '<' updates
//     (so NaN/inf never lower it and a lone free edge gets REAL_MAX);
//   - column sums: prefix (prior + c_0 + ... + c_{k-1}) then suffix ((0 + c_{d-1}) + ... );
//   - hard decision posterior <= 0.
// Layout: edges are numbered column-major (rows ascending within a column), so a bit's edges
// are the contiguous range col_ptr[b]..col_ptr[b+1]; row_edges lists edge ids per check.

#ifndef REAL
#define REAL double
#endif
#ifndef REAL_MAX
#define REAL_MAX 1.7976931348623157e308
#endif
#define BLOCK 256

__device__ __forceinline__ void check_phase(
    int m, const int* row_ptr, const int* row_edges, const int* edge_bit,
    const unsigned char* fr, const unsigned char* syn, const REAL* V, REAL* C)
{
    for (int c = threadIdx.x; c < m; c += BLOCK) {
        const int k0 = row_ptr[c], k1 = row_ptr[c + 1];
        REAL min1 = REAL_MAX, min2 = REAL_MAX;
        int arg = -1, neg = syn[c];
        for (int k = k0; k < k1; ++k) {
            const int e = row_edges[k];
            if (!fr[edge_bit[e]]) continue;
            const REAL v = V[e];
            neg += (v <= 0);
            const REAL a = fabs(v);
            if (a < min1) { min2 = min1; min1 = a; arg = e; }
            else if (a < min2) { min2 = a; }
        }
        for (int k = k0; k < k1; ++k) {
            const int e = row_edges[k];
            if (!fr[edge_bit[e]]) continue;
            const REAL mag = (e == arg) ? min2 : min1;
            C[e] = ((neg + (V[e] <= 0)) & 1) ? -mag : mag;
        }
    }
}

__device__ __forceinline__ void bit_phase(
    int n, const int* col_ptr, const REAL* prior, const unsigned char* fr,
    REAL* V, const REAL* C, REAL* LS, unsigned char* D)
{
    for (int b = threadIdx.x; b < n; b += BLOCK) {
        if (!fr[b]) continue;
        const int k0 = col_ptr[b], k1 = col_ptr[b + 1];
        REAL t = prior[b];
        for (int e = k0; e < k1; ++e) { V[e] = t; t += C[e]; }
        LS[b] += t;
        D[b] = (t <= 0);
        REAL s = 0;
        for (int e = k1 - 1; e >= k0; --e) { V[e] += s; s += C[e]; }
    }
}

__device__ __forceinline__ int any_unsatisfied(
    int m, const int* row_ptr, const int* row_edges, const int* edge_bit,
    const unsigned char* fr, const unsigned char* syn, const unsigned char* D)
{
    int bad = 0;
    for (int c = threadIdx.x; c < m; c += BLOCK) {
        int parity = syn[c];
        for (int k = row_ptr[c]; k < row_ptr[c + 1]; ++k) {
            const int b = edge_bit[row_edges[k]];
            parity ^= (fr[b] & D[b]);
        }
        bad |= parity;
    }
    return __syncthreads_or(bad);
}

// argmin over free branchable bits of |LS|, strict '<' from REAL_MAX, lowest index on ties.
__device__ __forceinline__ int branch_bit(int n, const unsigned char* fr, const unsigned char* branchable, const REAL* LS)
{
    __shared__ REAL best_val[BLOCK];
    __shared__ int best_idx[BLOCK];
    REAL best = REAL_MAX;
    int idx = 0x7fffffff;
    for (int b = threadIdx.x; b < n; b += BLOCK) {
        if (!(fr[b] && branchable[b])) continue;
        const REAL a = fabs(LS[b]);
        if (a < best) { best = a; idx = b; }
    }
    best_val[threadIdx.x] = best;
    best_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const REAL v = best_val[threadIdx.x + stride];
            const int i = best_idx[threadIdx.x + stride];
            if (v < best_val[threadIdx.x] || (v == best_val[threadIdx.x] && i < best_idx[threadIdx.x])) {
                best_val[threadIdx.x] = v;
                best_idx[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    return best_idx[0] == 0x7fffffff ? -1 : best_idx[0];
}

extern "C" __global__ void __launch_bounds__(BLOCK) bp_runs(
    const int m, const int n, const int max_iters, const int want_summary,
    const int* __restrict__ row_ptr, const int* __restrict__ row_edges,
    const int* __restrict__ col_ptr, const int* __restrict__ edge_bit,
    const REAL* __restrict__ prior, const unsigned char* __restrict__ branchable,
    const unsigned char* __restrict__ free_all, const unsigned char* __restrict__ syn_all,
    REAL* __restrict__ v2c_all, REAL* __restrict__ c2v_all,
    REAL* __restrict__ ls_all, unsigned char* __restrict__ dec_all,
    int* __restrict__ converged_at, double* __restrict__ score_out, int* __restrict__ branch_out)
{
    const size_t r = blockIdx.x;
    const int E = col_ptr[n];
    const unsigned char* fr = free_all + r * n;
    const unsigned char* syn = syn_all + r * m;
    REAL* V = v2c_all + r * E;
    REAL* C = c2v_all + r * E;
    REAL* LS = ls_all + r * n;
    unsigned char* D = dec_all + r * n;

    for (int b = threadIdx.x; b < n; b += BLOCK) { LS[b] = 0; D[b] = 0; }
    __syncthreads();
    int done_at = 0;
    for (int it = 1; it <= max_iters; ++it) {
        check_phase(m, row_ptr, row_edges, edge_bit, fr, syn, V, C);
        __syncthreads();
        bit_phase(n, col_ptr, prior, fr, V, C, LS, D);
        __syncthreads();
        if (!any_unsatisfied(m, row_ptr, row_edges, edge_bit, fr, syn, D)) { done_at = it; break; }
    }
    if (threadIdx.x == 0) converged_at[r] = done_at;
    if (!want_summary) return;
    __syncthreads();
    const int branch = branch_bit(n, fr, branchable, LS);
    if (threadIdx.x == 0) {
        // Sequential left-to-right sum, as the C++ loop does (fixed bits contribute +0.0).
        double s = 0.0;
        for (int b = 0; b < n; ++b) s += (double)fabs(LS[b]);
        score_out[r] = s / (double)(done_at ? done_at : max_iters);
        branch_out[r] = branch;
    }
}
