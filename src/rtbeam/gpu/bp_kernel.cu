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

// Score (sequential left-to-right sum of |LS| / iterations, as the C++ loop; fixed bits add +0.0)
// and next branch bit of one run; called by every thread of the block.
__device__ __forceinline__ void summarize(int n, int max_iters, int done_at, const unsigned char* fr,
                                          const unsigned char* branchable, const REAL* LS, double* score, int* branch)
{
    const int bit = branch_bit(n, fr, branchable, LS);
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int b = 0; b < n; ++b) s += (double)fabs(LS[b]);
        *score = s / (double)(done_at ? done_at : max_iters);
        *branch = bit;
    }
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
    summarize(n, max_iters, done_at, fr, branchable, LS, score_out + r, branch_out + r);
}

// Summary only, for backends that run BP elsewhere (the OptiX mapping); one block per run.
extern "C" __global__ void __launch_bounds__(BLOCK) bp_summary(
    const int n, const int max_iters, const unsigned char* __restrict__ branchable,
    const unsigned char* __restrict__ free_all, const REAL* __restrict__ ls_all,
    const int* __restrict__ converged_at, double* __restrict__ score_out, int* __restrict__ branch_out)
{
    const size_t r = blockIdx.x;
    summarize(n, max_iters, converged_at[r], free_all + r * n, branchable, ls_all + r * n, score_out + r, branch_out + r);
}

// ---- S1: frontier freezing (one fused kernel per batch of beam-child runs) ----------------
// After `warm` full iterations (0 = start from the parent's messages), a free bit is frozen
// for the rest of the run if all its bit-to-check messages agree in sign with |v| >= tau, none
// of its checks is unsatisfied by the current hard decisions, and none touches `seed` (the bit
// this child just fixed). A frozen bit keeps its messages, posterior and decision; its
// posterior is recovered exactly from its messages: sum_e v_e = d*prior + (d-1)*sum_e c_e.
// Each check stores a summary of its frozen edges once (min1/min2/argmin, sign and decision
// parity); every later iteration touches only live checks (>= 1 frontier edge) and frontier
// bits, through compacted per-run lists. tau = inf makes every free bit frontier, and the
// kernel then reproduces bp_runs bit for bit. refresh > 0: after every `refresh` pruned
// iterations, one full iteration updates all bits and the frontier is selected again.
// sizes_out records the first selection.

__device__ __forceinline__ void full_iteration(
    int m, int n, const int* row_ptr, const int* row_edges, const int* col_ptr, const int* edge_bit,
    const REAL* prior, const unsigned char* fr, const unsigned char* syn, REAL* V, REAL* C, REAL* LS,
    unsigned char* D)
{
    check_phase(m, row_ptr, row_edges, edge_bit, fr, syn, V, C);
    __syncthreads();
    bit_phase(n, col_ptr, prior, fr, V, C, LS, D);
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(BLOCK) bp_runs_frontier(
    const int m, const int n, const int max_iters, const int want_summary, const int warm, const double tau,
    const int refresh,
    const int* __restrict__ row_ptr, const int* __restrict__ row_edges,
    const int* __restrict__ col_ptr, const int* __restrict__ edge_bit, const int* __restrict__ edge_check,
    const REAL* __restrict__ prior, const unsigned char* __restrict__ branchable,
    const unsigned char* __restrict__ free_all, const unsigned char* __restrict__ syn_all,
    const int* __restrict__ seed_all,
    REAL* __restrict__ v2c_all, REAL* __restrict__ c2v_all,
    REAL* __restrict__ ls_all, unsigned char* __restrict__ dec_all,
    int* __restrict__ converged_at, double* __restrict__ score_out, int* __restrict__ branch_out,
    REAL* __restrict__ post_all, unsigned char* __restrict__ front_all, unsigned char* __restrict__ hot_all,
    REAL* __restrict__ fzmin_all, int* __restrict__ fzarg_all, int* __restrict__ fzbits_all,
    int* __restrict__ live_all, int* __restrict__ lptr_all, int* __restrict__ fe_all, int* __restrict__ fb_all,
    int* __restrict__ sizes_out)
{
    const size_t r = blockIdx.x;
    const int E = col_ptr[n];
    const unsigned char* fr = free_all + r * n;
    const unsigned char* syn = syn_all + r * m;
    REAL* V = v2c_all + r * E;
    REAL* C = c2v_all + r * E;
    REAL* LS = ls_all + r * n;
    unsigned char* D = dec_all + r * n;
    REAL* P = post_all + r * n;
    unsigned char* front = front_all + r * n;
    unsigned char* hot = hot_all + r * m;
    REAL* fzmin = fzmin_all + r * 2 * m;
    int* fzarg = fzarg_all + r * m;
    int* fzbits = fzbits_all + r * m;
    int* live = live_all + r * m;
    int* lptr = lptr_all + r * 2 * m;
    int* fe = fe_all + r * E;
    int* fb = fb_all + r * n;
    __shared__ int s_live, s_fe, s_fb, s_stuck;
    const REAL thr = (REAL)tau;
    const bool all = isinf(tau);  // no freezing: exact bp_runs

    if (threadIdx.x == 0) { s_live = 0; s_fe = 0; s_fb = 0; s_stuck = 0; }
    for (int b = threadIdx.x; b < n; b += BLOCK) { LS[b] = 0; D[b] = 0; }
    __syncthreads();
    int done_at = 0, it = 1;
    for (; it <= warm && it <= max_iters; ++it) {
        full_iteration(m, n, row_ptr, row_edges, col_ptr, edge_bit, prior, fr, syn, V, C, LS, D);
        if (!any_unsatisfied(m, row_ptr, row_edges, edge_bit, fr, syn, D)) { done_at = it; break; }
    }
    bool first = true;
    while (!done_at && it <= max_iters) {
        if (threadIdx.x == 0) { s_live = 0; s_fe = 0; s_fb = 0; s_stuck = 0; }
        // 1. posteriors, decisions and the message criterion; reset hot checks
        for (int c = threadIdx.x; c < m; c += BLOCK) hot[c] = 0;
        for (int b = threadIdx.x; b < n; b += BLOCK) {
            front[b] = 0;
            if (!fr[b]) continue;
            const int k0 = col_ptr[b], k1 = col_ptr[b + 1], d = k1 - k0;
            REAL sum = 0;
            int negs = 0, small = 0;
            for (int e = k0; e < k1; ++e) {
                const REAL v = V[e];
                sum += v;
                negs += (v <= 0);
                small |= !(fabs(v) >= thr);
            }
            const REAL post = d >= 2 ? prior[b] + (sum - (REAL)d * prior[b]) / (REAL)(d - 1) : prior[b];
            P[b] = post;
            D[b] = (post <= 0);
            front[b] = all || small || d < 2 || (negs != 0 && negs != d) || !isfinite(post);
        }
        __syncthreads();
        // 2. hot checks: unsatisfied now, or touching the bit this child just fixed
        const int seed = seed_all[r];
        if (seed >= 0)
            for (int e = col_ptr[seed] + threadIdx.x; e < col_ptr[seed + 1]; e += BLOCK) hot[edge_check[e]] = 1;
        for (int c = threadIdx.x; c < m; c += BLOCK) {
            int parity = syn[c];
            for (int k = row_ptr[c]; k < row_ptr[c + 1]; ++k) {
                const int b = edge_bit[row_edges[k]];
                parity ^= (fr[b] & D[b]);
            }
            if (parity) hot[c] = 1;
        }
        __syncthreads();
        // 3. every free bit of a hot check joins the frontier
        for (int b = threadIdx.x; b < n; b += BLOCK) {
            if (!fr[b] || front[b]) continue;
            for (int e = col_ptr[b]; e < col_ptr[b + 1]; ++e)
                if (hot[edge_check[e]]) { front[b] = 1; break; }
        }
        __syncthreads();
        // 4. frozen summaries per check, compacted live checks / frontier edges / frontier bits
        for (int c = threadIdx.x; c < m; c += BLOCK) {
            const int k0 = row_ptr[c], k1 = row_ptr[c + 1];
            REAL min1 = REAL_MAX, min2 = REAL_MAX;
            int arg = -1, neg = syn[c], dpar = syn[c], cnt = 0;
            for (int k = k0; k < k1; ++k) {
                const int e = row_edges[k], b = edge_bit[e];
                if (!fr[b]) continue;
                if (front[b]) { ++cnt; continue; }
                const REAL v = V[e];
                neg += (v <= 0);
                dpar ^= D[b];
                const REAL a = fabs(v);
                if (a < min1) { min2 = min1; min1 = a; arg = e; }
                else if (a < min2) { min2 = a; }
            }
            fzmin[2 * c] = min1;
            fzmin[2 * c + 1] = min2;
            fzarg[c] = arg;
            fzbits[c] = (neg & 1) | ((dpar & 1) << 1);
            if (cnt) {
                const int slot = atomicAdd(&s_live, 1), base = atomicAdd(&s_fe, cnt);
                live[slot] = c;
                lptr[2 * slot] = base;
                lptr[2 * slot + 1] = base + cnt;
                int w = base;
                for (int k = k0; k < k1; ++k) {
                    const int e = row_edges[k], b = edge_bit[e];
                    if (fr[b] && front[b]) fe[w++] = e;
                }
            } else if (dpar & 1) {
                s_stuck = 1;  // unsatisfiable: no free bit left on an unsatisfied check
            }
        }
        for (int b = threadIdx.x; b < n; b += BLOCK)
            if (fr[b] && front[b]) fb[atomicAdd(&s_fb, 1)] = b;
        __syncthreads();
        const int nlive = s_live, nfb = s_fb, stuck = s_stuck;
        if (threadIdx.x == 0 && first) {
            sizes_out[3 * r] = nlive;
            sizes_out[3 * r + 1] = s_fe;
            sizes_out[3 * r + 2] = nfb;
        }
        first = false;
        int pruned = 0;
        for (; it <= max_iters && (refresh <= 0 || pruned < refresh); ++it) {
            for (int i = threadIdx.x; i < nlive; i += BLOCK) {
                const int c = live[i], j0 = lptr[2 * i], j1 = lptr[2 * i + 1];
                REAL min1 = fzmin[2 * c], min2 = fzmin[2 * c + 1];
                int arg = fzarg[c], neg = fzbits[c] & 1;
                for (int j = j0; j < j1; ++j) {
                    const int e = fe[j];
                    const REAL v = V[e];
                    neg += (v <= 0);
                    const REAL a = fabs(v);
                    if (a < min1) { min2 = min1; min1 = a; arg = e; }
                    else if (a < min2) { min2 = a; }
                }
                for (int j = j0; j < j1; ++j) {
                    const int e = fe[j];
                    const REAL mag = (e == arg) ? min2 : min1;
                    C[e] = ((neg + (V[e] <= 0)) & 1) ? -mag : mag;
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < nfb; i += BLOCK) {
                const int b = fb[i], k0 = col_ptr[b], k1 = col_ptr[b + 1];
                REAL t = prior[b];
                for (int e = k0; e < k1; ++e) { V[e] = t; t += C[e]; }
                LS[b] += t;
                D[b] = (t <= 0);
                REAL s = 0;
                for (int e = k1 - 1; e >= k0; --e) { V[e] += s; s += C[e]; }
            }
            __syncthreads();
            ++pruned;
            int bad = stuck;
            for (int i = threadIdx.x; i < nlive; i += BLOCK) {
                int parity = fzbits[live[i]] >> 1;
                for (int j = lptr[2 * i]; j < lptr[2 * i + 1]; ++j) parity ^= D[edge_bit[fe[j]]];
                bad |= parity;
            }
            if (!__syncthreads_or(bad)) { done_at = it; break; }
        }
        for (int b = threadIdx.x; b < n; b += BLOCK)
            if (fr[b] && !front[b]) LS[b] += P[b] * (REAL)pruned;
        if (done_at || it > max_iters) break;
        // refresh: one full iteration updates every bit, then the frontier is chosen again
        __syncthreads();
        full_iteration(m, n, row_ptr, row_edges, col_ptr, edge_bit, prior, fr, syn, V, C, LS, D);
        if (!any_unsatisfied(m, row_ptr, row_edges, edge_bit, fr, syn, D)) done_at = it;
        ++it;
    }
    if (threadIdx.x == 0) converged_at[r] = done_at;
    if (!want_summary) return;
    __syncthreads();
    summarize(n, max_iters, done_at, fr, branchable, LS, score_out + r, branch_out + r);
}
