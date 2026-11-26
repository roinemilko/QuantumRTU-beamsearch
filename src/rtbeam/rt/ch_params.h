// Launch parameters of the closest-hit min-sum programs (ch_programs.cu), mirrored by
// rtbeam.rt.ch_launcher._ChParams. Layout: 13 × u64 then 6 × i32 (no padding).
#pragma once

enum ChMode {
    CH_CHECK = 0,     // row flags (syndrome test of the previous iteration) + min1/min2 by closest hit
    CH_SYNDROME = 1,  // row flags only: syndrome test
    CH_MIN_ONLY = 2,  // min1/min2 by closest hit only (microbenchmark: pure RT-core work)
    CH_BRANCH = 3,    // one ray per run over the bit-reliability scene: branch bit = argmin |LLR sum|
};

struct ChParams {
    unsigned long long gas;  // OptixTraversableHandle (filled in by the host library)
    unsigned long long v2c, syn_all, flags, row_ptr, min1, min2, arg, neg, unsat, done_at, hit_t, spare;
    int mode, m, n, num_edges, it, pad;
};
