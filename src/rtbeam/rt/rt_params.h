// Launch parameters shared by the OptiX device programs and the host library.
// Layout: 11 × u64 then 6 × i32 (no padding); mirrored by rtbeam.rt.launcher._Params.
#pragma once

enum RtMode {
    RT_CHECK = 0,      // row rays: check-node update (and syndrome test of the previous iteration)
    RT_BIT = 1,        // column rays: bit-node update
    RT_SYNDROME = 2,   // row rays: syndrome test only
    RT_COUNT_ROWS = 3, // row rays, run 0 only: write hit count per check to `unsat`
    RT_COUNT_COLS = 4, // column rays, run 0 only: write hit count per bit to `unsat`
};

struct RtParams {
    unsigned long long gas;  // OptixTraversableHandle (filled in by the host library)
    unsigned long long edge_bit, prior, free_all, syn_all, v2c, c2v, ls, dec, unsat, done_at;
    int mode, m, n, num_edges, it, pad;
};
