// Launch parameters shared by the OptiX device programs and the host library.
// Layout: 17 × u64 then 7 × i32; ABI rounds the 164 bytes of fields up to 168 bytes.
// Mirrored by rtbeam.rt.launcher._Params.
#pragma once

enum RtMode {
    RT_CHECK = 0,      // row rays: check-node update (and syndrome test of the previous iteration)
    RT_BIT = 1,        // column rays: bit-node update
    RT_SYNDROME = 2,   // row rays: syndrome test only
    RT_COUNT_ROWS = 3, // row rays, run 0 only: write hit count per check to `unsat`
    RT_COUNT_COLS = 4, // column rays, run 0 only: write hit count per bit to `unsat`
    RT_BUILD_ROWS = 5, // one-time row-neighbor cache construction via BVH traversal
    RT_BUILD_COLS = 6, // one-time column-neighbor cache construction via BVH traversal
    RT_BIT_COMPACT = 7, // compacted (bit, run) ray tasks
};

struct RtParams {
    unsigned long long gas;  // OptixTraversableHandle (filled in by the host library)
    unsigned long long edge_bit, check_coord, bit_coord, prior, free_all, syn_all, v2c, c2v, ls, dec;
    unsigned long long row_neighbors, col_neighbors, row_counts, col_counts, unsat, done_at, active_tasks;
    int mode, m, n, num_edges, it, cache_neighbors, pad;
};
