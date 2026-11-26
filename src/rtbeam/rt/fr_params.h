// Launch parameters of the frontier-scene programs (fr_programs.cu), mirrored by
// rtbeam.rt.frontier._FrParams. Layout: 11 × u64 then 4 × i32.
#pragma once

struct FrParams {
    unsigned long long gas;  // filled in by the host library
    unsigned long long live_run, live_check, prim_edge, v2c, c2v, syn, fz_min1, fz_min2, fz_arg, fz_neg;
    int m, num_edges, live_count, pad;
};
