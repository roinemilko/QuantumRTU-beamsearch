"""R2: does the RT core's closest-hit search get cheaper as BP proceeds, and what does refitting
(instead of rebuilding) cost in trace speed?

Messages after k CUDA BP iterations on golden p = 0.001 shots (converged runs keep their last
messages). For each k: share of "frontier" edges (|v2c| < 6), GPU time of the pure RT check
pass (both closest-hit rays for every check) on a freshly built BVH and on a BVH built for the
k = 0 messages and refitted to the k-iteration messages. Writes results/ch_traversal.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cupy as cp

from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, BpLauncher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.ch_launcher import ChLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.001"
RUNS = (1, 64, 256)
ITERS = (0, 1, 2, 3, 5, 10, 20, 30)
FRONTIER = 6.0


def main() -> None:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    cuda = BpLauncher(graph, prior, FLOAT32)
    fresh, refit = ChLauncher(graph, prior, rebuild=True), ChLauncher(graph, prior, rebuild=False)
    report: dict[str, dict[str, float]] = {}
    for runs in RUNS:
        syn = cp.asarray(shots.syndromes[:runs])
        free = cp.ones((runs, graph.num_bits), cp.uint8)
        refit.prepare(cuda.initial_messages(runs), free)  # BVH topology from the k = 0 messages
        for k in ITERS:
            v2c = cuda.initial_messages(runs)
            converged = 0.0
            if k:
                out = cuda.run(v2c, free, syn, k, False)
                v2c, converged = out.v2c, float((out.converged_at > 0).mean())
            slot_fresh = fresh.prepare(v2c, free)
            slot_refit = refit.prepare(v2c, free)
            row = {"frontier_edge_share": float((cp.abs(v2c) < FRONTIER).mean()), "converged_share": converged,
                   "trace_fresh_bvh_us": gpu(lambda: fresh.minimums(slot_fresh, v2c, runs)),
                   "trace_refitted_bvh_us": gpu(lambda: refit.minimums(slot_refit, v2c, runs))}
            report[f"{runs}/{k}"] = row
            print(runs, k, {key: round(v, 3) for key, v in row.items()}, flush=True)
    (ROOT / "results" / "ch_traversal.json").write_text(json.dumps(report, indent=2))


def gpu(fn: Callable[[], object], repeats: int = 30) -> float:
    fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return float(1e3 * cp.cuda.get_elapsed_time(start, stop) / repeats)


if __name__ == "__main__":
    main()
