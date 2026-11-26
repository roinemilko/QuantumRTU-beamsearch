"""R0: cost of message-encoded geometry (closest-hit min-sum), BB[[144,12,12]], RTX 5070.

For realistic message magnitudes, CUDA BP runs a few iterations on golden p = 0.001 syndromes.
Per batch size it times (CUDA events, mean of repeats): writing depths + full BVH build,
BVH refit, and the pure RT-core check pass (two closest-hit rays per check and run). The
min1/min2 from the RT core are checked against NumPy. Writes results/ch_refit.json.
Usage: bench_ch_refit.py [repeats]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import cupy as cp
import numpy as np

from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, BpLauncher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.ch_launcher import ChLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.001"
RUNS = (1, 8, 16, 64, 128, 256)
WARM_ITERS = (0, 3)


def main(repeats: int) -> None:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    cuda, ch = BpLauncher(graph, prior, FLOAT32), ChLauncher(graph, prior)
    report: dict[str, dict[str, float]] = {}
    for runs in RUNS:
        syn = cp.asarray(shots.syndromes[:runs])
        free = cp.ones((runs, graph.num_bits), cp.uint8)
        for warm in WARM_ITERS:
            v2c = cuda.initial_messages(runs)
            if warm:
                v2c = cuda.run(v2c, free, syn, warm, False).v2c
            build = timed(lambda: ch.prepare(v2c, free), repeats)
            slot = ch.prepare(v2c, free)
            refit = timed(lambda: ch.accel(slot, update=True), repeats)
            trace = timed(lambda: ch.minimums(slot, v2c, runs), repeats)
            ok = verify(graph, v2c, ch.minimums(slot, v2c, runs))
            row = {"build_with_depths_us": build, "refit_us": refit, "check_pass_rt_only_us": trace,
                   "gas_mbytes": ch.gas_bytes(slot) / 2**20, "triangles": runs * graph.num_edges, "minimums_exact": ok}
            report[f"{runs}/warm{warm}"] = row
            print(runs, warm, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    (ROOT / "results" / "ch_refit.json").write_text(json.dumps(report, indent=2))


def timed(fn: Callable[[], object], repeats: int) -> float:
    fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return float(1e3 * cp.cuda.get_elapsed_time(start, stop) / repeats)


def verify(graph: TannerGraph, v2c: cp.ndarray, result: tuple[cp.ndarray, cp.ndarray, cp.ndarray]) -> float:
    """Fraction of (run, check) whose RT min1/min2 equal the fp32 values."""
    mags = np.abs(v2c.get())[:, graph.row_order]
    starts = graph.row_starts
    min1, min2, _ = (a.get() for a in result)
    good = total = 0
    for c, (lo, hi) in enumerate(zip(starts, np.append(starts[1:], graph.num_edges))):
        row = np.sort(mags[:, lo:hi], axis=1)
        good += int(np.sum((row[:, 0] == min1[:, c]) & (row[:, 1] == min2[:, c])))
        total += row.shape[0]
    return good / total


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
