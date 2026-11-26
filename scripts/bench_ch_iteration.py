"""R1: one BP iteration with closest-hit min-sum vs CUDA and the baseline RT mapping (fp32).

Random syndromes (no convergence), 30 iterations, wall time per iteration incl. host syncs,
plus the GPU time of each closest-hit stage. Writes results/ch_iteration.json.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cupy as cp
import numpy as np

from rtbeam.gpu.beam import FLOAT32, BpLauncher, Launcher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.ch_launcher import ChLauncher
from rtbeam.rt.launcher import RtLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
STEM = "BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12"
RUNS = (1, 8, 64, 256)
ITERS = 30


def main() -> None:
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{STEM}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    launchers: dict[str, Launcher] = {"cuda": BpLauncher(graph, prior, FLOAT32), "rt": RtLauncher(graph, prior, FLOAT32),
                 "ch_refit": ChLauncher(graph, prior, rebuild=False), "ch_refit_check4": ChLauncher(graph, prior, rebuild=False, check_every=4)}
    rng = np.random.default_rng(0)
    report = {}
    for runs in RUNS:
        syn = cp.asarray(rng.integers(0, 2, (runs, graph.num_checks), dtype=np.uint8))
        free = cp.ones((runs, graph.num_bits), cp.uint8)
        row: dict[str, float] = {}
        for name, launcher in launchers.items():
            out = launcher.run(launcher.initial_messages(runs), free, syn, ITERS, False)
            assert not bool((out.converged_at > 0).any()), "random syndromes should not converge"
            row[f"{name}_us_per_iter"] = 1e6 * wall(lambda: launcher.run(launcher.initial_messages(runs), free, syn, ITERS, False)) / ITERS
        ch = launchers["ch_refit"]
        assert isinstance(ch, ChLauncher)
        v2c = ch.initial_messages(runs)
        slot = ch.prepare(v2c, free)
        row["ch_refit_gpu_us"] = gpu(lambda: ch.accel(slot, update=True))
        row["ch_min_trace_gpu_us"] = gpu(lambda: ch.minimums(slot, v2c, runs))
        report[str(runs)] = row
        print(runs, {k: round(v, 1) for k, v in row.items()}, flush=True)
    (ROOT / "results" / "ch_iteration.json").write_text(json.dumps(report, indent=2))


def wall(fn: Callable[[], object], repeats: int = 3) -> float:
    fn()
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def gpu(fn: Callable[[], object], repeats: int = 20) -> float:
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
