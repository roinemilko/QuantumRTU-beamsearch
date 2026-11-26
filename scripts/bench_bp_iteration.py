"""Cost of one BP iteration: CUDA kernel vs RT (OptiX) mapping, isolated from the beam search.

Random syndromes (BP never converges) so every run does all ``ITERS`` iterations. Reports GPU-side
time per iteration for each batch size (runs = shots or beam children launched together):
CUDA = whole bp_runs kernel / ITERS; RT = check-pass + bit-pass launch times (CUDA events), and
RT end-to-end = RtLauncher.run wall time / ITERS (incl. per-iteration convergence sync).
Writes results/bp_iteration.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from rtbeam.gpu.beam import FLOAT32, BpLauncher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.launcher import RtLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
STEM = "BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12"
RUNS = (1, 8, 64, 256, 1024, 2048)
ITERS = 30


def main() -> None:
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{STEM}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    cuda = BpLauncher(graph, prior, FLOAT32)
    setup_seconds = {}
    def make_rt(name: str, **options: object) -> RtLauncher:
        start = time.perf_counter()
        launcher = RtLauncher(graph, prior, FLOAT32, **options)
        setup_seconds[name] = time.perf_counter() - start
        return launcher

    variants = {
        "baseline": make_rt("baseline"),
        "cached_neighbors": make_rt("cached_neighbors", cache_neighbors=True),
        "check_every_4": make_rt("check_every_4", check_every=4),
        "check_every_8": make_rt("check_every_8", check_every=8),
        "rcm_geometry": make_rt("rcm_geometry", reorder_geometry=True),
        "cache_check4_rcm": make_rt("cache_check4_rcm", check_every=4, cache_neighbors=True,
                                     reorder_geometry=True),
    }
    print("setup_seconds", json.dumps(setup_seconds), flush=True)
    rng = np.random.default_rng(0)
    report = {}
    for runs in RUNS:
        syndromes = cp.asarray(rng.integers(0, 2, (runs, graph.num_checks), dtype=np.uint8))
        free = cp.ones((runs, graph.num_bits), cp.uint8)
        row = {"cuda_us_per_iter": 1e6 * wall(cuda, syndromes, free) / ITERS}
        for name, launcher in variants.items():
            row[f"rt_{name}_us_per_iter"] = 1e6 * wall(launcher, syndromes, free) / ITERS
        row["rt_us_per_iter"] = row["rt_baseline_us_per_iter"]  # preserve Task B plot schema
        row.update({f"rt_{k}_us": 1e6 * v for k, v in variants["baseline"].pass_seconds(syndromes).items()})
        report[str(runs)] = row
        print(runs, {k: round(v, 1) for k, v in row.items()}, flush=True)
    (ROOT / "results" / "bp_iteration.json").write_text(json.dumps(report, indent=2))


def wall(launcher: BpLauncher | RtLauncher, syndromes: cp.ndarray, free: cp.ndarray) -> float:
    best = np.inf
    for _ in range(4):
        v2c = launcher.initial_messages(syndromes.shape[0])
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        runs = launcher.run(v2c, free, syndromes, ITERS, want_summary=False)
        assert int(runs.converged_at.max()) == 0, "a random syndrome converged; timing would be short"
        best = min(best, time.perf_counter() - start)
    return best


if __name__ == "__main__":
    main()
