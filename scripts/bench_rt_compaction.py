"""Compare masked rectangular vs compact active-bit OptiX dispatch.

Run on the Aalto GPU host with ``source env.sh && python scripts/bench_rt_compaction.py``.
Measures end-to-end BP time (including ``flatnonzero`` compaction), verifies outputs, and plots.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from rtbeam.gpu.beam import FLOAT32
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.launcher import RtLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
STEM = "BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12"
BATCHES = (8, 64, 256, 1024)
FIXED = (1, 4, 16, 64)
ITERS = 4


def main() -> None:
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{STEM}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    baseline = RtLauncher(graph, prior, FLOAT32)
    cached = RtLauncher(graph, prior, FLOAT32, cache_neighbors=True)
    compact = RtLauncher(graph, prior, FLOAT32, cache_neighbors=True, compact_dispatch=True)
    rng = np.random.default_rng(42)
    report = {"fixed_bits": {}, "batch_sizes": list(BATCHES), "iterations": ITERS}
    for fixed in FIXED:
        report["fixed_bits"][str(fixed)] = {}
        for runs in BATCHES:
            syn = cp.asarray(rng.integers(0, 2, (runs, graph.num_checks), dtype=np.uint8))
            free_np = np.ones((runs, graph.num_bits), dtype=np.uint8)
            for r in range(runs):
                free_np[r, (np.arange(fixed) + r) % graph.num_bits] = 0
            free = cp.asarray(free_np)
            def execute(launcher: RtLauncher):
                return launcher.run(launcher.initial_messages(runs), free, syn, ITERS, True)
            a, b, c = execute(baseline), execute(cached), execute(compact)
            cp.cuda.Device().synchronize()
            for result in (b, c):
                assert cp.array_equal(a.v2c, result.v2c) and cp.array_equal(a.decisions, result.decisions)
                assert cp.array_equal(a.converged_at, result.converged_at)
                assert cp.array_equal(a.llr_sums, result.llr_sums)
                assert cp.array_equal(a.score, result.score) and cp.array_equal(a.branch, result.branch)
            times = {}
            for name, launcher in (("baseline", baseline), ("cached", cached), ("compacted", compact)):
                samples = []
                for _ in range(6):
                    start = time.perf_counter()
                    execute(launcher)
                    cp.cuda.Device().synchronize()
                    samples.append(time.perf_counter() - start)
                times[name + "_ms"] = 1000 * float(np.median(samples[1:]))
            times["active_fraction"] = float(free_np.mean())
            report["fixed_bits"][str(fixed)][str(runs)] = times
            print(f"fixed={fixed:2} runs={runs:4} active={times['active_fraction']:.4f} "
                  f"baseline={times['baseline_ms']:.3f} cached={times['cached_ms']:.3f} "
                  f"compacted={times['compacted_ms']:.3f} ms", flush=True)
    (ROOT / "results" / "rt_compaction.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, len(FIXED), figsize=(13, 3.5), sharey=True, layout="constrained")
    for ax, fixed in zip(axes, FIXED):
        rows = report["fixed_bits"][str(fixed)]
        x = np.asarray(BATCHES)
        for name, label in (("baseline", "RTU"), ("cached", "Cached edge IDs"),
                            ("compacted", "Cached + compacted dispatch")):
            ax.plot(x, [rows[str(n)][name + "_ms"] for n in BATCHES], marker="o", label=label)
        ax.set_xscale("log", base=2); ax.set_title(f"{fixed} fixed bits/run")
        ax.set_xlabel("Runs in launch"); ax.grid(True, alpha=.25)
    axes[0].set_ylabel(f"BP time ({ITERS} iterations, ms)")
    axes[-1].legend(frameon=False)
    fig.suptitle("RTU BP dispatch variants, including compaction cost")
    fig.savefig(ROOT / "results" / "rt_compaction.png", dpi=180)


if __name__ == "__main__":
    main()
