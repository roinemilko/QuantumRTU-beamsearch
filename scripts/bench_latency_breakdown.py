"""Batch-one latency split: GPU time of the initial-BP kernel, GPU time of the beam-round kernels,
and everything else (host planner, input building, copies, launches, syncs), for the unpruned
CUDA decoder and S1 freezing configurations.

Shots: the first SHOTS non-trivial shots of LER batch 0 at p (beam8). GPU times come from CUDA
events around each BpLauncher.run call (one sync per call; the decoder syncs there anyway when it
reads the results). Writes results/latency_breakdown.json (keys "<basis>/<p>/<config>").
Usage: bench_latency_breakdown.py [--p 0.001 0.003]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import stim

from rtbeam.decoders import PRESETS
from rtbeam.gpu.beam import FLOAT32, BpLauncher, Freezing, GpuBeamSearch
from rtbeam.ler import sample_batch
from rtbeam.problem import DecodingProblem

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
SHOTS = 300
CONFIGS: dict[str, Freezing | None] = {
    "unpruned": None, "tau32": Freezing(32.0), "tau32_w3": Freezing(32.0, 3), "tau32_w3_r5": Freezing(32.0, 3, refresh=5),
    "tau32_w3_init5": Freezing(32.0, 3, initial_warm=5),
}


class Timer:
    """Wraps BpLauncher.run; records GPU milliseconds per call, split initial (no summary) / rounds."""

    def __init__(self) -> None:
        self.initial: list[float] = []
        self.rounds: list[float] = []
        original = BpLauncher.run

        def run(launcher: BpLauncher, *args: Any, **kwargs: Any) -> Any:
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            out = original(launcher, *args, **kwargs)
            stop.record()
            stop.synchronize()
            want_summary = kwargs.get("want_summary", args[4] if len(args) > 4 else False)
            (self.rounds if want_summary else self.initial).append(cp.cuda.get_elapsed_time(start, stop))
            return out

        setattr(BpLauncher, "run", run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", nargs="+", default=["0.001", "0.003"])
    ap.add_argument("--basis", default="z")
    a = ap.parse_args()
    out = ROOT / "results" / "latency_breakdown.json"
    report: dict[str, dict[str, float]] = json.loads(out.read_text()) if out.exists() else {}
    timer = Timer()
    for p in a.p:
        path = CIRCUITS / f"BB[[144,12,12]],memory_{a.basis.upper()},error_rate={p},syndrome_rounds=12.stim"
        problem = DecodingProblem.from_stim_file(path)
        dets, _ = sample_batch(stim.Circuit.from_file(str(path)), path.stem, 0, 8192)
        shots = dets[dets.any(axis=1)][:SHOTS]
        for name, freezing in CONFIGS.items():
            decoder = GpuBeamSearch(problem, PRESETS["beam8_230"], FLOAT32, freezing=freezing)
            decoder.decode_batch(shots[:32])
            wall, initial, rounds, count = [], [], [], []
            for syndrome in shots:
                timer.initial.clear()
                timer.rounds.clear()
                start = time.perf_counter()
                decoder.decode_batch(syndrome[None, :])
                wall.append(1e3 * (time.perf_counter() - start))
                initial.append(sum(timer.initial))
                rounds.append(sum(timer.rounds))
                count.append(len(timer.rounds))
            w, i, r = np.array(wall), np.array(initial), np.array(rounds)
            row = {"shots": len(w), "wall_mean_ms": float(w.mean()), "wall_p50_ms": float(np.median(w)),
                   "wall_p99_ms": float(np.percentile(w, 99)), "initial_gpu_mean_ms": float(i.mean()),
                   "rounds_gpu_mean_ms": float(r.mean()), "other_mean_ms": float((w - i - r).mean()),
                   "rounds_mean": float(np.mean(count)),
                   "other_per_round_ms": float((w - i - r).sum() / max(1, sum(count))),
                   "gpu_per_round_ms": float(r.sum() / max(1, sum(count)))}
            report[f"{a.basis}/{p}/{name}"] = row
            print(f"{a.basis}/{p}/{name}", {k: round(v, 3) for k, v in row.items()}, flush=True)
            out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
