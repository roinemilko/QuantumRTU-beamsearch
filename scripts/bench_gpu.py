"""Latency (batch = 1) and throughput (batch sweep) of the CUDA decoder on the golden shot sets.

Latency is wall time per ``decode_batch`` call incl. host<->device copies and Python control,
i.e. comparable to the CPU baseline's per-shot time. Writes results/gpu_baseline.json
(backend cuda) or results/rt_baseline.json (backend rt).
Usage: bench_gpu.py [latency_shots] [cuda|cuda-fr|rt|rt-ch|rt-cache|rt-check4|rt-check8|rt-rcm|rt-cache-check4-rcm]
                    [--cases KEY=PRESET+PRESET,...] [--tag TAG] [--fp32-only]
``cuda-fr`` = CUDA with the recommended S1 freezing, Freezing(32, refresh=5, initial_warm=5).
``--cases`` replaces the default golden sets; ``--tag`` writes {backend}_baseline_{tag}.json instead
(so earlier outputs are kept).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rtbeam.decoders import PRESETS, Bits
from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, FLOAT64, Freezing, GpuBeamSearch, Precision
from rtbeam.problem import DecodingProblem
from rtbeam.timing import LatencySummary

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
CASES = (("bb144_z_p0.0005", ("beam8_230", "beam32_340")),
         ("bb144_z_p0.001", ("beam8_230", "beam32_340", "beam64_640")))
BATCHES = (1, 16, 64, 256, 1024, 4096, 10_000)
RT_VARIANTS: dict[str, dict[str, object]] = {
    "rt": {},
    "rt-cache": {"rt_cache_neighbors": True},
    "rt-check4": {"rt_check_every": 4},
    "rt-check8": {"rt_check_every": 8},
    "rt-rcm": {"rt_reorder_geometry": True},
    "rt-cache-check4-rcm": {
        "rt_cache_neighbors": True, "rt_check_every": 4, "rt_reorder_geometry": True,
    },
}


def main(latency_shots: int, backend: str, cases: tuple[tuple[str, tuple[str, ...]], ...] = CASES,
         tag: str | None = None, fp32_only: bool = False) -> None:
    options: dict[str, object]
    if backend == "cuda":
        implementation, options = "cuda", {}
    elif backend == "cuda-fr":
        implementation, options = "cuda", {"freezing": Freezing(32.0, refresh=5, initial_warm=5)}
    elif backend == "rt-ch":
        implementation, options = "rt-ch", {}
    elif backend in RT_VARIANTS:
        implementation, options = "rt", RT_VARIANTS[backend]
    else:
        raise ValueError(f"unknown backend variant: {backend}")
    report: dict[str, object] = {"host": host_info()}
    for key, presets in cases:
        shots = ShotSet.load(ROOT / "data" / "golden" / key / "shots.npz")
        problem = DecodingProblem.from_stim_file(CIRCUITS / f"{shots.stem}.stim")
        for preset in presets:
            for precision in (FLOAT32,) if implementation == "rt-ch" or fp32_only else (FLOAT32, FLOAT64):
                name = f"{key}/{preset}/{precision.name}"
                setup_start = time.perf_counter()
                decoder = GpuBeamSearch(problem, PRESETS[preset], precision, backend=implementation, **options)
                setup_seconds = time.perf_counter() - setup_start
                result = bench(decoder, shots.syndromes,
                               latency_shots if precision is FLOAT32 else latency_shots // 5)
                result["decoder_setup_seconds"] = setup_seconds
                report[name] = result
                print(name, json.dumps(report[name]), flush=True)
    name = "gpu_baseline" if backend == "cuda" else f"{backend}_baseline"
    (ROOT / "results" / (f"{name}_{tag}.json" if tag else f"{name}.json")).write_text(json.dumps(report, indent=2))


def bench(decoder: GpuBeamSearch, syndromes: Bits, latency_shots: int) -> dict[str, object]:
    decoder.decode_batch(syndromes[:512])  # compile every kernel (incl. beam rounds) + warm up
    seconds = np.array([timed(decoder, syndromes[i:i + 1]) for i in range(latency_shots)])
    throughput = {str(b): syndromes[:b].shape[0] / best_of(decoder, syndromes[:b], 2 if b <= 1024 else 1)
                  for b in BATCHES if b <= syndromes.shape[0]}
    return {"latency_shots": latency_shots, "latency": LatencySummary.of(seconds).as_ms(),
            "throughput_shots_per_s": throughput}


def timed(decoder: GpuBeamSearch, batch: Bits) -> float:
    start = time.perf_counter()
    decoder.decode_batch(batch)
    return time.perf_counter() - start


def best_of(decoder: GpuBeamSearch, batch: Bits, repeats: int) -> float:
    return min(timed(decoder, batch) for _ in range(repeats))


def host_info() -> dict[str, str]:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu": smi, "loadavg": " ".join(f"{x:.1f}" for x in os.getloadavg()), "cpus": str(os.cpu_count())}


def parse_cases(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """'bb144_z_p0.002_10k=beam8_230+beam32_340,...' -> ((key, presets), ...)."""
    return tuple((key, tuple(presets.split("+"))) for key, presets in (item.split("=") for item in text.split(",")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("latency_shots", nargs="?", type=int, default=10_000)
    ap.add_argument("backend", nargs="?", default="cuda")
    ap.add_argument("--cases", type=parse_cases, default=CASES)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fp32-only", action="store_true")
    a = ap.parse_args(sys.argv[1:])
    main(a.latency_shots, a.backend, a.cases, a.tag, a.fp32_only)
