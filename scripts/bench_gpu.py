"""Latency (batch = 1) and throughput (batch sweep) of the CUDA decoder on the golden shot sets.

Latency is wall time per ``decode_batch`` call incl. host<->device copies and Python control,
i.e. comparable to the CPU baseline's per-shot time. Writes results/gpu_baseline.json.
Usage: bench_gpu.py [latency_shots]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rtbeam.decoders import PRESETS
from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, FLOAT64, GpuBeamSearch, Precision
from rtbeam.problem import DecodingProblem
from rtbeam.timing import LatencySummary

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
CASES = (("bb144_z_p0.0005", ("beam8_230", "beam32_340")),
         ("bb144_z_p0.001", ("beam8_230", "beam32_340", "beam64_640")))
BATCHES = (1, 16, 64, 256, 1024, 4096, 10_000)


def main(latency_shots: int) -> None:
    report: dict[str, object] = {"host": host_info()}
    for key, presets in CASES:
        shots = ShotSet.load(ROOT / "data" / "golden" / key / "shots.npz")
        problem = DecodingProblem.from_stim_file(CIRCUITS / f"{shots.stem}.stim")
        for preset in presets:
            for precision in (FLOAT32, FLOAT64):
                name = f"{key}/{preset}/{precision.name}"
                report[name] = bench(GpuBeamSearch(problem, PRESETS[preset], precision), shots.syndromes,
                                     latency_shots if precision is FLOAT32 else latency_shots // 5)
                print(name, json.dumps(report[name]), flush=True)
    (ROOT / "results" / "gpu_baseline.json").write_text(json.dumps(report, indent=2))


def bench(decoder: GpuBeamSearch, syndromes: np.ndarray, latency_shots: int) -> dict[str, object]:
    decoder.decode_batch(syndromes[:512])  # compile every kernel (incl. beam rounds) + warm up
    seconds = np.array([timed(decoder, syndromes[i:i + 1]) for i in range(latency_shots)])
    throughput = {str(b): syndromes[:b].shape[0] / best_of(decoder, syndromes[:b], 2 if b <= 1024 else 1)
                  for b in BATCHES if b <= syndromes.shape[0]}
    return {"latency_shots": latency_shots, "latency": LatencySummary.of(seconds).as_ms(),
            "throughput_shots_per_s": throughput}


def timed(decoder: GpuBeamSearch, batch: np.ndarray) -> float:
    start = time.perf_counter()
    decoder.decode_batch(batch)
    return time.perf_counter() - start


def best_of(decoder: GpuBeamSearch, batch: np.ndarray, repeats: int) -> float:
    return min(timed(decoder, batch) for _ in range(repeats))


def host_info() -> dict[str, str]:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu": smi, "loadavg": " ".join(f"{x:.1f}" for x in os.getloadavg()), "cpus": str(os.cpu_count())}


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
