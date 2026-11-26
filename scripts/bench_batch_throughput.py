"""Throughput sweep over increasing shot batch sizes for CPU, CUDA, and RTU backends.

Run on the GPU host: python scripts/bench_batch_throughput.py cpu|cuda|cuda-fr|rt|rt-cache [tag]
    [--case KEY] [--max-batch N]
The 10,000 seeded shots of the golden set (default p=0.001) are cycled for larger batches. Output is one
JSON per backend, results/throughput_{backend}.json, or results/throughput_{backend}_{tag}.json when a tag
is given (e.g. ``cpu laptop`` for the CPU sweep on the laptop, which has no GPU stack). ``cuda-fr`` = CUDA
with the recommended S1 freezing, Freezing(32, refresh=5, initial_warm=5). ``--max-batch`` drops larger
batch sizes (the sequential CPU is flat in batch size, and slow at high p).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rtbeam.decoders import PRESETS, Decoder
from rtbeam.golden import ShotSet
from rtbeam.ionq import IonQBeamSearch
from rtbeam.problem import DecodingProblem

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.001"
PRESET = "beam8_230"
BATCHES = (1, 16, 64, 256, 1024, 4096, 10_000, 20_000, 50_000, 100_000)
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"


def main(backend: str, tag: str | None = None, case: str = CASE, max_batch: int | None = None) -> None:
    shotset = ShotSet.load(ROOT / "data" / "golden" / case / "shots.npz")
    problem = DecodingProblem.from_stim_file(CIRCUITS / f"{shotset.stem}.stim")
    setup_start = time.perf_counter()
    decoder = IonQBeamSearch(problem, PRESETS[PRESET]) if backend == "cpu" else gpu_decoder(problem, backend)
    setup_s = time.perf_counter() - setup_start
    syndromes = shotset.syndromes
    # Warm native code, caches, and CPU frequency before recording any batch size.
    decoder.decode_batch(syndromes[:min(1_024, len(syndromes))])
    report: dict[str, object] = {
        "backend": backend, "case": case, "preset": PRESET,
        "host": host_info(), "decoder_setup_seconds": setup_s,
        "source_shots": len(syndromes), "batch_throughput_shots_per_s": {},
        "batch_elapsed_seconds": {}, "batch_repeats": {},
    }
    # Measure a mid-size workload first so startup/frequency ramp does not favor
    # the larger points simply because they were measured later.
    measurement_order = (1_024, 10_000, 4_096, 256, 64, 16, 1, 20_000, 50_000, 100_000)
    for batch in (b for b in measurement_order if max_batch is None or b <= max_batch):
        # For small batches, repeat across at least 2,048 shots so the result is not
        # determined by an unusually easy/hard prefix of the seeded shot set.
        repeats = max(1, (max(2_048, batch) + batch - 1) // batch)
        elapsed = []
        for rep in range(repeats):
            start_idx = (rep * batch) % len(syndromes)
            idx = (start_idx + np.arange(batch)) % len(syndromes)
            sample = syndromes[idx]
            start = time.perf_counter()
            decoder.decode_batch(sample)
            elapsed.append(time.perf_counter() - start)
        # Aggregate rate across the representative sample (per-call time can vary
        # substantially with syndrome difficulty, especially on the CPU baseline).
        seconds = float(sum(elapsed))
        measured_shots = batch * repeats
        report["batch_throughput_shots_per_s"][str(batch)] = measured_shots / seconds
        report["batch_elapsed_seconds"][str(batch)] = seconds
        report.setdefault("batch_measured_shots", {})[str(batch)] = measured_shots
        report["batch_repeats"][str(batch)] = repeats
        print(backend, batch, json.dumps({"measured_shots": measured_shots,
              "total_seconds": seconds, "shots_per_s": measured_shots / seconds,
              "calls": repeats}), flush=True)
    report["tag"] = tag
    out = ROOT / "results" / (f"throughput_{backend}_{tag}.json" if tag else f"throughput_{backend}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}", flush=True)
    del decoder
    gc.collect()


def gpu_decoder(problem: DecodingProblem, backend: str) -> Decoder:
    from rtbeam.gpu.beam import FLOAT32, Freezing, GpuBeamSearch  # GPU stack only when needed: the laptop runs cpu only
    if backend == "cuda":
        return GpuBeamSearch(problem, PRESETS[PRESET], FLOAT32, backend="cuda")
    if backend == "cuda-fr":
        return GpuBeamSearch(problem, PRESETS[PRESET], FLOAT32, backend="cuda",
                             freezing=Freezing(32.0, refresh=5, initial_warm=5))
    if backend == "rt":
        return GpuBeamSearch(problem, PRESETS[PRESET], FLOAT32, backend="rt")
    if backend == "rt-cache":
        return GpuBeamSearch(problem, PRESETS[PRESET], FLOAT32, backend="rt", rt_cache_neighbors=True)
    raise ValueError(f"unknown backend: {backend}")


def host_info() -> dict[str, str]:
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        smi = "none"
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), platform.processor())
    return {"gpu": smi, "cpu": platform.processor(), "cpu_model": model, "loadavg": " ".join(f"{x:.1f}" for x in os.getloadavg()),
            "cpus": str(os.cpu_count())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=("cpu", "cuda", "cuda-fr", "rt", "rt-cache"))
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--case", default=CASE)
    ap.add_argument("--max-batch", type=int, default=None)
    a = ap.parse_args(sys.argv[1:])
    main(a.backend, a.tag, a.case, a.max_batch)
