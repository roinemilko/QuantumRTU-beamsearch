"""S1: frontier freezing inside the CUDA beam search: accuracy and speed versus the unpruned decoder.

Shots: batch 0 of the LER sampler (the same shots results/ler/* start with) at each p, beam8.
Per (tau, warm) and the unpruned reference: failures (observable mismatch), corrections identical
to the reference, unsatisfied corrections, batch wall time, mean work-list sizes per child run
(live checks, frontier edges and their share of all edges, frontier bits), and batch-one latency
over the first LATENCY_SHOTS non-trivial shots. Writes results/s1_freezing.json.
Usage: bench_s1.py [--p 0.003 0.004] [--basis z]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import stim
from numpy.typing import NDArray

from rtbeam.decoders import PRESETS
from rtbeam.gpu.beam import FLOAT32, Freezing, GpuBeamSearch
from rtbeam.ler import _gf2, sample_batch
from rtbeam.problem import DecodingProblem

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
PRESET = "beam8_230"
SHOTS = 4096
LATENCY_SHOTS = 400
CONFIGS: dict[str, tuple[Freezing, ...]] = {
    "sweep": tuple(Freezing(t, w) for t, w in ((2, 0), (4, 0), (6, 0), (8, 0), (12, 0), (4, 3), (6, 3), (8, 3))),
    "refresh": (Freezing(32), Freezing(32, refresh=5), Freezing(32, 3), Freezing(32, 3, refresh=5), Freezing(64, 3),
                Freezing(8, 3), Freezing(32, 3, initial_warm=5)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", nargs="+", default=["0.003", "0.004"])
    ap.add_argument("--basis", default="z")
    ap.add_argument("--configs", choices=sorted(CONFIGS), default="sweep")
    a = ap.parse_args()
    out = ROOT / "results" / "s1_freezing.json"
    report: dict[str, dict[str, float]] = json.loads(out.read_text()) if out.exists() else {}
    for p in a.p:
        path = CIRCUITS / f"BB[[144,12,12]],memory_{a.basis.upper()},error_rate={p},syndrome_rounds=12.stim"
        problem = DecodingProblem.from_stim_file(path)
        dets, obs = sample_batch(stim.Circuit.from_file(str(path)), path.stem, 0, SHOTS)
        latency_shots = dets[dets.any(axis=1)][:LATENCY_SHOTS]
        reference = None
        for config in (None, *CONFIGS[a.configs]):
            decoder = GpuBeamSearch(problem, PRESETS[PRESET], FLOAT32, freezing=config)
            decoder.decode_batch(dets[:64])  # compile + warm up
            decoder.frontier_totals[:] = 0
            start = time.perf_counter()
            corrections = decoder.decode_batch(dets).corrections
            seconds = time.perf_counter() - start
            if reference is None:
                reference = corrections
            runs, live, edges, bits = (int(x) for x in decoder.frontier_totals)
            row = {"failures": int((_gf2(problem.observables, corrections) != obs).any(axis=1).sum()),
                   "unsatisfied": int((_gf2(problem.check_matrix, corrections) != dets).any(axis=1).sum()),
                   "identical_to_unpruned": float((corrections == reference).all(axis=1).mean()),
                   "batch_seconds": seconds, "shots_per_second": SHOTS / seconds, "child_runs": runs,
                   "live_checks_per_run": live / max(runs, 1), "frontier_edges_per_run": edges / max(runs, 1),
                   "frontier_edge_share": edges / max(runs, 1) / problem.check_matrix.nnz,
                   "frontier_bits_per_run": bits / max(runs, 1)}
            row.update(latency(decoder, latency_shots))
            key = f"{a.basis}/{p}/" + (label(config) if config else "unpruned")
            report[key] = row
            print(key, {k: round(v, 4) for k, v in row.items()}, flush=True)
            out.write_text(json.dumps(report, indent=2))


def label(f: Freezing) -> str:
    """Same naming as the existing keys: tau8_warm0, plus _init5 / _r5 when set."""
    return (f"tau{f.tau:g}_warm{f.warm}" + (f"_init{f.initial_warm}" if f.initial_warm is not None else "")
            + (f"_r{f.refresh}" if f.refresh else ""))


def latency(decoder: GpuBeamSearch, shots: NDArray[np.uint8]) -> dict[str, float]:
    seconds = np.empty(len(shots))
    for i, syndrome in enumerate(shots):
        start = time.perf_counter()
        decoder.decode_batch(syndrome[None, :])
        seconds[i] = time.perf_counter() - start
    ms = seconds * 1e3
    return {"latency_mean_ms": float(ms.mean()), "latency_p50_ms": float(np.percentile(ms, 50)),
            "latency_p99_ms": float(np.percentile(ms, 99)), "latency_max_ms": float(ms.max())}


if __name__ == "__main__":
    main()
