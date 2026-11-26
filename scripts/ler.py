"""LER sweep for the BB[[144,12,12]] memory-X/Z circuits (IonQ's), any backend, resumable.

Each (p, basis, preset) job writes results/ler/<key>/<preset>__<backend>.json after every batch;
rerunning continues where it stopped. Shots are paired across backends (see rtbeam.ler).
Usage: ler.py --backend gpu-fp32 [--presets beam8_230 ...] [--p 0.002 ...] [--workers 2]
S1 frontier freezing (CUDA only): --tau 6 [--warm 2] names the backend gpu-fp32-fr6w2 and writes
to results/ler_s1/; --match gpu-fp32 decodes exactly the shots that backend's tally covers
(capped by --max-shots).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from rtbeam.decoders import PRESETS, Decoder
from rtbeam.ler import Budget, run
from rtbeam.problem import DecodingProblem

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
OUT = ROOT / "results" / "ler"
OUT_S1 = ROOT / "results" / "ler_s1"
P_VALUES = ("0.001", "0.002", "0.003", "0.004", "0.005", "0.006")
SHOT_CAP = {"0.001": 2 * 10**7}  # 100 failures would need ~1e8-1e9 shots per job here
BACKENDS = ("gpu-fp32", "gpu-fp64", "rt-fp32", "rt-fp64", "rt-ch-fp32", "ionq")
KINDS = {"gpu": "cuda", "rt": "rt", "rt-ch": "rt-ch"}


def circuit_path(p: str, basis: str) -> Path:
    return CIRCUITS / f"BB[[144,12,12]],memory_{basis.upper()},error_rate={p},syndrome_rounds=12.stim"


def job_key(p: str, basis: str) -> str:
    return f"bb144_{basis}_p{p}"


def make_decoder(backend: str, problem: DecodingProblem, preset: str,
                 freezing: tuple[float, int, int | None, int] | None = None) -> Decoder:
    if backend == "ionq":
        from rtbeam.ionq import IonQBeamSearch
        return IonQBeamSearch(problem, PRESETS[preset])
    from rtbeam.gpu.beam import FLOAT32, FLOAT64, Freezing, GpuBeamSearch
    kind, precision = backend.rsplit("-", 1)
    return GpuBeamSearch(problem, PRESETS[preset], FLOAT64 if precision == "fp64" else FLOAT32, backend=KINDS[kind],
                         freezing=Freezing(*freezing) if freezing else None)


def job(args: tuple[str, str, str, str, Budget, tuple[float, int, int | None, int] | None, str | None]) -> str:
    backend, p, basis, preset, budget, freezing, match = args
    path = circuit_path(p, basis)
    name = backend
    if freezing:
        name += (f"-fr{freezing[0]:g}w{freezing[1]}" + (f"i{freezing[2]}" if freezing[2] is not None else "")
                 + (f"r{freezing[3]}" if freezing[3] else ""))
    out = (OUT_S1 if freezing else OUT) / job_key(p, basis) / f"{preset}__{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if match:
        reference = json.loads((OUT / job_key(p, basis) / f"{preset}__{match}.json").read_text())
        budget = Budget(10**12, min(reference["shots"], budget.max_shots), reference["batch"])
    decoder = make_decoder(backend, DecodingProblem.from_stim_file(path), preset, freezing)
    t = run(path, decoder, f"{preset}/{name}", budget, out)
    return (f"{job_key(p, basis)} {preset} {name}: {t.errors} fails / {t.shots:,} shots "
            f"(unsat {t.unsatisfied}), {t.shots / t.seconds:,.0f} shots/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default="gpu-fp32")
    ap.add_argument("--presets", nargs="+", default=["beam8_230", "beam32_340", "beam64_640"])
    ap.add_argument("--p", nargs="+", default=list(P_VALUES))
    ap.add_argument("--bases", nargs="+", default=["x", "z"])
    ap.add_argument("--max-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=None, help="overrides the per-p default cap")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tau", type=float, default=None, help="S1 frontier freezing threshold (CUDA backends)")
    ap.add_argument("--warm", type=int, default=0, help="S1: full iterations before freezing")
    ap.add_argument("--initial-warm", type=int, default=None, help="S1: also freeze initial BP after this many iterations")
    ap.add_argument("--refresh", type=int, default=0, help="S1: full iteration + new frontier every N pruned iterations")
    ap.add_argument("--match", default=None, help="decode the same shots as this backend's existing tally")
    a = ap.parse_args()
    freezing = (a.tau, a.warm, a.initial_warm, a.refresh) if a.tau is not None else None
    jobs = [(a.backend, p, basis, preset,
             Budget(a.max_errors, a.max_shots or SHOT_CAP.get(p, 10**9), a.batch), freezing, a.match)
            for p in sorted(a.p, reverse=True) for preset in a.presets for basis in a.bases]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for line in pool.imap_unordered(job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
