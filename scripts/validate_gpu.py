"""Check the CUDA decoder against IonQ's recorded outputs on the golden shot sets.

Usage: validate_gpu.py [max_shots] [cuda|rt]. float64 must be bit-identical; float32 is reported as
agreement rate, syndrome-satisfied rate and logical failures; for the RT backend, float32 is also
compared with the CUDA float32 decoder (should be identical). Writes results/{gpu,rt}_validation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from rtbeam.decoders import PRESETS
from rtbeam.golden import DecoderRun, ShotSet
from rtbeam.gpu.beam import FLOAT32, FLOAT64, GpuBeamSearch, Precision
from rtbeam.problem import DecodingProblem

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden"
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"


def main(max_shots: int, backend: str) -> None:
    report: dict[str, dict[str, object]] = {}
    for folder in sorted(p for p in GOLDEN.iterdir() if p.is_dir()):
        shots = ShotSet.load(folder / "shots.npz")
        problem = DecodingProblem.from_stim_file(CIRCUITS / f"{shots.stem}.stim")
        for run_file in sorted(folder.glob("beam*.npz")):
            expected = DecoderRun.load(run_file)
            for precision in (FLOAT64, FLOAT32):
                key = f"{folder.name}/{expected.decoder}/{precision.name}"
                report[key] = compare(problem, shots, expected, precision, max_shots, backend)
                print(key, report[key], flush=True)
    name = "gpu_validation.json" if backend == "cuda" else f"{backend}_validation.json"
    (ROOT / "results" / name).write_text(json.dumps(report, indent=2))


def compare(problem: DecodingProblem, shots: ShotSet, expected: DecoderRun, precision: Precision,
            max_shots: int, backend: str) -> dict[str, object]:
    count = min(max_shots, shots.syndromes.shape[0]) // (10 if "32res" in expected.decoder else 1)
    syndromes = shots.syndromes[:count]
    config = PRESETS[expected.decoder]
    actual = GpuBeamSearch(problem, config, precision, backend=backend).decode_batch(syndromes).corrections
    same = np.all(actual == expected.corrections[:count], axis=1)
    satisfied = np.all(problem.syndromes_of(actual) == syndromes, axis=1)
    failed = np.any(problem.logical_flips(actual) != shots.observable_flips[:count], axis=1)
    report: dict[str, object] = {
        "shots": count, "identical": int(same.sum()), "syndrome_satisfied": int(satisfied.sum()),
        "logical_failures": int(failed.sum()), "first_mismatches": np.flatnonzero(~same)[:10].tolist()}
    if backend != "cuda" and precision is FLOAT32:
        cuda = GpuBeamSearch(problem, config, precision).decode_batch(syndromes).corrections
        report["identical_to_cuda_float32"] = int(np.all(actual == cuda, axis=1).sum())
    return report


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000, sys.argv[2] if len(sys.argv) > 2 else "cuda")
