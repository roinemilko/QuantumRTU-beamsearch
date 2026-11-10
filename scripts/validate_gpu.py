"""Check the CUDA decoder against IonQ's recorded outputs on the golden shot sets.

Usage: validate_gpu.py [max_shots]. float64 must be bit-identical; float32 is reported as
agreement rate, syndrome-satisfied rate and logical failures. Writes results/gpu_validation.json.
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


def main(max_shots: int) -> None:
    report: dict[str, dict[str, object]] = {}
    for folder in sorted(p for p in GOLDEN.iterdir() if p.is_dir()):
        shots = ShotSet.load(folder / "shots.npz")
        problem = DecodingProblem.from_stim_file(CIRCUITS / f"{shots.stem}.stim")
        for run_file in sorted(folder.glob("beam*.npz")):
            expected = DecoderRun.load(run_file)
            for precision in (FLOAT64, FLOAT32):
                key = f"{folder.name}/{expected.decoder}/{precision.name}"
                report[key] = compare(problem, shots, expected, precision, max_shots)
                print(key, report[key], flush=True)
    (ROOT / "results" / "gpu_validation.json").write_text(json.dumps(report, indent=2))


def compare(problem: DecodingProblem, shots: ShotSet, expected: DecoderRun, precision: Precision,
            max_shots: int) -> dict[str, object]:
    count = min(max_shots, shots.syndromes.shape[0]) // (10 if "32res" in expected.decoder else 1)
    syndromes = shots.syndromes[:count]
    actual = GpuBeamSearch(problem, PRESETS[expected.decoder], precision).decode_batch(syndromes).corrections
    same = np.all(actual == expected.corrections[:count], axis=1)
    satisfied = np.all(problem.syndromes_of(actual) == syndromes, axis=1)
    failed = np.any(problem.logical_flips(actual) != shots.observable_flips[:count], axis=1)
    return {"shots": count, "identical": int(same.sum()), "syndrome_satisfied": int(satisfied.sum()),
            "logical_failures": int(failed.sum()), "first_mismatches": np.flatnonzero(~same)[:10].tolist()}


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
