"""Check the NumPy reference against IonQ's recorded outputs on the golden shot sets.

Usage: validate_reference.py [max_shots_per_run]. Shots run in parallel worker processes.
Writes results/reference_validation.json.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rtbeam.decoders import PRESETS
from rtbeam.golden import DecoderRun, ShotSet
from rtbeam.problem import DecodingProblem
from rtbeam.reference.beam import ReferenceBeamSearch
from rtbeam.search import ShotOutcome

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden"
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
SHOT_CAP = {"beam8_230": 1.0, "beam32_340": 0.5, "beam64_640": 0.2, "beam64_32res_640": 0.1}

_decoder: ReferenceBeamSearch | None = None


def main(max_shots: int) -> None:
    report = {}
    for folder in sorted(p for p in GOLDEN.iterdir() if p.is_dir()):
        shots = ShotSet.load(folder / "shots.npz")
        for run_file in sorted(folder.glob("beam*.npz")):
            report[f"{folder.name}/{run_file.stem}"] = validate(shots, DecoderRun.load(run_file), max_shots)
    (ROOT / "results" / "reference_validation.json").write_text(json.dumps(report, indent=2))


def validate(shots: ShotSet, expected: DecoderRun, max_shots: int) -> dict[str, object]:
    count = max(1, int(max_shots * SHOT_CAP[expected.decoder]))
    outcomes = decode_parallel(shots.stem, expected.decoder, shots.syndromes[:count])
    same = [np.array_equal(o.correction, c) for o, c in zip(outcomes, expected.corrections[:count])]
    result = {
        "shots": count, "identical": int(sum(same)),
        "mismatched_shots": [i for i, s in enumerate(same) if not s],
        "rounds_histogram": dict(sorted(Counter(o.rounds for o in outcomes).items())),
    }
    print(f"{shots.stem} {expected.decoder}: {result['identical']}/{count} identical, "
          f"rounds {result['rounds_histogram']}", flush=True)
    return result


def decode_parallel(stem: str, preset: str, syndromes: np.ndarray) -> list[ShotOutcome]:
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(stem, preset)) as pool:
        return list(pool.map(_decode, syndromes, chunksize=4))


def _init_worker(stem: str, preset: str) -> None:
    global _decoder
    os.environ["OMP_NUM_THREADS"] = "1"
    _decoder = ReferenceBeamSearch(DecodingProblem.from_stim_file(CIRCUITS / f"{stem}.stim"), PRESETS[preset])


def _decode(syndrome: np.ndarray) -> ShotOutcome:
    assert _decoder is not None
    return _decoder.decode_shot(syndrome)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
