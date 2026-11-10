"""Sample seeded shot sets, decode them with the CPU baselines, record outputs and timings.

Writes data/golden/<key>/shots.npz and data/golden/<key>/<decoder>.npz, plus
results/cpu_baseline.json. Run on an otherwise idle machine: timings are single-core.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtbeam.baselines import BpOsdBaseline
from rtbeam.decoders import PRESETS, Decoder
from rtbeam.golden import DecoderRun, ShotSet
from rtbeam.ionq import IonQBeamSearch
from rtbeam.problem import DecodingProblem
from rtbeam.timing import LatencySummary

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
GOLDEN = ROOT / "data" / "golden"
BB144_Z = "BB[[144,12,12]],memory_Z,error_rate={p},syndrome_rounds=12"


@dataclass(frozen=True)
class Job:
    """One shot set and the decoders to run on it."""

    key: str
    stem: str
    shots: int
    seed: int
    decoders: tuple[str, ...]


JOBS = (
    Job("bb144_z_p0.0005", BB144_Z.format(p="0.0005"), 10_000, 1001, ("beam8_230", "beam32_340", "bp30+osd")),
    Job("bb144_z_p0.001", BB144_Z.format(p="0.001"), 10_000, 1002,
        ("beam8_230", "beam32_340", "beam64_640", "bp30+osd")),
    Job("bb144_z_p0.001_32res", BB144_Z.format(p="0.001"), 1_000, 1003, ("beam64_32res_640",)),
    Job("bb144_z_p0.002", BB144_Z.format(p="0.002"), 2_000, 1004, ("beam8_230", "beam32_340")),
)


def main(selected: list[str]) -> None:
    summaries = {job.key: run_job(job) for job in JOBS if not selected or job.key in selected}
    out = ROOT / "results" / "cpu_baseline.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps(previous | summaries, indent=2))


def run_job(job: Job) -> dict[str, dict[str, float]]:
    folder = GOLDEN / job.key
    folder.mkdir(parents=True, exist_ok=True)
    problem = DecodingProblem.from_stim_file(CIRCUITS / f"{job.stem}.stim")
    shots = ShotSet.sample(CIRCUITS / f"{job.stem}.stim", job.shots, job.seed)
    shots.save(folder / "shots.npz")
    return {name: run_decoder(problem, shots, name, folder) for name in job.decoders}


def run_decoder(problem: DecodingProblem, shots: ShotSet, name: str, folder: Path) -> dict[str, float]:
    batch = make_decoder(problem, name).decode_batch(shots.syndromes)
    DecoderRun(name, batch.corrections, batch.seconds).save(folder / f"{name}.npz")
    satisfied = np.all(problem.syndromes_of(batch.corrections) == shots.syndromes, axis=1)
    failed = np.any(problem.logical_flips(batch.corrections) != shots.observable_flips, axis=1)
    summary = LatencySummary.of(batch.seconds).as_ms() | {
        "shots": float(shots.syndromes.shape[0]),
        "syndrome_satisfied_frac": float(satisfied.mean()), "logical_failures": float(failed.sum()),
    }
    print(f"{shots.stem} {name}: {summary}", flush=True)
    return summary


def make_decoder(problem: DecodingProblem, name: str) -> Decoder:
    return BpOsdBaseline(problem) if name == "bp30+osd" else IonQBeamSearch(problem, PRESETS[name])


if __name__ == "__main__":
    main(sys.argv[1:])
