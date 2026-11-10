"""Logical error rate (LER) by Monte Carlo with paired, reproducible shots.

Shots come in fixed-size batches; batch ``k`` of a circuit is sampled with a seed derived
from the circuit name and ``k`` only, so every decoder sees the *same* shots and runs can be
compared shot by shot (paired comparison) and resumed. Counting follows IonQ's
``simulation_functions.py``: a shot fails if any predicted observable differs; per-round
LER = failures / shots / rounds, and the X and Z memory experiments are summed.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

from rtbeam.decoders import Bits, Decoder
from rtbeam.problem import DecodingProblem


@dataclass(frozen=True)
class Budget:
    """Stop after ``max_errors`` failures or ``max_shots`` shots, whichever comes first."""

    max_errors: int = 100
    max_shots: int = 10**8
    batch: int = 4096


@dataclass
class Tally:
    """Running totals of one (circuit, decoder) run; ``failed`` holds global shot indices."""

    circuit: str
    decoder: str
    batch: int
    batches: int = 0
    shots: int = 0
    errors: int = 0
    unsatisfied: int = 0
    seconds: float = 0.0
    failed: list[int] = field(default_factory=list)

    @classmethod
    def load_or_new(cls, path: Path, circuit: str, decoder: str, batch: int) -> Tally:
        if path.exists():
            tally = cls(**json.loads(path.read_text()))
            assert (tally.circuit, tally.decoder, tally.batch) == (circuit, decoder, batch), f"{path} mismatch"
            return tally
        return cls(circuit, decoder, batch)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)

    def done(self, budget: Budget) -> bool:
        return self.errors >= budget.max_errors or self.shots >= budget.max_shots


def batch_seed(circuit: str, k: int) -> int:
    """Seed of batch ``k`` of ``circuit``: independent of the decoder."""
    return zlib.crc32(circuit.encode()) * 1_000_003 + k


def sample_batch(sampler_circuit: stim.Circuit, circuit: str, k: int, shots: int) -> tuple[Bits, Bits]:
    sampler = sampler_circuit.compile_detector_sampler(seed=batch_seed(circuit, k))
    dets, obs = sampler.sample(shots, separate_observables=True)
    return dets.astype(np.uint8), obs.astype(np.uint8)


def run(circuit_path: Path, decoder: Decoder, decoder_name: str, budget: Budget, out: Path) -> Tally:
    """Sample and decode batches until the budget is spent; checkpoints ``out`` after each batch."""
    circuit = stim.Circuit.from_file(str(circuit_path))
    problem = DecodingProblem.from_stim_file(circuit_path)
    tally = Tally.load_or_new(out, circuit_path.stem, decoder_name, budget.batch)
    while not tally.done(budget):
        dets, obs = sample_batch(circuit, tally.circuit, tally.batches, budget.batch)
        start = time.perf_counter()
        corrections = decoder.decode_batch(dets).corrections
        tally.seconds += time.perf_counter() - start
        wrong = (_gf2(problem.observables, corrections) != obs).any(axis=1)
        unsat = (_gf2(problem.check_matrix, corrections) != dets).any(axis=1)
        tally.failed += (tally.batches * budget.batch + np.flatnonzero(wrong)).tolist()
        tally.errors += int(wrong.sum())
        tally.unsatisfied += int(unsat.sum())
        tally.shots += budget.batch
        tally.batches += 1
        tally.save(out)
    return tally


def _gf2(matrix: sp.csr_matrix, rows: Bits) -> Bits:
    """``rows · matrixᵀ mod 2`` with ``rows`` sparse (corrections are low weight)."""
    product = sp.csr_matrix(rows, dtype=np.int32) @ matrix.T.astype(np.int32)
    bits: Bits = (product.toarray() % 2).astype(np.uint8)
    return bits
