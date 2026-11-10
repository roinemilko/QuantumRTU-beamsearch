"""Fixed, seeded shot sets and recorded decoder outputs used as regression oracles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
from numpy.typing import NDArray

from rtbeam.decoders import Bits


@dataclass(frozen=True)
class ShotSet:
    """Sampled detector outcomes with their true observable flips."""

    stem: str
    seed: int
    syndromes: Bits
    observable_flips: Bits

    @classmethod
    def sample(cls, circuit_path: Path, shots: int, seed: int) -> ShotSet:
        """Sample ``shots`` shots from a ``.stim`` circuit; deterministic in ``seed``."""
        sampler = stim.Circuit.from_file(str(circuit_path)).compile_detector_sampler(seed=seed)
        dets, obs = sampler.sample(shots, separate_observables=True)
        return cls(circuit_path.stem, seed, dets.astype(np.uint8), obs.astype(np.uint8))

    @classmethod
    def load(cls, path: Path) -> ShotSet:
        with np.load(path, allow_pickle=False) as f:
            return cls(str(f["stem"]), int(f["seed"]), f["syndromes"], f["observable_flips"])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, stem=self.stem, seed=self.seed,
                            syndromes=self.syndromes, observable_flips=self.observable_flips)


@dataclass(frozen=True)
class DecoderRun:
    """One decoder's corrections and per-shot wall times on a :class:`ShotSet`."""

    decoder: str
    corrections: Bits
    seconds: NDArray[np.float64]

    @classmethod
    def load(cls, path: Path) -> DecoderRun:
        with np.load(path, allow_pickle=False) as f:
            n = int(f["num_faults"])
            bits = np.unpackbits(f["corrections_packed"], axis=1, count=n)
            return cls(str(f["decoder"]), bits, f["seconds"])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, decoder=self.decoder, num_faults=self.corrections.shape[1],
                            corrections_packed=np.packbits(self.corrections, axis=1),
                            seconds=self.seconds)
