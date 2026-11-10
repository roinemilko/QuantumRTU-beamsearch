"""Decoder interface shared by every backend, and IonQ's beam-search parameter presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Bits = NDArray[np.uint8]


@dataclass(frozen=True)
class BeamConfig:
    """Beam-search parameters, named as in IonQ's decoder.

    Worst-case BP iterations along one path: ``initial_iters + max_rounds * iters_per_round``.
    """

    max_rounds: int = 10
    beam_width: int = 8
    num_results: int = 1
    initial_iters: int = 30
    iters_per_round: int = 20


PRESETS: dict[str, BeamConfig] = {
    "beam8_230": BeamConfig(),
    "beam32_340": BeamConfig(beam_width=32, initial_iters=40, iters_per_round=30),
    "beam64_640": BeamConfig(max_rounds=20, beam_width=64, initial_iters=40, iters_per_round=30),
    "beam64_32res_640": BeamConfig(
        max_rounds=20, beam_width=64, num_results=32, initial_iters=40, iters_per_round=30
    ),
}


@dataclass(frozen=True)
class DecodeBatch:
    """Result of decoding a batch: ``corrections`` (shots × faults) and per-shot wall time.

    ``seconds`` is NaN where a backend only measures the batch as a whole.
    """

    corrections: Bits
    seconds: NDArray[np.float64]


class Decoder(Protocol):
    """Anything that maps a batch of syndromes (shots × detectors) to corrections."""

    def decode_batch(self, syndromes: Bits) -> DecodeBatch:
        """Decode every row of ``syndromes``; shots are independent."""
        ...
