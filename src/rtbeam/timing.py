"""Latency statistics in the form IonQ's paper reports them."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LatencySummary:
    """Per-shot latency statistics in seconds.

    ``p999`` follows IonQ's ``decoding_time``: sorted times at index
    ``int(0.999 * n) - 1``.
    """

    mean: float
    p50: float
    p99: float
    p999: float
    max: float

    @classmethod
    def of(cls, seconds: NDArray[np.float64]) -> LatencySummary:
        ordered = np.sort(seconds)
        n = ordered.size
        return cls(
            mean=float(ordered.mean()), p50=float(ordered[n // 2]),
            p99=float(ordered[max(int(0.99 * n) - 1, 0)]),
            p999=float(ordered[max(int(0.999 * n) - 1, 0)]), max=float(ordered[-1]),
        )

    def as_ms(self) -> dict[str, float]:
        """Fields converted to milliseconds, keyed ``<field>_ms``."""
        return {f"{k}_ms": v * 1e3 for k, v in asdict(self).items()}
