"""Flooding min-sum BP over K independent runs, matching IonQ's C++ arithmetic bit for bit.

Conventions copied from ``beam_search.hpp`` (each is load-bearing for exactness):

- a bit-to-check message ``<= 0`` counts as negative (zero is negative);
- a check whose other free edges are all absent sends magnitude ``DBL_MAX``;
- ``NaN``/``inf`` magnitudes never lower the running minimum (``abs < DBL_MAX`` test);
- column sums run sequentially in edge order: prefix ``Λ + c_0 + … + c_{k-1}`` plus
  suffix ``(0 + c_{d-1}) + … + c_{k+1}``; numpy's ``cumsum`` is sequential, ``sum`` is not;
- hard decision is ``posterior <= 0``;
- prior LLRs use libm ``log`` (``math.log``), not numpy's SIMD ``log``, which may differ by an ulp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rtbeam.tanner import TannerGraph

Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]
Bits = NDArray[np.uint8]

DBL_MAX = float(np.finfo(np.float64).max)


@dataclass(frozen=True)
class Iteration:
    """Outcome of one BP iteration for K runs.

    ``v2c`` (K × edges) keeps its input value on edges of fixed bits;
    ``posterior`` is meaningful only on free bits; ``decisions`` is False on fixed bits;
    ``satisfied[k]`` says whether run ``k``'s free-bit decisions reproduce its syndrome.
    """

    v2c: Floats
    posterior: Floats
    decisions: Bools
    satisfied: Bools


class MinSum:
    """Min-sum update rules for one Tanner graph and prior vector; ``step`` is pure."""

    def __init__(self, graph: TannerGraph, priors: Floats) -> None:
        self.__graph = graph
        self.__llr = np.array([math.log((1 - p) / p) for p in priors.tolist()])
        self.__csr_check = graph.edge_check[graph.row_order]

    @property
    def prior_llr(self) -> Floats:
        """``log((1-p)/p)`` per bit, as IonQ computes it."""
        return self.__llr

    def initial_messages(self, runs: int) -> Floats:
        """Bit-to-check messages before the first iteration: each bit's prior LLR."""
        return np.tile(self.__llr[self.__graph.edge_bit], (runs, 1))

    def step(self, v2c: Floats, free_bits: Bools, syndromes: Bits) -> Iteration:
        """One flooding iteration. Fixed bits (``free_bits`` False) are absent from the graph."""
        free_edges = free_bits[:, self.__graph.edge_bit]
        c2v = self.__check_update(v2c, free_edges, syndromes)
        posterior, fresh = self.__bit_update(c2v)
        decisions = free_bits & (posterior <= 0)
        satisfied = self.__satisfied(decisions, syndromes)
        return Iteration(np.where(free_edges, fresh, v2c), posterior, decisions, satisfied)

    def __check_update(self, v2c: Floats, free_edges: Bools, syndromes: Bits) -> Floats:
        order = self.__graph.row_order
        v, free = v2c[:, order], free_edges[:, order]
        magnitude = np.where(free & ~np.isnan(v), np.abs(v), np.inf)
        negative = free & (v <= 0)
        c2v = np.zeros_like(v2c)
        c2v[:, order] = np.where(free, self.__signs(negative, syndromes) * self.__exclusive_min(magnitude), 0.0)
        return c2v

    def __exclusive_min(self, magnitude: Floats) -> Floats:
        starts, rows = self.__graph.row_starts, self.__csr_check
        min1 = np.minimum.reduceat(magnitude, starts, axis=1)
        is_min = magnitude == min1[:, rows]
        unique = is_min & (np.add.reduceat(is_min.astype(np.int32), starts, axis=1)[:, rows] == 1)
        min2 = np.minimum.reduceat(np.where(unique, np.inf, magnitude), starts, axis=1)
        exclusive: Floats = np.minimum(np.where(unique, min2[:, rows], min1[:, rows]), DBL_MAX)
        return exclusive

    def __signs(self, negative: Bools, syndromes: Bits) -> Floats:
        negatives = np.add.reduceat(negative.astype(np.int32), self.__graph.row_starts, axis=1)
        parity = (negatives + syndromes)[:, self.__csr_check] + negative
        return np.where(parity % 2 == 0, 1.0, -1.0)

    def __bit_update(self, c2v: Floats) -> tuple[Floats, Floats]:
        runs, slots = c2v.shape[0], self.__graph.col_slots
        incoming = np.concatenate([c2v, np.zeros((runs, 1))], axis=1)[:, slots]
        prior = np.broadcast_to(self.__llr[None, :, None], (runs, slots.shape[0], 1))
        prefix = np.cumsum(np.concatenate([prior, incoming], axis=2), axis=2)
        suffix = np.cumsum(incoming[:, :, ::-1], axis=2)[:, :, ::-1]
        after = np.concatenate([suffix[:, :, 1:], np.zeros_like(suffix[:, :, :1])], axis=2)
        return prefix[:, :, -1], self.__scatter_columns(prefix[:, :, :-1] + after)

    def __scatter_columns(self, per_slot: Floats) -> Floats:
        slots = self.__graph.col_slots
        valid = slots < self.__graph.num_edges
        out = np.empty((per_slot.shape[0], self.__graph.num_edges))
        out[:, slots[valid]] = per_slot[:, valid]
        return out

    def __satisfied(self, decisions: Bools, syndromes: Bits) -> Bools:
        flips = decisions[:, self.__graph.edge_bit][:, self.__graph.row_order]
        parity = np.add.reduceat(flips.astype(np.int32), self.__graph.row_starts, axis=1) % 2
        satisfied: Bools = np.all(parity == syndromes, axis=1)
        return satisfied
