"""Edge-indexed Tanner graph layouts shared by every decoder backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

Index = NDArray[np.int64]


@dataclass(frozen=True)
class GraphStats:
    """Size and degree summary of a Tanner graph."""

    checks: int
    bits: int
    edges: int
    check_degree_min: int
    check_degree_max: int
    check_degree_mean: float
    bit_degree_min: int
    bit_degree_max: int
    bit_degree_mean: float
    bits_degree_le2: int


@dataclass(frozen=True)
class TannerGraph:
    """Tanner graph of a check matrix with one id per edge (nonzero of H).

    Edge ids follow column-major order with rows ascending inside a column:
    the order IonQ's decoder stores warm-start messages in, and the order its
    column sums run in (which matters for bit-exact floating point).

    - ``edge_check[e]``, ``edge_bit[e]``: endpoints of edge ``e``.
    - ``row_order``: edge ids sorted by (check, bit); check ``c`` owns the slice
      starting at ``row_starts[c]``. Every check has at least one edge.
    - ``col_slots[b, k]``: ``k``-th edge of bit ``b``, padded with ``num_edges``.
    """

    num_checks: int
    num_bits: int
    edge_check: Index
    edge_bit: Index
    row_order: Index
    row_starts: Index
    col_slots: Index
    bit_degree: Index

    @classmethod
    def from_check_matrix(cls, h: sp.csr_matrix) -> TannerGraph:
        """Build layouts from ``h``; raises ``ValueError`` if a check has no edges."""
        csc = sp.csc_matrix(h)
        csc.sort_indices()
        edge_check = csc.indices.astype(np.int64)
        edge_bit = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr)).astype(np.int64)
        check_degree = np.bincount(edge_check, minlength=csc.shape[0])
        if np.any(check_degree == 0):
            raise ValueError("check matrix has a row without nonzeros")
        return cls(
            num_checks=int(csc.shape[0]), num_bits=int(csc.shape[1]),
            edge_check=edge_check, edge_bit=edge_bit,
            row_order=np.lexsort((edge_bit, edge_check)).astype(np.int64),
            row_starts=np.concatenate(([0], np.cumsum(check_degree)[:-1])).astype(np.int64),
            col_slots=_column_slots(csc.indptr.astype(np.int64)),
            bit_degree=np.diff(csc.indptr).astype(np.int64),
        )

    @property
    def num_edges(self) -> int:
        return int(self.edge_check.size)

    def stats(self) -> GraphStats:
        """Degree summary; cheap."""
        check_degree = np.bincount(self.edge_check, minlength=self.num_checks)
        return GraphStats(
            checks=self.num_checks, bits=self.num_bits, edges=self.num_edges,
            check_degree_min=int(check_degree.min()), check_degree_max=int(check_degree.max()),
            check_degree_mean=float(check_degree.mean()),
            bit_degree_min=int(self.bit_degree.min()), bit_degree_max=int(self.bit_degree.max()),
            bit_degree_mean=float(self.bit_degree.mean()),
            bits_degree_le2=int(np.count_nonzero(self.bit_degree <= 2)),
        )


def _column_slots(indptr: Index) -> Index:
    degree = np.diff(indptr)
    num_edges = int(indptr[-1])
    slots = np.full((degree.size, int(degree.max())), num_edges, dtype=np.int64)
    bits = np.repeat(np.arange(degree.size), degree)
    slots[bits, np.arange(num_edges) - indptr[bits]] = np.arange(num_edges)
    return slots
