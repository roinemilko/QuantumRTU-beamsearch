"""Backend-agnostic control logic of IonQ's beam search: which children to run, which to keep.

Backends (NumPy reference, CUDA, OptiX) only run masked BP and report a summary per run;
:class:`ShotPlanner` turns summaries into the next round exactly as ``beam_search.hpp`` does,
including its quirks:

- the first branch bit may be bit 0 even if its degree is <= 2;
- children are visited parent-by-parent in descending score order, value 0 before 1;
  the first child (in that order) whose BP converges counts first, whatever its iteration;
- pruning keeps the ``beam_width`` largest ``(score, insertion index)`` pairs, so on equal
  scores the later child survives;
- if initial BP converged and ``num_results > 1``, the root skips the value-0 child
  (C++ reads an all-zero vector there);
- a path with no branchable free bit reuses the last chosen branch bit;
- when nothing converges, the last initial-BP hard decision is returned.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from rtbeam.decoders import BeamConfig, Bits

Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]
Fixings = tuple[tuple[int, int], ...]
DBL_MAX = float(np.finfo(np.float64).max)


@dataclass(frozen=True)
class Path:
    """A beam entry: its fixed bits, the bit it branches on next, and its stored messages.

    ``converged_value`` is ``next_bit``'s value in the path's own converged solution
    (-1 if its BP did not converge); the child repeating that value is skipped.
    ``messages`` is an opaque, backend-owned handle to the path's bit-to-check messages.
    """

    fixed: Fixings
    next_bit: int
    converged_value: int
    messages: object


@dataclass(frozen=True)
class Child:
    """A path extended by fixing its ``next_bit`` to ``value``."""

    parent: Path
    value: int

    @property
    def fixed(self) -> Fixings:
        return self.parent.fixed + ((self.parent.next_bit, self.value),)


@dataclass(frozen=True)
class RootSummary:
    """Initial (unmasked) BP of one shot: hard decisions, summed posteriors, messages handle."""

    converged: bool
    decisions: Bits
    llr_sums: Floats
    messages: object


@dataclass(frozen=True)
class ChildSummary:
    """What a backend reports for one child's masked BP run.

    ``score``: sum over free bits of ``|Σ_iter posterior|`` (sequential, bit order) divided by
    iterations run. ``branch``: argmin of that magnitude over free bits with degree > 2 and
    magnitude < DBL_MAX, lowest index on ties, -1 if none. ``decisions``: free-bit hard
    decisions at convergence (fixed bits are filled in here); may be None if not converged.
    """

    converged: bool
    score: float
    branch: int
    decisions: Bits | None
    messages: object


@dataclass(frozen=True)
class ShotOutcome:
    """Decoder output for one shot plus how far the search went.

    ``rounds`` is 0 when initial BP decided the shot, else the beam rounds run;
    ``solutions`` counts converged solutions seen.
    """

    correction: Bits
    rounds: int
    solutions: int


class ChildInputs:
    """Builds per-child BP inputs (free-bit masks, residual syndromes) from fixings."""

    def __init__(self, check_matrix: sp.csr_matrix) -> None:
        self.__columns = sp.csc_matrix(check_matrix, dtype=np.int32)
        self.__num_bits = int(check_matrix.shape[1])

    def free_bits(self, children: list[Child]) -> Bools:
        """(children × bits) mask, False on each child's fixed bits."""
        free = np.ones((len(children), self.__num_bits), dtype=np.bool_)
        for k, child in enumerate(children):
            free[k, [bit for bit, _ in child.fixed]] = False
        return free

    def syndromes(self, syndrome: Bits, children: list[Child]) -> Bits:
        """``syndrome`` with the checks of each child's bits fixed to 1 toggled (children × checks)."""
        ones = np.zeros((self.__num_bits, len(children)), dtype=np.int32)
        for k, child in enumerate(children):
            ones[[bit for bit, value in child.fixed if value], k] = 1
        flips = (np.asarray(self.__columns @ ones) % 2).T.astype(np.uint8)
        return np.ascontiguousarray(np.bitwise_xor(syndrome[None, :], flips))


class ShotPlanner:
    """Search state for one shot; mirrors the C++ loop's shared variables.

    Protocol: :meth:`start` once, then while :meth:`children` is non-empty run them and pass
    their summaries (same order) to :meth:`select`; finally :meth:`outcome`.
    """

    def __init__(self, config: BeamConfig, prior_llr: Floats, branchable: Bools) -> None:
        self.__config = config
        self.__prior_llr = prior_llr
        self.__branchable = branchable
        self.__best: Bits | None = None
        self.__best_weight = DBL_MAX
        self.__solutions = 0
        self.__last_branch = 0
        self.__round = 0
        self.__done = False
        self.__fallback: Bits | None = None
        self.__explore: list[Path] = []

    def start(self, root: RootSummary) -> None:
        """Consume initial BP; may finish the shot immediately."""
        self.__fallback = root.decisions
        self.__done = root.converged and self.__accept(root.decisions)
        self.__last_branch = self.__initial_branch(root.llr_sums)
        if not self.__done:
            self.__explore = [Path((), self.__last_branch, 0 if root.converged else -1, root.messages)]

    def children(self) -> list[Child]:
        """Children to run this round (visiting order); empty once the search has ended."""
        if self.__done or self.__round >= self.__config.max_rounds:
            return []
        return [Child(p, v) for p in self.__explore for v in (0, 1) if p.converged_value != v]

    def select(self, children: list[Child], summaries: list[ChildSummary]) -> None:
        """Apply one round's results: accept solutions in visiting order, prune to the beam."""
        self.__round += 1
        beam = _Beam(self.__config.beam_width)
        for child, summary in zip(children, summaries):
            decoding = _with_fixings(summary.decisions, child.fixed) if summary.converged else None
            if decoding is not None and self.__accept(decoding):
                self.__done, self.__explore = True, []
                return
            if not beam.rejects(summary.score):
                beam.insert(summary.score, self.__stored_path(child, summary, decoding))
        self.__explore = beam.ranked()

    def outcome(self) -> ShotOutcome:
        assert self.__fallback is not None, "start() not called"
        correction = self.__best if self.__best is not None else self.__fallback
        return ShotOutcome(correction, self.__round, self.__solutions)

    def __initial_branch(self, llr_sums: Floats) -> int:
        candidates = np.where(self.__branchable, np.abs(llr_sums), np.inf)
        candidates[0] = abs(llr_sums[0])
        return int(np.argmin(candidates))

    def __stored_path(self, child: Child, summary: ChildSummary, decoding: Bits | None) -> Path:
        if summary.branch >= 0:
            self.__last_branch = summary.branch
        converged_value = int(decoding[self.__last_branch]) if decoding is not None else -1
        return Path(child.fixed, self.__last_branch, converged_value, summary.messages)

    def __accept(self, decoding: Bits) -> bool:
        weight = sequential_sum(np.where(decoding.astype(np.bool_), self.__prior_llr, 0.0))
        if weight < self.__best_weight:
            self.__best, self.__best_weight = decoding, weight
        self.__solutions += 1
        return self.__solutions == self.__config.num_results


class _Beam:
    """C++ ``priority_queue<pair<score, insertion index>, greater>`` bounded to ``width``."""

    def __init__(self, width: int) -> None:
        self.__width = width
        self.__heap: list[tuple[float, int, Path]] = []
        self.__inserted = 0

    def rejects(self, score: float) -> bool:
        return len(self.__heap) >= self.__width and score < self.__heap[0][0]

    def insert(self, score: float, path: Path) -> None:
        heapq.heappush(self.__heap, (score, self.__inserted, path))
        self.__inserted += 1
        if len(self.__heap) > self.__width:
            heapq.heappop(self.__heap)

    def ranked(self) -> list[Path]:
        """Paths by descending (score, insertion index): the next round's visiting order."""
        return [entry[2] for entry in sorted(self.__heap, key=lambda e: (e[0], e[1]), reverse=True)]


def _with_fixings(decisions: Bits | None, fixed: Fixings) -> Bits:
    assert decisions is not None, "a converged child must report its decisions"
    decoding = decisions.astype(np.uint8)
    for bit, value in fixed:
        decoding[bit] = value
    return decoding


def sequential_sum(values: Floats) -> float:
    """Left-to-right float64 sum (what a C++ loop computes); ``np.sum`` is pairwise."""
    return float(np.cumsum(values)[-1]) if values.size else 0.0
