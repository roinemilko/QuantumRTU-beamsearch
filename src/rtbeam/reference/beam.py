"""IonQ's beam search, with each round's children run as one vectorised BP batch.

Semantics follow ``beam_search.hpp`` exactly, including its quirks:

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
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from rtbeam.decoders import BeamConfig, Bits, DecodeBatch
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import DBL_MAX, Bools, Floats, MinSum
from rtbeam.tanner import TannerGraph

Counts = NDArray[np.int64]
Fixings = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BpStart:
    """Initial state of K independent masked BP runs."""

    v2c: Floats
    free_bits: Bools
    syndromes: Bits


@dataclass(frozen=True)
class BpRun:
    """Outcome of K masked BP runs; each run stops at its first converging iteration.

    ``converged_at[k]`` is that iteration (0 = never); ``decisions`` and ``v2c`` are taken there
    or after the last iteration; ``llr_sums`` add posteriors of free bits over the iterations run.
    """

    converged_at: Counts
    iterations: Counts
    decisions: Bools
    llr_sums: Floats
    v2c: Floats


def run_bp(kernel: MinSum, start: BpStart, max_iters: int) -> BpRun:
    """Run up to ``max_iters`` iterations of every run in ``start``."""
    v2c, free = start.v2c.copy(), start.free_bits
    converged_at = np.zeros(free.shape[0], dtype=np.int64)
    llr_sums, decisions = np.zeros(free.shape), np.zeros(free.shape, dtype=np.bool_)
    for it in range(1, max_iters + 1):
        live = np.flatnonzero(converged_at == 0)
        if live.size == 0:
            break
        step = kernel.step(v2c[live], free[live], start.syndromes[live])
        v2c[live], decisions[live] = step.v2c, step.decisions
        llr_sums[live] += np.where(free[live], step.posterior, 0.0)
        converged_at[live[step.satisfied]] = it
    iterations = np.where(converged_at > 0, converged_at, max_iters)
    return BpRun(converged_at, iterations, decisions, llr_sums, v2c)


@dataclass(frozen=True)
class Path:
    """A beam entry: its fixed bits, the bit it branches on next, and its stored messages.

    ``converged_value`` is ``next_bit``'s value in the path's own converged solution
    (-1 if its BP did not converge); the child repeating that value is skipped.
    """

    fixed: Fixings
    next_bit: int
    converged_value: int
    v2c: Floats


@dataclass(frozen=True)
class Child:
    """A path extended by fixing its ``next_bit`` to ``value``."""

    parent: Path
    value: int

    @property
    def fixed(self) -> Fixings:
        return self.parent.fixed + ((self.parent.next_bit, self.value),)


@dataclass(frozen=True)
class ShotOutcome:
    """Decoder output for one shot plus how far the search went.

    ``rounds`` is 0 when initial BP decided the shot, else the beam rounds run;
    ``solutions`` counts converged solutions seen.
    """

    correction: Bits
    rounds: int
    solutions: int


class ReferenceBeamSearch:
    """Bit-exact NumPy re-implementation of IonQ's ``BeamSearchDecoder``."""

    def __init__(self, problem: DecodingProblem, config: BeamConfig) -> None:
        graph = TannerGraph.from_check_matrix(problem.check_matrix)
        self.__context = _Context(
            kernel=MinSum(graph, problem.priors), config=config,
            columns=sp.csc_matrix(problem.check_matrix, dtype=np.int32),
            branchable=graph.bit_degree > 2,
        )

    def decode_batch(self, syndromes: Bits) -> DecodeBatch:
        """Decode each shot in turn; ``seconds`` is per-shot wall time (not a performance target)."""
        corrections = np.zeros((syndromes.shape[0], self.__context.branchable.size), dtype=np.uint8)
        seconds = np.zeros(syndromes.shape[0])
        for shot, syndrome in enumerate(syndromes):
            start = time.perf_counter()
            corrections[shot] = self.decode_shot(syndrome).correction
            seconds[shot] = time.perf_counter() - start
        return DecodeBatch(corrections, seconds)

    def decode_shot(self, syndrome: Bits) -> ShotOutcome:
        """Decode one shot; an all-zero syndrome returns all zeros without running BP (as IonQ)."""
        if not syndrome.any():
            return ShotOutcome(np.zeros(self.__context.branchable.size, dtype=np.uint8), 0, 0)
        return _ShotSearch(self.__context, syndrome).run()


@dataclass(frozen=True)
class _Context:
    kernel: MinSum
    config: BeamConfig
    columns: sp.csc_matrix
    branchable: Bools


class _ShotSearch:
    """Mutable search state for one shot; mirrors the C++ loop's shared variables."""

    def __init__(self, context: _Context, syndrome: Bits) -> None:
        self.__ctx = context
        self.__syndrome = syndrome
        self.__best: Bits | None = None
        self.__best_weight = DBL_MAX
        self.__solutions = 0
        self.__last_branch = 0

    def run(self) -> ShotOutcome:
        root, fallback, done = self.__initial_bp()
        explore = [] if done else [root]
        for round_index in range(self.__ctx.config.max_rounds):
            if done or not explore:
                return self.__outcome(fallback, round_index)
            children = [Child(p, v) for p in explore for v in (0, 1) if p.converged_value != v]
            run = run_bp(self.__ctx.kernel, self.__start(children), self.__ctx.config.iters_per_round)
            explore, done = self.__select(children, run)
        return self.__outcome(fallback, self.__ctx.config.max_rounds)

    def __initial_bp(self) -> tuple[Path, Bits, bool]:
        n = self.__ctx.branchable.size
        start = BpStart(self.__ctx.kernel.initial_messages(1), np.ones((1, n), dtype=np.bool_), self.__syndrome[None])
        run = run_bp(self.__ctx.kernel, start, self.__ctx.config.initial_iters)
        decoding = run.decisions[0].astype(np.uint8)
        converged = bool(run.converged_at[0])
        done = converged and self.__accept(decoding)
        self.__last_branch = self.__initial_branch(run.llr_sums[0])
        return Path((), self.__last_branch, 0 if converged else -1, run.v2c[0]), decoding, done

    def __initial_branch(self, llr_sums: Floats) -> int:
        candidates = np.where(self.__ctx.branchable, np.abs(llr_sums), np.inf)
        candidates[0] = abs(llr_sums[0])
        return int(np.argmin(candidates))

    def __start(self, children: list[Child]) -> BpStart:
        free = np.ones((len(children), self.__ctx.branchable.size), dtype=np.bool_)
        syndromes = np.tile(self.__syndrome, (len(children), 1))
        for k, child in enumerate(children):
            free[k, [bit for bit, _ in child.fixed]] = False
            syndromes[k] ^= self.__flipped_checks([bit for bit, value in child.fixed if value])
        return BpStart(np.stack([c.parent.v2c for c in children]), free, syndromes)

    def __flipped_checks(self, bits: list[int]) -> Bits:
        indicator = np.zeros(self.__ctx.branchable.size, dtype=np.int32)
        indicator[bits] = 1
        return (np.asarray(self.__ctx.columns @ indicator) % 2).astype(np.uint8)

    def __select(self, children: list[Child], run: BpRun) -> tuple[list[Path], bool]:
        beam = _Beam(self.__ctx.config.beam_width)
        for k, child in enumerate(children):
            decoding = _with_fixings(run.decisions[k], child.fixed)
            if run.converged_at[k] and self.__accept(decoding):
                return [], True
            score = _sequential_sum(np.abs(run.llr_sums[k])) / float(run.iterations[k])
            if not beam.rejects(score):
                beam.insert(score, self.__stored_path(child, run, k, decoding))
        return beam.ranked(), False

    def __stored_path(self, child: Child, run: BpRun, k: int, decoding: Bits) -> Path:
        free = np.ones(self.__ctx.branchable.size, dtype=np.bool_)
        free[[bit for bit, _ in child.fixed]] = False
        branch = self.__next_branch(run.llr_sums[k], free)
        converged_value = int(decoding[branch]) if run.converged_at[k] else -1
        return Path(child.fixed, branch, converged_value, run.v2c[k])

    def __next_branch(self, llr_sums: Floats, free: Bools) -> int:
        magnitude = np.abs(llr_sums)
        candidates = self.__ctx.branchable & free & (magnitude < DBL_MAX)
        if candidates.any():
            self.__last_branch = int(np.argmin(np.where(candidates, magnitude, np.inf)))
        return self.__last_branch

    def __accept(self, decoding: Bits) -> bool:
        weight = _sequential_sum(np.where(decoding.astype(np.bool_), self.__ctx.kernel.prior_llr, 0.0))
        if weight < self.__best_weight:
            self.__best, self.__best_weight = decoding, weight
        self.__solutions += 1
        return self.__solutions == self.__ctx.config.num_results

    def __outcome(self, fallback: Bits, rounds: int) -> ShotOutcome:
        correction = self.__best if self.__best is not None else fallback
        return ShotOutcome(correction, rounds, self.__solutions)


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


def _with_fixings(decisions: Bools, fixed: Fixings) -> Bits:
    decoding = decisions.astype(np.uint8)
    for bit, value in fixed:
        decoding[bit] = value
    return decoding


def _sequential_sum(values: Floats) -> float:
    return float(np.cumsum(values)[-1]) if values.size else 0.0
