"""NumPy backend of IonQ's beam search: each round's children run as one vectorised BP batch.

Control logic (and its documented quirks) lives in :mod:`rtbeam.search`; this module only runs
masked min-sum BP and summarises each run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from rtbeam.decoders import BeamConfig, Bits, DecodeBatch
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import DBL_MAX, Bools, Floats, MinSum
from rtbeam.search import (Child, ChildInputs, ChildSummary, RootSummary, ShotOutcome, ShotPlanner,
                           sequential_sum)
from rtbeam.tanner import TannerGraph

Counts = NDArray[np.int64]


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


class ReferenceBeamSearch:
    """Bit-exact NumPy re-implementation of IonQ's ``BeamSearchDecoder``."""

    def __init__(self, problem: DecodingProblem, config: BeamConfig) -> None:
        graph = TannerGraph.from_check_matrix(problem.check_matrix)
        self.__kernel = MinSum(graph, problem.priors)
        self.__inputs = ChildInputs(problem.check_matrix)
        self.__branchable = graph.bit_degree > 2
        self.__config = config

    def decode_batch(self, syndromes: Bits) -> DecodeBatch:
        """Decode each shot in turn; ``seconds`` is per-shot wall time (not a performance target)."""
        corrections = np.zeros((syndromes.shape[0], self.__branchable.size), dtype=np.uint8)
        seconds = np.zeros(syndromes.shape[0])
        for shot, syndrome in enumerate(syndromes):
            start = time.perf_counter()
            corrections[shot] = self.decode_shot(syndrome).correction
            seconds[shot] = time.perf_counter() - start
        return DecodeBatch(corrections, seconds)

    def decode_shot(self, syndrome: Bits) -> ShotOutcome:
        """Decode one shot; an all-zero syndrome returns all zeros without running BP (as IonQ)."""
        n = self.__branchable.size
        if not syndrome.any():
            return ShotOutcome(np.zeros(n, dtype=np.uint8), 0, 0)
        planner = ShotPlanner(self.__config, self.__kernel.prior_llr, self.__branchable)
        start = BpStart(self.__kernel.initial_messages(1), np.ones((1, n), dtype=np.bool_), syndrome[None])
        root = run_bp(self.__kernel, start, self.__config.initial_iters)
        planner.start(RootSummary(bool(root.converged_at[0]), root.decisions[0].astype(np.uint8),
                                  root.llr_sums[0], root.v2c[0]))
        while children := planner.children():
            planner.select(children, self.__run_children(syndrome, children))
        return planner.outcome()

    def __run_children(self, syndrome: Bits, children: list[Child]) -> list[ChildSummary]:
        free = self.__inputs.free_bits(children)
        start = BpStart(np.stack([cast(Floats, c.parent.messages) for c in children]), free,
                        self.__inputs.syndromes(syndrome, children))
        run = run_bp(self.__kernel, start, self.__config.iters_per_round)
        return [self.__summary(run, free, k) for k in range(len(children))]

    def __summary(self, run: BpRun, free: Bools, k: int) -> ChildSummary:
        magnitude = np.abs(run.llr_sums[k])
        candidates = self.__branchable & free[k] & (magnitude < DBL_MAX)
        branch = int(np.argmin(np.where(candidates, magnitude, np.inf))) if candidates.any() else -1
        converged = bool(run.converged_at[k])
        return ChildSummary(converged, sequential_sum(magnitude) / float(run.iterations[k]), branch,
                            run.decisions[k].astype(np.uint8) if converged else None, run.v2c[k])
