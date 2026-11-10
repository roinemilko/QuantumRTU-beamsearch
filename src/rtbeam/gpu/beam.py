"""CUDA beam-search decoder (Task A baseline): BP on the GPU, beam control on the host.

- Initial BP of every shot in a batch runs as one kernel launch (one thread block per shot).
- Each beam round, the children of *all* still-active shots run as one launch; the GPU returns
  per-child convergence, score and next branch bit; the host's :class:`rtbeam.search.ShotPlanner`
  (shared with the NumPy reference) decides what survives. Messages never leave the GPU.
- ``float64`` reproduces IonQ's decoder bit for bit; ``float32`` is the performance mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from rtbeam.decoders import BeamConfig, Bits, DecodeBatch
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.search import Child, ChildInputs, ChildSummary, RootSummary, ShotPlanner
from rtbeam.tanner import TannerGraph

DeviceArray = Any  # cupy.ndarray (CuPy ships no type stubs)
_SOURCE = Path(__file__).with_name("bp_kernel.cu").read_text()
_BLOCK = 256


@dataclass(frozen=True)
class Precision:
    """Floating-point type the kernel runs in."""

    name: str
    dtype: type[np.floating[Any]]
    real_max: str


FLOAT64 = Precision("float64", np.float64, "1.7976931348623157e308")
FLOAT32 = Precision("float32", np.float32, "3.402823466e38f")


@dataclass(frozen=True)
class DeviceRuns:
    """Device-resident outputs of one batch of BP runs (row r = run r)."""

    v2c: DeviceArray
    converged_at: DeviceArray
    decisions: DeviceArray
    llr_sums: DeviceArray
    score: DeviceArray
    branch: DeviceArray


class BpLauncher:
    """Owns the graph on the device and launches ``bp_runs`` in chunks of at most ``max_runs``."""

    def __init__(self, graph: TannerGraph, prior_llr: NDArray[np.float64], precision: Precision, max_runs: int = 2048) -> None:
        real = "double" if precision.dtype is np.float64 else "float"
        module = cp.RawModule(code=_SOURCE, options=("--fmad=false", f"-DREAL={real}", f"-DREAL_MAX={precision.real_max}"))
        self.__kernel = module.get_function("bp_runs")
        self.__dtype, self.__max_runs = precision.dtype, max_runs
        self.__m, self.__n, self.__e = graph.num_checks, graph.num_bits, graph.num_edges
        self.__arrays = (
            _i32(np.append(graph.row_starts, graph.num_edges)), _i32(graph.row_order),
            _i32(np.append(0, np.cumsum(graph.bit_degree))), _i32(graph.edge_bit),
            cp.asarray(prior_llr, dtype=precision.dtype), cp.asarray(graph.bit_degree > 2, dtype=cp.uint8),
        )
        self.__initial_v2c = cp.asarray(prior_llr[graph.edge_bit], dtype=precision.dtype)

    def initial_messages(self, runs: int) -> DeviceArray:
        return cp.tile(self.__initial_v2c, (runs, 1))

    def run(self, v2c: DeviceArray, free: DeviceArray, syndromes: DeviceArray, max_iters: int,
            want_summary: bool) -> DeviceRuns:
        """Run BP on ``v2c`` (runs × edges; in place if C-contiguous); inputs are uint8 (runs × bits / checks)."""
        runs = v2c.shape[0]
        v2c, free, syndromes = (cp.ascontiguousarray(a) for a in (v2c, free, syndromes))
        out = DeviceRuns(v2c, cp.zeros(runs, cp.int32), cp.zeros((runs, self.__n), cp.uint8),
                         cp.zeros((runs, self.__n), self.__dtype), cp.zeros(runs, cp.float64), cp.zeros(runs, cp.int32))
        for lo in range(0, runs, self.__max_runs):
            self.__launch(slice(lo, min(lo + self.__max_runs, runs)), out, (free, syndromes), (max_iters, want_summary))
        return out

    def __launch(self, rows: slice, out: DeviceRuns, inputs: tuple[DeviceArray, DeviceArray],
                 limits: tuple[int, bool]) -> None:
        count = rows.stop - rows.start
        scratch = cp.empty((count, self.__e), self.__dtype)
        args = (np.int32(self.__m), np.int32(self.__n), np.int32(limits[0]), np.int32(limits[1]), *self.__arrays,
                inputs[0][rows], inputs[1][rows], out.v2c[rows], scratch, out.llr_sums[rows], out.decisions[rows],
                out.converged_at[rows], out.score[rows], out.branch[rows])
        self.__kernel((count,), (_BLOCK,), args)


class GpuBeamSearch:
    """IonQ's beam search with all BP on the GPU; see module docstring."""

    def __init__(self, problem: DecodingProblem, config: BeamConfig, precision: Precision = FLOAT32,
                 shot_chunk: int = 1024, beam_memory: int = 2 * 2**30) -> None:
        """``beam_memory`` bounds the device bytes of one beam phase: at most that many
        bytes' worth of shots (≈3 message arrays of 2·beam_width runs each) search together."""
        graph = TannerGraph.from_check_matrix(problem.check_matrix)
        prior_llr = MinSum(graph, problem.priors).prior_llr
        self.__launcher = BpLauncher(graph, prior_llr, precision)
        self.__inputs = ChildInputs(problem.check_matrix)
        self.__planner_args = (config, prior_llr, graph.bit_degree > 2)
        self.__config, self.__shot_chunk, self.__n = config, shot_chunk, graph.num_bits
        per_shot = 3 * 2 * config.beam_width * graph.num_edges * np.dtype(precision.dtype).itemsize
        self.__beam_shots = max(1, beam_memory // per_shot)

    def decode_batch(self, syndromes: Bits) -> DecodeBatch:
        """Decode all shots together; blocks until done. ``seconds`` is NaN (batch-level timing only)."""
        corrections = np.zeros((syndromes.shape[0], self.__n), dtype=np.uint8)
        shots = np.flatnonzero(syndromes.any(axis=1))
        for lo in range(0, shots.size, self.__shot_chunk):
            chunk = shots[lo:lo + self.__shot_chunk]
            corrections[chunk] = self.__decode_chunk(syndromes[chunk])
        return DecodeBatch(corrections, np.full(syndromes.shape[0], np.nan))

    def __decode_chunk(self, syndromes: Bits) -> Bits:
        runs = syndromes.shape[0]
        roots = self.__launcher.run(self.__launcher.initial_messages(runs), cp.ones((runs, self.__n), cp.uint8),
                                    cp.asarray(syndromes), self.__config.initial_iters, want_summary=False)
        corrections = cast(Bits, roots.decisions.get())
        planners = list(self.__start_planners(roots, corrections).items())
        for lo in range(0, len(planners), self.__beam_shots):
            group = planners[lo:lo + self.__beam_shots]
            while work := [(shot, p, cs) for shot, p in group if (cs := p.children())]:
                self.__run_round(syndromes, work)
            for shot, planner in group:
                corrections[shot] = planner.outcome().correction
        return corrections

    def __start_planners(self, roots: DeviceRuns, decisions: Bits) -> dict[int, ShotPlanner]:
        converged = roots.converged_at.get() > 0
        pending = np.flatnonzero(~converged | (self.__config.num_results > 1))
        llr_sums = roots.llr_sums[pending].get().astype(np.float64) if pending.size else None
        planners = {}
        for j, shot in enumerate(pending.tolist()):
            planners[shot] = ShotPlanner(*self.__planner_args)
            assert llr_sums is not None
            planners[shot].start(RootSummary(bool(converged[shot]), decisions[shot], llr_sums[j], (roots.v2c, shot)))
        return planners

    def __run_round(self, syndromes: Bits, work: list[tuple[int, ShotPlanner, list[Child]]]) -> None:
        children = [c for _, _, cs in work for c in cs]
        free = np.concatenate([self.__inputs.free_bits(cs) for _, _, cs in work]).astype(np.uint8)
        residual = np.concatenate([self.__inputs.syndromes(syndromes[shot], cs) for shot, _, cs in work])
        runs = self.__launcher.run(_gather_messages(children), cp.asarray(free), cp.asarray(residual),
                                   self.__config.iters_per_round, want_summary=True)
        summaries = _summaries(runs)
        offset = 0
        for _, planner, cs in work:
            planner.select(cs, summaries[offset:offset + len(cs)])
            offset += len(cs)


def _summaries(runs: DeviceRuns) -> list[ChildSummary]:
    converged, score, branch = runs.converged_at.get() > 0, runs.score.get(), runs.branch.get()
    hits = np.flatnonzero(converged)
    decisions = dict(zip(hits.tolist(), runs.decisions[hits].get())) if hits.size else {}
    return [ChildSummary(bool(converged[k]), float(score[k]), int(branch[k]), decisions.get(k), (runs.v2c, k))
            for k in range(converged.size)]


def _gather_messages(children: list[Child]) -> DeviceArray:
    handles = [cast(tuple[DeviceArray, int], c.parent.messages) for c in children]
    out = cp.empty((len(children), handles[0][0].shape[1]), handles[0][0].dtype)
    groups: dict[int, tuple[DeviceArray, list[int], list[int]]] = {}
    for position, (array, row) in enumerate(handles):
        _, positions, rows = groups.setdefault(id(array), (array, [], []))
        positions.append(position)
        rows.append(row)
    for array, positions, rows in groups.values():
        out[cp.asarray(positions)] = array[cp.asarray(rows)]
    return out


def _i32(values: NDArray[np.int64]) -> DeviceArray:
    return cp.asarray(values, dtype=cp.int32)
