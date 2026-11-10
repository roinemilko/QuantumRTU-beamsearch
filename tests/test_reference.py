"""The NumPy reference must reproduce IonQ's decoder bit for bit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from beam_search_decoder import BeamSearchDecoder

from rtbeam.decoders import PRESETS, BeamConfig
from rtbeam.golden import ShotSet
from rtbeam.ionq import IonQBeamSearch
from rtbeam.problem import DecodingProblem
from rtbeam.reference.beam import BpStart, ReferenceBeamSearch, run_bp
from rtbeam.reference.minsum import MinSum
from rtbeam.tanner import TannerGraph

CIRCUITS = Path(__file__).resolve().parents[1] / "external" / "BeamSearchDecoder" / "StimCircuit"
STRESS = CIRCUITS / "BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim"


@pytest.fixture(scope="module")
def problem() -> DecodingProblem:
    return DecodingProblem.from_stim_file(STRESS)


@pytest.fixture(scope="module")
def shots() -> ShotSet:
    return ShotSet.sample(STRESS, 24, seed=11)


def test_tanner_layouts_cover_every_edge_once(problem: DecodingProblem) -> None:
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    rebuilt = sp.csr_matrix((np.ones(graph.num_edges), (graph.edge_check, graph.edge_bit)), shape=problem.check_matrix.shape)
    assert (rebuilt != problem.check_matrix).nnz == 0
    assert np.array_equal(np.sort(graph.row_order), np.arange(graph.num_edges))
    assert np.array_equal(np.sort(graph.col_slots[graph.col_slots < graph.num_edges]), np.arange(graph.num_edges))


def test_initial_bp_posteriors_are_bit_exact(problem: DecodingProblem, shots: ShotSet) -> None:
    config = BeamConfig()
    ionq = BeamSearchDecoder(pcm=problem.check_matrix, error_channel=list(problem.priors))
    kernel = MinSum(TannerGraph.from_check_matrix(problem.check_matrix), problem.priors)
    compared = 0
    for syndrome in shots.syndromes:
        start = BpStart(kernel.initial_messages(1), np.ones((1, problem.num_faults), np.bool_), syndrome[None])
        converged_at = int(run_bp(kernel, start, config.initial_iters).converged_at[0])
        if converged_at == 0 or not syndrome.any():
            continue
        ionq.decode(syndrome)
        assert ionq.iter == converged_at
        assert np.array_equal(_posterior_after(kernel, start, converged_at), ionq.log_prob_ratios)
        compared += 1
    assert compared >= 5


def _posterior_after(kernel: MinSum, start: BpStart, iterations: int) -> np.ndarray:
    v2c = start.v2c
    for _ in range(iterations):
        step = kernel.step(v2c, start.free_bits, start.syndromes)
        v2c = step.v2c
    return step.posterior[0]


@pytest.mark.parametrize("preset", ["beam8_230", "beam64_32res_640"])
def test_reference_matches_ionq(problem: DecodingProblem, shots: ShotSet, preset: str) -> None:
    config = PRESETS[preset]
    subset = shots.syndromes[:12] if preset == "beam8_230" else shots.syndromes[:3]
    expected = IonQBeamSearch(problem, config).decode_batch(subset).corrections
    actual = ReferenceBeamSearch(problem, config).decode_batch(subset).corrections
    assert np.array_equal(actual, expected)
