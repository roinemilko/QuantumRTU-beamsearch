"""S1 frontier-freezing kernel: with tau = inf it must reproduce bp_runs bit for bit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1)
except Exception:  # noqa: BLE001 - no GPU on this host
    pytest.skip("no CUDA device", allow_module_level=True)

from rtbeam.decoders import PRESETS  # noqa: E402
from rtbeam.golden import ShotSet  # noqa: E402
from rtbeam.gpu.beam import FLOAT32, FLOAT64, BpLauncher, Freezing, GpuBeamSearch, Precision  # noqa: E402
from rtbeam.problem import DecodingProblem  # noqa: E402
from rtbeam.reference.minsum import MinSum  # noqa: E402
from rtbeam.tanner import TannerGraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.002"


def _problem() -> tuple[ShotSet, DecodingProblem]:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    return shots, DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")


@pytest.mark.parametrize("precision", [FLOAT32, FLOAT64], ids=lambda p: p.name)
@pytest.mark.parametrize(("warm", "refresh"), [(0, 0), (3, 0), (0, 4)])
def test_infinite_tau_equals_bp_runs(precision: Precision, warm: int, refresh: int) -> None:
    shots, problem = _problem()
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    runs = 200
    syn = cp.asarray(shots.syndromes[:runs])
    rng = np.random.default_rng(7)
    free_np = (rng.random((runs, graph.num_bits)) > 0.002).astype(np.uint8)
    seeds_np = rng.integers(0, graph.num_bits, runs).astype(np.int32)
    seeds_np[::5] = -1
    free_np[np.arange(runs), seeds_np] = np.where(seeds_np >= 0, 0, free_np[np.arange(runs), seeds_np])
    free = cp.asarray(free_np)
    plain = BpLauncher(graph, prior, precision)
    start = plain.run(plain.initial_messages(runs), cp.ones_like(free), syn, 7, False).v2c
    expected = plain.run(start.copy(), free, syn, 20, True)
    frontier = BpLauncher(graph, prior, precision, freezing=Freezing(float("inf"), warm, refresh=refresh))
    actual = frontier.run(start.copy(), free, syn, 20, True, seeds=cp.asarray(seeds_np))
    for name in ("v2c", "converged_at", "decisions", "llr_sums", "score", "branch"):
        np.testing.assert_array_equal(getattr(expected, name).get(), getattr(actual, name).get(), err_msg=name)


def test_infinite_tau_beam_search_unchanged() -> None:
    shots, problem = _problem()
    hard = shots.syndromes[:400]
    plain = GpuBeamSearch(problem, PRESETS["beam8_230"], FLOAT32).decode_batch(hard).corrections
    frozen = GpuBeamSearch(problem, PRESETS["beam8_230"], FLOAT32, freezing=Freezing(float("inf"))).decode_batch(hard)
    np.testing.assert_array_equal(plain, frozen.corrections)


@pytest.mark.parametrize("freezing", [Freezing(6.0), Freezing(6.0, refresh=5), Freezing(6.0, initial_warm=5)])
def test_finite_tau_runs_and_prunes(freezing: Freezing) -> None:
    shots, problem = _problem()
    decoder = GpuBeamSearch(problem, PRESETS["beam8_230"], FLOAT32, freezing=freezing)
    corrections = decoder.decode_batch(shots.syndromes[:400]).corrections
    assert corrections.shape == (400, problem.check_matrix.shape[1])
    runs, _, edges, _ = decoder.frontier_totals
    assert runs > 0 and edges < runs * problem.check_matrix.nnz
