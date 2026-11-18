"""RT (OptiX) backend vs the CUDA kernel; skipped without CuPy, a GPU or the OptiX headers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001 (no driver / no device)
    pytest.skip("no CUDA device", allow_module_level=True)

from rtbeam.golden import ShotSet  # noqa: E402
from rtbeam.gpu.beam import FLOAT32, FLOAT64, BpLauncher, Precision  # noqa: E402
from rtbeam.problem import DecodingProblem  # noqa: E402
from rtbeam.reference.minsum import MinSum  # noqa: E402
from rtbeam.rt.launcher import RtLauncher  # noqa: E402
from rtbeam.tanner import TannerGraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Setup = tuple[DecodingProblem, TannerGraph, "np.ndarray[tuple[int, ...], np.dtype[np.float64]]"]
STEM = "BB[[144,12,12]],memory_Z,error_rate=0.002,syndrome_rounds=12"


@pytest.fixture(scope="module")
def setup() -> Setup:
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{STEM}.stim")
    return problem, TannerGraph.from_check_matrix(problem.check_matrix), MinSum(
        TannerGraph.from_check_matrix(problem.check_matrix), problem.priors).prior_llr


def test_rays_enumerate_exactly_the_tanner_edges(setup: Setup) -> None:
    problem, graph, prior = setup
    rows, cols = RtLauncher(graph, prior, FLOAT32).hit_counts()
    np.testing.assert_array_equal(rows, np.diff(problem.check_matrix.indptr))
    np.testing.assert_array_equal(cols, graph.bit_degree)


@pytest.mark.parametrize("precision", [FLOAT64, FLOAT32], ids=lambda p: p.name)
def test_rt_bp_equals_cuda_bp(setup: Setup, precision: Precision) -> None:
    problem, graph, prior = setup
    shots = ShotSet.load(ROOT / "data" / "golden" / "bb144_z_p0.002" / "shots.npz").syndromes[:256]
    rng = np.random.default_rng(7)
    free = (rng.random((shots.shape[0], graph.num_bits)) > 0.01).astype(np.uint8)  # exercise the mask
    outs = []
    for launcher in (BpLauncher(graph, prior, precision), RtLauncher(graph, prior, precision)):
        runs = launcher.run(launcher.initial_messages(shots.shape[0]), cp.asarray(free), cp.asarray(shots), 30, True)
        outs.append([a.get() for a in (runs.v2c, runs.converged_at, runs.decisions, runs.llr_sums, runs.score,
                                       runs.branch)])
    for cuda, rt in zip(*outs):
        np.testing.assert_array_equal(cuda, rt)
