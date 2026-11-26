"""Closest-hit min-sum (RT core computes check-node minimums) equals the CUDA float32 kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1)
except Exception:  # noqa: BLE001 - no GPU on this host
    pytest.skip("no CUDA device", allow_module_level=True)

from rtbeam.golden import ShotSet  # noqa: E402
from rtbeam.gpu.beam import FLOAT32, BpLauncher  # noqa: E402
from rtbeam.problem import DecodingProblem  # noqa: E402
from rtbeam.reference.minsum import MinSum  # noqa: E402
from rtbeam.rt.ch_launcher import ChLauncher  # noqa: E402
from rtbeam.tanner import TannerGraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.002"


@pytest.mark.parametrize("options", [{"rebuild": True}, {"rebuild": False}, {"check_every": 4, "max_runs": 64}])
def test_closest_hit_bp_equals_cuda_float32(options: dict[str, object]) -> None:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    syn = cp.asarray(shots.syndromes[:200])
    rng = np.random.default_rng(5)
    free = cp.asarray((rng.random((200, graph.num_bits)) > 0.01).astype(np.uint8))
    cuda = BpLauncher(graph, prior, FLOAT32)
    expected = cuda.run(cuda.initial_messages(200), free, syn, 30, True)
    ch = ChLauncher(graph, prior, **options)  # type: ignore[arg-type]
    actual = ch.run(ch.initial_messages(200), free, syn, 30, True)
    for name in ("v2c", "converged_at", "decisions", "llr_sums", "score", "branch"):
        np.testing.assert_array_equal(getattr(expected, name).get(), getattr(actual, name).get(), err_msg=name)
