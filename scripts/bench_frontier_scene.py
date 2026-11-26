"""S3: same frontier pruning, two ways to find a check's frontier edges: per-round RT scene
(BVH over frontier triangles, row rays) vs per-round CUDA compacted lists.

Start state: messages after k CUDA BP iterations on golden p = 0.001 shots (a beam round
boundary). Per (runs, k): frontier sizes, per-round setup GPU time (shared part; CUDA lists;
RT lists + triangles + BVH build), per-iteration GPU time of each check pass and of the shared
bit pass, a 20-iteration round total for each variant, and the unpruned CUDA kernel's time
for 20 iterations as reference. Verifies both variants give identical messages.
Writes results/frontier_scene.json, or the path given as the first argument.
Since 2026-11-26 also the unpruned RT baseline (RtLauncher) for 20 iterations on the same random syndromes
as the unpruned CUDA reference (``round_unpruned_rt_us``), so the plot can show what pruning saves for RT too.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import cupy as cp
import numpy as np

from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, BpLauncher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.frontier import Frontier
from rtbeam.rt.launcher import RtLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.001"
RUNS = (1, 16, 64, 256)
WARM = (1, 3, 10)
TAU = 6.0
ROUND_ITERS = 20


def main() -> None:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    cuda, fr, rt = BpLauncher(graph, prior, FLOAT32), Frontier(graph, prior), RtLauncher(graph, prior, FLOAT32)
    report: dict[str, dict[str, float]] = {}
    for runs in RUNS:
        syn = cp.asarray(shots.syndromes[:runs])
        free = cp.ones((runs, graph.num_bits), cp.uint8)
        noise = cp.asarray(np.random.default_rng(0).integers(0, 2, (runs, graph.num_checks), dtype=np.uint8))
        unpruned = gpu(lambda: cuda.run(cuda.initial_messages(runs), free, noise, ROUND_ITERS, False), 5)
        unpruned_rt = gpu(lambda: rt.run(rt.initial_messages(runs), free, noise, ROUND_ITERS, False), 5)
        for k in WARM:
            start = cuda.run(cuda.initial_messages(runs), free, syn, k, False).v2c
            rnd = fr.begin_round(start.copy(), syn, TAU)
            fr.build_scene(rnd)
            verify(fr, start, syn)
            shared = gpu(lambda: fr.begin_round(start.copy(), syn, TAU))
            scene = gpu(lambda: fr.build_scene(rnd))
            check_cuda, check_rt, bit = gpu(lambda: fr.check_cuda(rnd)), gpu(lambda: fr.check_rt(rnd)), gpu(lambda: fr.bits(rnd))
            row = {"live_checks_per_run": rnd.live_run.size / runs, "frontier_edges_per_run": rnd.fe_edges.size / runs,
                   "frontier_bits_per_run": rnd.fb_run.size / runs, "frontier_edge_share": rnd.fe_edges.size / (runs * graph.num_edges),
                   "setup_shared_us": shared, "setup_rt_scene_us": scene, "check_cuda_us": check_cuda, "check_rt_us": check_rt,
                   "bit_pass_us": bit, "round_cuda_us": shared + ROUND_ITERS * (check_cuda + bit),
                   "round_rt_us": shared + scene + ROUND_ITERS * (check_rt + bit), "round_unpruned_cuda_us": unpruned,
                   "round_unpruned_rt_us": unpruned_rt}
            report[f"{runs}/{k}"] = row
            print(runs, k, {key: round(v, 2) for key, v in row.items()}, flush=True)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "frontier_scene.json"
    out.write_text(json.dumps(report, indent=2))


def verify(fr: Frontier, start: cp.ndarray, syn: cp.ndarray) -> None:
    """Three frontier iterations with each check pass give identical messages."""
    out = []
    for check in (fr.check_cuda, fr.check_rt):
        rnd = fr.begin_round(start.copy(), syn, TAU)
        fr.build_scene(rnd)
        for _ in range(3):
            check(rnd)
            fr.bits(rnd)
        out.append((rnd.v2c.get(), rnd.c2v.get()))
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1]), "RT and CUDA frontier passes differ"


def gpu(fn: Callable[[], object], repeats: int = 20) -> float:
    fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return float(1e3 * cp.cuda.get_elapsed_time(start, stop) / repeats)


if __name__ == "__main__":
    main()
