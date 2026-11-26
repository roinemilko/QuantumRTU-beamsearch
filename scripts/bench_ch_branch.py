"""R3: beam branch-bit selection (argmin |LLR sum| over free bits of degree > 2) by one closest-hit
ray per run, vs the CUDA block reduction in ``bp_summary``.

LLR sums come from 30 CUDA BP iterations on golden p = 0.002 shots with ~1% random fixed bits
(like beam children). Reports agreement with CUDA's choice and GPU times: RT (scene kernel +
BVH refit + trace), a CuPy masked argmin (branch only) and CUDA ``bp_summary`` (branch bit +
serial fp64 score sum). Writes results/ch_branch.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cupy as cp
import numpy as np

from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, BpLauncher
from rtbeam.problem import DecodingProblem
from rtbeam.reference.minsum import MinSum
from rtbeam.rt.ch_launcher import ChLauncher
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
CASE = "bb144_z_p0.002"
RUNS = (1, 16, 64, 256, 1024)
ITERS = 30


def main() -> None:
    shots = ShotSet.load(ROOT / "data" / "golden" / CASE / "shots.npz")
    problem = DecodingProblem.from_stim_file(ROOT / "external" / "BeamSearchDecoder" / "StimCircuit" / f"{shots.stem}.stim")
    graph = TannerGraph.from_check_matrix(problem.check_matrix)
    prior = MinSum(graph, problem.priors).prior_llr
    cuda = BpLauncher(graph, prior, FLOAT32)
    ch = ChLauncher(graph, prior, rebuild=False)
    summary = cp.RawModule(code=(ROOT / "src" / "rtbeam" / "gpu" / "bp_kernel.cu").read_text(),
                           options=("--fmad=false", "-DREAL=float", f"-DREAL_MAX={FLOAT32.real_max}")).get_function("bp_summary")
    branchable = cp.asarray(graph.bit_degree > 2, dtype=cp.uint8)
    rng = np.random.default_rng(3)
    report: dict[str, dict[str, float]] = {}
    for runs in RUNS:
        idx = np.arange(runs) % shots.syndromes.shape[0]
        syn = cp.asarray(shots.syndromes[idx])
        free = cp.asarray((rng.random((runs, graph.num_bits)) > 0.01).astype(np.uint8))
        out = cuda.run(cuda.initial_messages(runs), free, syn, ITERS, True)
        rt_branch = ch.branch_bits(free, out.llr_sums)
        agree = float((rt_branch == out.branch).mean())
        ls = cp.abs(out.llr_sums)
        tie_value_ok = float((ls[cp.arange(runs), rt_branch] == ls[cp.arange(runs), out.branch]).mean())
        score, branch = cp.empty(runs, cp.float64), cp.empty(runs, cp.int32)
        eligible = (free != 0) & (branchable[None, :] != 0)
        row = {"agreement": agree, "same_min_value": tie_value_ok,
               "cuda_branch_only_us": gpu(lambda: cp.where(eligible, ls, cp.inf).argmin(axis=1)),
               "rt_branch_us": gpu(lambda: ch.branch_bits(free, out.llr_sums)),
               "cuda_summary_us": gpu(lambda: summary((runs,), (256,), (np.int32(graph.num_bits), np.int32(ITERS), branchable,
                                                                       free, out.llr_sums, out.converged_at, score, branch)))}
        report[str(runs)] = row
        print(runs, {k: round(v, 4) for k, v in row.items()}, flush=True)
    (ROOT / "results" / "ch_branch.json").write_text(json.dumps(report, indent=2))


def gpu(fn: Callable[[], object], repeats: int = 30) -> float:
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
