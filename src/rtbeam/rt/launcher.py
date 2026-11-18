"""RT-core (OptiX) backend for masked min-sum BP: drop-in replacement for ``BpLauncher``.

The Tanner graph is geometry (one triangle per edge, see ``rt_programs.cu``); every BP
iteration is a row-ray launch (check nodes) and a column-ray launch (bit nodes) over all runs.
The syndrome test of iteration ``it`` rides on the row launch of ``it + 1``; a run whose test
passes is marked converged and its later rays return at once, so results equal the CUDA
kernel's (bit for bit at the same precision). Score and branch bit come from the CUDA
``bp_summary`` kernel. The host library and PTX are compiled on first use into ``.cache/rt``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from rtbeam.gpu.beam import DeviceArray, DeviceRuns, Precision
from rtbeam.tanner import TannerGraph

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_OPTIX_INCLUDE = Path(os.environ.get("RTBEAM_OPTIX_INCLUDE", _ROOT / "external" / "optix-dev" / "include"))
_CACHE = _ROOT / ".cache" / "rt"
_CUDA = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
_CHECK, _BIT, _SYNDROME, _COUNT_ROWS, _COUNT_COLS = range(5)
_BLOCK = 256


class _Params(ctypes.Structure):
    """Mirror of ``RtParams`` in rt_params.h."""

    _fields_ = [(name, ctypes.c_uint64) for name in (
        "gas", "edge_bit", "prior", "free_all", "syn_all", "v2c", "c2v", "ls", "dec", "unsat", "done_at")] + [
        (name, ctypes.c_int32) for name in ("mode", "m", "n", "num_edges", "it", "pad")]


class RtLauncher:
    """Owns the edge-triangle GAS and the OptiX pipeline; same interface as ``BpLauncher``."""

    def __init__(self, graph: TannerGraph, prior_llr: NDArray[np.float64], precision: Precision,
                 max_runs: int = 2048, check_every: int = 1) -> None:
        cp.zeros(1)  # make sure CuPy's CUDA context exists and is current
        self.__lib = _library()
        self.__err = ctypes.create_string_buffer(8192)
        max_hits = int(max(np.diff(np.append(graph.row_starts, graph.num_edges)).max(), graph.bit_degree.max()))
        ptx = _ptx(precision, max_hits)
        vertices = _triangles(graph)
        handle = self.__lib.rtbp_create(vertices.ctypes.data, graph.num_edges, ptx, len(ptx), self.__err,
                                        len(self.__err))
        if not handle:
            raise RuntimeError(f"OptiX setup failed: {self.__err.value.decode()}")
        self.__handle = handle
        kernel_source = (_HERE.parent / "gpu" / "bp_kernel.cu").read_text()
        real = "double" if precision.dtype is np.float64 else "float"
        module = cp.RawModule(code=kernel_source, options=("--fmad=false", f"-DREAL={real}",
                                                           f"-DREAL_MAX={precision.real_max}"))
        self.__summary = module.get_function("bp_summary")
        self.__mark = cp.ElementwiseKernel("int32 u, int32 it", "int32 d", "if (d == 0 && u == 0) d = it;", "rt_mark")
        self.__dtype, self.__max_runs, self.__check_every = precision.dtype, max_runs, check_every
        self.__m, self.__n, self.__e = graph.num_checks, graph.num_bits, graph.num_edges
        self.__edge_bit = cp.asarray(graph.edge_bit, dtype=cp.int32)
        self.__prior = cp.asarray(prior_llr, dtype=precision.dtype)
        self.__branchable = cp.asarray(graph.bit_degree > 2, dtype=cp.uint8)
        self.__initial_v2c = cp.asarray(prior_llr[graph.edge_bit], dtype=precision.dtype)

    def initial_messages(self, runs: int) -> DeviceArray:
        return cp.tile(self.__initial_v2c, (runs, 1))

    def hit_counts(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Edges found by each check's row ray and each bit's column ray (enumeration test)."""
        rows, cols = cp.zeros(self.__m, cp.int32), cp.zeros(self.__n, cp.int32)
        self.__launch(_COUNT_ROWS, 0, self.__m, 1, unsat=rows)
        self.__launch(_COUNT_COLS, 0, self.__n, 1, unsat=cols)
        return rows.get(), cols.get()

    def pass_seconds(self, syndromes: DeviceArray, repeats: int = 20) -> dict[str, float]:
        """Mean GPU time of one row-ray (check) and one column-ray (bit) launch over all runs."""
        runs = syndromes.shape[0]
        state = {"free_all": cp.ones((runs, self.__n), cp.uint8), "syn_all": cp.ascontiguousarray(syndromes),
                 "v2c": self.initial_messages(runs), "c2v": cp.zeros((runs, self.__e), self.__dtype),
                 "ls": cp.zeros((runs, self.__n), self.__dtype), "dec": cp.zeros((runs, self.__n), cp.uint8),
                 "unsat": cp.zeros(runs, cp.int32), "done_at": cp.zeros(runs, cp.int32)}
        times = {}
        for name, mode, width in (("check_pass", _CHECK, self.__m), ("bit_pass", _BIT, self.__n)):
            self.__launch(mode, 2, width, runs, **state)  # warm-up
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            for _ in range(repeats):
                self.__launch(mode, 2, width, runs, **state)
            stop.record()
            stop.synchronize()
            times[name] = cp.cuda.get_elapsed_time(start, stop) / 1e3 / repeats
        return times

    def run(self, v2c: DeviceArray, free: DeviceArray, syndromes: DeviceArray, max_iters: int,
            want_summary: bool) -> DeviceRuns:
        """Same contract as ``BpLauncher.run``."""
        runs = v2c.shape[0]
        v2c, free, syndromes = (cp.ascontiguousarray(a) for a in (v2c, free, syndromes))
        out = DeviceRuns(v2c, cp.zeros(runs, cp.int32), cp.zeros((runs, self.__n), cp.uint8),
                         cp.zeros((runs, self.__n), self.__dtype), cp.zeros(runs, cp.float64), cp.zeros(runs, cp.int32))
        for lo in range(0, runs, self.__max_runs):
            self.__bp(slice(lo, min(lo + self.__max_runs, runs)), out, free, syndromes, max_iters)
        if want_summary:
            self.__summary((runs,), (_BLOCK,), (np.int32(self.__n), np.int32(max_iters), self.__branchable, free,
                                                 out.llr_sums, out.converged_at, out.score, out.branch))
        return out

    def __bp(self, rows: slice, out: DeviceRuns, free: DeviceArray, syndromes: DeviceArray, max_iters: int) -> None:
        count = rows.stop - rows.start
        done, unsat = out.converged_at[rows], cp.zeros(count, cp.int32)
        state = {"free_all": free[rows], "syn_all": syndromes[rows], "v2c": out.v2c[rows],
                 "c2v": cp.empty((count, self.__e), self.__dtype), "ls": out.llr_sums[rows],
                 "dec": out.decisions[rows], "unsat": unsat, "done_at": done}
        for it in range(1, max_iters + 1):
            unsat.fill(0)
            self.__launch(_CHECK, it, self.__m, count, **state)  # also tests iteration it - 1
            if it > 1:
                self.__mark(unsat, np.int32(it - 1), done)
                if (it - 1) % self.__check_every == 0 and not bool((done == 0).any()):
                    return
            self.__launch(_BIT, it, self.__n, count, **state)
        unsat.fill(0)
        self.__launch(_SYNDROME, max_iters, self.__m, count, **state)
        self.__mark(unsat, np.int32(max_iters), done)

    def __launch(self, mode: int, it: int, width: int, height: int, **arrays: DeviceArray) -> None:
        pointers = {k: int(a.data.ptr) for k, a in arrays.items()}
        params = _Params(edge_bit=self.__edge_bit.data.ptr, prior=self.__prior.data.ptr, mode=mode, m=self.__m,
                         n=self.__n, num_edges=self.__e, it=it, **pointers)
        if not self.__lib.rtbp_launch(self.__handle, ctypes.byref(params), width, height, self.__err, len(self.__err)):
            raise RuntimeError(f"optixLaunch failed: {self.__err.value.decode()}")


def _triangles(graph: TannerGraph) -> NDArray[np.float32]:
    """Triangle e around (edge_bit[e], edge_check[e], 0) in the plane x + y = const (9 floats each)."""
    v, c = graph.edge_bit.astype(np.float32), graph.edge_check.astype(np.float32)
    h = np.float32(0.25)
    low, top = np.full_like(v, -h), np.full_like(v, 0.4)
    corners = [np.stack((v - h, c + h, low), axis=1), np.stack((v + h, c - h, low), axis=1), np.stack((v, c, top), axis=1)]
    return np.ascontiguousarray(np.stack(corners, axis=1).reshape(-1, 9), dtype=np.float32)


def _library() -> ctypes.CDLL:
    source = _HERE / "rt_host.cpp"
    target = _CACHE / f"librtbp_{_digest(source, _HERE / 'rt_params.h')}.so"
    if not target.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        _run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", str(source), f"-I{_OPTIX_INCLUDE}", f"-I{_CUDA / 'include'}",
              f"-L{_CUDA / 'lib64' / 'stubs'}", "-lcuda", "-ldl", "-o", str(target)])
    lib = ctypes.CDLL(str(target))
    lib.rtbp_create.restype = ctypes.c_void_p
    lib.rtbp_create.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                                ctypes.c_size_t]
    lib.rtbp_launch.restype = ctypes.c_int
    lib.rtbp_launch.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Params), ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p,
                                ctypes.c_size_t]
    lib.rtbp_params_size.restype = ctypes.c_size_t
    assert lib.rtbp_params_size() == ctypes.sizeof(_Params), "RtParams layout mismatch"
    return lib


def _ptx(precision: Precision, max_hits: int) -> bytes:
    real = "double" if precision.dtype is np.float64 else "float"
    defines = [f"-DREAL={real}", f"-DREAL_MAX={precision.real_max}", f"-DMAX_HITS={max_hits}"]
    source = _HERE / "rt_programs.cu"
    target = _CACHE / f"rt_programs_{_digest(source, _HERE / 'rt_params.h', extra=' '.join(defines))}.ptx"
    if not target.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        _run([str(_CUDA / "bin" / "nvcc"), "-ptx", "-arch=compute_75", "--fmad=false", "-O3", "-std=c++17",
              f"-I{_OPTIX_INCLUDE}", f"-I{_HERE}", *defines, str(source), "-o", str(target)])
    return target.read_bytes()


def _digest(*paths: Path, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"build failed: {' '.join(cmd)}\n{result.stderr}")
