"""S3: frontier pruning with two enumeration mechanisms, RT scene vs CUDA compacted lists.

Pruning (shared by both): at a beam-round boundary the bits whose messages are all confident
(|v2c| >= tau on every edge) are frozen for the round; their contribution to each check is
summarised once (``fr_frozen``). Every iteration then updates only the frontier: each live
(run, check), i.e. one with at least one frontier edge, and each frontier bit.

The variants differ only in how a live check finds its frontier edges:
- ``cuda``: a compacted edge list per round (prefix sum), one thread per live check;
- ``rt``: a per-round scene with one triangle per frontier edge (BVH built once per round),
  one row ray per live check; the RT core's traversal enumerates the edges.
The bit pass (``fr_bit``) is the same CUDA kernel in both. Both give bit-identical messages.
This measures the mechanism; whether freezing keeps the LER is a separate experiment (S1).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from rtbeam.gpu.beam import DeviceArray
from rtbeam.rt.ch_launcher import _library
from rtbeam.rt.launcher import _CACHE, _CUDA, _OPTIX_INCLUDE, _digest, _run
from rtbeam.tanner import TannerGraph

_HERE = Path(__file__).resolve().parent
_BLOCK = 256


class _FrParams(ctypes.Structure):
    """Mirror of ``FrParams`` in fr_params.h."""

    _fields_ = [(name, ctypes.c_uint64) for name in (
        "gas", "live_run", "live_check", "prim_edge", "v2c", "c2v", "syn", "fz_min1", "fz_min2", "fz_arg", "fz_neg")] + [
        (name, ctypes.c_int32) for name in ("m", "num_edges", "live_count", "pad")]


@dataclass
class Round:
    """Per-round state shared by both variants (frontier decision + frozen summaries + work lists)."""

    runs: int
    v2c: DeviceArray
    c2v: DeviceArray
    syn: DeviceArray
    frozen: tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray]
    live_run: DeviceArray
    live_check: DeviceArray
    fe_run: DeviceArray
    fe_edges: DeviceArray
    fe_ptr: DeviceArray
    fb_run: DeviceArray
    fb_bit: DeviceArray


class Frontier:
    """Both variants of the frontier-restricted BP iteration on one Tanner graph (float32)."""

    def __init__(self, graph: TannerGraph, prior_llr: NDArray[np.float64]) -> None:
        cp.zeros(1)
        self.__lib = _library()
        self.__err = ctypes.create_string_buffer(8192)
        ptx = _ptx()
        handle = self.__lib.chbp_create(ptx, len(ptx), ctypes.sizeof(_FrParams), b"__raygen__rows", b"", b"__anyhit__collect",
                                        b"__miss__none", 2, self.__err, len(self.__err))
        if not handle:
            raise RuntimeError(f"OptiX setup failed: {self.__err.value.decode()}")
        assert self.__lib.chbp_fr_params_size() == ctypes.sizeof(_FrParams), "FrParams layout mismatch"
        self.__handle = handle
        module = cp.RawModule(code=(_HERE / "fr_kernels.cu").read_text(), options=("--fmad=false", f"-I{_HERE}"))
        self.__frozen_k, self.__check_k, self.__bit_k, self.__geom_k = (
            module.get_function(k) for k in ("fr_frozen", "fr_check_cuda", "fr_bit", "fr_geometry"))
        self.__m, self.__n, self.__e = graph.num_checks, graph.num_bits, graph.num_edges
        self.__row_ptr = cp.asarray(np.append(graph.row_starts, graph.num_edges), dtype=cp.int32)
        self.__row_order = cp.asarray(graph.row_order, dtype=cp.int32)
        self.__csr_check = cp.asarray(graph.edge_check[graph.row_order], dtype=cp.int32)
        self.__col_ptr = cp.asarray(np.append(0, np.cumsum(graph.bit_degree)), dtype=cp.int32)
        self.__edge_bit = cp.asarray(graph.edge_bit, dtype=cp.int32)
        self.__edge_check = cp.asarray(graph.edge_check, dtype=cp.int32)
        self.__prior = cp.asarray(prior_llr, dtype=cp.float32)
        self.__verts = cp.empty((0, 9), cp.float32)
        self.triangles = 0

    # ---- per round ------------------------------------------------------------------------
    def begin_round(self, v2c: DeviceArray, syn: DeviceArray, tau: float) -> Round:
        """Shared per-round work: frontier decision, frozen summaries, live-check and bit lists."""
        runs, m, n, E = v2c.shape[0], self.__m, self.__n, self.__e
        small = cp.abs(v2c) < tau
        bits = cp.zeros((runs, n), cp.int32)
        rows = cp.repeat(cp.arange(runs, dtype=cp.int32), E)
        cp.add.at(bits, (rows, cp.tile(self.__edge_bit, runs)), small.ravel().astype(cp.int32))
        frontier_bits = bits > 0
        frontier = cp.ascontiguousarray(frontier_bits[:, self.__edge_bit]).astype(cp.uint8)
        fz = (cp.empty((runs, m), cp.float32), cp.empty((runs, m), cp.float32), cp.empty((runs, m), cp.int32),
              cp.empty((runs, m), cp.uint8))
        self.__frozen_k(_grid(runs * m), (_BLOCK,), (np.int32(m), np.int32(E), np.int32(runs), self.__row_ptr,
                                                     self.__row_order, frontier, v2c, *fz))
        pos = cp.flatnonzero(frontier[:, self.__row_order]).astype(cp.int32)  # (run, csr position), grouped by check
        fe_run, slot = pos // E, pos % E
        fe_edges = self.__row_order[slot]
        key = fe_run * m + self.__csr_check[slot]
        counts = cp.bincount(key, minlength=runs * m)
        live = cp.flatnonzero(counts).astype(cp.int32)
        fe_ptr = cp.concatenate((cp.zeros(1, cp.int32), cp.cumsum(counts[live]).astype(cp.int32)))
        fb = cp.flatnonzero(frontier_bits).astype(cp.int32)
        return Round(runs, v2c, cp.zeros_like(v2c), syn, fz, live // m, live % m, fe_run, fe_edges, fe_ptr,
                     fb // n, fb % n)

    def build_scene(self, rnd: Round) -> None:
        """RT-only per-round work: frontier triangles + BVH build."""
        count = int(rnd.fe_edges.size)
        if count > self.__verts.shape[0]:
            self.__verts = cp.empty((count, 9), cp.float32)
        if count:
            self.__geom_k(_grid(count), (_BLOCK,), (np.int32(count), rnd.fe_run, rnd.fe_edges, self.__edge_bit,
                                                     self.__edge_check, self.__verts))
            if not self.__lib.chbp_accel(self.__handle, 0, self.__verts.data.ptr, count, 0, self.__err, len(self.__err)):
                raise RuntimeError(f"optixAccelBuild failed: {self.__err.value.decode()}")
        self.triangles = count

    # ---- per iteration --------------------------------------------------------------------
    def check_cuda(self, rnd: Round) -> None:
        count = int(rnd.live_run.size)
        if count:
            self.__check_k(_grid(count), (_BLOCK,), (np.int32(count), np.int32(self.__m), np.int32(self.__e), rnd.live_run,
                                                      rnd.live_check, rnd.fe_ptr, rnd.fe_edges, rnd.v2c, rnd.c2v, rnd.syn,
                                                      *rnd.frozen))

    def check_rt(self, rnd: Round) -> None:
        count = int(rnd.live_run.size)
        if not count:
            return
        params = _FrParams(live_run=rnd.live_run.data.ptr, live_check=rnd.live_check.data.ptr,
                           prim_edge=rnd.fe_edges.data.ptr, v2c=rnd.v2c.data.ptr, c2v=rnd.c2v.data.ptr,
                           syn=rnd.syn.data.ptr, fz_min1=rnd.frozen[0].data.ptr, fz_min2=rnd.frozen[1].data.ptr,
                           fz_arg=rnd.frozen[2].data.ptr, fz_neg=rnd.frozen[3].data.ptr, m=self.__m,
                           num_edges=self.__e, live_count=count)
        if not self.__lib.chbp_launch(self.__handle, 0, ctypes.addressof(params), count, 1, self.__err, len(self.__err)):
            raise RuntimeError(f"optixLaunch failed: {self.__err.value.decode()}")

    def bits(self, rnd: Round) -> None:
        count = int(rnd.fb_run.size)
        if count:
            self.__bit_k(_grid(count), (_BLOCK,), (np.int32(count), np.int32(self.__n), np.int32(self.__e), rnd.fb_run,
                                                    rnd.fb_bit, self.__col_ptr, self.__prior, rnd.v2c, rnd.c2v))


def _grid(threads: int) -> tuple[int]:
    return ((threads + _BLOCK - 1) // _BLOCK,)


def _ptx() -> bytes:
    source = _HERE / "fr_programs.cu"
    target = _CACHE / f"fr_programs_{_digest(source, _HERE / 'fr_params.h', _HERE / 'fr_common.h')}.ptx"
    if not target.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        _run([str(_CUDA / "bin" / "nvcc"), "-ptx", "-arch=compute_75", "--fmad=false", "-O3", "-std=c++17",
              f"-I{_OPTIX_INCLUDE}", f"-I{_HERE}", str(source), "-o", str(target)])
    return target.read_bytes()
