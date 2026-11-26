"""Closest-hit min-sum: the RT core computes the check-node minimums (fp32 only).

Each BP run's messages are geometry: the message on edge e of run r is a triangle at depth
``|v2c[r][e]|`` in the tube of (check(e), r) (see ``ch_programs.cu``). Per iteration:

1. refit the GAS to the current message magnitudes (OptiX ``OPERATION_UPDATE``);
2. check launch, one raygen per (check, run): two closest-hit rays give min1/argmin and min2
   in hardware; a byte per edge gives the sign parity and the syndrome test of the last iteration;
3. mark converged runs, one host sync ("all done?") every ``check_every`` iterations;
4. CUDA bit kernel: rebuilds c2v from (min1, min2, argmin, parity), does the column sums in
   CSC order and writes the new depths and flags.

The arithmetic equals ``bp_kernel.cu`` (float), so results match ``BpLauncher(FLOAT32)`` as long
as the RT core orders the magnitudes the same way as fp32 comparison does (ties are harmless).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from rtbeam.gpu.beam import FLOAT32, DeviceArray, DeviceRuns, Precision
from rtbeam.rt.launcher import _CACHE, _CUDA, _OPTIX_INCLUDE, _digest, _run
from rtbeam.tanner import TannerGraph

_HERE = Path(__file__).resolve().parent
_CHECK, _SYNDROME, _MIN_ONLY, _BRANCH = range(4)
_BRANCH_SLOTS = 16
_BLOCK = 256


class _ChParams(ctypes.Structure):
    """Mirror of ``ChParams`` in ch_params.h."""

    _fields_ = [(name, ctypes.c_uint64) for name in (
        "gas", "v2c", "syn_all", "flags", "row_ptr", "min1", "min2", "arg", "neg", "unsat", "done_at", "hit_t",
        "spare")] + [(name, ctypes.c_int32) for name in ("mode", "m", "n", "num_edges", "it", "pad")]


@dataclass
class _Slot:
    """Geometry of one batch capacity: vertex buffer + flags, GAS id in the host library."""

    index: int
    runs: int
    vertices: DeviceArray
    flags: DeviceArray
    triangles: int
    built: bool = False


class ChLauncher:
    """Same interface as ``BpLauncher``; float32 only.

    ``rebuild``: rebuild the BVH from scratch at the start of every ``run`` (else only refit).
    ``check_every``: host convergence test every k iterations (semantics unchanged).
    """

    def __init__(self, graph: TannerGraph, prior_llr: NDArray[np.float64], precision: Precision = FLOAT32,
                 max_runs: int = 256, check_every: int = 1, rebuild: bool = True) -> None:
        if precision.dtype is not np.float32:
            raise ValueError("closest-hit min-sum is float32 only (RT hardware orders fp32 distances)")
        cp.zeros(1)
        self.__lib = _library()
        self.__err = ctypes.create_string_buffer(8192)
        ptx = _ptx()
        handle = self.__lib.chbp_create(ptx, len(ptx), ctypes.sizeof(_ChParams), b"__raygen__check", b"__closesthit__min",
                                        b"__anyhit__skip", b"__miss__none", 3, self.__err, len(self.__err))
        if not handle:
            raise RuntimeError(f"OptiX setup failed: {self.__err.value.decode()}")
        self.__handle = handle
        module = cp.RawModule(code=(_HERE / "ch_kernels.cu").read_text(), options=("--fmad=false",))
        self.__layout, self.__init, self.__bit, self.__branch_scene = (
            module.get_function(k) for k in ("ch_layout", "ch_init", "ch_bit", "ch_branch_scene"))
        summary = cp.RawModule(code=(_HERE.parent / "gpu" / "bp_kernel.cu").read_text(),
                               options=("--fmad=false", "-DREAL=float", f"-DREAL_MAX={FLOAT32.real_max}"))
        self.__summary = summary.get_function("bp_summary")
        self.__mark = cp.ElementwiseKernel("int32 u, int32 it", "int32 d", "if (d == 0 && u == 0) d = it;", "ch_mark")
        self.__max_runs, self.__check_every, self.__rebuild = max_runs, max(1, check_every), rebuild
        self.__m, self.__n, self.__e = graph.num_checks, graph.num_bits, graph.num_edges
        csr_pos = np.empty(graph.num_edges, np.int32)
        csr_pos[graph.row_order] = np.arange(graph.num_edges, dtype=np.int32)
        self.__row_ptr = cp.asarray(np.append(graph.row_starts, graph.num_edges), dtype=cp.int32)
        self.__col_ptr = cp.asarray(np.append(0, np.cumsum(graph.bit_degree)), dtype=cp.int32)
        self.__edge_check = cp.asarray(graph.edge_check, dtype=cp.int32)
        self.__edge_bit = cp.asarray(graph.edge_bit, dtype=cp.int32)
        self.__csr_pos = cp.asarray(csr_pos)
        self.__prior = cp.asarray(prior_llr, dtype=cp.float32)
        self.__branchable = cp.asarray(graph.bit_degree > 2, dtype=cp.uint8)
        self.__initial_v2c = cp.asarray(prior_llr[graph.edge_bit], dtype=cp.float32)
        self.__slots: dict[int, _Slot] = {}
        self.__branch_slots: dict[int, _Slot] = {}
        self.hit_t: DeviceArray | None = None  # set to a (runs, m) float32 array to record ray-1 distances

    def initial_messages(self, runs: int) -> DeviceArray:
        return cp.tile(self.__initial_v2c, (runs, 1))

    def run(self, v2c: DeviceArray, free: DeviceArray, syndromes: DeviceArray, max_iters: int,
            want_summary: bool) -> DeviceRuns:
        """Same contract as ``BpLauncher.run``."""
        runs = v2c.shape[0]
        v2c, free, syndromes = (cp.ascontiguousarray(a) for a in (v2c, free, syndromes))
        out = DeviceRuns(v2c, cp.zeros(runs, cp.int32), cp.zeros((runs, self.__n), cp.uint8),
                         cp.zeros((runs, self.__n), cp.float32), cp.zeros(runs, cp.float64), cp.zeros(runs, cp.int32))
        for lo in range(0, runs, self.__max_runs):
            self.__bp(slice(lo, min(lo + self.__max_runs, runs)), out, free, syndromes, max_iters)
        if want_summary:
            self.__summary((runs,), (_BLOCK,), (np.int32(self.__n), np.int32(max_iters), self.__branchable, free,
                                                 out.llr_sums, out.converged_at, out.score, out.branch))
        return out

    # ---- pieces, public for the microbenchmarks (R0) -------------------------------------
    def prepare(self, v2c: DeviceArray, free: DeviceArray) -> _Slot:
        """Write depths/flags for ``v2c`` into the capacity slot and build (or refit) its GAS."""
        runs = v2c.shape[0]
        slot = self.__slot(runs)
        self.__init(_grid(runs * self.__e), (_BLOCK,), (np.int32(self.__e), np.int32(self.__n), np.int32(runs),
                                                         self.__edge_bit, self.__csr_pos, free, v2c, slot.vertices,
                                                         slot.flags))
        self.accel(slot, update=slot.built and not self.__rebuild)
        return slot

    def accel(self, slot: _Slot, update: bool) -> None:
        if not self.__lib.chbp_accel(self.__handle, slot.index, slot.vertices.data.ptr, slot.triangles, int(update), self.__err,
                                     len(self.__err)):
            raise RuntimeError(f"optixAccelBuild failed: {self.__err.value.decode()}")
        slot.built = True

    def minimums(self, slot: _Slot, v2c: DeviceArray, runs: int) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
        """min1, min2, argmin per (run, check) from the current GAS only (pure RT-core work)."""
        min1, min2 = cp.empty((runs, self.__m), cp.float32), cp.empty((runs, self.__m), cp.float32)
        arg = cp.empty((runs, self.__m), cp.int32)
        self.__launch(slot, _MIN_ONLY, 0, runs, v2c=v2c, min1=min1, min2=min2, arg=arg)
        return min1, min2, arg

    def branch_bits(self, free: DeviceArray, llr_sums: DeviceArray) -> DeviceArray:
        """R3: branch bit per run (argmin |LLR sum| over free bits of degree > 2, -1 if none) by one
        closest-hit ray per run. Ties may pick a different index than CUDA's lowest-index rule."""
        runs = llr_sums.shape[0]
        capacity = 1 << max(0, (runs - 1).bit_length())
        slot = self.__branch_slots.get(capacity)
        if slot is None:
            slot = _Slot(_BRANCH_SLOTS + capacity.bit_length() - 1, capacity,
                         cp.empty((capacity * self.__n, 9), cp.float32), cp.empty(0, cp.uint8), capacity * self.__n)
            self.__branch_slots[capacity] = slot
        self.__branch_scene(_grid(capacity * self.__n), (_BLOCK,), (np.int32(self.__n), np.int32(runs),
                                                                     np.int32(capacity), free, self.__branchable,
                                                                     llr_sums, slot.vertices))
        self.accel(slot, update=slot.built and not self.__rebuild)
        branch = cp.empty(runs, cp.int32)
        self.__launch(slot, _BRANCH, 0, runs, width=1, arg=branch)
        return branch

    def gas_bytes(self, slot: _Slot) -> int:
        return int(self.__lib.chbp_gas_bytes(self.__handle, slot.index))

    # ---- internals ------------------------------------------------------------------------
    def __bp(self, rows: slice, out: DeviceRuns, free: DeviceArray, syndromes: DeviceArray, max_iters: int) -> None:
        count = rows.stop - rows.start
        v2c, fr, syn = out.v2c[rows], free[rows], syndromes[rows]
        done, unsat = out.converged_at[rows], cp.zeros(count, cp.int32)
        slot = self.prepare(v2c, fr)
        state = {"v2c": v2c, "syn_all": syn, "flags": slot.flags, "row_ptr": self.__row_ptr,
                 "min1": cp.empty((count, self.__m), cp.float32), "min2": cp.empty((count, self.__m), cp.float32),
                 "arg": cp.empty((count, self.__m), cp.int32), "neg": cp.empty((count, self.__m), cp.uint8),
                 "unsat": unsat, "done_at": done}
        bit_args = (np.int32(self.__m), np.int32(self.__n), np.int32(self.__e), np.int32(count), self.__col_ptr,
                    self.__edge_check, self.__csr_pos, self.__prior, fr, done, state["min1"], state["min2"],
                    state["arg"], state["neg"], v2c, out.llr_sums[rows], out.decisions[rows], slot.vertices, slot.flags)
        for it in range(1, max_iters + 1):
            if it > 1:
                self.accel(slot, update=True)
            unsat.fill(0)
            self.__launch(slot, _CHECK, it, count, **state)
            if it > 1:
                self.__mark(unsat, np.int32(it - 1), done)
                if (it - 1) % self.__check_every == 0 and not bool((done == 0).any()):
                    return
            self.__bit(_grid(count * self.__n), (_BLOCK,), bit_args)
        unsat.fill(0)
        self.__launch(slot, _SYNDROME, max_iters, count, **state)
        self.__mark(unsat, np.int32(max_iters), done)

    def __slot(self, runs: int) -> _Slot:
        capacity = 1 << max(0, (runs - 1).bit_length())
        slot = self.__slots.get(capacity)
        if slot is None:
            vertices = cp.empty((capacity * self.__e, 9), cp.float32)
            slot = _Slot(len(self.__slots), capacity, vertices, cp.zeros((capacity, self.__e), cp.uint8),
                         capacity * self.__e)
            self.__slots[capacity] = slot
            self.__far(slot, 0)
        elif runs < capacity:  # unused runs of the slot: far away, never traced
            self.__far(slot, runs)
        return slot

    def __far(self, slot: _Slot, first: int) -> None:
        self.__layout(_grid((slot.runs - first) * self.__e), (_BLOCK,),
                      (np.int32(self.__e), np.int32(first), np.int32(slot.runs), self.__edge_check, slot.vertices))

    def __launch(self, slot: _Slot, mode: int, it: int, runs: int, width: int = 0, **arrays: DeviceArray) -> None:
        pointers = {k: int(a.data.ptr) for k, a in arrays.items()}
        if self.hit_t is not None:
            pointers["hit_t"] = int(self.hit_t.data.ptr)
        params = _ChParams(mode=mode, m=self.__m, n=self.__n, num_edges=self.__e, it=it, **pointers)
        if not self.__lib.chbp_launch(self.__handle, slot.index, ctypes.addressof(params), width or self.__m, runs, self.__err,
                                      len(self.__err)):
            raise RuntimeError(f"optixLaunch failed: {self.__err.value.decode()}")


def _grid(threads: int) -> tuple[int]:
    return ((threads + _BLOCK - 1) // _BLOCK,)


def _library() -> ctypes.CDLL:
    """Generic OptiX host library (``ch_host.cpp``), shared with the frontier-scene experiment."""
    source = _HERE / "ch_host.cpp"
    target = _CACHE / f"libchbp_{_digest(source, _HERE / 'ch_params.h', _HERE / 'fr_params.h')}.so"
    if not target.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        _run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", str(source), f"-I{_OPTIX_INCLUDE}", f"-I{_CUDA / 'include'}",
              f"-L{_CUDA / 'lib64' / 'stubs'}", "-lcuda", "-ldl", "-o", str(target)])
    lib = ctypes.CDLL(str(target))
    lib.chbp_create.restype = ctypes.c_void_p
    lib.chbp_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.chbp_accel.restype = ctypes.c_int
    lib.chbp_accel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_size_t]
    lib.chbp_launch.restype = ctypes.c_int
    lib.chbp_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                                ctypes.c_char_p, ctypes.c_size_t]
    lib.chbp_gas_bytes.restype = ctypes.c_size_t
    lib.chbp_gas_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.chbp_ch_params_size.restype = ctypes.c_size_t
    lib.chbp_fr_params_size.restype = ctypes.c_size_t
    assert lib.chbp_ch_params_size() == ctypes.sizeof(_ChParams), "ChParams layout mismatch"
    return lib


def _ptx() -> bytes:
    source = _HERE / "ch_programs.cu"
    target = _CACHE / f"ch_programs_{_digest(source, _HERE / 'ch_params.h')}.ptx"
    if not target.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        _run([str(_CUDA / "bin" / "nvcc"), "-ptx", "-arch=compute_75", "--fmad=false", "-O3", "-std=c++17",
              f"-I{_OPTIX_INCLUDE}", f"-I{_HERE}", str(source), "-o", str(target)])
    return target.read_bytes()
