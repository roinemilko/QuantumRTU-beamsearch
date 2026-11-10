"""Decoding problems: a stim detector error model as check matrix, priors and observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim
from numpy.typing import NDArray
from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices

Bits = NDArray[np.uint8]


@dataclass(frozen=True)
class DecodingProblem:
    """Binary decoding problem ``H·e = s`` built from a detector error model.

    ``check_matrix`` is (detectors × fault mechanisms), ``observables`` is
    (logical observables × fault mechanisms), both CSR with uint8 entries;
    ``priors[j]`` is the probability of fault ``j``. Matrices are built the same
    way IonQ's ``BeamSearch`` builds them, so decoders see identical inputs.
    """

    name: str
    check_matrix: sp.csr_matrix
    priors: NDArray[np.float64]
    observables: sp.csr_matrix

    @classmethod
    def from_stim_file(cls, path: Path) -> DecodingProblem:
        """Build from a ``.stim`` circuit file (hyperedges kept undecomposed)."""
        dem = stim.Circuit.from_file(str(path)).detector_error_model()
        mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        return cls(
            name=path.stem,
            check_matrix=sp.csr_matrix(mats.check_matrix, dtype=np.uint8),
            priors=np.asarray(mats.priors, dtype=np.float64),
            observables=sp.csr_matrix(mats.observables_matrix, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path) -> DecodingProblem:
        """Load a problem written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            return cls(
                name=str(f["name"]),
                check_matrix=_csr_from(f, "h", int(f["n"])),
                priors=f["priors"],
                observables=_csr_from(f, "l", int(f["n"])),
            )

    def save(self, path: Path) -> None:
        """Write to a compressed ``.npz`` (no pickles)."""
        h, obs = self.check_matrix, self.observables
        np.savez_compressed(
            path, name=self.name, n=h.shape[1], priors=self.priors,
            h_indptr=h.indptr, h_indices=h.indices, h_rows=h.shape[0],
            l_indptr=obs.indptr, l_indices=obs.indices, l_rows=obs.shape[0],
        )

    @property
    def num_detectors(self) -> int:
        return int(self.check_matrix.shape[0])

    @property
    def num_faults(self) -> int:
        return int(self.check_matrix.shape[1])

    def syndromes_of(self, errors: Bits) -> Bits:
        """Detector pattern ``H·e mod 2`` for each row of ``errors`` (shots × faults)."""
        return _gf2_products(self.check_matrix, errors)

    def logical_flips(self, errors: Bits) -> Bits:
        """Observable flips ``L·e mod 2`` for each row of ``errors`` (shots × faults)."""
        return _gf2_products(self.observables, errors)


def _csr_from(f: np.lib.npyio.NpzFile, key: str, n: int) -> sp.csr_matrix:
    indptr, indices = f[f"{key}_indptr"], f[f"{key}_indices"]
    data = np.ones(len(indices), dtype=np.uint8)
    return sp.csr_matrix((data, indices, indptr), shape=(int(f[f"{key}_rows"]), n))


def _gf2_products(matrix: sp.csr_matrix, rows: Bits) -> Bits:
    products = matrix.astype(np.int32) @ rows.astype(np.int32).T
    return (np.asarray(products).T % 2).astype(np.uint8)
