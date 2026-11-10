"""Adapter for IonQ's single-threaded C++ beam-search decoder (the correctness oracle)."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
from beam_search_decoder import BeamSearchDecoder

from rtbeam.decoders import BeamConfig, Bits, DecodeBatch
from rtbeam.problem import DecodingProblem


class IonQBeamSearch:
    """IonQ's decoder behind the :class:`rtbeam.decoders.Decoder` interface.

    Shots run one after another on the calling thread; ``seconds`` times only the
    C++ ``decode`` call (incl. the Cython copy), comparable to IonQ's own timing.
    """

    def __init__(self, problem: DecodingProblem, config: BeamConfig) -> None:
        self.__decoder = BeamSearchDecoder(
            pcm=problem.check_matrix, error_channel=list(problem.priors), **asdict(config)
        )
        self.__num_faults = problem.num_faults

    def decode_batch(self, syndromes: Bits) -> DecodeBatch:
        """Decode each shot; blocks for the whole batch."""
        corrections = np.zeros((syndromes.shape[0], self.__num_faults), dtype=np.uint8)
        seconds = np.zeros(syndromes.shape[0])
        for shot, syndrome in enumerate(syndromes):
            start = time.perf_counter()
            corrections[shot] = self.__decoder.decode(syndrome)
            seconds[shot] = time.perf_counter() - start
        return DecodeBatch(corrections, seconds)
