"""CPU baseline decoders other than IonQ's beam search."""

from __future__ import annotations

import time

import numpy as np
from stimbposd.bp_osd import create_decoder

from rtbeam.decoders import Bits, DecodeBatch
from rtbeam.problem import DecodingProblem


class BpOsdBaseline:
    """``bp30+osd`` exactly as IonQ's paper configures it via stimbposd/ldpc.

    Min-sum BP (30 iterations) followed by order-10 combination-sweep OSD;
    single-threaded, shots decoded sequentially.
    """

    def __init__(self, problem: DecodingProblem) -> None:
        h = problem.check_matrix
        self.__decoder = create_decoder(
            pcm=h, priors=problem.priors, max_iter=30, bp_method="min_sum",
            osd_order=min(10, h.shape[1] - h.shape[0]), osd_method="osd_cs",
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
