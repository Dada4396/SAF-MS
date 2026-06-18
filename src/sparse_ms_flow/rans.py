"""Small static-model range asymmetric numeral system implementation."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


RANS_LOWER_BOUND = 1 << 23


@dataclass(frozen=True)
class RansPayload:
    """Self-contained data needed to reconstruct an integer array."""

    shape: Tuple[int, ...]
    minimum: int
    cdf: np.ndarray
    state: int
    stream: bytes


def normalized_frequencies(counts: np.ndarray, precision: int) -> np.ndarray:
    """Normalize positive counts to positive frequencies summing to 2**precision."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("counts must be a non-empty vector of positive integers")
    total = 1 << precision
    if counts.size > total:
        raise ValueError("precision is too small for the symbol alphabet")

    exact = counts.astype(np.float64) * total / counts.sum()
    frequencies = np.maximum(1, np.floor(exact).astype(np.int64))
    difference = int(total - frequencies.sum())

    if difference > 0:
        order = np.argsort(-(exact - np.floor(exact)), kind="stable")
        for index in order[:difference]:
            frequencies[index] += 1
    elif difference < 0:
        order = np.argsort(exact - np.floor(exact), kind="stable")
        remaining = -difference
        for index in order:
            removable = min(remaining, int(frequencies[index] - 1))
            frequencies[index] -= removable
            remaining -= removable
            if remaining == 0:
                break
        if remaining:
            raise ValueError("unable to normalize symbol frequencies")
    return frequencies


def encode_indices(indices: np.ndarray, cdf: np.ndarray, precision: int) -> tuple[int, bytes]:
    """Encode zero-based symbol indices using a static cumulative distribution."""
    frequencies = np.diff(cdf)
    state = RANS_LOWER_BOUND
    stream = bytearray()
    for symbol in np.asarray(indices, dtype=np.int64)[::-1]:
        start = int(cdf[symbol])
        frequency = int(frequencies[symbol])
        maximum = ((RANS_LOWER_BOUND >> precision) << 8) * frequency
        while state >= maximum:
            stream.append(state & 0xFF)
            state >>= 8
        state = ((state // frequency) << precision) + state % frequency + start
    return state, bytes(stream)


def decode_indices(
    count: int,
    cdf: np.ndarray,
    precision: int,
    state: int,
    stream: bytes,
) -> np.ndarray:
    """Decode a static-model rANS stream into zero-based symbol indices."""
    total_mask = (1 << precision) - 1
    frequencies = np.diff(cdf)
    output = np.empty(count, dtype=np.int64)
    cursor = len(stream) - 1
    for index in range(count):
        slot = state & total_mask
        symbol = int(np.searchsorted(cdf, slot, side="right") - 1)
        output[index] = symbol
        state = int(frequencies[symbol]) * (state >> precision) + slot - int(cdf[symbol])
        while state < RANS_LOWER_BOUND and cursor >= 0:
            state = (state << 8) | stream[cursor]
            cursor -= 1
    if cursor != -1 or state != RANS_LOWER_BOUND:
        raise ValueError("rANS stream is truncated or contains trailing data")
    return output
