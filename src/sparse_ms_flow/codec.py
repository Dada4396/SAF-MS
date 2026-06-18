"""Public array codec backed by a static rANS model."""

import math

import numpy as np

from .rans import RansPayload, decode_indices, encode_indices, normalized_frequencies


class RansCodec:
    """Losslessly encode integer NumPy arrays with an embedded static model."""

    def __init__(self, precision: int = 16) -> None:
        if not 1 <= precision <= 20:
            raise ValueError("precision must be between 1 and 20")
        self.precision = precision

    def encode(self, symbols: np.ndarray) -> RansPayload:
        array = np.asarray(symbols)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("symbols must contain integers")
        if array.size == 0:
            raise ValueError("symbols must not be empty")
        if np.issubdtype(array.dtype, np.unsignedinteger):
            if int(array.max()) > np.iinfo(np.int64).max:
                raise ValueError("symbols must fit in signed 64-bit integers")

        flattened = array.astype(np.int64, copy=False).ravel()
        alphabet, indices, counts = np.unique(
            flattened, return_inverse=True, return_counts=True
        )
        frequencies = normalized_frequencies(counts, self.precision)
        cdf = np.concatenate(([0], np.cumsum(frequencies))).astype(np.int64)
        state, stream = encode_indices(indices, cdf, self.precision)

        minimum = int(alphabet[0])
        is_contiguous = int(alphabet[-1]) - minimum == alphabet.size - 1
        if not is_contiguous:
            cdf = np.column_stack((np.concatenate((alphabet, [0])), cdf))
            minimum = 0
        return RansPayload(tuple(array.shape), minimum, cdf, state, stream)

    def decode(self, payload: RansPayload) -> np.ndarray:
        cdf = np.asarray(payload.cdf, dtype=np.int64)
        if cdf.ndim == 2:
            if cdf.shape[1] != 2 or cdf.shape[0] < 2:
                raise ValueError("rANS CDF has an invalid sparse-alphabet shape")
            alphabet = cdf[:-1, 0]
            cumulative = cdf[:, 1]
            if np.any(alphabet[1:] <= alphabet[:-1]):
                raise ValueError("rANS CDF alphabet must be strictly increasing")
        else:
            if cdf.ndim != 1 or cdf.size < 2:
                raise ValueError("rANS CDF must be a one-dimensional cumulative table")
            alphabet = np.arange(cdf.size - 1, dtype=np.int64)
            cumulative = cdf
        if (
            cumulative[0] != 0
            or cumulative[-1] != 1 << self.precision
            or np.any(np.diff(cumulative) <= 0)
        ):
            raise ValueError("rANS CDF must increase from zero to 2 ** precision")
        if not isinstance(payload.shape, tuple) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in payload.shape
        ):
            raise ValueError("rANS payload shape must contain nonnegative integers")
        count = math.prod(payload.shape)
        if count < 1 or count > np.iinfo(np.intp).max:
            raise ValueError("rANS payload shape has an invalid element count")
        if not isinstance(payload.state, int) or payload.state < 1:
            raise ValueError("rANS payload state must be a positive integer")
        if not isinstance(payload.stream, bytes):
            raise ValueError("rANS payload stream must contain bytes")
        indices = decode_indices(
            count, cumulative, self.precision, payload.state, payload.stream
        )
        return (alphabet[indices] + payload.minimum).reshape(payload.shape)
