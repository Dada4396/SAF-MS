"""Validated NumPy data loading and deterministic dataset splitting."""

from os import PathLike
from typing import Tuple, Union

import numpy as np


Pathish = Union[str, PathLike[str]]
WindowSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]


def load_spectrum_windows(path: Pathish) -> np.ndarray:
    """Load spectrum windows as a contiguous ``samples x 2 x 512`` array."""
    windows = np.load(path, allow_pickle=False)
    if windows.ndim != 3:
        raise ValueError(
            "expected NumPy data shaped samples x 2 x 512 or samples x 512 x 2"
        )
    if windows.shape[1:] == (512, 2):
        windows = windows.transpose(0, 2, 1)
    elif windows.shape[1:] != (2, 512):
        raise ValueError(
            "expected NumPy data shaped samples x 2 x 512 or samples x 512 x 2"
        )
    if len(windows) == 0:
        raise ValueError("spectrum data must contain at least one sample")
    if not np.issubdtype(windows.dtype, np.number) or np.issubdtype(
        windows.dtype, np.complexfloating
    ):
        raise ValueError("spectrum values must be real integers")
    if not np.isfinite(windows).all():
        raise ValueError("spectrum values must be finite integers")
    if np.issubdtype(windows.dtype, np.floating) and not np.equal(
        windows, np.trunc(windows)
    ).all():
        raise ValueError("spectrum values must be integer-valued")
    lower = int(windows.min())
    upper = int(windows.max())
    bounds = np.iinfo(np.int64)
    if lower < bounds.min or upper > bounds.max:
        raise ValueError("spectrum values must fit in signed 64-bit integers")
    return np.ascontiguousarray(windows, dtype=np.int64)


def split_windows(windows: np.ndarray, seed: int = 0) -> WindowSplit:
    """Shuffle and split windows into deterministic 80/10/10 partitions."""
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[1:] != (2, 512):
        raise ValueError("windows must have shape samples x 2 x 512")
    order = np.random.default_rng(seed).permutation(len(windows))
    train_end = len(windows) * 8 // 10
    validation_end = train_end + len(windows) // 10
    shuffled = windows[order]
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )
