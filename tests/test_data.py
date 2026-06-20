import numpy as np
import pytest

from saf_ms.data import load_spectrum_windows, split_windows


def test_loader_accepts_both_supported_layouts(tmp_path):
    windows = np.arange(3 * 2 * 512).reshape(3, 2, 512)
    channel_first = tmp_path / "channel_first.npy"
    channel_last = tmp_path / "channel_last.npy"
    np.save(channel_first, windows)
    np.save(channel_last, windows.transpose(0, 2, 1))

    assert np.array_equal(load_spectrum_windows(channel_first), windows)
    assert np.array_equal(load_spectrum_windows(channel_last), windows)


def test_loader_rejects_an_unsupported_shape(tmp_path):
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros((4, 512)))

    with pytest.raises(ValueError, match="samples x 2 x 512"):
        load_spectrum_windows(path)


def test_loader_rejects_non_integer_values(tmp_path):
    path = tmp_path / "fractional.npy"
    windows = np.zeros((1, 2, 512), dtype=np.float64)
    windows[0, 0, 0] = 0.5
    np.save(path, windows)

    with pytest.raises(ValueError, match="integer"):
        load_spectrum_windows(path)


def test_loader_rejects_empty_data(tmp_path):
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((0, 2, 512), dtype=np.int64))

    with pytest.raises(ValueError, match="sample"):
        load_spectrum_windows(path)


def test_seeded_split_is_reproducible():
    windows = np.arange(100 * 2 * 512).reshape(100, 2, 512)

    first = split_windows(windows, seed=17)
    second = split_windows(windows, seed=17)

    assert [len(partition) for partition in first] == [80, 10, 10]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
