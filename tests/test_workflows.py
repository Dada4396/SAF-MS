import numpy as np
import pytest

from saf_ms.model import SAFMSConfig, SAFMSModel
from saf_ms.workflows import (
    decode_array,
    encode_array,
    load_checkpoint,
    load_container,
    save_checkpoint,
    save_container,
)


def test_checkpoint_and_numeric_container_round_trip(tmp_path):
    config = SAFMSConfig(
        sequence_length=64,
        levels=2,
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    model = SAFMSModel(config)
    windows = np.arange(2 * 64, dtype=np.int64).reshape(1, 2, 64)
    checkpoint = tmp_path / "model.pt"
    container = tmp_path / "windows.safms"

    save_checkpoint(model, checkpoint)
    restored_model = load_checkpoint(checkpoint)
    save_container(container, encode_array(restored_model, windows, 12), 12)
    precision, payloads = load_container(container)
    restored = decode_array(restored_model, payloads, precision)

    assert np.array_equal(restored, windows)


def test_container_rejects_an_invalid_latent_count(tmp_path):
    container = tmp_path / "invalid.safms"
    with open(container, "wb") as handle:
        np.savez(
            handle,
            precision=np.array(12, dtype=np.int64),
            latent_count=np.array(65, dtype=np.int64),
        )

    with pytest.raises(ValueError, match="latent_count"):
        load_container(container)
