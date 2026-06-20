import pytest
import torch

from saf_ms.flow_layers import InterleavedSqueeze1d
from saf_ms.model import SAFMSConfig, SAFMSModel


def test_interleaved_squeeze_round_trip():
    layer = InterleavedSqueeze1d(factor=4)
    inputs = torch.arange(2 * 2 * 16).reshape(2, 2, 16).float()

    reconstructed = layer.inverse(layer(inputs))

    assert torch.equal(reconstructed, inputs)


def test_three_level_flow_is_exactly_invertible():
    config = SAFMSConfig(
        sequence_length=512,
        levels=3,
        couplings_per_level=1,
        model_dim=32,
        num_heads=4,
        feedforward_dim=64,
        transformer_layers=1,
    )
    model = SAFMSModel(config)
    inputs = torch.randint(0, 4096, (2, 2, 512)).float()

    latents = model.encode(inputs)
    reconstructed = model.decode(latents)

    assert [tuple(latent.shape) for latent in latents] == [
        (2, 4, 128),
        (2, 8, 32),
        (2, 32, 8),
    ]
    assert torch.equal(reconstructed, inputs)


def test_configuration_rejects_incompatible_sequence_length():
    with pytest.raises(ValueError, match="divisible"):
        SAFMSConfig(sequence_length=500, levels=3)
