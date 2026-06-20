import math

import torch

from saf_ms.attention import SparsePeakAttention


def test_selects_high_intensity_key_peaks():
    layer = SparsePeakAttention(16, 4, 4, 0.25, 0.7)
    scores = torch.tensor([[0.1, 3.0, 0.2, 7.0, 1.0, 0.3, 5.0, 0.4]])

    indices = layer.select_key_indices(scores)

    assert indices.shape == (1, math.ceil(8 * 0.25))
    assert set(indices[0].tolist()) == {3, 6}


def test_local_window_tracks_boundaries():
    layer = SparsePeakAttention(16, 4, 4, 0.25, 0.7)

    indices, valid = layer.build_local_index(6, torch.device("cpu"))

    assert indices[0].tolist() == [0, 0, 0, 1]
    assert valid[0].tolist() == [False, False, True, True]
    assert indices[3].tolist() == [1, 2, 3, 4]
    assert valid[3].all().item()


def test_attention_preserves_sequence_shape_and_gradients():
    layer = SparsePeakAttention(32, 4, 8, 0.25, 0.7)
    inputs = torch.randn(2, 32, 32, requires_grad=True)
    scores = torch.randn(2, 32)

    outputs = layer(inputs, scores)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
