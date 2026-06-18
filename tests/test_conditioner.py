import torch

from sparse_ms_flow.conditioner import SparseFlowConditioner


def test_conditioner_preserves_length_and_gradients():
    model = SparseFlowConditioner(
        input_channels=4,
        output_channels=4,
        model_dim=32,
        num_heads=4,
        feedforward_dim=64,
        num_layers=2,
        window_size=8,
        key_peak_ratio=0.25,
        fusion_alpha=0.7,
    )
    inputs = torch.randn(2, 4, 32, requires_grad=True)

    outputs = model(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == (2, 4, 32)
    assert torch.isfinite(outputs).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
