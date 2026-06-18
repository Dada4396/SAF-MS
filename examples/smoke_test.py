"""Run a synthetic forward/backward pass and exact flow round trip."""

import torch

from sparse_ms_flow.model import SparseMSFlowConfig, SparseMSFlowModel
from sparse_ms_flow.workflows import synthetic_inputs


def main() -> None:
    config = SparseMSFlowConfig(
        sequence_length=64,
        levels=2,
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    model = SparseMSFlowModel(config)
    inputs = synthetic_inputs(config, batch_size=1)
    bits = model.bits_per_value(inputs)
    bits.backward()
    reconstructed = model.decode(model.encode(inputs))
    print(f"bits_per_value={float(bits.detach()):.6f}")
    print(f"exact_round_trip={torch.equal(reconstructed, inputs)}")


if __name__ == "__main__":
    main()
