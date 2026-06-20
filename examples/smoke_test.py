"""Run a synthetic forward/backward pass and exact flow round trip."""

import torch

from saf_ms.model import SAFMSConfig, SAFMSModel
from saf_ms.workflows import synthetic_inputs


def main() -> None:
    config = SAFMSConfig(
        sequence_length=64,
        levels=2,
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    model = SAFMSModel(config)
    inputs = synthetic_inputs(config, batch_size=1)
    bits = model.bits_per_value(inputs)
    bits.backward()
    reconstructed = model.decode(model.encode(inputs))
    print(f"bits_per_value={float(bits.detach()):.6f}")
    print(f"exact_round_trip={torch.equal(reconstructed, inputs)}")


if __name__ == "__main__":
    main()
