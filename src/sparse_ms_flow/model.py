"""Top-level SparseMSFlow model definitions."""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from .distributions import DiscretizedLogisticEntropyModel
from .flow_layers import MultiScaleIntegerFlow


@dataclass(frozen=True)
class SparseMSFlowConfig:
    """Architecture configuration for SparseMSFlow."""

    input_channels: int = 2
    sequence_length: int = 512
    levels: int = 3
    couplings_per_level: int = 1
    squeeze_factor: int = 4
    model_dim: int = 128
    num_heads: int = 4
    feedforward_dim: int = 256
    transformer_layers: int = 1
    window_size: int = 32
    key_peak_ratio: float = 0.16
    fusion_alpha: float = 0.7
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_channels != 2:
            raise ValueError("SparseMSFlow expects two input channels")
        integer_fields = {
            "sequence_length": self.sequence_length,
            "levels": self.levels,
            "couplings_per_level": self.couplings_per_level,
            "squeeze_factor": self.squeeze_factor,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "transformer_layers": self.transformer_layers,
            "window_size": self.window_size,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields.values()
        ):
            raise ValueError("architecture dimensions must be integers")
        if self.levels < 1 or self.couplings_per_level < 1:
            raise ValueError("levels and couplings_per_level must be positive")
        if self.squeeze_factor < 2 or self.squeeze_factor % 2:
            raise ValueError("squeeze_factor must be a positive even integer")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        divisor = self.squeeze_factor ** self.levels
        if self.sequence_length % divisor:
            raise ValueError(
                "sequence_length must be divisible by squeeze_factor ** levels"
            )
        if self.model_dim < 1 or self.num_heads < 1:
            raise ValueError("model_dim and num_heads must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.feedforward_dim < 1 or self.transformer_layers < 1:
            raise ValueError("feedforward_dim and transformer_layers must be positive")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < self.key_peak_ratio <= 1.0:
            raise ValueError("key_peak_ratio must be in (0, 1]")
        if not 0.0 <= self.fusion_alpha <= 1.0:
            raise ValueError("fusion_alpha must be in [0, 1]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SparseMSFlowModel(nn.Module):
    """Multi-scale integer flow conditioned by sparse peak attention."""

    def __init__(self, config: Optional[SparseMSFlowConfig] = None):
        super().__init__()
        self.config = config or SparseMSFlowConfig()
        self.flow = MultiScaleIntegerFlow(self.config)
        self.entropy_models = nn.ModuleList(
            DiscretizedLogisticEntropyModel(channels)
            for channels in self._latent_channels()
        )

    def encode(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        self._validate_inputs(inputs)
        return self.flow.encode(inputs.float())

    def decode(self, latents: List[torch.Tensor]) -> torch.Tensor:
        return self.flow.decode(latents)

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return self.encode(inputs)

    def bits_per_value(self, inputs: torch.Tensor) -> torch.Tensor:
        """Estimate mean coding cost under the learned latent distributions."""
        latents = self.encode(inputs)
        total_bits = sum(
            entropy_model.bits(latent)
            for entropy_model, latent in zip(self.entropy_models, latents)
        )
        return total_bits / inputs.numel()

    def _latent_channels(self) -> List[int]:
        channels = self.config.input_channels * self.config.squeeze_factor
        latent_channels = []
        for level in range(self.config.levels):
            if level == self.config.levels - 1:
                latent_channels.append(channels)
            else:
                channels //= 2
                latent_channels.append(channels)
                channels *= self.config.squeeze_factor
        return latent_channels

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        expected = (self.config.input_channels, self.config.sequence_length)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                "inputs must have shape (batch, {}, {})".format(*expected)
            )
