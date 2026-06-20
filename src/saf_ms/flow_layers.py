"""Reversible integer flow layers for one-dimensional peak sequences."""

from typing import List

import torch
from torch import nn

from .conditioner import SparseFlowConditioner


def straight_through_round(inputs: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use the identity gradient."""
    return inputs + (torch.round(inputs) - inputs).detach()


class InterleavedSqueeze1d(nn.Module):
    """Move neighboring sequence values into interleaved channels."""

    def __init__(self, factor: int = 4) -> None:
        super().__init__()
        if factor <= 1:
            raise ValueError("factor must be greater than one")
        self.factor = factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, channels, sequence)")
        batch, channels, length = inputs.shape
        if length % self.factor:
            raise ValueError("sequence length must be divisible by squeeze factor")
        reduced_length = length // self.factor
        return inputs.reshape(
            batch,
            channels,
            reduced_length,
            self.factor,
        ).permute(0, 3, 1, 2).reshape(
            batch,
            channels * self.factor,
            reduced_length,
        )

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, channels, sequence)")
        batch, channels, length = inputs.shape
        if channels % self.factor:
            raise ValueError("channel count must be divisible by squeeze factor")
        original_channels = channels // self.factor
        return inputs.reshape(
            batch,
            self.factor,
            original_channels,
            length,
        ).permute(0, 2, 3, 1).reshape(
            batch,
            original_channels,
            length * self.factor,
        )


class PairSwap(nn.Module):
    """Swap channel-pair halves while preserving mass/intensity ordering."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, length = inputs.shape
        if channels % 4:
            raise ValueError("channel count must be divisible by four")
        pairs = inputs.reshape(batch, channels // 2, 2, length)
        pairs = torch.roll(pairs, shifts=channels // 4, dims=1)
        return pairs.reshape(batch, channels, length)

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)


class IntegerAdditiveCoupling(nn.Module):
    """Exactly invertible additive coupling with integer-valued shifts."""

    def __init__(
        self,
        channels: int,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        transformer_layers: int,
        window_size: int,
        key_peak_ratio: float,
        fusion_alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("channels must be divisible by four")
        self.split_channels = channels // 2
        self.conditioner = SparseFlowConditioner(
            input_channels=self.split_channels,
            output_channels=channels - self.split_channels,
            model_dim=model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=transformer_layers,
            window_size=window_size,
            key_peak_ratio=key_peak_ratio,
            fusion_alpha=fusion_alpha,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fixed, transformed = torch.split(
            inputs,
            [self.split_channels, inputs.shape[1] - self.split_channels],
            dim=1,
        )
        shift = straight_through_round(self.conditioner(fixed))
        return torch.cat([fixed, transformed + shift], dim=1)

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        fixed, transformed = torch.split(
            inputs,
            [self.split_channels, inputs.shape[1] - self.split_channels],
            dim=1,
        )
        shift = torch.round(self.conditioner(fixed))
        return torch.cat([fixed, transformed - shift], dim=1)


class IntegerFlowBlock(nn.Module):
    """A sequence of integer couplings and pair-preserving permutations."""

    def __init__(
        self,
        channels: int,
        couplings: int,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        transformer_layers: int,
        window_size: int,
        key_peak_ratio: float,
        fusion_alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.couplings = nn.ModuleList([
            IntegerAdditiveCoupling(
                channels,
                model_dim,
                num_heads,
                feedforward_dim,
                transformer_layers,
                window_size,
                key_peak_ratio,
                fusion_alpha,
                dropout,
            )
            for _ in range(couplings)
        ])
        self.permutation = PairSwap()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for coupling in self.couplings:
            hidden = coupling(hidden)
            hidden = self.permutation(hidden)
        return hidden

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for coupling in reversed(self.couplings):
            hidden = self.permutation.inverse(hidden)
            hidden = coupling.inverse(hidden)
        return hidden


class MultiScaleIntegerFlow(nn.Module):
    """Three-level flow with factor-out latents between levels."""

    def __init__(self, config: object) -> None:
        super().__init__()
        self.input_channels = config.input_channels
        self.sequence_length = config.sequence_length
        self.squeeze = InterleavedSqueeze1d(config.squeeze_factor)
        self.levels = nn.ModuleList()

        channels = config.input_channels * config.squeeze_factor
        for _ in range(config.levels):
            self.levels.append(IntegerFlowBlock(
                channels=channels,
                couplings=config.couplings_per_level,
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                feedforward_dim=config.feedforward_dim,
                transformer_layers=config.transformer_layers,
                window_size=config.window_size,
                key_peak_ratio=config.key_peak_ratio,
                fusion_alpha=config.fusion_alpha,
                dropout=config.dropout,
            ))
            channels = channels // 2 * config.squeeze_factor

    def encode(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.squeeze(inputs)
        latents: List[torch.Tensor] = []
        for index, level in enumerate(self.levels):
            hidden = level(hidden)
            if index < len(self.levels) - 1:
                hidden, factor = hidden.chunk(2, dim=1)
                latents.append(factor)
                hidden = self.squeeze(hidden)
        latents.append(hidden)
        return latents

    def decode(self, latents: List[torch.Tensor]) -> torch.Tensor:
        if len(latents) != len(self.levels):
            raise ValueError("latent count must match the number of flow levels")
        hidden = latents[-1]
        for index in range(len(self.levels) - 1, -1, -1):
            hidden = self.levels[index].inverse(hidden)
            if index > 0:
                hidden = self.squeeze.inverse(hidden)
                hidden = torch.cat([hidden, latents[index - 1]], dim=1)
        return self.squeeze.inverse(hidden)
