"""Sparse Transformer conditioner used by integer coupling layers."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import SparsePeakAttention


class MassEncoder(nn.Module):
    """Encode one or more mass-like channels with sinusoidal wavelengths."""

    def __init__(
        self,
        dim_model: int,
        min_wavelength: float = 1e-3,
        max_wavelength: float = 1e4,
    ) -> None:
        super().__init__()
        sin_dim = dim_model // 2
        cos_dim = dim_model - sin_dim
        sin_scale = torch.logspace(
            math.log10(min_wavelength),
            math.log10(max_wavelength),
            sin_dim,
        )
        cos_scale = torch.logspace(
            math.log10(min_wavelength),
            math.log10(max_wavelength),
            cos_dim,
        )
        self.register_buffer("sin_scale", sin_scale)
        self.register_buffer("cos_scale", cos_scale)

    def forward(self, masses: torch.Tensor) -> torch.Tensor:
        angles_sin = masses.unsqueeze(-1) / self.sin_scale
        angles_cos = masses.unsqueeze(-1) / self.cos_scale
        encoded = torch.cat([torch.sin(angles_sin), torch.cos(angles_cos)], dim=-1)
        return encoded.mean(dim=2)


class SparseTransformerLayer(nn.Module):
    """Transformer encoder block backed by sparse peak attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        window_size: int,
        key_peak_ratio: float,
        fusion_alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = SparsePeakAttention(
            model_dim,
            num_heads,
            window_size,
            key_peak_ratio,
            fusion_alpha,
            dropout,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs + self.dropout(self.attention(inputs, scores)))
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        return self.norm2(hidden + self.dropout(feedforward))


class SparseFlowConditioner(nn.Module):
    """Predict additive coupling shifts from interleaved peak features."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        model_dim: int = 128,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        num_layers: int = 1,
        window_size: int = 32,
        key_peak_ratio: float = 0.16,
        fusion_alpha: float = 0.7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels < 2 or input_channels % 2:
            raise ValueError("input_channels must be an even number of at least two")
        self.mass_encoder = MassEncoder(model_dim)
        self.intensity_projection = nn.Linear(input_channels // 2, model_dim)
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                model_dim,
                num_heads,
                feedforward_dim,
                window_size,
                key_peak_ratio,
                fusion_alpha,
                dropout,
            )
            for _ in range(num_layers)
        ])
        hidden_channels = max(64, model_dim)
        self.output_network = nn.Sequential(
            nn.Conv1d(model_dim + input_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1),
        )
        nn.init.zeros_(self.output_network[-1].weight)
        nn.init.zeros_(self.output_network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, channels, sequence)")
        mass_features = inputs[:, 0::2, :].transpose(1, 2)
        intensity_features = inputs[:, 1::2, :].transpose(1, 2)
        hidden = self.mass_encoder(mass_features)
        hidden = hidden + self.intensity_projection(intensity_features)
        scores = intensity_features.abs().mean(dim=-1)
        for layer in self.layers:
            hidden = layer(hidden, scores)

        pooled = hidden.amax(dim=1).unsqueeze(-1).expand(-1, -1, inputs.shape[-1])
        return self.output_network(torch.cat([pooled, inputs], dim=1))
