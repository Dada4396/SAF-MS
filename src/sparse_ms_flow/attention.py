"""Indexed sparse attention for mass-spectrometry peak sequences."""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SparsePeakAttention(nn.Module):
    """Fuse centered local attention with high-intensity key-peak attention."""

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        window_size: int = 32,
        key_peak_ratio: float = 0.16,
        fusion_alpha: float = 0.7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < key_peak_ratio <= 1.0:
            raise ValueError("key_peak_ratio must be in (0, 1]")
        if not 0.0 <= fusion_alpha <= 1.0:
            raise ValueError("fusion_alpha must be in [0, 1]")

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.window_size = window_size
        self.key_peak_ratio = key_peak_ratio
        self.fusion_alpha = fusion_alpha
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)
        self.output = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def select_key_indices(self, intensity_scores: torch.Tensor) -> torch.Tensor:
        """Select the highest-scoring peak positions for each sample."""
        if intensity_scores.ndim != 2:
            raise ValueError("intensity_scores must have shape (batch, sequence)")
        sequence_length = intensity_scores.shape[1]
        if sequence_length == 0:
            raise ValueError("sequence length must be positive")
        key_count = max(1, math.ceil(sequence_length * self.key_peak_ratio))
        return torch.topk(
            intensity_scores,
            k=key_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

    def build_local_index(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return centered local indices and an out-of-range validity mask."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        positions = torch.arange(sequence_length, device=device).unsqueeze(1)
        offsets = torch.arange(self.window_size, device=device)
        offsets = offsets - self.window_size // 2
        raw_indices = positions + offsets.unsqueeze(0)
        valid = (raw_indices >= 0) & (raw_indices < sequence_length)
        return raw_indices.clamp(0, sequence_length - 1), valid

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        inputs: torch.Tensor,
        intensity_scores: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, sequence, features)")
        if inputs.shape[-1] != self.dim_model:
            raise ValueError("input feature dimension does not match dim_model")
        if intensity_scores.shape != inputs.shape[:2]:
            raise ValueError("intensity_scores must match batch and sequence dimensions")

        batch_size, sequence_length, _ = inputs.shape
        queries = self._split_heads(self.query(inputs))
        keys = self._split_heads(self.key(inputs))
        values = self._split_heads(self.value(inputs))

        local_indices, local_valid = self.build_local_index(
            sequence_length,
            inputs.device,
        )
        local_keys = keys[:, :, local_indices, :]
        local_values = values[:, :, local_indices, :]
        local_logits = torch.einsum(
            "bhnd,bhnkd->bhnk",
            queries,
            local_keys,
        ) * self.scale
        local_logits = local_logits.masked_fill(
            ~local_valid.unsqueeze(0).unsqueeze(0),
            torch.finfo(local_logits.dtype).min,
        )
        local_weights = self.dropout(F.softmax(local_logits, dim=-1))
        local_context = torch.einsum(
            "bhnk,bhnkd->bhnd",
            local_weights,
            local_values,
        )

        key_indices = self.select_key_indices(intensity_scores)
        key_count = key_indices.shape[1]
        gather_indices = key_indices[:, None, :, None].expand(
            batch_size,
            self.num_heads,
            key_count,
            self.head_dim,
        )
        key_peak_keys = torch.gather(keys, dim=2, index=gather_indices)
        key_peak_values = torch.gather(values, dim=2, index=gather_indices)
        key_logits = torch.einsum(
            "bhnd,bhmd->bhnm",
            queries,
            key_peak_keys,
        ) * self.scale
        key_weights = self.dropout(F.softmax(key_logits, dim=-1))
        key_context = torch.einsum(
            "bhnm,bhmd->bhnd",
            key_weights,
            key_peak_values,
        )

        context = (
            self.fusion_alpha * local_context
            + (1.0 - self.fusion_alpha) * key_context
        )
        context = context.permute(0, 2, 1, 3).contiguous().reshape(
            batch_size,
            sequence_length,
            self.dim_model,
        )
        return self.output(context)
