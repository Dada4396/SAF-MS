"""Probability utilities used by SparseMSFlow entropy models."""

import math
from typing import Union

import torch
from torch import nn


TensorLike = Union[float, torch.Tensor]


def discretized_logistic_log_prob(
    values: torch.Tensor,
    mean: TensorLike,
    log_scale: TensorLike,
) -> torch.Tensor:
    """Return stable log masses for unit-width discretized logistic bins."""
    if not torch.is_floating_point(values):
        values = values.float()
    mean_tensor = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
    log_scale_tensor = torch.as_tensor(
        log_scale, dtype=values.dtype, device=values.device
    )
    inverse_scale = torch.exp(-log_scale_tensor).clamp(max=1e6)
    upper = (values - mean_tensor + 0.5) * inverse_scale
    lower = (values - mean_tensor - 0.5) * inverse_scale
    probability = torch.sigmoid(upper) - torch.sigmoid(lower)
    epsilon = torch.finfo(values.dtype).tiny
    return torch.log(probability.clamp_min(epsilon))


class DiscretizedLogisticEntropyModel(nn.Module):
    """Independent learned logistic parameters for one latent tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return discretized_logistic_log_prob(values, self.mean, self.log_scale)

    def bits(self, values: torch.Tensor) -> torch.Tensor:
        return -self(values).sum() / math.log(2.0)
