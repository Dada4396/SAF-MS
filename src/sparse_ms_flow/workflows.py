"""Shared training, evaluation, checkpoint, and coding workflows."""

from dataclasses import asdict
from os import PathLike
from typing import Iterable, List, Mapping, Optional, Union

import numpy as np
import torch

from .codec import RansCodec
from .config import TrainingSettings
from .model import SparseMSFlowConfig, SparseMSFlowModel
from .rans import RansPayload


Pathish = Union[str, PathLike[str]]


def synthetic_inputs(
    config: SparseMSFlowConfig,
    batch_size: int = 2,
    seed: int = 17,
) -> torch.Tensor:
    """Create a deterministic integer-valued batch without external data."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        4096,
        (batch_size, config.input_channels, config.sequence_length),
        generator=generator,
    ).float()


def train_model(
    model: SparseMSFlowModel,
    settings: TrainingSettings,
    windows: Optional[np.ndarray] = None,
    max_steps: Optional[int] = None,
) -> List[float]:
    """Run a bounded optimization loop and return step losses."""
    steps = settings.max_steps if max_steps is None else max_steps
    if steps < 1:
        raise ValueError("max_steps must be positive")
    device = torch.device(settings.device)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = np.random.default_rng(settings.seed)
    losses = []

    for _ in range(steps):
        if windows is None:
            batch = synthetic_inputs(
                model.config, settings.batch_size, settings.seed + len(losses)
            )
        else:
            indices = generator.integers(0, len(windows), size=settings.batch_size)
            batch = torch.from_numpy(windows[indices]).float()
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.bits_per_value(batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.no_grad()
def evaluate_model(model: SparseMSFlowModel, inputs: torch.Tensor) -> Mapping[str, object]:
    """Report coding cost and exact integer-flow reconstruction status."""
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    latents = model.encode(inputs)
    reconstructed = model.decode(latents)
    return {
        "bits_per_value": float(model.bits_per_value(inputs).cpu()),
        "exact_round_trip": bool(torch.equal(reconstructed, inputs)),
    }


def save_checkpoint(model: SparseMSFlowModel, path: Pathish) -> None:
    """Save architecture metadata and a model state dictionary."""
    torch.save(
        {"config": asdict(model.config), "model_state": model.state_dict()},
        path,
    )


def load_checkpoint(path: Pathish, device: str = "cpu") -> SparseMSFlowModel:
    """Load a SparseMSFlow checkpoint onto an explicit device."""
    document = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(document, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    try:
        config = SparseMSFlowConfig(**document["config"])
        state = document["model_state"]
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint is missing model config or state") from error
    model = SparseMSFlowModel(config).to(device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def encode_array(
    model: SparseMSFlowModel,
    windows: np.ndarray,
    precision: int = 16,
) -> List[RansPayload]:
    """Transform an array into flow latents and rANS-code each latent."""
    device = next(model.parameters()).device
    inputs = torch.from_numpy(np.asarray(windows)).float().to(device)
    codec = RansCodec(precision)
    return [
        codec.encode(latent.round().to(torch.int64).cpu().numpy())
        for latent in model.encode(inputs)
    ]


@torch.no_grad()
def decode_array(
    model: SparseMSFlowModel,
    payloads: Iterable[RansPayload],
    precision: int = 16,
) -> np.ndarray:
    """Decode rANS latents and invert the integer flow."""
    payload_list = list(payloads)
    expected_shapes = _latent_shapes(model.config, payload_list)
    actual_shapes = [payload.shape for payload in payload_list]
    if actual_shapes != expected_shapes:
        raise ValueError(f"latent shapes must match model architecture: {expected_shapes}")
    device = next(model.parameters()).device
    codec = RansCodec(precision)
    latents = [
        torch.from_numpy(codec.decode(payload)).float().to(device)
        for payload in payload_list
    ]
    return model.decode(latents).round().to(torch.int64).cpu().numpy()


def save_container(
    path: Pathish,
    payloads: Iterable[RansPayload],
    precision: int = 16,
) -> None:
    """Save rANS payloads in a pickle-free NumPy container."""
    payload_list = list(payloads)
    fields = {
        "precision": np.array(precision, dtype=np.int64),
        "latent_count": np.array(len(payload_list), dtype=np.int64),
    }
    for index, payload in enumerate(payload_list):
        prefix = f"latent_{index}_"
        fields[prefix + "shape"] = np.asarray(payload.shape, dtype=np.int64)
        fields[prefix + "minimum"] = np.array(payload.minimum, dtype=np.int64)
        fields[prefix + "cdf"] = np.asarray(payload.cdf, dtype=np.int64)
        fields[prefix + "state"] = np.array(payload.state, dtype=np.uint64)
        fields[prefix + "stream"] = np.frombuffer(payload.stream, dtype=np.uint8)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **fields)


def load_container(path: Pathish) -> tuple[int, List[RansPayload]]:
    """Load rANS payloads from a pickle-free NumPy container."""
    with np.load(path, allow_pickle=False) as fields:
        try:
            precision = int(fields["precision"])
            latent_count = int(fields["latent_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("container is missing scalar precision metadata") from error
        RansCodec(precision)
        if not 1 <= latent_count <= 64:
            raise ValueError("container latent_count must be between 1 and 64")
        payloads = []
        for index in range(latent_count):
            prefix = f"latent_{index}_"
            try:
                shape_array = fields[prefix + "shape"]
                cdf = fields[prefix + "cdf"]
                stream = fields[prefix + "stream"]
                if shape_array.ndim != 1 or not 1 <= shape_array.size <= 8:
                    raise ValueError("shape metadata must have between 1 and 8 axes")
                if stream.ndim != 1 or stream.dtype != np.uint8:
                    raise ValueError("stream metadata must be a byte vector")
                payloads.append(RansPayload(
                    shape=tuple(int(value) for value in shape_array),
                    minimum=int(fields[prefix + "minimum"]),
                    cdf=cdf.copy(),
                    state=int(fields[prefix + "state"]),
                    stream=stream.tobytes(),
                ))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"container latent {index} metadata is invalid") from error
    return precision, payloads


def _latent_shapes(
    config: SparseMSFlowConfig,
    payloads: List[RansPayload],
) -> List[tuple[int, int, int]]:
    if len(payloads) != config.levels:
        raise ValueError("latent count must match the model flow levels")
    first_shape = payloads[0].shape
    if len(first_shape) != 3 or first_shape[0] < 1:
        raise ValueError("latent tensors must have batch, channel, and sequence axes")
    batch_size = first_shape[0]
    channels = config.input_channels * config.squeeze_factor
    sequence_length = config.sequence_length // config.squeeze_factor
    expected = []
    for level in range(config.levels):
        if level < config.levels - 1:
            channels //= 2
            expected.append((batch_size, channels, sequence_length))
            channels *= config.squeeze_factor
            sequence_length //= config.squeeze_factor
        else:
            expected.append((batch_size, channels, sequence_length))
    return expected
