"""YAML-backed model and training configuration."""

from dataclasses import dataclass
from os import PathLike
from typing import Mapping, Tuple, Union

import yaml

from .model import SparseMSFlowConfig


Pathish = Union[str, PathLike[str]]


@dataclass(frozen=True)
class TrainingSettings:
    """Small, environment-neutral training defaults."""

    batch_size: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 100
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")


def load_config(path: Pathish) -> Tuple[SparseMSFlowConfig, TrainingSettings]:
    """Load strict model and training dataclasses from a YAML mapping."""
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, Mapping):
        raise ValueError("configuration root must be a mapping")
    unexpected = set(document) - {"model", "training"}
    if unexpected:
        names = ", ".join(sorted(str(name) for name in unexpected))
        raise ValueError(f"unexpected configuration sections: {names}")

    model_values = _mapping_section(document, "model")
    training_values = _mapping_section(document, "training")
    try:
        return SparseMSFlowConfig(**model_values), TrainingSettings(**training_values)
    except TypeError as error:
        raise ValueError(f"invalid configuration field: {error}") from error


def _mapping_section(document: Mapping[object, object], name: str) -> Mapping[str, object]:
    section = document.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return dict(section)
