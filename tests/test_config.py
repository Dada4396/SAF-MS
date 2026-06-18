from pathlib import Path

import pytest

from sparse_ms_flow.config import TrainingSettings, load_config
from sparse_ms_flow.model import SparseMSFlowConfig


def test_default_yaml_builds_model_and_training_settings():
    model, training = load_config(Path("configs/default.yaml"))

    assert isinstance(model, SparseMSFlowConfig)
    assert isinstance(training, TrainingSettings)
    assert model.sequence_length == 512
    assert model.levels == 3
    assert training.batch_size > 0
    assert training.learning_rate > 0


def test_config_rejects_unknown_sections(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("model: {}\ntraining: {}\nunexpected: {}\n", encoding="utf-8")

    try:
        load_config(path)
    except ValueError as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unknown configuration sections must be rejected")


@pytest.mark.parametrize(
    "values",
    [
        {"squeeze_factor": 0},
        {"model_dim": 15, "num_heads": 4},
        {"key_peak_ratio": 0.0},
        {"dropout": 1.0},
    ],
)
def test_model_config_rejects_invalid_architecture(values):
    with pytest.raises(ValueError):
        SparseMSFlowConfig(**values)
