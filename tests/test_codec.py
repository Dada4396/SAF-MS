import numpy as np
import torch

from sparse_ms_flow.codec import RansCodec
from sparse_ms_flow.distributions import discretized_logistic_log_prob
from sparse_ms_flow.model import SparseMSFlowConfig, SparseMSFlowModel


def test_discretized_logistic_is_finite():
    values = torch.arange(-5, 6).float()

    log_prob = discretized_logistic_log_prob(values, 0.0, 0.0)

    assert torch.isfinite(log_prob).all()
    assert (log_prob <= 0).all()


def test_rans_round_trip():
    symbols = np.array([4, 2, 4, 1, 0, 4, 3, 2], dtype=np.int64)
    codec = RansCodec(precision=12)

    encoded = codec.encode(symbols)
    decoded = codec.decode(encoded)

    assert np.array_equal(decoded, symbols)


def test_rans_round_trip_preserves_integer_shapes_and_alphabets():
    codec = RansCodec(precision=12)
    arrays = [
        np.array([-9, 100, -9, 4], dtype=np.int64),
        np.arange(24, dtype=np.int64).reshape(2, 3, 4),
        np.full((3, 2), 7, dtype=np.int64),
    ]

    for symbols in arrays:
        assert np.array_equal(codec.decode(codec.encode(symbols)), symbols)


def test_rans_round_trip_supports_full_int64_range():
    symbols = np.array(
        [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max],
        dtype=np.int64,
    )
    codec = RansCodec(precision=12)

    assert np.array_equal(codec.decode(codec.encode(symbols)), symbols)


def test_rans_rejects_invalid_cdf():
    from sparse_ms_flow.rans import RansPayload

    payload = RansPayload(
        shape=(2,),
        minimum=0,
        cdf=np.array([0, 3, 2], dtype=np.int64),
        state=1 << 23,
        stream=b"",
    )

    with np.testing.assert_raises_regex(ValueError, "CDF"):
        RansCodec(precision=12).decode(payload)


def test_model_bits_per_value_is_finite_and_differentiable():
    config = SparseMSFlowConfig(
        sequence_length=64,
        levels=2,
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    model = SparseMSFlowModel(config)
    inputs = torch.randint(0, 256, (2, 2, 64)).float()

    bits = model.bits_per_value(inputs)
    bits.backward()

    assert bits.ndim == 0
    assert torch.isfinite(bits)
    assert any(parameter.grad is not None for parameter in model.entropy_models.parameters())
