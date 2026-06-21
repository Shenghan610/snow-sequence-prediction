import pytest
import torch

from snow_attractor.benchmark_models import (
    MODEL_REGISTRY,
    ConvLSTM,
    ConvLSTMUNet,
    build_benchmark_model,
)


@pytest.mark.parametrize("name", tuple(MODEL_REGISTRY))
def test_benchmark_forward_contract(name):
    model = build_benchmark_model(
        name,
        input_steps=4,
        context_dim=12,
        spatial_context_channels=5,
        internal_size=32,
        dropout=0.0,
    )
    inputs = torch.rand(1, 4, 1, 32, 32)
    prior = torch.rand(1, 2, 32, 32)
    context = torch.rand(1, 12)
    spatial_context = torch.rand(1, 5, 32, 32)
    outputs = model(inputs, prior, context, spatial_context)
    assert outputs["prediction"].shape == (1, 1, 32, 32)
    assert outputs["coverage_prediction"].shape == (1,)
    assert torch.allclose(
        outputs["prediction"].mean((1, 2, 3)),
        outputs["coverage_prediction"],
        atol=1e-5,
    )


def test_convlstm_unet_is_distinct_from_convlstm():
    convlstm = build_benchmark_model(
        "convlstm",
        input_steps=4,
        context_dim=12,
        spatial_context_channels=5,
        internal_size=32,
        dropout=0.0,
    )
    convlstm_unet = build_benchmark_model(
        "convlstm_unet",
        input_steps=4,
        context_dim=12,
        spatial_context_channels=5,
        internal_size=32,
        dropout=0.0,
    )

    assert isinstance(convlstm, ConvLSTM)
    assert isinstance(convlstm_unet, ConvLSTMUNet)
    assert type(convlstm_unet) is not type(convlstm)
