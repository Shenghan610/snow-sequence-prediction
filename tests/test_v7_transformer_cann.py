import json
import subprocess
import sys

import torch

from run_qa_experiment import build_model
from snow_attractor.benchmark_models import ConvLSTMUNet
from snow_attractor.losses import AttractorEnergyLoss
from snow_attractor.training import load_config
from snow_attractor.transformer_cann import TransformerCANNLyapunovNet


def _context(batch: int, steps: int, context_dim: int) -> torch.Tensor:
    context = torch.zeros(batch, context_dim)
    context[:, :steps] = torch.linspace(0.2, 0.35, steps)
    context[:, steps : steps + steps - 1] = 0.01
    calendar_start = steps + (steps - 1) + 37
    context[:, calendar_start] = 0.0
    context[:, calendar_start + 1] = 1.0
    return context


def test_transformer_cann_lyapunov_forward_backward_contract():
    model = TransformerCANNLyapunovNet(
        input_steps=4,
        input_channels=3,
        context_dim=48,
        spatial_context_channels=7,
        internal_size=64,
        patch_size=8,
        embed_dim=32,
        attention_heads=4,
        temporal_layers=1,
        spatial_layers=1,
        window_size=2,
        attractor_iterations=2,
        coverage_bins=3,
        change_bins=3,
        season_bins=4,
        memory_tokens=3,
        dropout=0.0,
    )
    inputs = torch.rand(1, 4, 3, 64, 64)
    spatial_prior = torch.rand(1, 2, 64, 64)
    spatial_context = torch.rand(1, 7, 64, 64)
    target = torch.rand(1, 1, 64, 64)
    mask = torch.ones_like(target)

    outputs = model(
        inputs,
        spatial_prior,
        _context(1, 4, 48),
        spatial_context,
        coverage_mask=mask,
    )
    criterion = AttractorEnergyLoss(
        lyapunov_weight=0.02,
        energy_monotonicity_weight=0.02,
        trajectory_smoothness_weight=0.001,
    )
    losses = criterion(outputs, target, valid_mask=mask)
    losses["total"].backward()

    assert outputs["prediction"].shape == (1, 1, 64, 64)
    assert outputs["energy"].shape == (1, 3)
    assert outputs["lyapunov_delta"].shape == (1, 2)
    assert outputs["coordinate_trajectory"].shape == (1, 3, 4)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["lyapunov"])
    assert model.attractor_memory.anchors.grad is not None


def test_v7_config_is_256_and_uses_isolated_paths():
    config = load_config("configs/qa_experiment_v7.yaml")

    assert config["data"]["image_size"] == 256
    assert config["data"]["root"].endswith("data_qa_v7_256")
    assert config["project"]["artifacts_dir"].endswith("qa_experiment_v7")
    assert config["model"]["internal_size"] == 256
    assert config["model"]["patch_size"] == 8
    assert config["training"]["physical_batch_size"]["transformer_cann_lyapunov"] == 1
    assert config["training"]["effective_batch_size"] == 8
    assert config["splits"]["frozen_test"] == {
        "start": "2024-01-01",
        "end": "2024-12-31",
    }


def test_v7_convlstm_unet_builds_real_unet_model():
    config = load_config("configs/qa_experiment_v7.yaml")
    config["model"]["internal_size"] = 32

    class DatasetStub:
        sequence_length = 4
        input_channels = 3
        context_dim = 48
        spatial_context_channels = 7

    model = build_model("convlstm_unet", config, DatasetStub(), "dominance")

    assert isinstance(model, ConvLSTMUNet)


def test_v7_download_dry_run_only_plans_ali_2015_2024():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_qa_data.py",
            "submit",
            "--regions",
            "ali",
            "--start-year",
            "2015",
            "--end-year",
            "2024",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tasks = json.loads(result.stdout)
    names = [task["task_name"] for task in tasks]

    assert len(tasks) == 11
    assert names[0] == "snow-qa-ali-2015"
    assert names[-2] == "snow-qa-ali-2024"
    assert names[-1] == "terrain-srtm-ali"
    assert not any("tianshan" in name for name in names)
