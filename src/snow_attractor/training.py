"""Training helpers shared by the command line entry point and tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml

from .losses import AttractorEnergyLoss
from .model import AttractorEnergyUNet


def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict, device: torch.device) -> AttractorEnergyUNet:
    return AttractorEnergyUNet(**config["model"]).to(device)


def build_loss(config: dict) -> AttractorEnergyLoss:
    return AttractorEnergyLoss(**config["loss"])


def train_step(
    model: AttractorEnergyUNet,
    criterion: AttractorEnergyLoss,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    amp_enabled: bool = False,
) -> dict[str, float]:
    device = next(model.parameters()).device
    inputs = batch["inputs"].to(device)
    target = batch["target"].to(device)
    prior = batch["spatial_prior"].to(device)
    context = batch["context"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled and device.type == "cuda",
    ):
        spatial_context = batch.get("spatial_context")
        if spatial_context is not None:
            spatial_context = spatial_context.to(device)
        outputs = model(
            inputs,
            prior,
            context,
            spatial_context,
            land_mask=batch.get("land_mask").to(device)
            if batch.get("land_mask") is not None
            else None,
        )
        losses = criterion(
            outputs,
            target,
            valid_mask=batch.get("target_valid_mask").to(device)
            if batch.get("target_valid_mask") is not None
            else None,
            land_mask=batch.get("land_mask").to(device)
            if batch.get("land_mask") is not None
            else None,
        )
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {name: float(value.detach()) for name, value in losses.items()}
