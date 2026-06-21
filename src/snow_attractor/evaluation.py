"""Evaluation metrics for snow-map forecasts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)


@torch.no_grad()
def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    window = _gaussian_window(
        window_size,
        sigma,
        prediction.device,
        prediction.dtype,
    )
    padding = window_size // 2
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    valid_mask = valid_mask.to(dtype=prediction.dtype)
    support = F.conv2d(valid_mask, window, padding=padding)
    normalization = support.clamp_min(1e-6)
    mean_prediction = (
        F.conv2d(prediction * valid_mask, window, padding=padding)
        / normalization
    )
    mean_target = (
        F.conv2d(target * valid_mask, window, padding=padding)
        / normalization
    )
    prediction_variance = (
        F.conv2d(prediction.square() * valid_mask, window, padding=padding)
        / normalization
        - mean_prediction.square()
    ).clamp_min(0.0)
    target_variance = (
        F.conv2d(target.square() * valid_mask, window, padding=padding)
        / normalization
        - mean_target.square()
    ).clamp_min(0.0)
    covariance = (
        F.conv2d(
            prediction * target * valid_mask,
            window,
            padding=padding,
        )
        / normalization
        - mean_prediction * mean_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_prediction * mean_target + c1)
        * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        ).clamp_min(1e-8)
    )
    usable = support >= 0.9
    if not usable.any():
        return float("nan")
    return float(score[usable].mean())


@torch.no_grad()
def forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    snow_threshold: float = 0.10,
) -> dict[str, float]:
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    valid_mask = valid_mask.to(dtype=prediction.dtype)
    valid_count = valid_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    pixel_error = prediction - target
    predicted_coverage = (
        prediction * valid_mask
    ).flatten(1).sum(dim=1) / valid_count
    target_coverage = (
        target * valid_mask
    ).flatten(1).sum(dim=1) / valid_count
    coverage_error = predicted_coverage - target_coverage
    denominator = ((target_coverage - target_coverage.mean()) ** 2).sum()
    centered_prediction = predicted_coverage - predicted_coverage.mean()
    centered_target = target_coverage - target_coverage.mean()
    correlation_denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    )
    r2 = (
        float("nan")
        if denominator <= 1e-8
        else float(1.0 - coverage_error.square().sum() / denominator)
    )
    pearson = (
        float("nan")
        if correlation_denominator <= 1e-8
        else float((centered_prediction * centered_target).sum() / correlation_denominator)
    )
    absolute = (pixel_error.abs() * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    squared = (pixel_error.square() * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    predicted_snow = prediction >= snow_threshold
    target_snow = target >= snow_threshold
    valid_boolean = valid_mask.bool()
    true_positive = (predicted_snow & target_snow & valid_boolean).sum().float()
    false_positive = (predicted_snow & ~target_snow & valid_boolean).sum().float()
    false_negative = (~predicted_snow & target_snow & valid_boolean).sum().float()
    iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    f1 = 2.0 * true_positive / (
        2.0 * true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    return {
        "pixel_mae": float(absolute),
        "pixel_rmse": float(squared.sqrt()),
        "pixel_ssim": structural_similarity(prediction, target, valid_mask),
        "snow_iou": float(iou),
        "snow_f1": float(f1),
        "coverage_mae": float(coverage_error.abs().mean()),
        "coverage_rmse": float(coverage_error.square().mean().sqrt()),
        "coverage_r2": r2,
        "coverage_pearson": pearson,
    }


@torch.no_grad()
def baseline_predictions(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    last_day = inputs[:, -1, 0:1]
    recent_window = min(7, inputs.size(1))
    recent_mean = inputs[:, -recent_window:, 0:1].mean(dim=1)
    return {
        "last_day": last_day,
        "recent_7day_mean": recent_mean,
    }
