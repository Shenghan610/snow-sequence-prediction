"""Prediction and physically parameterized continuous-manifold objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_energy_monotonicity(energy: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if energy.size(1) < 2:
        return energy.new_zeros(())
    relative_change = (energy[:, 1:] - energy[:, :-1]) / (
        energy[:, :-1].abs() + epsilon
    )
    return F.relu(relative_change).mean()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dimensions: tuple[int, ...],
) -> torch.Tensor:
    weighted = values * mask
    return weighted.sum(dim=dimensions) / mask.sum(dim=dimensions).clamp_min(1.0)


def masked_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    kernel = prediction.new_ones(1, 1, window_size, window_size) / (window_size**2)
    padding = window_size // 2
    support = F.conv2d(mask, kernel, padding=padding)
    normalization = support.clamp_min(1e-6)
    mean_prediction = F.conv2d(prediction * mask, kernel, padding=padding) / normalization
    mean_target = F.conv2d(target * mask, kernel, padding=padding) / normalization
    prediction_second = (
        F.conv2d(prediction.square() * mask, kernel, padding=padding) / normalization
    )
    target_second = (
        F.conv2d(target.square() * mask, kernel, padding=padding) / normalization
    )
    cross = (
        F.conv2d(prediction * target * mask, kernel, padding=padding) / normalization
    )
    prediction_variance = (prediction_second - mean_prediction.square()).clamp_min(0)
    target_variance = (target_second - mean_target.square()).clamp_min(0)
    covariance = cross - mean_prediction * mean_target
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
    usable = (support >= 0.9).to(score.dtype)
    return 1.0 - (score * usable).sum() / usable.sum().clamp_min(1.0)


def soft_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.10,
    temperature: float = 0.05,
) -> torch.Tensor:
    predicted_snow = torch.sigmoid((prediction - threshold) / temperature) * mask
    target_snow = torch.sigmoid((target - threshold) / temperature) * mask
    intersection = (predicted_snow * target_snow).flatten(1).sum(dim=1)
    denominator = (
        predicted_snow.flatten(1).sum(dim=1)
        + target_snow.flatten(1).sum(dim=1)
    )
    return (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def sobel_edges(values: torch.Tensor) -> torch.Tensor:
    kernel_x = values.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = values.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(values, kernel_x, padding=1)
    grad_y = F.conv2d(values, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def boundary_band_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.10,
    radius: int = 3,
) -> torch.Tensor:
    snow = (target >= threshold).to(target.dtype) * mask
    pooled_max = F.max_pool2d(snow, kernel_size=3, stride=1, padding=1)
    pooled_min = -F.max_pool2d(-snow, kernel_size=3, stride=1, padding=1)
    edge = ((pooled_max - pooled_min) > 0).to(target.dtype) * mask
    if radius > 1:
        size = radius * 2 + 1
        edge = F.max_pool2d(edge, kernel_size=size, stride=1, padding=radius)
    return edge * mask


class AttractorEnergyLoss(nn.Module):
    def __init__(
        self,
        final_l1_weight: float = 1.0,
        final_mse_weight: float = 1.0,
        auxiliary_weight: float = 0.15,
        energy_alignment_weight: float = 0.0,
        energy_monotonicity_weight: float = 0.01,
        coverage_l1_weight: float = 0.0,
        coverage_mse_weight: float = 0.0,
        coordinate_coverage_weight: float = 0.5,
        coordinate_change_weight: float = 0.5,
        manifold_distance_weight: float = 0.05,
        manifold_first_order_weight: float = 0.001,
        manifold_second_order_weight: float = 0.005,
        noise_consistency_weight: float = 0.05,
        event_change_threshold: float = 0.01,
        event_weight: float = 1.5,
        accumulation_extra_weight: float = 0.5,
        accumulation_underprediction_weight: float = 2.0,
        coverage_delta_mse_weight: float = 0.0,
        coverage_direction_weight: float = 0.0,
        bounded_event_weighting: bool = False,
        event_weight_min: float = 0.5,
        event_weight_max: float = 2.0,
        direction_temperature: float = 0.01,
        coverage_regime_weight: float = 0.0,
        coverage_expert_weight: float = 0.0,
        stable_zero_weight: float = 0.0,
        coverage_base_l1_weight: float = 0.0,
        coverage_base_mse_weight: float = 0.0,
        coordinate_loss: str = "l1",
        ssim_weight: float = 0.0,
        dice_weight: float = 0.0,
        dominance_weight: float = 0.0,
        dominance_margin: float = 0.0,
        hopfield_entropy_weight: float = 0.0,
        trajectory_smoothness_weight: float = 0.0,
        lyapunov_weight: float = 0.0,
        boundary_sobel_weight: float = 0.0,
        boundary_band_weight: float = 0.0,
        boundary_band_radius: int = 3,
    ) -> None:
        super().__init__()
        self.final_l1_weight = final_l1_weight
        self.final_mse_weight = final_mse_weight
        self.auxiliary_weight = auxiliary_weight
        self.energy_alignment_weight = energy_alignment_weight
        self.energy_monotonicity_weight = energy_monotonicity_weight
        self.coverage_l1_weight = coverage_l1_weight
        self.coverage_mse_weight = coverage_mse_weight
        self.coordinate_coverage_weight = coordinate_coverage_weight
        self.coordinate_change_weight = coordinate_change_weight
        self.manifold_distance_weight = manifold_distance_weight
        self.manifold_first_order_weight = manifold_first_order_weight
        self.manifold_second_order_weight = manifold_second_order_weight
        self.noise_consistency_weight = noise_consistency_weight
        self.event_change_threshold = event_change_threshold
        self.event_weight = event_weight
        self.accumulation_extra_weight = accumulation_extra_weight
        self.accumulation_underprediction_weight = accumulation_underprediction_weight
        self.coverage_delta_mse_weight = coverage_delta_mse_weight
        self.coverage_direction_weight = coverage_direction_weight
        self.bounded_event_weighting = bounded_event_weighting
        self.event_weight_min = event_weight_min
        self.event_weight_max = event_weight_max
        self.direction_temperature = direction_temperature
        self.coverage_regime_weight = coverage_regime_weight
        self.coverage_expert_weight = coverage_expert_weight
        self.stable_zero_weight = stable_zero_weight
        self.coverage_base_l1_weight = coverage_base_l1_weight
        self.coverage_base_mse_weight = coverage_base_mse_weight
        self.coordinate_loss = coordinate_loss
        self.ssim_weight = ssim_weight
        self.dice_weight = dice_weight
        self.dominance_weight = dominance_weight
        self.dominance_margin = dominance_margin
        self.hopfield_entropy_weight = hopfield_entropy_weight
        self.trajectory_smoothness_weight = trajectory_smoothness_weight
        self.lyapunov_weight = lyapunov_weight
        self.boundary_sobel_weight = boundary_sobel_weight
        self.boundary_band_weight = boundary_band_weight
        self.boundary_band_radius = boundary_band_radius
        if coordinate_loss not in {"l1", "smooth_l1"}:
            raise ValueError(f"Unsupported coordinate_loss: {coordinate_loss}")

    def forward(
        self,
        outputs: dict,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        land_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = outputs["prediction"]
        energy = outputs["energy"]
        auxiliary_predictions = outputs["auxiliary_predictions"]
        assert isinstance(prediction, torch.Tensor)
        assert isinstance(energy, torch.Tensor)
        assert isinstance(auxiliary_predictions, list)

        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        valid_mask = valid_mask.to(dtype=prediction.dtype)
        target = torch.where(valid_mask.bool(), target, torch.zeros_like(target))
        spatial_dimensions = tuple(range(1, target.ndim))
        target_coverage = masked_mean(
            target,
            valid_mask,
            spatial_dimensions,
        )
        last_coverage = outputs.get("last_input_coverage")
        if last_coverage is None:
            last_coverage = target_coverage.detach()
        target_change = target_coverage - last_coverage
        event_strength = (
            target_change.abs() / max(self.event_change_threshold, 1e-6)
        ).clamp(0.0, 2.0)
        sample_weight = 1.0 + self.event_weight * event_strength
        sample_weight = sample_weight + self.accumulation_extra_weight * (
            target_change > self.event_change_threshold
        ).to(sample_weight.dtype)
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
        if self.bounded_event_weighting:
            sample_weight = sample_weight.clamp(
                self.event_weight_min,
                self.event_weight_max,
            )
        map_l1 = masked_mean(
            (prediction - target).abs(),
            valid_mask,
            spatial_dimensions,
        )
        map_mse = masked_mean(
            (prediction - target).square(),
            valid_mask,
            spatial_dimensions,
        )
        final_l1 = (sample_weight * map_l1).mean()
        final_mse = (sample_weight * map_mse).mean()
        predicted_coverage = masked_mean(
            prediction,
            valid_mask,
            spatial_dimensions,
        )
        coverage_error = predicted_coverage - target_coverage
        coverage_l1 = (sample_weight * coverage_error.abs()).mean()
        coverage_mse = (sample_weight * coverage_error.square()).mean()
        temporal_base = outputs.get("coverage_temporal_base")
        if temporal_base is None:
            coverage_base_l1 = coverage_error.new_zeros(())
            coverage_base_mse = coverage_error.new_zeros(())
        else:
            base_error = temporal_base - target_coverage
            coverage_base_l1 = (sample_weight * base_error.abs()).mean()
            coverage_base_mse = (sample_weight * base_error.square()).mean()
        predicted_change = outputs.get("coverage_delta")
        if predicted_change is None:
            predicted_change = predicted_coverage - last_coverage
        coverage_delta_mse = (
            sample_weight * (predicted_change - target_change).square()
        ).mean()
        regime_target = torch.full_like(target_change, 2, dtype=torch.long)
        regime_target[target_change > self.event_change_threshold] = 0
        regime_target[target_change < -self.event_change_threshold] = 1
        regime_logits = outputs.get("coverage_regime_logits")
        coverage_regime = (
            F.cross_entropy(regime_logits.float(), regime_target)
            if regime_logits is not None and regime_logits.size(1) >= 3
            else coverage_error.new_zeros(())
        )
        expert_deltas = outputs.get("coverage_expert_deltas")
        coverage_expert = (
            F.smooth_l1_loss(
                expert_deltas.gather(1, regime_target[:, None]).squeeze(1),
                target_change,
            )
            if expert_deltas is not None and expert_deltas.size(1) >= 3
            else coverage_error.new_zeros(())
        )
        stable_mask = target_change.abs() <= self.event_change_threshold
        stable_zero = (
            predicted_change[stable_mask].square().mean()
            if stable_mask.any()
            else coverage_error.new_zeros(())
        )
        direction_mask = target_change.abs() > self.event_change_threshold
        if direction_mask.any():
            direction_sign = target_change[direction_mask].sign().float()
            temperature = max(self.direction_temperature, 1e-6)
            direction_margin = (
                predicted_change[direction_mask].float()
                * direction_sign
                / temperature
            ).clamp(-20.0, 20.0)
            direction_loss = temperature * F.softplus(-direction_margin).mean()
        else:
            direction_loss = coverage_error.new_zeros(())
        accumulation_mask = (
            target_change > self.event_change_threshold
        ).to(coverage_error.dtype)
        accumulation_underprediction = (
            accumulation_mask * (target_coverage - predicted_coverage).clamp_min(0.0)
        ).sum() / accumulation_mask.sum().clamp_min(1.0)
        auxiliary_losses: list[torch.Tensor] = []
        per_sample_errors: list[torch.Tensor] = []
        step_count = len(auxiliary_predictions)
        for step, auxiliary in enumerate(auxiliary_predictions, start=1):
            auxiliary_target = F.interpolate(
                target,
                size=auxiliary.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            auxiliary_mask = F.interpolate(
                valid_mask,
                size=auxiliary.shape[-2:],
                mode="nearest",
            )
            step_loss = masked_mean(
                (auxiliary - auxiliary_target).abs(),
                auxiliary_mask,
                spatial_dimensions,
            ).mean()
            auxiliary_losses.append((step / step_count) * step_loss)
            per_sample_errors.append(
                masked_mean(
                    (auxiliary - auxiliary_target).abs(),
                    auxiliary_mask,
                    spatial_dimensions,
                )
            )
        if auxiliary_losses:
            auxiliary_loss = torch.stack(auxiliary_losses).sum() / sum(
                range(1, step_count + 1)
            ) * step_count
        else:
            auxiliary_loss = coverage_error.new_zeros(())

        manifold_coordinate = outputs.get("manifold_coordinate")
        if manifold_coordinate is None:
            manifold_coordinate = target_coverage.new_zeros(target_coverage.size(0), 2)
        if self.coordinate_loss == "smooth_l1":
            coverage_coordinate_error = F.smooth_l1_loss(
                manifold_coordinate[:, 0],
                target_coverage,
                reduction="none",
            )
            change_coordinate_error = F.smooth_l1_loss(
                manifold_coordinate[:, 1],
                target_change,
                reduction="none",
            )
        else:
            coverage_coordinate_error = (
                manifold_coordinate[:, 0] - target_coverage
            ).abs()
            change_coordinate_error = (
                manifold_coordinate[:, 1] - target_change
            ).abs()
        coordinate_coverage = (sample_weight * coverage_coordinate_error).mean()
        coordinate_change = (sample_weight * change_coordinate_error).mean()
        manifold_distance = energy[:, -1].mean()
        energy_alignment = manifold_distance
        energy_monotonicity = relative_energy_monotonicity(energy)
        manifold_first_order = outputs.get("manifold_first_order", energy.new_zeros(()))
        manifold_second_order = outputs.get("manifold_second_order", energy.new_zeros(()))
        clean_state = outputs.get("clean_manifold_state")
        noisy_state = outputs.get("noisy_manifold_state")
        noise_consistency = (
            F.smooth_l1_loss(noisy_state, clean_state.detach())
            if clean_state is not None and noisy_state is not None
            else energy.new_zeros(())
        )
        ssim = masked_ssim_loss(prediction, target, valid_mask)
        dice = soft_dice_loss(prediction, target, valid_mask)
        edge_mask = boundary_band_mask(
            target,
            valid_mask,
            radius=self.boundary_band_radius,
        )
        boundary_sobel = masked_mean(
            (sobel_edges(prediction) - sobel_edges(target)).abs(),
            valid_mask,
            spatial_dimensions,
        ).mean()
        boundary_band = masked_mean(
            (prediction - target).abs(),
            edge_mask,
            spatial_dimensions,
        ).mean()
        backbone = outputs.get("backbone_prediction")
        if backbone is None:
            dominance = prediction.new_zeros(())
            dominance_map = prediction.new_zeros(())
            dominance_coverage = prediction.new_zeros(())
        else:
            backbone_map_l1 = masked_mean(
                (backbone - target).abs(),
                valid_mask,
                spatial_dimensions,
            )
            backbone_map_mse = masked_mean(
                (backbone - target).square(),
                valid_mask,
                spatial_dimensions,
            )
            backbone_coverage = masked_mean(
                backbone,
                valid_mask,
                spatial_dimensions,
            )
            margin = prediction.new_tensor(self.dominance_margin)
            map_error = map_l1 + map_mse
            map_reference = (backbone_map_l1 + backbone_map_mse).detach()
            coverage_reference = (
                backbone_coverage - target_coverage
            ).abs().detach()
            dominance_map = F.relu(map_error - map_reference + margin).mean()
            dominance_coverage = F.relu(
                coverage_error.abs() - coverage_reference + margin
            ).mean()
            dominance = dominance_map + dominance_coverage
        hopfield_entropy = outputs.get("hopfield_entropy")
        entropy_loss = (
            hopfield_entropy.mean()
            if hopfield_entropy is not None and hopfield_entropy.numel()
            else prediction.new_zeros(())
        )
        coordinate_trajectory = outputs.get("coordinate_trajectory")
        if coordinate_trajectory is not None and coordinate_trajectory.size(1) >= 3:
            second_difference = (
                coordinate_trajectory[:, 2:, :2]
                - 2.0 * coordinate_trajectory[:, 1:-1, :2]
                + coordinate_trajectory[:, :-2, :2]
            )
            trajectory_smoothness = second_difference.square().mean()
        else:
            trajectory_smoothness = prediction.new_zeros(())
        lyapunov_delta = outputs.get("lyapunov_delta")
        lyapunov = (
            F.relu(lyapunov_delta).mean()
            if lyapunov_delta is not None and lyapunov_delta.numel()
            else prediction.new_zeros(())
        )

        total = (
            self.final_l1_weight * final_l1
            + self.final_mse_weight * final_mse
            + self.auxiliary_weight * auxiliary_loss
            + self.energy_alignment_weight * energy_alignment
            + self.energy_monotonicity_weight * energy_monotonicity
            + self.coverage_l1_weight * coverage_l1
            + self.coverage_mse_weight * coverage_mse
            + self.coordinate_coverage_weight * coordinate_coverage
            + self.coordinate_change_weight * coordinate_change
            + self.manifold_distance_weight * manifold_distance
            + self.manifold_first_order_weight * manifold_first_order
            + self.manifold_second_order_weight * manifold_second_order
            + self.noise_consistency_weight * noise_consistency
            + self.accumulation_underprediction_weight * accumulation_underprediction
            + self.coverage_delta_mse_weight * coverage_delta_mse
            + self.coverage_direction_weight * direction_loss
            + self.coverage_regime_weight * coverage_regime
            + self.coverage_expert_weight * coverage_expert
            + self.stable_zero_weight * stable_zero
            + self.coverage_base_l1_weight * coverage_base_l1
            + self.coverage_base_mse_weight * coverage_base_mse
            + self.ssim_weight * ssim
            + self.dice_weight * dice
            + self.dominance_weight * dominance
            + self.hopfield_entropy_weight * entropy_loss
            + self.trajectory_smoothness_weight * trajectory_smoothness
            + self.lyapunov_weight * lyapunov
            + self.boundary_sobel_weight * boundary_sobel
            + self.boundary_band_weight * boundary_band
        )
        return {
            "total": total,
            "final_l1": final_l1,
            "final_mse": final_mse,
            "auxiliary": auxiliary_loss,
            "energy_alignment": energy_alignment,
            "energy_monotonicity": energy_monotonicity,
            "coverage_l1": coverage_l1,
            "coverage_mse": coverage_mse,
            "coordinate_coverage": coordinate_coverage,
            "coordinate_change": coordinate_change,
            "manifold_distance": manifold_distance,
            "manifold_first_order": manifold_first_order,
            "manifold_second_order": manifold_second_order,
            "noise_consistency": noise_consistency,
            "accumulation_underprediction": accumulation_underprediction,
            "coverage_delta_mse": coverage_delta_mse,
            "coverage_direction": direction_loss,
            "coverage_regime": coverage_regime,
            "coverage_expert": coverage_expert,
            "stable_zero": stable_zero,
            "coverage_base_l1": coverage_base_l1,
            "coverage_base_mse": coverage_base_mse,
            "ssim": ssim,
            "dice": dice,
            "dominance": dominance,
            "dominance_map": dominance_map,
            "dominance_coverage": dominance_coverage,
            "hopfield_entropy": entropy_loss,
            "trajectory_smoothness": trajectory_smoothness,
            "lyapunov": lyapunov,
            "boundary_sobel": boundary_sobel,
            "boundary_band": boundary_band,
        }
