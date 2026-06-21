"""Transformer continuous-attractor model with Lyapunov-style energy."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hopfield_cann import SeasonalContinuousAttractorMemory, _masked_coverage
from .model import ConvBlock


class TransformerEncoderBlock(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        update = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + update
        return tokens + self.ffn(self.norm2(tokens))


def _window_partition(inputs: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = inputs.shape
    return (
        inputs.view(batch, height // window, window, width // window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window * window, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window: int,
    batch: int,
    height: int,
    width: int,
) -> torch.Tensor:
    channels = windows.size(-1)
    return (
        windows.view(batch, height // window, width // window, window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


class WindowTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.window = window
        self.block = TransformerEncoderBlock(channels, heads, dropout)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if height % self.window or width % self.window:
            raise ValueError("token grid must be divisible by window_size")
        batch = tokens.size(0)
        maps = tokens.reshape(batch, height, width, -1)
        windows = _window_partition(maps, self.window)
        windows = self.block(windows)
        return _window_reverse(windows, self.window, batch, height, width).reshape(
            batch,
            height * width,
            -1,
        )


class LyapunovAttractorUpdate(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        attraction_rate_init: float,
    ) -> None:
        super().__init__()
        self.memory_norm = nn.LayerNorm(channels)
        self.state_norm = nn.LayerNorm(channels)
        self.memory_attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.step_size = nn.Parameter(torch.tensor(0.10))
        self.attraction_rate = nn.Parameter(torch.tensor(attraction_rate_init))

    def forward(
        self,
        state: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        step = self.step_size.clamp(0.02, 0.25)
        update = self.memory_attention(
            self.state_norm(state),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            need_weights=False,
        )[0]
        state = state + step * update
        attraction = self.attraction_rate.clamp(0.005, 0.20)
        state = state + attraction * (
            memory_tokens.mean(dim=1, keepdim=True) - state.mean(dim=1, keepdim=True)
        )
        return state + step * self.ffn(self.ffn_norm(state))


class BoundaryRefiner(nn.Module):
    """High-resolution residual refiner for snowline boundary detail."""

    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        self.stem = ConvBlock(input_channels, 48, dropout * 0.25)
        self.local = ConvBlock(48, 48, dropout * 0.25)
        self.dilated = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.25),
        )
        self.fuse = ConvBlock(96, 48, dropout * 0.25)
        self.out = nn.Conv2d(48, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = self.local(self.stem(features))
        context = self.dilated(local)
        return self.out(self.fuse(torch.cat([local, context], dim=1)))


def _calibrate_map_to_coverage(
    prediction: torch.Tensor,
    target_coverage: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        current = prediction.flatten(1).mean(dim=1)
    else:
        mask = mask.to(dtype=prediction.dtype)
        current = (prediction * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(
            dim=1
        ).clamp_min(1.0)
    target = target_coverage.clamp(0.0, 1.0)
    lower = prediction * (target / current.clamp_min(1e-6))[:, None, None, None]
    upper = 1.0 - (1.0 - prediction) * (
        (1.0 - target) / (1.0 - current).clamp_min(1e-6)
    )[:, None, None, None]
    return torch.where(
        (target <= current)[:, None, None, None],
        lower,
        upper,
    ).clamp(0.0, 1.0)


class TransformerCANNLyapunovNet(nn.Module):
    """256-friendly spatiotemporal Transformer with continuous attractor dynamics."""

    def __init__(
        self,
        input_steps: int = 20,
        input_channels: int = 3,
        context_dim: int = 60,
        spatial_context_channels: int = 0,
        internal_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 192,
        attention_heads: int = 6,
        temporal_layers: int = 3,
        spatial_layers: int = 4,
        window_size: int = 4,
        attractor_iterations: int = 4,
        coverage_bins: int = 9,
        change_bins: int = 9,
        season_bins: int = 12,
        memory_tokens: int = 8,
        change_scale: float = 0.20,
        coordinate_step_scale: float = 0.04,
        max_residual: float = 0.40,
        lyapunov_coordinate_weight: float = 0.05,
        attraction_rate_init: float = 0.05,
        enable_boundary_refiner: bool = False,
        boundary_residual_scale: float = 0.08,
        coverage_calibration: bool = False,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if internal_size % patch_size:
            raise ValueError("internal_size must be divisible by patch_size")
        token_size = internal_size // patch_size
        if token_size % window_size:
            raise ValueError("token grid must be divisible by window_size")
        self.input_steps = input_steps
        self.input_channels = input_channels
        self.context_dim = context_dim
        self.spatial_context_channels = spatial_context_channels
        self.internal_size = internal_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.token_size = token_size
        self.attractor_iterations = attractor_iterations
        self.change_scale = change_scale
        self.coordinate_step_scale = coordinate_step_scale
        self.max_residual = max_residual
        self.lyapunov_coordinate_weight = lyapunov_coordinate_weight
        self.enable_boundary_refiner = enable_boundary_refiner
        self.boundary_residual_scale = boundary_residual_scale
        self.coverage_calibration = coverage_calibration

        context_channels = embed_dim
        condition_channels = 2 + spatial_context_channels
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_channels, context_channels),
        )
        self.frame_encoder = nn.Conv2d(
            input_channels + condition_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.condition_encoder = nn.Conv2d(
            condition_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, input_steps, 1, embed_dim)
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, attention_heads, dropout)
                for _ in range(temporal_layers)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [
                WindowTransformerBlock(embed_dim, attention_heads, window_size, dropout)
                for _ in range(spatial_layers)
            ]
        )
        self.context_film = nn.Linear(context_channels, embed_dim * 2)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)
        self.attractor_memory = SeasonalContinuousAttractorMemory(
            embed_dim,
            coverage_bins,
            change_bins,
            season_bins,
            memory_tokens,
            change_scale,
        )
        self.attractor_update = LyapunovAttractorUpdate(
            embed_dim,
            attention_heads,
            dropout,
            attraction_rate_init,
        )
        coordinate_dim = embed_dim + context_channels + 4
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(coordinate_dim),
            nn.Linear(coordinate_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.coordinate_refiner = nn.Sequential(
            nn.LayerNorm(coordinate_dim),
            nn.Linear(coordinate_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)
        nn.init.zeros_(self.coordinate_refiner[-1].weight)
        nn.init.zeros_(self.coordinate_refiner[-1].bias)
        decoder_channels = max(embed_dim // 2, 32)
        self.decoder = nn.Sequential(
            ConvBlock(embed_dim, decoder_channels, dropout * 0.5),
            ConvBlock(decoder_channels, decoder_channels // 2, dropout * 0.25),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )
        self.direct_decoder = nn.Sequential(
            ConvBlock(embed_dim, decoder_channels, dropout * 0.5),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.auxiliary_head = nn.Conv2d(embed_dim, 1, 1)
        self.fusion_gate = nn.Conv2d(1, 1, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -2.0)
        boundary_channels = 6 + condition_channels
        self.boundary_refiner = (
            BoundaryRefiner(boundary_channels, dropout)
            if enable_boundary_refiner
            else None
        )

    def _prepare_condition(
        self,
        inputs: torch.Tensor,
        spatial_prior: torch.Tensor,
        spatial_context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _, height, width = inputs.shape
        prior = F.interpolate(
            spatial_prior,
            size=(self.internal_size, self.internal_size),
            mode="bilinear",
            align_corners=False,
        )
        if self.spatial_context_channels:
            if spatial_context is None:
                spatial_context = inputs.new_zeros(
                    batch,
                    self.spatial_context_channels,
                    height,
                    width,
                )
            if spatial_context.size(1) != self.spatial_context_channels:
                raise ValueError(
                    f"Expected {self.spatial_context_channels} spatial context "
                    f"channels, received {spatial_context.size(1)}"
                )
            spatial_context = F.interpolate(
                spatial_context,
                size=(self.internal_size, self.internal_size),
                mode="bilinear",
                align_corners=False,
            )
            condition = torch.cat([prior, spatial_context], dim=1)
        else:
            condition = prior
        return prior, condition

    def _season_features(self, context: torch.Tensor) -> torch.Tensor:
        calendar_start = self.input_steps + (self.input_steps - 1) + 37
        if context.size(1) >= calendar_start + 2:
            season = context[:, calendar_start : calendar_start + 2]
            return season / season.norm(dim=1, keepdim=True).clamp_min(1e-6)
        fallback = context.new_zeros(context.size(0), 2)
        fallback[:, 1] = 1.0
        return fallback

    def _last_coverage(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.size(1) >= self.input_steps:
            return context[:, self.input_steps - 1].clamp(0.0, 1.0)
        return inputs[:, -1, 0].flatten(1).mean(dim=1).clamp(0.0, 1.0)

    def _coordinate(
        self,
        global_state: torch.Tensor,
        encoded_context: torch.Tensor,
        last_coverage: torch.Tensor,
        season: torch.Tensor,
        previous_change: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous = (
            torch.zeros_like(last_coverage)
            if previous_change is None
            else previous_change
        )
        features = torch.cat(
            [
                global_state,
                encoded_context,
                last_coverage[:, None],
                previous[:, None],
                season,
            ],
            dim=1,
        )
        if previous_change is None:
            change = self.change_scale * torch.tanh(
                self.coordinate_head(features).squeeze(1)
            )
        else:
            change = (
                previous
                + self.coordinate_step_scale
                * torch.tanh(self.coordinate_refiner(features).squeeze(1))
            ).clamp(-self.change_scale, self.change_scale)
        coverage = (last_coverage + change).clamp(0.0, 1.0)
        return torch.cat([coverage[:, None], change[:, None], season], dim=1)

    def _lyapunov_energy(
        self,
        state: torch.Tensor,
        memory_tokens: torch.Tensor,
        coordinate: torch.Tensor,
        prior_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        state_term = (
            state.mean(dim=1) - memory_tokens.mean(dim=1)
        ).square().mean(dim=1)
        coordinate_term = (
            coordinate[:, :2] - prior_coordinate[:, :2]
        ).square().mean(dim=1)
        return state_term + self.lyapunov_coordinate_weight * coordinate_term

    def _decode(
        self,
        state: torch.Tensor,
        height: int,
        width: int,
        last_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = state.transpose(1, 2).reshape(
            state.size(0),
            self.embed_dim,
            self.token_size,
            self.token_size,
        )
        latent = F.interpolate(
            latent,
            size=last_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = self.max_residual * torch.tanh(self.decoder(latent))
        residual_prediction = (last_map + residual).clamp(0.0, 1.0)
        direct_prediction = torch.sigmoid(self.direct_decoder(latent))
        gate = torch.sigmoid(self.fusion_gate(residual))
        prediction = (
            (1.0 - gate) * residual_prediction + gate * direct_prediction
        ).clamp(0.0, 1.0)
        if prediction.shape[-2:] != (height, width):
            prediction = F.interpolate(
                prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            residual_prediction = F.interpolate(
                residual_prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            direct_prediction = F.interpolate(
                direct_prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            gate = F.interpolate(
                gate,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return prediction, residual_prediction, direct_prediction, gate

    def forward(
        self,
        inputs: torch.Tensor,
        spatial_prior: torch.Tensor,
        context: torch.Tensor,
        spatial_context: Optional[torch.Tensor] = None,
        state_noise_std: float = 0.0,
        ablation_mode: str = "none",
        land_mask: Optional[torch.Tensor] = None,
        coverage_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if ablation_mode not in {"none"}:
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        batch, steps, channels, height, width = inputs.shape
        if (steps, channels) != (self.input_steps, self.input_channels):
            raise ValueError(
                f"Expected temporal shape {(self.input_steps, self.input_channels)}, "
                f"received {(steps, channels)}"
            )
        resized = F.interpolate(
            inputs.reshape(batch * steps, channels, height, width),
            size=(self.internal_size, self.internal_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch,
            steps,
            channels,
            self.internal_size,
            self.internal_size,
        )
        _, condition = self._prepare_condition(inputs, spatial_prior, spatial_context)
        condition_tokens = self.condition_encoder(condition).flatten(2).transpose(1, 2)
        frame_condition = condition.unsqueeze(1).expand(-1, steps, -1, -1, -1)
        frame_inputs = torch.cat([resized, frame_condition], dim=2).reshape(
            batch * steps,
            channels + condition.size(1),
            self.internal_size,
            self.internal_size,
        )
        encoded = self.frame_encoder(frame_inputs)
        token_height, token_width = encoded.shape[-2:]
        if (token_height, token_width) != (self.token_size, self.token_size):
            raise ValueError("Unexpected token grid size")
        tokens = encoded.flatten(2).transpose(1, 2).reshape(
            batch,
            steps,
            token_height * token_width,
            self.embed_dim,
        )
        tokens = tokens + self.temporal_position[:, :steps]
        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch * token_height * token_width,
            steps,
            self.embed_dim,
        )
        for block in self.temporal_blocks:
            temporal = block(temporal)
        state = temporal[:, -1].reshape(batch, token_height * token_width, self.embed_dim)
        state = state + condition_tokens
        encoded_context = self.context_encoder(context)
        gamma, beta = self.context_film(encoded_context).chunk(2, dim=1)
        state = state * (1.0 + 0.1 * torch.tanh(gamma)[:, None])
        state = state + beta[:, None]
        for block in self.spatial_blocks:
            state = block(state, token_height, token_width)
        if state_noise_std > 0:
            state = state + state_noise_std * torch.randn_like(state)

        last_coverage = self._last_coverage(inputs, context)
        season = self._season_features(context)
        prior_coordinate = torch.cat(
            [
                last_coverage[:, None],
                torch.zeros_like(last_coverage[:, None]),
                season,
            ],
            dim=1,
        )
        coordinate = self._coordinate(
            state.mean(dim=1),
            encoded_context,
            last_coverage,
            season,
        )
        trajectory = [coordinate]
        memory_tokens = self.attractor_memory.retrieve(coordinate)
        energies = [
            self._lyapunov_energy(state, memory_tokens, coordinate, prior_coordinate)
        ]
        update_norms = []
        auxiliary_predictions = []
        for _ in range(self.attractor_iterations):
            previous = state
            state = self.attractor_update(state, memory_tokens)
            coordinate = self._coordinate(
                state.mean(dim=1),
                encoded_context,
                last_coverage,
                season,
                previous_change=coordinate[:, 1],
            )
            trajectory.append(coordinate)
            memory_tokens = self.attractor_memory.retrieve(coordinate)
            energies.append(
                self._lyapunov_energy(state, memory_tokens, coordinate, prior_coordinate)
            )
            update_norms.append(
                (state - previous).flatten(1).norm(dim=1)
                / previous.flatten(1).norm(dim=1).clamp_min(1e-6)
            )
            latent = state.transpose(1, 2).reshape(
                batch,
                self.embed_dim,
                token_height,
                token_width,
            )
            auxiliary_predictions.append(
                torch.sigmoid(self.auxiliary_head(latent))
            )

        last_map = resized[:, -1, 0:1]
        prediction, residual_prediction, direct_prediction, gate = self._decode(
            state,
            height,
            width,
            last_map,
        )
        effective_mask = coverage_mask if coverage_mask is not None else land_mask
        global_prediction = prediction
        boundary_residual = prediction.new_zeros(prediction.shape)
        if self.boundary_refiner is not None:
            previous_map = resized[:, -2, 0:1] if resized.size(1) > 1 else last_map
            recent_change = last_map - previous_map
            high_res_condition = F.interpolate(
                condition,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            boundary_features = torch.cat(
                [
                    last_map,
                    global_prediction,
                    residual_prediction,
                    direct_prediction,
                    gate,
                    recent_change,
                    high_res_condition,
                ],
                dim=1,
            )
            boundary_residual = self.boundary_residual_scale * torch.tanh(
                self.boundary_refiner(boundary_features)
            )
            prediction = (global_prediction + boundary_residual).clamp(0.0, 1.0)
            if self.coverage_calibration:
                target_coverage = _masked_coverage(global_prediction, effective_mask)
                prediction = _calibrate_map_to_coverage(
                    prediction,
                    target_coverage,
                    effective_mask,
                )
        coverage_prediction = _masked_coverage(prediction, effective_mask)
        backbone_prediction = F.interpolate(
            last_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        backbone_coverage = _masked_coverage(backbone_prediction, effective_mask)
        energy = torch.stack(energies, dim=1)
        trajectory_tensor = torch.stack(trajectory, dim=1)
        update_norm_tensor = (
            torch.stack(update_norms, dim=1)
            if update_norms
            else prediction.new_zeros(batch, 0)
        )
        first_order, second_order = self.attractor_memory.regularization()
        lyapunov_delta = (
            energy[:, 1:] - energy[:, :-1]
            if energy.size(1) > 1
            else energy.new_zeros(batch, 0)
        )
        zero = prediction.new_zeros(batch)
        return {
            "prediction": prediction,
            "raw_prediction": prediction,
            "global_prediction": global_prediction,
            "boundary_residual": boundary_residual,
            "refined_prediction": prediction,
            "coverage_prediction": coverage_prediction,
            "coverage_delta": coverage_prediction - last_coverage,
            "coverage_delta_raw": coverage_prediction - last_coverage,
            "coverage_temporal_base": backbone_coverage,
            "coverage_temporal_residual": coverage_prediction - last_coverage,
            "coverage_attractor_correction": zero,
            "coverage_correction_scale": zero,
            "coverage_regime_logits": prediction.new_zeros(batch, 0),
            "coverage_regime_weights": prediction.new_zeros(batch, 0),
            "coverage_expert_deltas": prediction.new_zeros(batch, 0),
            "coverage_weather_gate": prediction.new_zeros(batch, 0),
            "coverage_candidates": prediction.new_zeros(batch, 0),
            "coverage_weights": prediction.new_zeros(batch, 0),
            "coverage_residual": zero,
            "last_input_coverage": last_coverage,
            "input_steps": self.input_steps,
            "energy": energy,
            "manifold_energy": energy,
            "manifold_distance": energy,
            "manifold_coordinate": coordinate[:, :2],
            "coordinate_trajectory": trajectory_tensor,
            "lyapunov_delta": lyapunov_delta,
            "memory_tokens": memory_tokens,
            "manifold_first_order": first_order,
            "manifold_second_order": second_order,
            "clean_manifold_state": state.mean(dim=1),
            "noisy_manifold_state": state.mean(dim=1).detach(),
            "auxiliary_predictions": auxiliary_predictions,
            "attention": prediction.new_zeros(batch, 0, 0, 0),
            "update_norms": update_norm_tensor,
            "residual_prediction": residual_prediction,
            "direct_prediction": direct_prediction,
            "fusion_gate": gate,
            "residual_gate": gate,
            "backbone_prediction": backbone_prediction,
            "temporal_global_state": state.mean(dim=1),
            "temporal_fusion_gate": prediction.new_zeros(batch, 0, 0, 0),
            "accumulation_gate": torch.sigmoid((coordinate[:, 1] - 0.005) / 0.01),
        }

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
