"""Attractor-energy U-Net with direct/residual adaptive prediction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TemporalConvLSTMCell(nn.Module):
    """ConvLSTM cell used by the internal short-term temporal encoder."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            3,
            padding=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (inputs.size(0), self.hidden_channels, *inputs.shape[-2:])
            hidden = inputs.new_zeros(shape)
            cell = inputs.new_zeros(shape)
        else:
            hidden, cell = state
        gates = self.gates(torch.cat([inputs, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
            input_gate
        ) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class TemporalConvLSTMEncoder(nn.Module):
    """Encode map history with the same structure as the fair ConvLSTM baseline."""

    def __init__(
        self,
        input_channels: int,
        condition_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.frame_encoder = nn.Sequential(
            ConvBlock(input_channels, 32, dropout * 0.25),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, dropout * 0.5),
            nn.MaxPool2d(2),
        )
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 64, 3, stride=4, padding=1),
            nn.GELU(),
        )
        self.recurrent = TemporalConvLSTMCell(128, hidden_channels)

    def forward(
        self,
        frames: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_features = self.condition_encoder(condition)
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        for step in range(frames.size(1)):
            encoded = torch.cat(
                [self.frame_encoder(frames[:, step]), condition_features],
                dim=1,
            )
            state = self.recurrent(encoded, state)
        assert state is not None
        hidden, _ = state
        return hidden, hidden.mean(dim=(2, 3))


class HybridCoverageHead(nn.Module):
    """Single residual coverage head over temporal and attractor representations."""

    def __init__(
        self,
        temporal_dim: int,
        attractor_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        max_delta: float = 0.12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_delta = max_delta
        fusion_dim = temporal_dim + attractor_dim + context_dim + 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        temporal_state: torch.Tensor,
        attractor_state: torch.Tensor,
        encoded_context: torch.Tensor,
        manifold_coordinate: torch.Tensor,
        last_coverage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat(
            [
                temporal_state,
                attractor_state,
                encoded_context,
                manifold_coordinate,
            ],
            dim=1,
        )
        coverage_delta = self.max_delta * torch.tanh(self.head(fused).squeeze(1))
        coverage_prediction = (last_coverage + coverage_delta).clamp(0.0, 1.0)
        return {
            "coverage_prediction": coverage_prediction,
            "coverage_delta": coverage_delta,
            "coverage_delta_raw": coverage_delta,
        }


class TemporalAnchorCoverageHead(nn.Module):
    """ConvLSTM coverage anchor with an optional bounded attractor correction."""

    def __init__(
        self,
        context_dim: int,
        temporal_dim: int,
        attractor_dim: int,
        encoded_context_dim: int,
        correction_mode: str = "none",
        correction_hidden_dim: int = 128,
        max_correction: float = 0.02,
        stable_correction: float = 0.005,
        event_scale: float = 0.02,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if correction_mode not in {"none", "fixed", "smooth_event"}:
            raise ValueError(f"Unsupported correction mode: {correction_mode}")
        self.correction_mode = correction_mode
        self.max_correction = max_correction
        self.stable_correction = stable_correction
        self.event_scale = event_scale
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(temporal_dim * 2),
            nn.Linear(temporal_dim * 2, 96),
            nn.GELU(),
            nn.Linear(96, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.anchor_head[-2].weight)
        nn.init.zeros_(self.anchor_head[-2].bias)
        correction_dim = attractor_dim + encoded_context_dim + 2
        self.correction_head: Optional[nn.Module]
        if correction_mode == "none":
            self.correction_head = None
        else:
            self.correction_head = nn.Sequential(
                nn.LayerNorm(correction_dim),
                nn.Linear(correction_dim, correction_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(correction_hidden_dim, 1),
            )
            nn.init.zeros_(self.correction_head[-1].weight)
            nn.init.zeros_(self.correction_head[-1].bias)

    def forward(
        self,
        temporal_state: torch.Tensor,
        context: torch.Tensor,
        attractor_state: torch.Tensor,
        encoded_context: torch.Tensor,
        manifold_coordinate: torch.Tensor,
        last_coverage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temporal_context = self.context(context)
        temporal_delta = 0.12 * self.anchor_head(
            torch.cat([temporal_state, temporal_context], dim=1)
        ).squeeze(1)
        temporal_base = (last_coverage + temporal_delta).clamp(0.0, 1.0)
        correction_scale = temporal_base.new_zeros(temporal_base.shape)
        correction = temporal_base.new_zeros(temporal_base.shape)
        if self.correction_head is not None:
            if self.correction_mode == "fixed":
                correction_scale = torch.full_like(
                    temporal_base,
                    self.max_correction,
                )
            else:
                event_strength = torch.tanh(
                    temporal_delta.abs() / max(self.event_scale, 1e-6)
                )
                correction_scale = self.stable_correction + (
                    self.max_correction - self.stable_correction
                ) * event_strength
            correction_features = torch.cat(
                [
                    attractor_state,
                    encoded_context,
                    manifold_coordinate,
                ],
                dim=1,
            )
            correction = correction_scale * torch.tanh(
                self.correction_head(correction_features).squeeze(1)
            )
        coverage_prediction = (temporal_base + correction).clamp(0.0, 1.0)
        return {
            "coverage_prediction": coverage_prediction,
            "coverage_delta": coverage_prediction - last_coverage,
            "coverage_delta_raw": temporal_delta + correction,
            "coverage_temporal_base": temporal_base,
            "coverage_attractor_correction": correction,
            "coverage_correction_scale": correction_scale,
        }


class FiLM(nn.Module):
    def __init__(self, context_channels: int, feature_channels: int):
        super().__init__()
        self.projection = nn.Linear(context_channels, feature_channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(context).chunk(2, dim=1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while scaling its backward gradient."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("gradient scale must be between 0 and 1")
    detached = value.detach()
    return detached + scale * (value - detached)


class CoverageDynamicsHead(nn.Module):
    """Predict coverage change with either v1 regression or v2 supervised experts."""

    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        latent_dim: int,
        encoded_context_dim: int,
        hidden_dim: int = 128,
        expert_count: int = 3,
        max_delta: float = 0.12,
        stable_max_delta: float = 0.01,
        delta_deadband: float = 0.005,
        weather_feature_dim: int = 0,
        weather_hidden_dim: int = 64,
        mode: str = "supervised_v2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in {"continuous_moe_v1", "single_gru", "supervised_v2"}:
            raise ValueError(f"Unsupported coverage dynamics mode: {mode}")
        if expert_count != 3 and mode != "single_gru":
            raise ValueError("supervised dynamics head requires exactly three experts")
        minimum_context = input_steps + (input_steps - 1)
        if context_dim < minimum_context:
            raise ValueError(
                f"context_dim must be at least {minimum_context} for coverage dynamics"
            )
        self.input_steps = input_steps
        self.expert_count = expert_count
        self.max_delta = max_delta
        self.stable_max_delta = stable_max_delta
        self.delta_deadband = delta_deadband
        self.weather_feature_dim = weather_feature_dim
        self.mode = mode
        self.history_encoder = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.weather_encoder: Optional[nn.Module]
        if weather_feature_dim > 0:
            self.weather_encoder = nn.Sequential(
                nn.LayerNorm(weather_feature_dim * 4),
                nn.Linear(weather_feature_dim * 4, weather_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.weather_gate = nn.Sequential(
                nn.Linear(weather_hidden_dim, weather_hidden_dim),
                nn.Sigmoid(),
            )
            weather_output_dim = weather_hidden_dim
        else:
            self.weather_encoder = None
            self.weather_gate = None
            weather_output_dim = 0
        fusion_dim = (
            hidden_dim + latent_dim + encoded_context_dim + 2 + weather_output_dim
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        output_count = 1 if mode == "single_gru" else 3
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(output_count)
            ]
        )
        self.gate = nn.Linear(hidden_dim, output_count)
        if mode in {"single_gru", "supervised_v2"}:
            nn.init.zeros_(self.experts[-1][-1].weight)
            nn.init.zeros_(self.experts[-1][-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        context: torch.Tensor,
        global_state: torch.Tensor,
        encoded_context: torch.Tensor,
        manifold_coordinate: torch.Tensor,
        weather_context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        coverage_history = context[:, : self.input_steps]
        changes = context[
            :, self.input_steps : self.input_steps + self.input_steps - 1
        ]
        change_history = torch.cat(
            [torch.zeros_like(coverage_history[:, :1]), changes],
            dim=1,
        )
        sequence = torch.stack([coverage_history, change_history], dim=-1)
        _, hidden = self.history_encoder(sequence)
        fusion_parts = [
            hidden[-1],
            global_state,
            encoded_context,
            manifold_coordinate,
        ]
        weather_gate = context.new_zeros(context.size(0), 0)
        if self.weather_encoder is not None:
            if weather_context is None:
                weather_context = context.new_zeros(
                    context.size(0),
                    self.weather_feature_dim * 4,
                )
            weather_encoded = self.weather_encoder(weather_context)
            assert self.weather_gate is not None
            weather_gate = self.weather_gate(weather_encoded)
            fusion_parts.append(weather_encoded * weather_gate)
        fused = self.fusion(torch.cat(fusion_parts, dim=1))
        expert_logits = torch.cat([expert(fused) for expert in self.experts], dim=1)
        if self.mode == "supervised_v2":
            expert_deltas = torch.stack(
                [
                    self.max_delta * torch.sigmoid(expert_logits[:, 0]),
                    -self.max_delta * torch.sigmoid(expert_logits[:, 1]),
                    self.stable_max_delta * torch.tanh(expert_logits[:, 2]),
                ],
                dim=1,
            )
        else:
            expert_deltas = self.max_delta * torch.tanh(expert_logits)
        regime_logits = self.gate(fused)
        regime_weights = F.softmax(regime_logits, dim=1)
        raw_coverage_delta = (expert_deltas * regime_weights).sum(dim=1)
        if self.mode == "supervised_v2":
            coverage_delta = raw_coverage_delta.sign() * F.relu(
                raw_coverage_delta.abs() - self.delta_deadband
            )
        else:
            coverage_delta = raw_coverage_delta
        coverage_prediction = (
            coverage_history[:, -1] + coverage_delta
        ).clamp(0.0, 1.0)
        return {
            "coverage_prediction": coverage_prediction,
            "coverage_delta": coverage_delta,
            "coverage_delta_raw": raw_coverage_delta,
            "coverage_regime_logits": regime_logits,
            "coverage_regime_weights": regime_weights,
            "coverage_expert_deltas": expert_deltas,
            "coverage_weather_gate": weather_gate,
        }


class ContinuousAttractorMemory(nn.Module):
    def __init__(
        self,
        channels: int,
        coverage_bins: int,
        change_bins: int,
        memory_tokens: int,
        change_scale: float,
    ) -> None:
        super().__init__()
        if coverage_bins < 2 or change_bins < 2:
            raise ValueError("Continuous memory requires at least two bins per coordinate")
        self.coverage_bins = coverage_bins
        self.change_bins = change_bins
        self.memory_tokens = memory_tokens
        self.change_scale = change_scale
        self.anchors = nn.Parameter(
            torch.empty(coverage_bins, change_bins, memory_tokens, channels)
        )
        nn.init.normal_(self.anchors, std=0.02)
        coordinate_template = torch.zeros(
            coverage_bins,
            change_bins,
            1,
            channels,
        )
        if channels >= 2:
            coordinate_template[..., 0] = torch.linspace(0.0, 0.1, coverage_bins).view(
                coverage_bins,
                1,
                1,
            )
            coordinate_template[..., 1] = torch.linspace(-0.1, 0.1, change_bins).view(
                1,
                change_bins,
                1,
            )
        self.register_buffer("coordinate_template", coordinate_template)

    def anchor_values(self) -> torch.Tensor:
        return self.anchors + self.coordinate_template

    def _grid_coordinates(
        self,
        coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coverage = coordinate[:, 0].clamp(0.0, 1.0) * (self.coverage_bins - 1)
        change = (
            (coordinate[:, 1].clamp(-self.change_scale, self.change_scale)
             / self.change_scale + 1.0)
            * 0.5
            * (self.change_bins - 1)
        )
        return coverage, change

    def retrieve(self, coordinate: torch.Tensor, mode: str) -> torch.Tensor:
        coverage, change = self._grid_coordinates(coordinate)
        anchors = self.anchor_values()
        if mode == "nearest":
            return anchors[coverage.round().long(), change.round().long()]
        if mode != "continuous":
            raise ValueError(f"Unsupported memory retrieval mode: {mode}")

        q0 = coverage.floor().long()
        d0 = change.floor().long()
        q1 = (q0 + 1).clamp_max(self.coverage_bins - 1)
        d1 = (d0 + 1).clamp_max(self.change_bins - 1)
        q_weight = (coverage - q0).view(-1, 1, 1)
        d_weight = (change - d0).view(-1, 1, 1)
        lower = anchors[q0, d0] * (1.0 - q_weight) + anchors[q1, d0] * q_weight
        upper = anchors[q0, d1] * (1.0 - q_weight) + anchors[q1, d1] * q_weight
        return lower * (1.0 - d_weight) + upper * d_weight

    def regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        first_differences = [
            self.anchors[1:] - self.anchors[:-1],
            self.anchors[:, 1:] - self.anchors[:, :-1],
        ]
        first_order = torch.stack(
            [difference.square().mean() for difference in first_differences]
        ).mean()
        second_differences = [
            self.anchors[2:] - 2.0 * self.anchors[1:-1] + self.anchors[:-2],
            self.anchors[:, 2:] - 2.0 * self.anchors[:, 1:-1] + self.anchors[:, :-2],
        ]
        second_order = torch.stack(
            [difference.square().mean() for difference in second_differences]
        ).mean()
        return first_order, second_order


class SharedAttractorBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        attraction_rate_init: float,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("bottleneck channels must be divisible by attention heads")
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(channels)
        self.memory_attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
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
        memory: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(state)
        update, attention = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        step = self.step_size.clamp(0.02, 0.30)
        state = state + step * update
        if memory is not None:
            memory_update, _ = self.memory_attention(
                self.memory_norm(state),
                memory,
                memory,
                need_weights=False,
            )
            state = state + step * memory_update
            attraction = self.attraction_rate.clamp(0.01, 0.20)
            state = state + attraction * (
                memory.mean(dim=1, keepdim=True) - state.mean(dim=1, keepdim=True)
            )
        state = state + step * 0.5 * self.feed_forward(self.feed_forward_norm(state))
        return state, attention


class AttractorEnergyUNet(nn.Module):
    def __init__(
        self,
        input_steps: int = 20,
        input_channels: int = 1,
        context_dim: int = 4,
        base_channels: int = 48,
        bottleneck_channels: int = 384,
        attractor_iterations: int = 4,
        attention_heads: int = 8,
        internal_size: int = 128,
        max_residual: float = 0.4,
        max_coverage_residual: float = 0.12,
        coverage_calibration: bool = True,
        spatial_context_channels: int = 0,
        attractor_mode: str = "continuous",
        coverage_bins: int = 9,
        change_bins: int = 9,
        memory_tokens: int = 8,
        change_scale: float = 0.20,
        manifold_noise_std: float = 0.05,
        attraction_rate_init: float = 0.05,
        accumulation_blend: float = 0.0,
        coordinate_mode: str = "absolute",
        use_change_coordinate: bool = True,
        spatial_context_mode: str = "all",
        coverage_candidate_bias: Optional[list[float]] = None,
        coverage_head_mode: str = "legacy",
        coverage_dynamics_hidden: int = 128,
        coverage_expert_count: int = 3,
        coverage_gradient_scale: float = 0.0,
        coverage_stable_max_delta: float = 0.01,
        coverage_delta_deadband: float = 0.005,
        coverage_dynamics_mode: str = "supervised_v2",
        coverage_calibration_mode: str = "logit",
        coverage_calibration_passes: int = 2,
        coverage_calibration_detach: bool = False,
        weather_feature_dim: int = 0,
        weather_mode: str = "full",
        weather_selected_indices: Optional[list[int]] = None,
        coverage_weather_encoder: bool = True,
        temporal_fusion_mode: str = "none",
        temporal_hidden_channels: int = 128,
        temporal_gate_init: float = -2.0,
        coverage_correction_max: float = 0.02,
        coverage_correction_stable: float = 0.005,
        coverage_correction_event_scale: float = 0.02,
        shared_backbone_residual: bool = False,
        shared_backbone_only: bool = False,
        residual_gate_init: float = -3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if internal_size % 8:
            raise ValueError("internal_size must be divisible by 8")
        self.input_steps = input_steps
        self.input_channels = input_channels
        self.bottleneck_channels = bottleneck_channels
        self.attractor_iterations = attractor_iterations
        self.internal_size = internal_size
        self.max_residual = max_residual
        self.max_coverage_residual = max_coverage_residual
        self.coverage_calibration = coverage_calibration
        self.spatial_context_channels = spatial_context_channels
        self.attractor_mode = attractor_mode
        self.change_scale = change_scale
        self.manifold_noise_std = manifold_noise_std
        self.accumulation_blend = accumulation_blend
        self.coordinate_mode = coordinate_mode
        self.use_change_coordinate = use_change_coordinate
        self.spatial_context_mode = spatial_context_mode
        self.coverage_head_mode = coverage_head_mode
        self.coverage_gradient_scale = coverage_gradient_scale
        self.coverage_calibration_mode = coverage_calibration_mode
        self.coverage_calibration_passes = coverage_calibration_passes
        self.coverage_calibration_detach = coverage_calibration_detach
        self.weather_feature_dim = weather_feature_dim
        self.weather_mode = weather_mode
        self.weather_selected_indices = tuple(weather_selected_indices or [])
        self.temporal_fusion_mode = temporal_fusion_mode
        self.temporal_hidden_channels = temporal_hidden_channels
        self.shared_backbone_residual = shared_backbone_residual
        self.shared_backbone_only = shared_backbone_only
        if coordinate_mode not in {"absolute", "residual"}:
            raise ValueError(f"Unsupported coordinate_mode: {coordinate_mode}")
        if spatial_context_mode not in {
            "all",
            "no_coordinates",
            "no_terrain",
            "no_weather",
            "none",
        }:
            raise ValueError(f"Unsupported spatial_context_mode: {spatial_context_mode}")
        if attractor_mode not in {"continuous", "nearest", "transformer", "none"}:
            raise ValueError(f"Unsupported attractor_mode: {attractor_mode}")
        if coverage_head_mode not in {
            "legacy",
            "dynamics",
            "hybrid",
            "temporal_anchor",
            "anchor_residual",
            "smooth_event_residual",
        }:
            raise ValueError(f"Unsupported coverage_head_mode: {coverage_head_mode}")
        if not 0.0 <= coverage_gradient_scale <= 1.0:
            raise ValueError("coverage_gradient_scale must be between 0 and 1")
        if weather_mode not in {"full", "compact", "no_target", "history_only"}:
            raise ValueError(f"Unsupported weather_mode: {weather_mode}")
        if coverage_calibration_mode not in {"logit", "ratio_v1"}:
            raise ValueError(
                f"Unsupported coverage calibration mode: {coverage_calibration_mode}"
            )
        if coverage_calibration_passes not in {1, 2}:
            raise ValueError("coverage_calibration_passes must be 1 or 2")
        if temporal_fusion_mode not in {
            "none",
            "coverage_only",
            "gated_bottleneck",
        }:
            raise ValueError(
                f"Unsupported temporal_fusion_mode: {temporal_fusion_mode}"
            )
        if shared_backbone_residual and temporal_fusion_mode == "none":
            raise ValueError(
                "Shared ConvLSTM residual mode requires a temporal encoder"
            )
        if shared_backbone_only and not shared_backbone_residual:
            raise ValueError(
                "Backbone-only mode requires shared_backbone_residual"
            )
        temporal_coverage_modes = {
            "hybrid",
            "temporal_anchor",
            "anchor_residual",
            "smooth_event_residual",
        }
        if coverage_head_mode in temporal_coverage_modes and temporal_fusion_mode == "none":
            raise ValueError("Temporal coverage head requires a temporal encoder")
        context_channels = 256
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_channels),
            nn.GELU(),
            nn.Linear(context_channels, context_channels),
            nn.GELU(),
        )
        fused_channels = input_steps * input_channels + 2 + spatial_context_channels
        self.encoder1 = ConvBlock(fused_channels, c1, dropout * 0.25)
        self.encoder2 = ConvBlock(c1, c2, dropout * 0.35)
        self.encoder3 = ConvBlock(c2, c3, dropout * 0.50)
        self.bottleneck = ConvBlock(c3, bottleneck_channels, dropout)
        self.pool = nn.MaxPool2d(2)
        self.temporal_encoder: Optional[TemporalConvLSTMEncoder]
        self.temporal_projector: Optional[nn.Module]
        self.temporal_gate: Optional[nn.Module]
        if temporal_fusion_mode != "none":
            self.temporal_encoder = TemporalConvLSTMEncoder(
                input_channels=input_channels,
                condition_channels=2 + spatial_context_channels,
                hidden_channels=temporal_hidden_channels,
                dropout=dropout,
            )
        else:
            self.temporal_encoder = None
        if temporal_fusion_mode == "gated_bottleneck":
            self.temporal_projector = nn.Sequential(
                nn.Conv2d(
                    temporal_hidden_channels,
                    bottleneck_channels,
                    3,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(
                    _group_count(bottleneck_channels),
                    bottleneck_channels,
                ),
                nn.GELU(),
            )
            self.temporal_gate = nn.Conv2d(
                bottleneck_channels * 2,
                bottleneck_channels,
                1,
            )
            nn.init.zeros_(self.temporal_gate.weight)
            nn.init.constant_(self.temporal_gate.bias, temporal_gate_init)
        else:
            self.temporal_projector = None
            self.temporal_gate = None
        if shared_backbone_residual:
            self.backbone_decoder = nn.Sequential(
                ConvBlock(temporal_hidden_channels, c2, dropout),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBlock(c2, c1, dropout * 0.5),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c1, 1, 1),
            )
        else:
            self.backbone_decoder = None

        self.encoder_films = nn.ModuleList(
            [
                FiLM(context_channels, c1),
                FiLM(context_channels, c2),
                FiLM(context_channels, c3),
                FiLM(context_channels, bottleneck_channels),
            ]
        )
        self.attractor = SharedAttractorBlock(
            bottleneck_channels,
            attention_heads,
            dropout,
            attraction_rate_init,
        )
        self.manifold_memory = ContinuousAttractorMemory(
            bottleneck_channels,
            coverage_bins,
            change_bins,
            memory_tokens,
            change_scale,
        )
        coordinate_dim = bottleneck_channels + context_channels
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(coordinate_dim),
            nn.Linear(coordinate_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)
        self.auxiliary_head = nn.Conv2d(bottleneck_channels, 1, 1)

        self.decoder3 = ConvBlock(bottleneck_channels + c3, c3, dropout * 0.5)
        self.decoder2 = ConvBlock(c3 + c2, c2, dropout * 0.35)
        self.decoder1 = ConvBlock(c2 + c1, c1, dropout * 0.25)
        self.decoder_films = nn.ModuleList(
            [
                FiLM(context_channels, c3),
                FiLM(context_channels, c2),
                FiLM(context_channels, c1),
            ]
        )
        self.residual_head = nn.Conv2d(c1, 1, 1)
        self.direct_head = nn.Conv2d(c1, 1, 1)
        self.fusion_gate = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(
            self.fusion_gate.bias,
            residual_gate_init if shared_backbone_residual else -2.0,
        )
        if shared_backbone_residual:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        coverage_head_dim = bottleneck_channels + context_channels
        self.coverage_weight_head = nn.Sequential(
            nn.LayerNorm(coverage_head_dim),
            nn.Linear(coverage_head_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
                nn.Linear(128, 7),
        )
        self.coverage_residual_head = nn.Sequential(
            nn.LayerNorm(coverage_head_dim),
            nn.Linear(coverage_head_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.coverage_weight_head[-1].weight)
        if coverage_candidate_bias is None:
            coverage_candidate_bias = [0.0, 0.1, 0.5, 0.0, 0.1, 0.1, 1.0]
        if len(coverage_candidate_bias) != 7:
            raise ValueError("coverage_candidate_bias must contain seven values")
        with torch.no_grad():
            self.coverage_weight_head[-1].bias.copy_(
                torch.tensor(coverage_candidate_bias)
            )
        nn.init.zeros_(self.coverage_residual_head[-2].weight)
        nn.init.zeros_(self.coverage_residual_head[-2].bias)
        self.hybrid_coverage_head: Optional[HybridCoverageHead]
        if coverage_head_mode == "hybrid":
            self.hybrid_coverage_head = HybridCoverageHead(
                temporal_dim=temporal_hidden_channels,
                attractor_dim=bottleneck_channels,
                context_dim=context_channels,
                hidden_dim=coverage_dynamics_hidden,
                max_delta=max_coverage_residual,
                dropout=dropout,
            )
        else:
            self.hybrid_coverage_head = None
        self.temporal_anchor_head: Optional[TemporalAnchorCoverageHead]
        anchor_modes = {
            "temporal_anchor": "none",
            "anchor_residual": "fixed",
            "smooth_event_residual": "smooth_event",
        }
        if coverage_head_mode in anchor_modes:
            self.temporal_anchor_head = TemporalAnchorCoverageHead(
                context_dim=context_dim,
                temporal_dim=temporal_hidden_channels,
                attractor_dim=bottleneck_channels,
                encoded_context_dim=context_channels,
                correction_mode=anchor_modes[coverage_head_mode],
                correction_hidden_dim=coverage_dynamics_hidden,
                max_correction=coverage_correction_max,
                stable_correction=coverage_correction_stable,
                event_scale=coverage_correction_event_scale,
                dropout=dropout,
            )
        else:
            self.temporal_anchor_head = None
        self.coverage_dynamics_head: Optional[CoverageDynamicsHead]
        if coverage_head_mode in temporal_coverage_modes:
            self.coverage_dynamics_head = None
        elif context_dim >= input_steps + (input_steps - 1):
            self.coverage_dynamics_head = CoverageDynamicsHead(
                input_steps=input_steps,
                context_dim=context_dim,
                latent_dim=bottleneck_channels,
                encoded_context_dim=context_channels,
                hidden_dim=coverage_dynamics_hidden,
                expert_count=coverage_expert_count,
                max_delta=max_coverage_residual,
                stable_max_delta=coverage_stable_max_delta,
                delta_deadband=coverage_delta_deadband,
                weather_feature_dim=(
                    (
                        len(self.weather_selected_indices)
                        if weather_mode == "compact"
                        else weather_feature_dim
                    )
                    if coverage_weather_encoder
                    else 0
                ),
                mode=coverage_dynamics_mode,
                dropout=dropout,
            )
        else:
            if coverage_head_mode == "dynamics":
                raise ValueError(
                    "Dynamics coverage head requires coverage and change histories"
                )
            self.coverage_dynamics_head = None

    def _tokenize(self, features: torch.Tensor) -> torch.Tensor:
        return features.flatten(2).transpose(1, 2)

    @staticmethod
    def _spatialize(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), height, width)

    @staticmethod
    def _linear_trend(values: torch.Tensor) -> torch.Tensor:
        x = torch.arange(values.size(1), device=values.device, dtype=values.dtype)
        x = x - x.mean()
        slope = ((values - values.mean(dim=1, keepdim=True)) * x).sum(dim=1)
        slope = slope / x.square().sum().clamp_min(1e-6)
        return values[:, -1] + slope

    def _coverage_candidates(
        self,
        context: torch.Tensor,
        manifold_coverage: torch.Tensor,
    ) -> torch.Tensor:
        coverage_history = context[:, : self.input_steps]
        return torch.stack(
            [
                coverage_history[:, -1],
                coverage_history[:, -3:].mean(dim=1),
                coverage_history[:, -7:].mean(dim=1),
                coverage_history[:, -14:].mean(dim=1),
                self._linear_trend(coverage_history[:, -7:]),
                self._linear_trend(coverage_history),
                manifold_coverage,
            ],
            dim=1,
        ).clamp(0.0, 1.0)

    @staticmethod
    def _calibrate_map_to_coverage(
        prediction: torch.Tensor,
        target_coverage: torch.Tensor,
    ) -> torch.Tensor:
        target = target_coverage.clamp(0.0, 1.0)
        logits = torch.logit(prediction.clamp(1e-6, 1.0 - 1e-6))
        lower = logits.new_full((logits.size(0),), -20.0)
        upper = logits.new_full((logits.size(0),), 20.0)
        with torch.no_grad():
            for _ in range(28):
                midpoint = (lower + upper) * 0.5
                current = torch.sigmoid(
                    logits + midpoint[:, None, None, None]
                ).flatten(1).mean(dim=1)
                lower = torch.where(current < target, midpoint, lower)
                upper = torch.where(current >= target, midpoint, upper)
        bias = ((lower + upper) * 0.5).detach()
        calibrated = torch.sigmoid(logits + bias[:, None, None, None])
        calibrated = torch.where(
            (target <= 0.0)[:, None, None, None],
            torch.zeros_like(calibrated),
            calibrated,
        )
        return torch.where(
            (target >= 1.0)[:, None, None, None],
            torch.ones_like(calibrated),
            calibrated,
        )

    @staticmethod
    def _ratio_calibrate_map_to_coverage(
        prediction: torch.Tensor,
        target_coverage: torch.Tensor,
    ) -> torch.Tensor:
        current = prediction.flatten(1).mean(dim=1)
        target = target_coverage.clamp(0.0, 1.0)
        lower = prediction * (target / current.clamp_min(1e-6))[
            :, None, None, None
        ]
        upper = 1.0 - (1.0 - prediction) * (
            (1.0 - target) / (1.0 - current).clamp_min(1e-6)
        )[:, None, None, None]
        return torch.where(
            (target <= current)[:, None, None, None],
            lower,
            upper,
        ).clamp(0.0, 1.0)

    def _apply_coverage_calibration(
        self,
        prediction: torch.Tensor,
        target_coverage: torch.Tensor,
    ) -> torch.Tensor:
        if self.coverage_calibration_mode == "ratio_v1":
            return self._ratio_calibrate_map_to_coverage(
                prediction,
                target_coverage,
            )
        return self._calibrate_map_to_coverage(prediction, target_coverage)

    def _weather_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.weather_feature_dim <= 0:
            return context, None
        base_dim = self.input_steps + (self.input_steps - 1) + 37 + 4
        required = base_dim + self.weather_feature_dim * 4
        if context.size(1) < required:
            return context, None
        sanitized = context.clone()
        weather = context[:, base_dim:required].reshape(
            context.size(0),
            4,
            self.weather_feature_dim,
        )
        if self.weather_mode == "no_target":
            weather = weather.clone()
            weather[:, 2:] = 0
        elif self.weather_mode == "compact":
            indices = torch.as_tensor(
                self.weather_selected_indices,
                device=context.device,
            )
            compact = weather.index_select(2, indices)
            masked = torch.zeros_like(weather)
            masked.index_copy_(2, indices, compact)
            weather = masked
        sanitized[:, base_dim:required] = weather.flatten(1)
        head_weather = (
            weather.index_select(
                2,
                torch.as_tensor(self.weather_selected_indices, device=context.device),
            )
            if self.weather_mode == "compact"
            else weather
        )
        return sanitized, head_weather.flatten(1)

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
        if ablation_mode not in {
            "none",
            "drop_temporal",
            "drop_attractor_coverage",
        }:
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        batch, steps, channels, height, width = inputs.shape
        if (steps, channels) != (self.input_steps, self.input_channels):
            raise ValueError(
                f"Expected temporal shape {(self.input_steps, self.input_channels)}, "
                f"received {(steps, channels)}"
            )
        resized_sequence = F.interpolate(
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
        resized = resized_sequence.reshape(
            batch,
            steps * channels,
            self.internal_size,
            self.internal_size,
        )
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
                    f"Expected {self.spatial_context_channels} spatial context channels, "
                    f"received {spatial_context.size(1)}"
                )
            spatial_context = F.interpolate(
                spatial_context,
                size=(self.internal_size, self.internal_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.spatial_context_mode != "all":
                spatial_context = spatial_context.clone()
                if self.spatial_context_mode == "no_coordinates":
                    spatial_context[:, :2] = 0
                elif self.spatial_context_mode == "no_terrain":
                    spatial_context[:, 2:7] = 0
                elif self.spatial_context_mode == "no_weather":
                    spatial_context[:, 7:] = 0
                elif self.spatial_context_mode == "none":
                    spatial_context.zero_()
        model_context, weather_context = self._weather_context(context)
        encoded_context = self.context_encoder(model_context)
        fused_parts = [resized, prior]
        if self.spatial_context_channels:
            assert spatial_context is not None
            fused_parts.append(spatial_context)
        fused = torch.cat(fused_parts, dim=1)
        temporal_hidden = resized.new_zeros(
            batch,
            self.temporal_hidden_channels,
            self.internal_size // 4,
            self.internal_size // 4,
        )
        temporal_global_state = temporal_hidden.mean(dim=(2, 3))
        if self.temporal_encoder is not None and ablation_mode != "drop_temporal":
            condition_parts = [prior]
            if self.spatial_context_channels:
                assert spatial_context is not None
                condition_parts.append(spatial_context)
            temporal_hidden, temporal_global_state = self.temporal_encoder(
                resized_sequence,
                torch.cat(condition_parts, dim=1),
            )
        last_map = resized[:, -channels:-channels + 1] if channels > 1 else resized[:, -1:]
        if self.shared_backbone_only:
            assert self.backbone_decoder is not None
            backbone_delta = self.max_residual * torch.tanh(
                self.backbone_decoder(temporal_hidden)
            )
            prediction = (last_map + backbone_delta).clamp(0.0, 1.0)
            prediction = F.interpolate(
                prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            effective_mask = coverage_mask if coverage_mask is not None else land_mask
            if effective_mask is None:
                effective_mask = torch.ones_like(prediction)
            else:
                effective_mask = effective_mask.to(dtype=prediction.dtype)
                if effective_mask.shape[-2:] != prediction.shape[-2:]:
                    effective_mask = F.interpolate(
                        effective_mask,
                        size=prediction.shape[-2:],
                        mode="nearest",
                    )
            denominator = effective_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            coverage_prediction = (
                prediction * effective_mask
            ).flatten(1).sum(dim=1) / denominator
            last_coverage = context[:, self.input_steps - 1].clamp(0.0, 1.0)
            zero = prediction.new_zeros(batch)
            zero_state = prediction.new_zeros(batch, self.bottleneck_channels)
            return {
                "prediction": prediction,
                "raw_prediction": prediction,
                "coverage_prediction": coverage_prediction,
                "coverage_delta": coverage_prediction - last_coverage,
                "coverage_delta_raw": coverage_prediction - last_coverage,
                "coverage_regime_logits": prediction.new_zeros(batch, 0),
                "coverage_regime_weights": prediction.new_zeros(batch, 0),
                "coverage_expert_deltas": prediction.new_zeros(batch, 0),
                "coverage_weather_gate": prediction.new_zeros(batch, 0),
                "coverage_temporal_residual": coverage_prediction - last_coverage,
                "coverage_temporal_base": coverage_prediction,
                "coverage_attractor_correction": zero,
                "coverage_correction_scale": zero,
                "temporal_global_state": temporal_global_state,
                "temporal_fusion_gate": prediction.new_zeros(
                    batch,
                    self.bottleneck_channels,
                    self.internal_size // 8,
                    self.internal_size // 8,
                ),
                "coverage_candidates": prediction.new_zeros(batch, 0),
                "coverage_weights": prediction.new_zeros(batch, 0),
                "coverage_residual": zero,
                "accumulation_gate": zero,
                "last_input_coverage": last_coverage,
                "input_steps": self.input_steps,
                "energy": prediction.new_zeros(batch, 1),
                "manifold_energy": prediction.new_zeros(batch, 1),
                "manifold_distance": prediction.new_zeros(batch, 1),
                "manifold_coordinate": torch.stack(
                    [coverage_prediction, coverage_prediction - last_coverage],
                    dim=1,
                ),
                "memory_tokens": prediction.new_zeros(batch, 0, 0),
                "manifold_first_order": prediction.new_zeros(()),
                "manifold_second_order": prediction.new_zeros(()),
                "clean_manifold_state": zero_state,
                "noisy_manifold_state": zero_state,
                "auxiliary_predictions": [prediction],
                "attention": prediction.new_zeros(batch, 0, 0, 0),
                "update_norms": prediction.new_zeros(batch, 0),
                "residual_prediction": prediction,
                "direct_prediction": prediction,
                "fusion_gate": torch.zeros_like(prediction),
                "residual_gate": torch.zeros_like(prediction),
                "backbone_prediction": prediction,
            }

        enc1 = self.encoder_films[0](self.encoder1(fused), encoded_context)
        enc2 = self.encoder_films[1](self.encoder2(self.pool(enc1)), encoded_context)
        enc3 = self.encoder_films[2](self.encoder3(self.pool(enc2)), encoded_context)
        bottleneck = self.encoder_films[3](
            self.bottleneck(self.pool(enc3)),
            encoded_context,
        )
        temporal_fusion_gate = bottleneck.new_zeros(bottleneck.shape)
        if (
            self.temporal_fusion_mode == "gated_bottleneck"
            and ablation_mode != "drop_temporal"
        ):
            assert self.temporal_projector is not None
            assert self.temporal_gate is not None
            temporal_features = self.temporal_projector(temporal_hidden)
            if temporal_features.shape[-2:] != bottleneck.shape[-2:]:
                temporal_features = F.interpolate(
                    temporal_features,
                    size=bottleneck.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            temporal_fusion_gate = torch.sigmoid(
                self.temporal_gate(
                    torch.cat([bottleneck, temporal_features], dim=1)
                )
            )
            bottleneck = bottleneck + temporal_fusion_gate * temporal_features

        token_height, token_width = bottleneck.shape[-2:]
        state = self._tokenize(bottleneck)
        coordinate_features = torch.cat([state.mean(dim=1), encoded_context], dim=1)
        last_coverage = context[:, self.input_steps - 1].clamp(0.0, 1.0)
        coordinate_value = self.coordinate_head(coordinate_features).squeeze(1)
        if self.coordinate_mode == "residual":
            manifold_change = self.change_scale * torch.tanh(coordinate_value)
            manifold_coverage = (last_coverage + manifold_change).clamp(0.0, 1.0)
        else:
            manifold_coverage = torch.sigmoid(coordinate_value)
        manifold_change = manifold_coverage - last_coverage
        memory_change = (
            manifold_change
            if self.use_change_coordinate
            else torch.zeros_like(manifold_change)
        )
        manifold_coordinate = torch.stack(
            [manifold_coverage, memory_change],
            dim=1,
        )
        uses_memory = self.attractor_mode in {"continuous", "nearest"}
        memory_tokens = (
            self.manifold_memory.retrieve(manifold_coordinate, self.attractor_mode)
            if uses_memory
            else state.new_zeros(batch, 0, state.size(-1))
        )
        if state_noise_std > 0:
            state = state + state_noise_std * torch.randn_like(state)

        def manifold_distance(current: torch.Tensor) -> torch.Tensor:
            if not uses_memory:
                return current.new_zeros(current.size(0))
            return (
                current.mean(dim=1) - memory_tokens.mean(dim=1)
            ).square().mean(dim=1)

        energy_values = [manifold_distance(state)]
        auxiliary_predictions = [
            torch.sigmoid(self.auxiliary_head(bottleneck))
        ]
        update_norms: list[torch.Tensor] = []
        attention = state.new_zeros(batch, 1, state.size(1), state.size(1))

        for _ in range(self.attractor_iterations):
            previous = state
            state, attention = self.attractor(
                state,
                memory_tokens if uses_memory else None,
            )
            update_norms.append(
                (state - previous).flatten(1).norm(dim=1)
                / previous.flatten(1).norm(dim=1).clamp_min(1e-6)
            )
            state_map = self._spatialize(state, token_height, token_width)
            auxiliary_predictions.append(torch.sigmoid(self.auxiliary_head(state_map)))
            energy_values.append(manifold_distance(state))

        clean_manifold_state = state.mean(dim=1)
        noisy_manifold_state = clean_manifold_state.detach()
        if self.training and uses_memory and self.manifold_noise_std > 0:
            noisy_state = self._tokenize(bottleneck)
            noisy_state = noisy_state + self.manifold_noise_std * torch.randn_like(noisy_state)
            for _ in range(self.attractor_iterations):
                noisy_state, _ = self.attractor(noisy_state, memory_tokens)
            noisy_manifold_state = noisy_state.mean(dim=1)

        refined = self._spatialize(state, token_height, token_width)
        dec3 = F.interpolate(refined, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.decoder_films[0](
            self.decoder3(torch.cat([dec3, enc3], dim=1)),
            encoded_context,
        )
        dec2 = F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.decoder_films[1](
            self.decoder2(torch.cat([dec2, enc2], dim=1)),
            encoded_context,
        )
        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.decoder_films[2](
            self.decoder1(torch.cat([dec1, enc1], dim=1)),
            encoded_context,
        )

        residual_delta = self.max_residual * torch.tanh(self.residual_head(dec1))
        residual_prediction = (last_map + residual_delta).clamp(0.0, 1.0)
        direct_prediction = torch.sigmoid(self.direct_head(dec1))
        fusion_gate = torch.sigmoid(self.fusion_gate(dec1))
        backbone_prediction = residual_prediction
        if self.shared_backbone_residual:
            assert self.backbone_decoder is not None
            backbone_delta = self.max_residual * torch.tanh(
                self.backbone_decoder(temporal_hidden)
            )
            backbone_prediction = (last_map + backbone_delta).clamp(0.0, 1.0)
            raw_prediction = (
                backbone_prediction + fusion_gate * residual_delta
            ).clamp(0.0, 1.0)
        else:
            raw_prediction = (
                (1.0 - fusion_gate) * residual_prediction
                + fusion_gate * direct_prediction
            )
        global_state = state.mean(dim=1)
        coverage_features = torch.cat([global_state, encoded_context], dim=1)
        coverage_candidates = self._coverage_candidates(context, manifold_coverage)
        coverage_weights = F.softmax(self.coverage_weight_head(coverage_features), dim=1)
        base_coverage = (coverage_candidates * coverage_weights).sum(dim=1)
        coverage_residual = (
            self.max_coverage_residual
            * self.coverage_residual_head(coverage_features).squeeze(1)
        )
        legacy_coverage_prediction = (base_coverage + coverage_residual).clamp(0.0, 1.0)
        accumulation_gate = torch.sigmoid((manifold_change - 0.005) / 0.01)
        accumulation_blend = self.accumulation_blend * accumulation_gate
        legacy_coverage_prediction = (
            (1.0 - accumulation_blend) * legacy_coverage_prediction
            + accumulation_blend * manifold_coverage
        ).clamp(0.0, 1.0)
        if self.coverage_head_mode == "hybrid":
            assert self.hybrid_coverage_head is not None
            coverage_global_state = global_state
            coverage_encoded_context = encoded_context
            coverage_manifold_coordinate = manifold_coordinate
            if ablation_mode == "drop_attractor_coverage":
                coverage_global_state = torch.zeros_like(global_state)
                coverage_encoded_context = torch.zeros_like(encoded_context)
                coverage_manifold_coordinate = torch.zeros_like(manifold_coordinate)
            dynamics = self.hybrid_coverage_head(
                temporal_global_state,
                scale_gradient(
                    coverage_global_state,
                    self.coverage_gradient_scale,
                ),
                scale_gradient(
                    coverage_encoded_context,
                    self.coverage_gradient_scale,
                ),
                scale_gradient(
                    coverage_manifold_coordinate,
                    self.coverage_gradient_scale,
                ),
                last_coverage,
            )
            coverage_prediction = dynamics["coverage_prediction"]
            dynamics.update(
                {
                    "coverage_regime_logits": coverage_prediction.new_zeros(batch, 0),
                    "coverage_regime_weights": coverage_prediction.new_zeros(batch, 0),
                    "coverage_expert_deltas": coverage_prediction.new_zeros(batch, 0),
                    "coverage_weather_gate": coverage_prediction.new_zeros(batch, 0),
                }
            )
        elif self.coverage_head_mode in {
            "temporal_anchor",
            "anchor_residual",
            "smooth_event_residual",
        }:
            assert self.temporal_anchor_head is not None
            temporal_last_coverage = inputs[:, -1, 0].flatten(1).mean(dim=1)
            dynamics = self.temporal_anchor_head(
                temporal_global_state,
                model_context,
                scale_gradient(global_state, self.coverage_gradient_scale),
                scale_gradient(encoded_context, self.coverage_gradient_scale),
                scale_gradient(manifold_coordinate, self.coverage_gradient_scale),
                temporal_last_coverage,
            )
            last_coverage = temporal_last_coverage
            coverage_prediction = dynamics["coverage_prediction"]
            dynamics.update(
                {
                    "coverage_regime_logits": coverage_prediction.new_zeros(batch, 0),
                    "coverage_regime_weights": coverage_prediction.new_zeros(batch, 0),
                    "coverage_expert_deltas": coverage_prediction.new_zeros(batch, 0),
                    "coverage_weather_gate": coverage_prediction.new_zeros(batch, 0),
                }
            )
        elif self.coverage_head_mode == "dynamics":
            assert self.coverage_dynamics_head is not None
            dynamics = self.coverage_dynamics_head(
                model_context,
                scale_gradient(global_state, self.coverage_gradient_scale),
                scale_gradient(encoded_context, self.coverage_gradient_scale),
                scale_gradient(manifold_coordinate, self.coverage_gradient_scale),
                weather_context,
            )
            coverage_prediction = dynamics["coverage_prediction"]
        else:
            coverage_prediction = legacy_coverage_prediction
            dynamics = {
                "coverage_delta": coverage_prediction - last_coverage,
                "coverage_delta_raw": coverage_prediction - last_coverage,
                "coverage_regime_logits": coverage_weights[:, :3].clamp_min(1e-8).log(),
                "coverage_regime_weights": coverage_weights[:, :3],
                "coverage_expert_deltas": coverage_candidates[:, :3] - last_coverage[:, None],
                "coverage_weather_gate": coverage_prediction.new_zeros(batch, 0),
            }
        prediction = raw_prediction
        calibration_coverage = (
            coverage_prediction.detach()
            if self.coverage_calibration_detach
            else coverage_prediction
        )
        if (
            self.coverage_calibration
            and not self.shared_backbone_residual
            and self.coverage_calibration_passes == 2
        ):
            prediction = self._apply_coverage_calibration(
                prediction,
                calibration_coverage,
            )
        prediction = F.interpolate(
            prediction,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if self.coverage_calibration and not self.shared_backbone_residual:
            prediction = self._apply_coverage_calibration(
                prediction,
                calibration_coverage,
            )
        backbone_prediction = F.interpolate(
            backbone_prediction,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if self.shared_backbone_residual:
            effective_mask = coverage_mask if coverage_mask is not None else land_mask
            if effective_mask is None:
                coverage_mask = torch.ones_like(prediction)
            else:
                coverage_mask = effective_mask.to(dtype=prediction.dtype)
                if coverage_mask.shape[-2:] != prediction.shape[-2:]:
                    coverage_mask = F.interpolate(
                        coverage_mask,
                        size=prediction.shape[-2:],
                        mode="nearest",
                    )
            denominator = coverage_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            coverage_prediction = (
                prediction * coverage_mask
            ).flatten(1).sum(dim=1) / denominator
            backbone_coverage = (
                backbone_prediction * coverage_mask
            ).flatten(1).sum(dim=1) / denominator
            dynamics["coverage_delta"] = coverage_prediction - last_coverage
            dynamics["coverage_delta_raw"] = dynamics["coverage_delta"]
            dynamics["coverage_temporal_base"] = backbone_coverage
        manifold_first_order, manifold_second_order = self.manifold_memory.regularization()

        return {
            "prediction": prediction,
            "raw_prediction": F.interpolate(
                raw_prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ),
            "coverage_prediction": coverage_prediction,
            "coverage_delta": dynamics["coverage_delta"],
            "coverage_delta_raw": dynamics["coverage_delta_raw"],
            "coverage_regime_logits": dynamics["coverage_regime_logits"],
            "coverage_regime_weights": dynamics["coverage_regime_weights"],
            "coverage_expert_deltas": dynamics["coverage_expert_deltas"],
            "coverage_weather_gate": dynamics["coverage_weather_gate"],
            "coverage_temporal_residual": dynamics["coverage_delta"],
            "coverage_temporal_base": dynamics.get(
                "coverage_temporal_base",
                coverage_prediction,
            ),
            "coverage_attractor_correction": dynamics.get(
                "coverage_attractor_correction",
                coverage_prediction.new_zeros(batch),
            ),
            "coverage_correction_scale": dynamics.get(
                "coverage_correction_scale",
                coverage_prediction.new_zeros(batch),
            ),
            "temporal_global_state": temporal_global_state,
            "temporal_fusion_gate": temporal_fusion_gate,
            "coverage_candidates": coverage_candidates,
            "coverage_weights": coverage_weights,
            "coverage_residual": coverage_residual,
            "accumulation_gate": accumulation_gate,
            "last_input_coverage": last_coverage,
            "input_steps": self.input_steps,
            "energy": torch.stack(energy_values, dim=1),
            "manifold_energy": torch.stack(energy_values, dim=1),
            "manifold_distance": torch.stack(energy_values, dim=1),
            "manifold_coordinate": manifold_coordinate,
            "memory_tokens": memory_tokens,
            "manifold_first_order": manifold_first_order,
            "manifold_second_order": manifold_second_order,
            "clean_manifold_state": clean_manifold_state,
            "noisy_manifold_state": noisy_manifold_state,
            "auxiliary_predictions": auxiliary_predictions,
            "attention": attention,
            "update_norms": (
                torch.stack(update_norms, dim=1)
                if update_norms
                else prediction.new_zeros(batch, 0)
            ),
            "residual_prediction": F.interpolate(
                residual_prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ),
            "direct_prediction": F.interpolate(
                direct_prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ),
            "fusion_gate": F.interpolate(
                fusion_gate,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ),
            "residual_gate": F.interpolate(
                fusion_gate,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ),
            "backbone_prediction": backbone_prediction,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
