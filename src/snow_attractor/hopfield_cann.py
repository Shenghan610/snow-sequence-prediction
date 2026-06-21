"""Hopfield continuous-attractor network for QA-aware snow forecasting."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConvBlock, _group_count


def _masked_coverage(
    prediction: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return prediction.flatten(1).mean(dim=1)
    mask = mask.to(dtype=prediction.dtype)
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask, size=prediction.shape[-2:], mode="nearest")
    return (prediction * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


class SeasonalContinuousAttractorMemory(nn.Module):
    """Continuous memory field over coverage, change, and circular season."""

    def __init__(
        self,
        channels: int,
        coverage_bins: int,
        change_bins: int,
        season_bins: int,
        memory_tokens: int,
        change_scale: float,
    ) -> None:
        super().__init__()
        if coverage_bins < 2 or change_bins < 2 or season_bins < 3:
            raise ValueError("Attractor memory needs >=2 coverage/change and >=3 season bins")
        self.coverage_bins = coverage_bins
        self.change_bins = change_bins
        self.season_bins = season_bins
        self.memory_tokens = memory_tokens
        self.change_scale = change_scale
        self.anchors = nn.Parameter(
            torch.empty(
                coverage_bins,
                change_bins,
                season_bins,
                memory_tokens,
                channels,
            )
        )
        nn.init.normal_(self.anchors, std=0.02)
        template = torch.zeros(
            coverage_bins,
            change_bins,
            season_bins,
            1,
            channels,
        )
        if channels >= 4:
            coverage_axis = torch.linspace(0.0, 0.1, coverage_bins)
            change_axis = torch.linspace(-0.1, 0.1, change_bins)
            season_axis = torch.linspace(
                0.0,
                2.0 * math.pi,
                season_bins + 1,
            )[:-1]
            template[..., 0] = coverage_axis.view(coverage_bins, 1, 1, 1)
            template[..., 1] = change_axis.view(1, change_bins, 1, 1)
            template[..., 2] = 0.05 * torch.sin(season_axis).view(
                1,
                1,
                season_bins,
                1,
            )
            template[..., 3] = 0.05 * torch.cos(season_axis).view(
                1,
                1,
                season_bins,
                1,
            )
        self.register_buffer("coordinate_template", template)

    def anchor_values(self) -> torch.Tensor:
        return self.anchors + self.coordinate_template

    def _grid_coordinates(
        self,
        coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coverage = coordinate[:, 0].clamp(0.0, 1.0) * (self.coverage_bins - 1)
        change = (
            (coordinate[:, 1].clamp(-self.change_scale, self.change_scale)
             / self.change_scale + 1.0)
            * 0.5
            * (self.change_bins - 1)
        )
        season = torch.atan2(coordinate[:, 2], coordinate[:, 3])
        season = torch.remainder(season, 2.0 * math.pi) / (2.0 * math.pi)
        return coverage, change, season * self.season_bins

    def retrieve(self, coordinate: torch.Tensor) -> torch.Tensor:
        coverage, change, season = self._grid_coordinates(coordinate)
        c0 = coverage.floor().long()
        d0 = change.floor().long()
        s0 = season.floor().long() % self.season_bins
        c1 = (c0 + 1).clamp_max(self.coverage_bins - 1)
        d1 = (d0 + 1).clamp_max(self.change_bins - 1)
        s1 = (s0 + 1) % self.season_bins
        wc = (coverage - c0).view(-1, 1, 1)
        wd = (change - d0).view(-1, 1, 1)
        ws = (season - season.floor()).view(-1, 1, 1)
        anchors = self.anchor_values()

        def read(ci: torch.Tensor, di: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
            return anchors[ci, di, si]

        c00 = read(c0, d0, s0) * (1.0 - wc) + read(c1, d0, s0) * wc
        c01 = read(c0, d0, s1) * (1.0 - wc) + read(c1, d0, s1) * wc
        c10 = read(c0, d1, s0) * (1.0 - wc) + read(c1, d1, s0) * wc
        c11 = read(c0, d1, s1) * (1.0 - wc) + read(c1, d1, s1) * wc
        lower = c00 * (1.0 - wd) + c10 * wd
        upper = c01 * (1.0 - wd) + c11 * wd
        return lower * (1.0 - ws) + upper * ws

    def regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        first = [
            self.anchors[1:] - self.anchors[:-1],
            self.anchors[:, 1:] - self.anchors[:, :-1],
            torch.roll(self.anchors, shifts=-1, dims=2) - self.anchors,
        ]
        first_order = torch.stack([item.square().mean() for item in first]).mean()
        second = [
            self.anchors[2:] - 2.0 * self.anchors[1:-1] + self.anchors[:-2],
            self.anchors[:, 2:] - 2.0 * self.anchors[:, 1:-1] + self.anchors[:, :-2],
            torch.roll(self.anchors, shifts=-1, dims=2)
            - 2.0 * self.anchors
            + torch.roll(self.anchors, shifts=1, dims=2),
        ]
        second_order = torch.stack([item.square().mean() for item in second]).mean()
        return first_order, second_order


class HopfieldAssociativeMemory(nn.Module):
    """Modern Hopfield-style associative retrieval over episodic and prototype tokens."""

    def __init__(
        self,
        channels: int,
        prototype_count: int,
        beta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.prototype_count = prototype_count
        self.beta = beta
        self.prototypes = nn.Parameter(torch.empty(prototype_count, channels))
        if prototype_count:
            nn.init.normal_(self.prototypes, std=0.02)
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.output = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

    def forward(
        self,
        query: torch.Tensor,
        episode: torch.Tensor,
        confidence: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = query.size(0)
        memory_parts = [episode]
        if self.prototype_count:
            memory_parts.append(self.prototypes.unsqueeze(0).expand(batch, -1, -1))
        memory = torch.cat(memory_parts, dim=1)
        scores = (
            self.beta
            * torch.matmul(
                self.query_norm(query),
                self.memory_norm(memory).transpose(1, 2),
            )
            / math.sqrt(query.size(-1))
        )
        if confidence is not None:
            prior_parts = [confidence.clamp_min(1e-4)]
            if self.prototype_count:
                prior_parts.append(confidence.new_ones(batch, self.prototype_count))
            prior = torch.cat(prior_parts, dim=1)
            scores = scores + prior.log().unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, memory)
        retrieved = self.output(retrieved)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(max(weights.size(-1), 2))
        prototype_mass = (
            weights[..., -self.prototype_count :].sum(dim=-1).mean(dim=1)
            if self.prototype_count
            else weights.new_zeros(batch)
        )
        return retrieved, entropy.mean(dim=1), weights.mean(dim=1), prototype_mass


class HopfieldCANNUpdateBlock(nn.Module):
    """Shared recurrent update combining self-attention, Hopfield retrieval, and CANN pull."""

    def __init__(
        self,
        channels: int,
        heads: int,
        prototype_count: int,
        dropout: float,
        attraction_rate_init: float,
        hopfield_beta: float,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.hopfield = HopfieldAssociativeMemory(
            channels,
            prototype_count,
            hopfield_beta,
            dropout,
        )
        self.attractor_norm = nn.LayerNorm(channels)
        self.attractor_attention = nn.MultiheadAttention(
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
        episode: torch.Tensor,
        confidence: torch.Tensor,
        attractor_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.step_size.clamp(0.02, 0.30)
        self_update, attention = self.self_attention(
            self.self_norm(state),
            self.self_norm(state),
            self.self_norm(state),
            need_weights=True,
            average_attn_weights=False,
        )
        hopfield_update, entropy, retrieval, prototype_mass = self.hopfield(
            state,
            episode,
            confidence,
        )
        attractor_update, _ = self.attractor_attention(
            self.attractor_norm(state),
            attractor_tokens,
            attractor_tokens,
            need_weights=False,
        )
        state = state + step * (self_update + hopfield_update + attractor_update)
        attraction = self.attraction_rate.clamp(0.01, 0.25)
        state = state + attraction * (
            attractor_tokens.mean(dim=1, keepdim=True) - state.mean(dim=1, keepdim=True)
        )
        state = state + step * self.ffn(self.ffn_norm(state))
        return state, entropy, retrieval, prototype_mass, attention


class HopfieldCANNForecastNet(nn.Module):
    """Next-day snow-map forecaster driven by Hopfield memory and CANN dynamics."""

    def __init__(
        self,
        input_steps: int = 20,
        input_channels: int = 3,
        context_dim: int = 60,
        base_channels: int = 32,
        bottleneck_channels: int = 256,
        attractor_iterations: int = 4,
        attention_heads: int = 8,
        internal_size: int = 128,
        max_residual: float = 0.40,
        max_coverage_residual: float = 0.12,
        spatial_context_channels: int = 0,
        coverage_bins: int = 9,
        change_bins: int = 9,
        season_bins: int = 12,
        memory_tokens: int = 8,
        hopfield_prototypes: int = 8,
        hopfield_beta: float = 1.0,
        change_scale: float = 0.20,
        coordinate_step_scale: float = 0.04,
        manifold_noise_std: float = 0.05,
        attraction_rate_init: float = 0.05,
        spatial_context_mode: str = "all",
        use_hopfield_episode: bool = True,
        use_hopfield_prototypes: bool = True,
        use_continuous_attractor: bool = True,
        use_season_coordinate: bool = True,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if internal_size % 8:
            raise ValueError("internal_size must be divisible by 8")
        if spatial_context_mode not in {
            "all",
            "no_coordinates",
            "no_terrain",
            "no_weather",
            "none",
        }:
            raise ValueError(f"Unsupported spatial_context_mode: {spatial_context_mode}")
        self.input_steps = input_steps
        self.input_channels = input_channels
        self.context_dim = context_dim
        self.bottleneck_channels = bottleneck_channels
        self.attractor_iterations = attractor_iterations
        self.internal_size = internal_size
        self.max_residual = max_residual
        self.max_coverage_residual = max_coverage_residual
        self.spatial_context_channels = spatial_context_channels
        self.change_scale = change_scale
        self.coordinate_step_scale = coordinate_step_scale
        self.manifold_noise_std = manifold_noise_std
        self.spatial_context_mode = spatial_context_mode
        self.use_hopfield_episode = use_hopfield_episode
        self.use_hopfield_prototypes = use_hopfield_prototypes
        self.use_continuous_attractor = use_continuous_attractor
        self.use_season_coordinate = use_season_coordinate
        context_channels = 256
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_channels),
            nn.GELU(),
            nn.Linear(context_channels, context_channels),
            nn.GELU(),
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_encoder = nn.Sequential(
            ConvBlock(input_channels + condition_channels, c1, dropout * 0.25),
            nn.MaxPool2d(2),
            ConvBlock(c1, c2, dropout * 0.35),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3, dropout * 0.50),
            nn.MaxPool2d(2),
            ConvBlock(c3, bottleneck_channels, dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=bottleneck_channels,
            hidden_size=bottleneck_channels,
            batch_first=True,
        )
        self.temporal_to_map = nn.Linear(bottleneck_channels, bottleneck_channels)
        self.state_film = nn.Linear(context_channels, bottleneck_channels * 2)
        nn.init.zeros_(self.state_film.weight)
        nn.init.zeros_(self.state_film.bias)
        self.attractor_memory = SeasonalContinuousAttractorMemory(
            bottleneck_channels,
            coverage_bins,
            change_bins,
            season_bins,
            memory_tokens,
            change_scale,
        )
        self.update_block = HopfieldCANNUpdateBlock(
            bottleneck_channels,
            attention_heads,
            hopfield_prototypes if use_hopfield_prototypes else 0,
            dropout,
            attraction_rate_init,
            hopfield_beta,
        )
        coordinate_dim = bottleneck_channels + context_channels + 4
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
        self.auxiliary_head = nn.Conv2d(bottleneck_channels, 1, 1)
        self.decoder = nn.Sequential(
            ConvBlock(bottleneck_channels, c3, dropout * 0.50),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c3, c2, dropout * 0.35),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c2, c1, dropout * 0.25),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, 1, 1),
        )
        self.direct_decoder = nn.Sequential(
            ConvBlock(bottleneck_channels, c2, dropout * 0.35),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c2, c1, dropout * 0.25),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, 1, 1),
        )
        self.fusion_gate = nn.Conv2d(1, 1, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -2.0)
        self.decoder_norm = nn.GroupNorm(_group_count(bottleneck_channels), bottleneck_channels)

    def _tokenize(self, features: torch.Tensor) -> torch.Tensor:
        return features.flatten(2).transpose(1, 2)

    @staticmethod
    def _spatialize(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), height, width)

    def _prepare_spatial_context(
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
            condition = torch.cat([prior, spatial_context], dim=1)
        else:
            condition = prior
        return prior, condition

    def _season_features(self, context: torch.Tensor) -> torch.Tensor:
        if not self.use_season_coordinate:
            fallback = context.new_zeros(context.size(0), 2)
            fallback[:, 1] = 1.0
            return fallback
        calendar_start = self.input_steps + (self.input_steps - 1) + 37
        if context.size(1) >= calendar_start + 2:
            season = context[:, calendar_start : calendar_start + 2]
            norm = season.norm(dim=1, keepdim=True).clamp_min(1e-6)
            return season / norm
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
        coordinate_context = torch.cat(
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
                self.coordinate_head(coordinate_context).squeeze(1)
            )
        else:
            change = (
                previous
                + self.coordinate_step_scale
                * torch.tanh(self.coordinate_refiner(coordinate_context).squeeze(1))
            ).clamp(-self.change_scale, self.change_scale)
        coverage = (last_coverage + change).clamp(0.0, 1.0)
        return torch.cat([coverage[:, None], change[:, None], season], dim=1)

    def _episode_confidence(
        self,
        resized_sequence: torch.Tensor,
        token_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, steps, channels, _, _ = resized_sequence.shape
        if channels >= 3:
            observed = resized_sequence[:, :, 1:2].clamp(0.0, 1.0)
            age = resized_sequence[:, :, 2:3].clamp(0.0, 1.0)
            confidence = observed * (1.0 - age)
        else:
            confidence = resized_sequence.new_ones(
                batch,
                steps,
                1,
                self.internal_size,
                self.internal_size,
            )
        confidence = F.interpolate(
            confidence.reshape(batch * steps, 1, self.internal_size, self.internal_size),
            size=token_size,
            mode="area",
        )
        return confidence.reshape(batch, steps * token_size[0] * token_size[1])

    def _decode_prediction(
        self,
        state: torch.Tensor,
        token_height: int,
        token_width: int,
        last_map: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        refined = self._spatialize(state, token_height, token_width)
        refined = self.decoder_norm(refined)
        residual = self.max_residual * torch.tanh(self.decoder(refined))
        if residual.shape[-2:] != last_map.shape[-2:]:
            residual = F.interpolate(
                residual,
                size=last_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        residual_prediction = (last_map + residual).clamp(0.0, 1.0)
        direct = torch.sigmoid(self.direct_decoder(refined))
        if direct.shape[-2:] != last_map.shape[-2:]:
            direct = F.interpolate(
                direct,
                size=last_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = torch.sigmoid(self.fusion_gate(residual))
        raw = ((1.0 - gate) * residual_prediction + gate * direct).clamp(0.0, 1.0)
        prediction = F.interpolate(
            raw,
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
        direct = F.interpolate(
            direct,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        gate = F.interpolate(gate, size=(height, width), mode="bilinear", align_corners=False)
        return prediction, residual_prediction, direct, gate

    def _run_recurrence(
        self,
        state: torch.Tensor,
        episode: torch.Tensor,
        confidence: torch.Tensor,
        encoded_context: torch.Tensor,
        last_coverage: torch.Tensor,
        season: torch.Tensor,
        token_height: int,
        token_width: int,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        coordinate = self._coordinate(
            state.mean(dim=1),
            encoded_context,
            last_coverage,
            season,
        )
        trajectory = [coordinate]
        def retrieve_attractor(
            current_state: torch.Tensor,
            current_coordinate: torch.Tensor,
        ) -> torch.Tensor:
            if self.use_continuous_attractor:
                return self.attractor_memory.retrieve(current_coordinate)
            return current_state.mean(dim=1, keepdim=True).detach().expand(
                -1,
                self.attractor_memory.memory_tokens,
                -1,
            )

        attractor_tokens = retrieve_attractor(state, coordinate)

        def energy(current_state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
            return (
                current_state.mean(dim=1) - memory.mean(dim=1)
            ).square().mean(dim=1)

        energies = [energy(state, attractor_tokens)]
        entropies = []
        prototype_masses = []
        retrieval = state.new_zeros(state.size(0), episode.size(1))
        attention = state.new_zeros(state.size(0), 1, state.size(1), state.size(1))
        update_norms = []
        auxiliary = [
            torch.sigmoid(
                self.auxiliary_head(self._spatialize(state, token_height, token_width))
            )
        ]
        for _ in range(self.attractor_iterations):
            previous = state
            state, entropy, retrieval, prototype_mass, attention = self.update_block(
                state,
                episode,
                confidence,
                attractor_tokens,
            )
            global_state = state.mean(dim=1)
            coordinate = self._coordinate(
                global_state,
                encoded_context,
                last_coverage,
                season,
                previous_change=coordinate[:, 1],
            )
            trajectory.append(coordinate)
            attractor_tokens = retrieve_attractor(state, coordinate)
            energies.append(energy(state, attractor_tokens))
            entropies.append(entropy)
            prototype_masses.append(prototype_mass)
            update_norms.append(
                (state - previous).flatten(1).norm(dim=1)
                / previous.flatten(1).norm(dim=1).clamp_min(1e-6)
            )
            auxiliary.append(
                torch.sigmoid(
                    self.auxiliary_head(
                        self._spatialize(state, token_height, token_width)
                    )
                )
            )
        return {
            "state": state,
            "coordinate": coordinate,
            "trajectory": torch.stack(trajectory, dim=1),
            "energy": torch.stack(energies, dim=1),
            "entropy": (
                torch.stack(entropies, dim=1)
                if entropies
                else state.new_zeros(state.size(0), 0)
            ),
            "prototype_mass": (
                torch.stack(prototype_masses, dim=1)
                if prototype_masses
                else state.new_zeros(state.size(0), 0)
            ),
            "retrieval": retrieval,
            "attention": attention,
            "update_norms": (
                torch.stack(update_norms, dim=1)
                if update_norms
                else state.new_zeros(state.size(0), 0)
            ),
            "memory_tokens": attractor_tokens,
            "auxiliary": auxiliary,
        }

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
        _, condition = self._prepare_spatial_context(inputs, spatial_prior, spatial_context)
        frame_condition = condition.unsqueeze(1).expand(-1, steps, -1, -1, -1)
        frame_inputs = torch.cat(
            [
                resized_sequence,
                frame_condition,
            ],
            dim=2,
        ).reshape(
            batch * steps,
            channels + condition.size(1),
            self.internal_size,
            self.internal_size,
        )
        frame_maps = self.frame_encoder(frame_inputs)
        token_height, token_width = frame_maps.shape[-2:]
        frame_maps = frame_maps.reshape(
            batch,
            steps,
            self.bottleneck_channels,
            token_height,
            token_width,
        )
        frame_globals = frame_maps.mean(dim=(3, 4))
        _, hidden = self.temporal_gru(frame_globals)
        temporal_global = hidden[-1]
        state_map = frame_maps[:, -1] + self.temporal_to_map(temporal_global)[
            :,
            :,
            None,
            None,
        ]
        encoded_context = self.context_encoder(context)
        gamma, beta = self.state_film(encoded_context).chunk(2, dim=1)
        state_map = state_map * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        state_map = state_map + beta[:, :, None, None]
        state = self._tokenize(state_map)
        if state_noise_std > 0:
            state = state + state_noise_std * torch.randn_like(state)
        episode = frame_maps.permute(0, 1, 3, 4, 2).reshape(
            batch,
            steps * token_height * token_width,
            self.bottleneck_channels,
        )
        confidence = self._episode_confidence(
            resized_sequence,
            (token_height, token_width),
        )
        if not self.use_hopfield_episode:
            episode = episode[:, :0]
            confidence = confidence[:, :0]
        last_coverage = self._last_coverage(inputs, context)
        season = self._season_features(context)
        recurrence = self._run_recurrence(
            state,
            episode,
            confidence,
            encoded_context,
            last_coverage,
            season,
            token_height,
            token_width,
        )
        refined_state = recurrence["state"]
        assert isinstance(refined_state, torch.Tensor)
        last_map = resized_sequence[:, -1, 0:1]
        prediction, residual_prediction, direct_prediction, gate = self._decode_prediction(
            refined_state,
            token_height,
            token_width,
            last_map,
            height,
            width,
        )
        effective_mask = coverage_mask if coverage_mask is not None else land_mask
        coverage_prediction = _masked_coverage(prediction, effective_mask)
        backbone_prediction = F.interpolate(
            last_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        backbone_coverage = _masked_coverage(backbone_prediction, effective_mask)
        first_order, second_order = self.attractor_memory.regularization()
        clean_state = refined_state.mean(dim=1)
        noisy_state = clean_state.detach()
        if self.training and self.manifold_noise_std > 0:
            noisy_recurrence = self._run_recurrence(
                state + self.manifold_noise_std * torch.randn_like(state),
                episode,
                confidence,
                encoded_context,
                last_coverage,
                season,
                token_height,
                token_width,
            )
            noisy_output = noisy_recurrence["state"]
            assert isinstance(noisy_output, torch.Tensor)
            noisy_state = noisy_output.mean(dim=1)
        energy = recurrence["energy"]
        coordinate = recurrence["coordinate"]
        trajectory = recurrence["trajectory"]
        entropy = recurrence["entropy"]
        update_norms = recurrence["update_norms"]
        memory_tokens = recurrence["memory_tokens"]
        auxiliary = recurrence["auxiliary"]
        assert isinstance(energy, torch.Tensor)
        assert isinstance(coordinate, torch.Tensor)
        assert isinstance(trajectory, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(update_norms, torch.Tensor)
        assert isinstance(memory_tokens, torch.Tensor)
        assert isinstance(auxiliary, list)
        zero = prediction.new_zeros(batch)
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
            "coverage_temporal_base": backbone_coverage,
            "coverage_attractor_correction": zero,
            "coverage_correction_scale": zero,
            "temporal_global_state": temporal_global,
            "temporal_fusion_gate": prediction.new_zeros(batch, 0, 0, 0),
            "coverage_candidates": prediction.new_zeros(batch, 0),
            "coverage_weights": prediction.new_zeros(batch, 0),
            "coverage_residual": zero,
            "accumulation_gate": torch.sigmoid((coordinate[:, 1] - 0.005) / 0.01),
            "last_input_coverage": last_coverage,
            "input_steps": self.input_steps,
            "energy": energy,
            "manifold_energy": energy,
            "manifold_distance": energy,
            "manifold_coordinate": coordinate[:, :2],
            "coordinate_trajectory": trajectory,
            "memory_tokens": memory_tokens,
            "manifold_first_order": first_order,
            "manifold_second_order": second_order,
            "clean_manifold_state": clean_state,
            "noisy_manifold_state": noisy_state,
            "auxiliary_predictions": auxiliary,
            "attention": recurrence["attention"],
            "update_norms": update_norms,
            "residual_prediction": residual_prediction,
            "direct_prediction": direct_prediction,
            "fusion_gate": gate,
            "residual_gate": gate,
            "backbone_prediction": backbone_prediction,
            "hopfield_entropy": entropy,
            "hopfield_prototype_mass": recurrence["prototype_mass"],
            "retrieval_weights": recurrence["retrieval"],
        }

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
