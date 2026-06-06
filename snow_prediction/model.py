import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SCALAR_FEATURE_DIM, SEASON_FEATURE_DIM
from .data import AliSnowDatasetRAM


class PriorSpatialEncoder(nn.Module):
    def __init__(self, in_channels, d_model, dropout=0.25):
        super().__init__()
        self.fused_channels = in_channels + 2
        self.num_tokens = 6
        spatial_dropout = min(float(dropout) * 0.5, 0.12)

        self.stem = nn.Sequential(
            nn.Conv2d(self.fused_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(spatial_dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(spatial_dropout),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.GELU()
        )
        self.grid_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_flat, prior_expanded):
        x_fused = torch.cat([x_flat, prior_expanded], dim=1)
        feat = self.stem(x_fused)

        global_token = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(2).transpose(1, 2)
        grid_tokens = self.grid_pool(feat).flatten(2).transpose(1, 2)

        prior_weight = prior_expanded.mean(dim=1, keepdim=True)
        prior_weight = F.interpolate(prior_weight, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        prior_weight = torch.clamp(prior_weight, min=0.0)
        focus_token = (feat * (prior_weight + 1e-3)).sum(dim=(2, 3), keepdim=True)
        focus_token = focus_token / ((prior_weight + 1e-3).sum(dim=(2, 3), keepdim=True) + 1e-6)
        focus_token = focus_token.flatten(2).transpose(1, 2)

        tokens = torch.cat([global_token, focus_token, grid_tokens], dim=1)
        return self.dropout(self.out_norm(tokens))


# ==========================================
# 4. 先验感知动力学注意力 (Prior-Aware Attractor)
# ==========================================
class PriorDynamicalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, gamma=0.12, dropout=0.20):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.beta = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(min(float(dropout) * 0.5, 0.12))
        self.out_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def get_energy(self, x, prior_weight_factor=None):
        b, n, _ = x.shape
        q = self.W_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.beta
        lse = torch.logsumexp(attn_scores, dim=-1)
        attention_energy = -lse.sum(dim=1)
        state_energy = 0.5 * x.pow(2).mean(dim=-1)
        energy_per_step = attention_energy + 0.02 * state_energy

        if prior_weight_factor is not None:
            energy_per_step = energy_per_step * prior_weight_factor

        return energy_per_step.mean(dim=1)

    def forward_step(self, x):
        residual = x
        x_norm = self.norm_attn(x)
        b, n, _ = x_norm.shape

        q = self.W_q(x_norm).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x_norm).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x_norm).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.beta
        attn_weights = self.attn_dropout(F.softmax(attn_logits, dim=-1))

        out = (attn_weights @ v).transpose(1, 2).reshape(b, n, self.dim)
        out = self.out_dropout(self.W_o(out))

        step = torch.clamp(self.gamma, 0.02, 0.35)
        new_x = residual + step * out
        new_x = new_x + step * 0.5 * self.ff(self.norm_ff(new_x))
        return new_x, attn_weights


# ==========================================
# 5. 基于真实空间先验的全局迭代预测器
# ==========================================
class DataDrivenSnowPredictor(nn.Module):
    def __init__(
            self,
            in_channels,
            d_model=128,
            iterations=8,
            season_dim=SEASON_FEATURE_DIM,
            scalar_dim=SCALAR_FEATURE_DIM,
            base_window=7,
            max_delta=0.40,
            hidden_dropout=0.25,
            feature_dropout=0.12,
            head_dropout=0.25,
            residual_gate_bias=-1.5
    ):
        super().__init__()
        self.iterations = iterations
        self.d_model = d_model
        self.scalar_dim = scalar_dim
        self.season_dim = season_dim
        self.base_window = base_window
        self.max_delta = max_delta
        self.last_aux = {}

        self.encoder = PriorSpatialEncoder(in_channels, d_model, dropout=hidden_dropout)
        self.season_proj = nn.Sequential(
            nn.LayerNorm(season_dim),
            nn.Dropout(feature_dropout),
            nn.Linear(season_dim, d_model)
        )
        self.token_type_emb = nn.Parameter(torch.randn(1, 1, self.encoder.num_tokens, d_model) * 0.02)
        self.scalar_input_norm = nn.LayerNorm(scalar_dim + season_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.head_dropout = nn.Dropout(head_dropout)
        self.scalar_encoder = nn.GRU(
            input_size=scalar_dim + season_dim,
            hidden_size=d_model // 2,
            num_layers=2,
            dropout=hidden_dropout,
            batch_first=True,
            bidirectional=True
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model))
        self.attractor = PriorDynamicalSelfAttention(d_model, num_heads=4, gamma=0.1, dropout=hidden_dropout)
        self.token_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

        baseline_feature_dim = 5 + 3
        head_input_dim = d_model * 3 + scalar_dim * 5 + season_dim * 3 + baseline_feature_dim
        self.baseline_weight_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 96),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(96, 5)
        )
        self.residual_direction_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 48),
            nn.GELU(),
            nn.Dropout(min(float(head_dropout) * 0.5, 0.15)),
            nn.Linear(48, 1),
            nn.Tanh()
        )
        self.residual_gate_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1)
        )
        self.map_context_channels = min(96, max(32, d_model // 2))
        self.map_context_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, self.map_context_channels),
            nn.GELU()
        )
        map_dropout = min(float(hidden_dropout) * 0.5, 0.12)
        self.heatmap_delta_head = nn.Sequential(
            nn.Conv2d(in_channels + 2 + self.map_context_channels, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Dropout2d(map_dropout),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Tanh()
        )
        nn.init.zeros_(self.baseline_weight_head[-1].weight)
        with torch.no_grad():
            self.baseline_weight_head[-1].bias.copy_(torch.tensor([0.2, 1.8, 0.1, -2.0, -1.0]))
        nn.init.normal_(self.residual_direction_head[-2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_direction_head[-2].bias)
        nn.init.zeros_(self.residual_gate_head[-1].weight)
        nn.init.constant_(self.residual_gate_head[-1].bias, residual_gate_bias)
        nn.init.zeros_(self.heatmap_delta_head[-2].weight)
        nn.init.zeros_(self.heatmap_delta_head[-2].bias)

    def _summarize_scalar_sequence(self, scalar_features):
        return torch.cat([
            scalar_features[:, -1, :],
            scalar_features.mean(dim=1),
            scalar_features.std(dim=1, unbiased=False),
            scalar_features.min(dim=1).values,
            scalar_features.max(dim=1).values
        ], dim=1)

    @staticmethod
    def _baseline_candidates(scalar_features, season_features=None, target_season=None, base_window=7):
        coverage_series = scalar_features[:, :, 0]
        delta_series = scalar_features[:, :, 4]
        last_coverage = coverage_series[:, -1]

        base_window = min(base_window, scalar_features.size(1))
        recent = coverage_series[:, -base_window:]
        mean7 = recent.mean(dim=1)

        trend7 = torch.clamp(last_coverage + delta_series[:, -base_window:].mean(dim=1), 0.0, 1.0)

        if season_features is None or target_season is None or target_season.size(1) < 6:
            target_clim = mean7
            anomaly_base = mean7
        else:
            target_clim = torch.clamp(target_season[:, 4], 0.0, 1.0)
            last_clim = torch.clamp(season_features[:, -1, 4], 0.0, 1.0)
            anomaly_base = torch.clamp(target_clim + (last_coverage - last_clim), 0.0, 1.0)

        return torch.stack([last_coverage, mean7, trend7, target_clim, anomaly_base], dim=1)

    def _fallback_scalar_features(self, x):
        first_band = x[:, :, 0, :, :]
        flat = first_band.flatten(2)
        coverage = flat.mean(dim=2)
        spatial_std = flat.std(dim=2, unbiased=False)
        snow_ratio = (flat > 0.01).float().mean(dim=2)
        strong_snow_ratio = (flat > 0.10).float().mean(dim=2)
        feature_batches = []
        for b in range(x.size(0)):
            feature_batches.append(
                AliSnowDatasetRAM._build_scalar_features(
                    coverage[b],
                    spatial_std[b],
                    snow_ratio[b],
                    strong_snow_ratio[b]
                )
            )
        base_features = torch.stack(feature_batches, dim=0)
        if base_features.size(-1) < self.scalar_dim:
            pad_dim = self.scalar_dim - base_features.size(-1)
            padding = x.new_zeros(x.size(0), x.size(1), pad_dim)
            base_features = torch.cat([base_features, padding], dim=-1)
        return base_features

    def forward(self, x, spatial_prior, season_features=None, scalar_features=None, target_season=None):
        B, T, C, H, W = x.shape
        if season_features is None:
            season_features = x.new_zeros(B, T, self.season_dim)
        if scalar_features is None:
            scalar_features = self._fallback_scalar_features(x)
        if target_season is None:
            target_season = season_features[:, -1, :]

        # 利用全局方差图生成全局动态约束权重
        activity_mask = spatial_prior[:, 1, :, :]
        prior_weight_factor = activity_mask.mean(dim=(1, 2)).unsqueeze(1)
        prior_weight_factor = torch.clamp(prior_weight_factor, min=0.1)

        x_flat = x.view(B * T, C, H, W)
        prior_expanded = spatial_prior.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, 2, H, W)

        spatial_tokens = self.encoder(x_flat, prior_expanded)
        token_count = spatial_tokens.size(1)
        h = spatial_tokens.view(B, T, token_count, self.d_model)
        h = h + self.pos_emb[:, :T, :].unsqueeze(2)
        h = h + self.token_type_emb[:, :, :token_count, :]
        h = h + self.season_proj(season_features).unsqueeze(2)
        h = h.view(B, T * token_count, self.d_model)

        energies = []
        energies.append(self.attractor.get_energy(h, prior_weight_factor))

        for t in range(self.iterations):
            h, attn_weights = self.attractor.forward_step(h)
            energies.append(self.attractor.get_energy(h, prior_weight_factor))

        scalar_encoder_input = torch.cat([scalar_features, season_features], dim=-1)
        scalar_encoder_input = self.feature_dropout(self.scalar_input_norm(scalar_encoder_input))
        scalar_context, _ = self.scalar_encoder(scalar_encoder_input)
        trend_state = scalar_context[:, -1, :]
        h_tokens = h.view(B, T, token_count, self.d_model)
        last_tokens = h_tokens[:, -1, :, :]
        token_weights = F.softmax(self.token_pool(last_tokens).squeeze(-1), dim=1)
        final_state = (last_tokens * token_weights.unsqueeze(-1)).sum(dim=1)
        sequence_state = h_tokens.mean(dim=(1, 2))
        scalar_summary = self._summarize_scalar_sequence(scalar_features)
        season_gap = target_season - season_features[:, -1, :]
        baseline_candidates = self._baseline_candidates(
            scalar_features,
            season_features=season_features,
            target_season=target_season,
            base_window=self.base_window
        )
        baseline_stats = torch.stack([
            baseline_candidates.mean(dim=1),
            baseline_candidates.std(dim=1, unbiased=False),
            baseline_candidates.max(dim=1).values - baseline_candidates.min(dim=1).values
        ], dim=1)
        residual_input = torch.cat([
            final_state,
            sequence_state,
            trend_state,
            scalar_summary,
            target_season,
            season_gap,
            season_features[:, -1, :],
            baseline_candidates,
            baseline_stats
        ], dim=1)
        residual_input = self.head_dropout(residual_input)
        baseline_weights = F.softmax(self.baseline_weight_head(residual_input), dim=1)
        base_coverage = (baseline_candidates * baseline_weights).sum(dim=1)
        residual_direction = self.residual_direction_head(residual_input).squeeze(-1)
        residual_gate = torch.sigmoid(self.residual_gate_head(residual_input)).squeeze(-1)
        delta = self.max_delta * residual_gate * residual_direction
        coverage_guidance = torch.clamp(base_coverage + delta, 0.0, 1.0)

        last_map = x[:, -1, 0:1, :, :]
        last_coverage = last_map.flatten(1).mean(dim=1)
        coverage_bias = (coverage_guidance - last_coverage).view(B, 1, 1, 1)

        map_context = self.map_context_head(residual_input).view(B, self.map_context_channels, 1, 1)
        map_context = map_context.expand(-1, -1, H, W)
        if spatial_prior.shape[-2:] != (H, W):
            spatial_prior_map = F.interpolate(spatial_prior, size=(H, W), mode="bilinear", align_corners=False)
        else:
            spatial_prior_map = spatial_prior
        map_input = torch.cat([x[:, -1, :, :, :], spatial_prior_map, map_context], dim=1)
        local_delta = self.max_delta * residual_gate.view(B, 1, 1, 1) * self.heatmap_delta_head(map_input)
        map_delta = coverage_bias + local_delta
        snow_map = torch.clamp(last_map + map_delta, 0.0, 1.0)
        snow_coverage = snow_map.flatten(1).mean(dim=1)
        self.last_aux = {
            "baseline_candidates": baseline_candidates.detach(),
            "baseline_weights": baseline_weights.detach(),
            "base_coverage": base_coverage.detach(),
            "coverage_guidance": coverage_guidance.detach(),
            "pred_coverage": snow_coverage.detach(),
            "coverage_bias": coverage_bias,
            "residual_gate": residual_gate,
            "residual_delta": map_delta,
            "local_residual_delta": local_delta
        }

        return snow_map, torch.stack(energies, dim=1), attn_weights


# ==========================================
# 6. 评估与可视化
# ==========================================
