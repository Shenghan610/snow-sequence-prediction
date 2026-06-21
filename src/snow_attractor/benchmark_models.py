"""Comparable neural baselines for next-day snow-cover map forecasting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConvBlock


def calibrate_map_to_coverage(
    prediction: torch.Tensor,
    target_coverage: torch.Tensor,
) -> torch.Tensor:
    current = prediction.flatten(1).mean(dim=1)
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


class CoverageConditioner(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, 96),
            nn.GELU(),
            nn.Linear(96, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        last_coverage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.context(context)
        residual = 0.12 * self.head(torch.cat([latent, encoded], dim=1)).squeeze(1)
        return (last_coverage + residual).clamp(0.0, 1.0), encoded


class BenchmarkBase(nn.Module):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        latent_dim: int,
        internal_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.spatial_context_channels = spatial_context_channels
        self.internal_size = internal_size
        self.coverage = CoverageConditioner(context_dim, latent_dim, dropout)

    def prepare(
        self,
        inputs: torch.Tensor,
        spatial_prior: torch.Tensor,
        spatial_context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = inputs.shape
        frames = F.interpolate(
            inputs.reshape(batch * steps, channels, height, width),
            size=(self.internal_size, self.internal_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, steps, channels, self.internal_size, self.internal_size)
        frames = frames[:, :, :1]
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
            spatial_context = F.interpolate(
                spatial_context,
                size=(self.internal_size, self.internal_size),
                mode="bilinear",
                align_corners=False,
            )
            condition = torch.cat([prior, spatial_context], dim=1)
        else:
            condition = prior
        return frames, condition

    def format_output(
        self,
        raw_prediction: torch.Tensor,
        latent: torch.Tensor,
        inputs: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        last_coverage = inputs[:, -1, 0].flatten(1).mean(dim=1)
        coverage_prediction, _ = self.coverage(latent, context, last_coverage)
        prediction = calibrate_map_to_coverage(
            F.interpolate(
                raw_prediction,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0),
            coverage_prediction,
        )
        zero_energy = prediction.new_zeros(prediction.size(0), 1)
        return {
            "prediction": prediction,
            "coverage_prediction": coverage_prediction,
            "last_input_coverage": last_coverage,
            "energy": zero_energy,
            "manifold_coordinate": torch.stack(
                [coverage_prediction, coverage_prediction - last_coverage],
                dim=1,
            ),
            "auxiliary_predictions": [prediction],
            "update_norms": prediction.new_zeros(prediction.size(0), 0),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class TemporalUNet(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            192,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.enc1 = ConvBlock(input_steps + condition_channels, 40, dropout * 0.25)
        self.enc2 = ConvBlock(40, 80, dropout * 0.5)
        self.enc3 = ConvBlock(80, 160, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(160, 192, dropout)
        self.dec3 = ConvBlock(192 + 160, 160, dropout)
        self.dec2 = ConvBlock(160 + 80, 80, dropout * 0.5)
        self.dec1 = ConvBlock(80 + 40, 40, dropout * 0.25)
        self.output = nn.Conv2d(40, 1, 1)

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        fused = torch.cat([frames[:, :, 0], condition], dim=1)
        enc1 = self.enc1(fused)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat([F.interpolate(bottleneck, size=enc3.shape[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[-2:]), enc1], 1))
        raw = (frames[:, -1, 0:1] + 0.4 * torch.tanh(self.output(dec1))).clamp(0.0, 1.0)
        return self.format_output(raw, bottleneck.mean((2, 3)), inputs, context)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            3,
            padding=1,
        )

    def forward(self, inputs, state):
        if state is None:
            shape = (inputs.size(0), self.hidden_channels, *inputs.shape[-2:])
            hidden, cell = inputs.new_zeros(shape), inputs.new_zeros(shape)
        else:
            hidden, cell = state
        gates = self.gates(torch.cat([inputs, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(
            candidate
        )
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTM(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        input_channels: int = 1,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            128,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
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
        self.recurrent = ConvLSTMCell(128, 128)
        self.decoder = nn.Sequential(
            ConvBlock(128, 96, dropout),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(96, 48, dropout * 0.5),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        condition_features = self.condition_encoder(condition)
        state = None
        for step in range(frames.size(1)):
            encoded = torch.cat(
                [self.frame_encoder(frames[:, step]), condition_features],
                dim=1,
            )
            state = self.recurrent(encoded, state)
        hidden, _ = state
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(hidden))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, hidden.mean((2, 3)), inputs, context)


class ConvLSTMUNet(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        input_channels: int = 1,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            192,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_enc1 = ConvBlock(input_channels, 32, dropout * 0.25)
        self.frame_enc2 = ConvBlock(32, 64, dropout * 0.5)
        self.frame_enc3 = ConvBlock(64, 96, dropout)
        self.pool = nn.MaxPool2d(2)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 96, 3, stride=4, padding=1),
            nn.GELU(),
        )
        self.recurrent = ConvLSTMCell(192, 192)
        self.dec3 = ConvBlock(192 + 96, 144, dropout)
        self.dec2 = ConvBlock(144 + 64, 80, dropout * 0.5)
        self.dec1 = ConvBlock(80 + 32, 40, dropout * 0.25)
        self.output = nn.Conv2d(40, 1, 1)

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        condition_features = self.condition_encoder(condition)
        state = None
        skip1 = skip2 = skip3 = None
        for step in range(frames.size(1)):
            skip1 = self.frame_enc1(frames[:, step])
            skip2 = self.frame_enc2(self.pool(skip1))
            skip3 = self.frame_enc3(self.pool(skip2))
            recurrent_input = torch.cat([skip3, condition_features], dim=1)
            state = self.recurrent(recurrent_input, state)
        hidden, _ = state
        assert skip1 is not None and skip2 is not None and skip3 is not None
        dec3 = self.dec3(
            torch.cat([F.interpolate(hidden, size=skip3.shape[-2:]), skip3], dim=1)
        )
        dec2 = self.dec2(
            torch.cat([F.interpolate(dec3, size=skip2.shape[-2:]), skip2], dim=1)
        )
        dec1 = self.dec1(
            torch.cat([F.interpolate(dec2, size=skip1.shape[-2:]), skip1], dim=1)
        )
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.output(dec1))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, hidden.mean((2, 3)), inputs, context)


class SpatiotemporalLSTMCell(nn.Module):
    """PredRNN-style cell with hidden, cell, and global spatiotemporal memory."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            3,
            padding=1,
        )
        self.memory_gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 3,
            3,
            padding=1,
        )
        self.output_gate = nn.Conv2d(
            hidden_channels * 3,
            hidden_channels,
            3,
            padding=1,
        )
        self.project = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]],
        memory: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (inputs.size(0), self.hidden_channels, *inputs.shape[-2:])
            hidden, cell = inputs.new_zeros(shape), inputs.new_zeros(shape)
        else:
            hidden, cell = state
        if memory is None:
            memory = torch.zeros_like(hidden)
        i, f, o, g = self.hidden_gates(torch.cat([inputs, hidden], dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        mi, mf, mg = self.memory_gates(torch.cat([inputs, memory], dim=1)).chunk(3, dim=1)
        memory = torch.sigmoid(mf) * memory + torch.sigmoid(mi) * torch.tanh(mg)
        output = torch.sigmoid(
            self.output_gate(torch.cat([inputs, cell, memory], dim=1))
        )
        hidden = output * torch.tanh(self.project(torch.cat([cell, memory], dim=1)))
        return hidden, cell, memory


class PredRNN(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        input_channels: int = 1,
        internal_size: int = 128,
        dropout: float = 0.1,
        hidden_channels: int = 128,
        layers: int = 2,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            hidden_channels,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_encoder = nn.Sequential(
            ConvBlock(input_channels, 48, dropout * 0.25),
            nn.MaxPool2d(2),
            ConvBlock(48, 96, dropout * 0.5),
            nn.MaxPool2d(2),
            nn.Conv2d(96, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, 3, stride=4, padding=1),
            nn.GELU(),
        )
        self.cells = nn.ModuleList(
            [
                SpatiotemporalLSTMCell(
                    hidden_channels if layer == 0 else hidden_channels,
                    hidden_channels,
                )
                for layer in range(layers)
            ]
        )
        self.decoder = nn.Sequential(
            ConvBlock(hidden_channels, 96, dropout),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(96, 48, dropout * 0.5),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        condition_features = self.condition_encoder(condition)
        states = [None] * len(self.cells)
        memory = None
        output = condition_features
        for step in range(frames.size(1)):
            output = self.frame_encoder(frames[:, step]) + condition_features
            for index, cell in enumerate(self.cells):
                hidden, cell_state, memory = cell(output, states[index], memory)
                states[index] = (hidden, cell_state)
                output = hidden
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(output))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, output.mean((2, 3)), inputs, context)


class PredRNNv2(PredRNN):
    """PredRNNv2-inspired variant with wider recurrence and a boundary refiner."""

    def __init__(self, *args, hidden_channels: int = 160, layers: int = 3, **kwargs) -> None:
        super().__init__(
            *args,
            hidden_channels=hidden_channels,
            layers=layers,
            **kwargs,
        )
        self.gradient_refiner = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **kwargs):
        outputs = super().forward(
            inputs,
            spatial_prior,
            context,
            spatial_context=spatial_context,
            **kwargs,
        )
        prediction = outputs["prediction"]
        edge = prediction - F.avg_pool2d(prediction, kernel_size=3, stride=1, padding=1)
        refined = (prediction + 0.08 * torch.tanh(self.gradient_refiner(edge))).clamp(
            0.0,
            1.0,
        )
        outputs["prediction"] = refined
        outputs["auxiliary_predictions"] = [prediction, refined]
        return outputs


class CNNGRU(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            192,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, dropout * 0.25),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, dropout * 0.5),
            nn.MaxPool2d(2),
            ConvBlock(64, 96, dropout),
        )
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 32, 3, stride=4, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(96, 192, num_layers=2, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(
            ConvBlock(96 + 32 + 192, 128, dropout),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64, dropout * 0.5),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        batch, steps = frames.shape[:2]
        encoded = self.encoder(frames.reshape(batch * steps, 1, *frames.shape[-2:]))
        encoded = encoded.reshape(batch, steps, *encoded.shape[1:])
        pooled = encoded.mean((-2, -1))
        _, hidden = self.gru(pooled)
        latent = hidden[-1]
        latent_map = latent[:, :, None, None].expand(-1, -1, *encoded.shape[-2:])
        condition_map = self.condition_encoder(condition)
        raw = (
            frames[:, -1, 0:1]
            + 0.4
            * torch.tanh(
                self.decoder(torch.cat([encoded[:, -1], condition_map, latent_map], dim=1))
            )
        ).clamp(0.0, 1.0)
        return self.format_output(raw, latent, inputs, context)


class InceptionBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
                for kernel in (3, 5, 7)
            ]
        )
        self.project = nn.Conv2d(hidden, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, inputs):
        expanded = F.gelu(self.expand(inputs))
        mixed = sum(layer(expanded) for layer in self.depthwise) / len(self.depthwise)
        return inputs + self.norm(self.project(F.gelu(mixed)))


class SimVPv2(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            192,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.encoder = nn.Sequential(
            ConvBlock(1, 48, dropout * 0.25),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.condition_encoder = nn.Conv2d(
            condition_channels,
            192,
            3,
            stride=4,
            padding=1,
        )
        self.temporal_reduce = nn.Conv2d(input_steps * 192, 192, 1)
        self.translator = nn.Sequential(*[InceptionBlock(192) for _ in range(6)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 1, 3, padding=1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        batch, steps = frames.shape[:2]
        encoded = self.encoder(frames.reshape(batch * steps, 1, *frames.shape[-2:]))
        encoded = encoded.reshape(batch, steps * encoded.size(1), *encoded.shape[-2:])
        latent_map = self.translator(
            self.temporal_reduce(encoded) + self.condition_encoder(condition)
        )
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(latent_map))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, latent_map.mean((2, 3)), inputs, context)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
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

    def forward(self, inputs):
        normalized = self.norm1(inputs)
        inputs = inputs + self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return inputs + self.ffn(self.norm2(inputs))


class EarthformerLite(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            192,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_encoder = nn.Conv2d(1, 192, 8, stride=8)
        self.condition_encoder = nn.Conv2d(condition_channels, 192, 8, stride=8)
        self.temporal_position = nn.Parameter(torch.zeros(1, input_steps, 1, 192))
        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(192, 6, dropout) for _ in range(2)]
        )
        self.spatial_blocks = nn.ModuleList(
            [TransformerBlock(192, 6, dropout) for _ in range(4)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        batch, steps = frames.shape[:2]
        encoded = self.frame_encoder(frames.reshape(batch * steps, 1, *frames.shape[-2:]))
        height, width = encoded.shape[-2:]
        tokens = encoded.flatten(2).transpose(1, 2).reshape(batch, steps, height * width, -1)
        tokens = tokens + self.temporal_position[:, :steps]
        temporal = tokens.permute(0, 2, 1, 3).reshape(batch * height * width, steps, -1)
        for block in self.temporal_blocks:
            temporal = block(temporal)
        spatial = temporal[:, -1].reshape(batch, height * width, -1)
        condition_tokens = self.condition_encoder(condition).flatten(2).transpose(1, 2)
        spatial = spatial + condition_tokens
        for block in self.spatial_blocks:
            spatial = block(spatial)
        latent_map = spatial.transpose(1, 2).reshape(batch, -1, height, width)
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(latent_map))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, spatial.mean(dim=1), inputs, context)


def window_partition(inputs: torch.Tensor, window: int) -> torch.Tensor:
    batch, channels, height, width = inputs.shape
    return (
        inputs.view(batch, channels, height // window, window, width // window, window)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window * window, channels)
    )


def window_reverse(
    windows: torch.Tensor,
    window: int,
    batch: int,
    height: int,
    width: int,
) -> torch.Tensor:
    channels = windows.size(-1)
    return (
        windows.view(batch, height // window, width // window, window, window, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, height, width)
    )


class WindowAttention(nn.Module):
    def __init__(self, channels: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.window = window
        self.block = TransformerBlock(channels, heads, dropout)

    def forward(self, inputs):
        batch, _, height, width = inputs.shape
        windows = self.block(window_partition(inputs, self.window))
        return window_reverse(windows, self.window, batch, height, width)


class SwinLSTM(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            128,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_encoder = nn.Conv2d(1, 128, 4, stride=4)
        self.condition_encoder = nn.Conv2d(condition_channels, 128, 4, stride=4)
        self.input_attention = WindowAttention(128, 4, 4, dropout)
        self.recurrent = ConvLSTMCell(256, 128)
        self.hidden_attention = WindowAttention(128, 4, 4, dropout)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        condition_features = self.condition_encoder(condition)
        state = None
        for step in range(frames.size(1)):
            encoded = self.input_attention(self.frame_encoder(frames[:, step]))
            hidden = (
                encoded.new_zeros(encoded.size(0), 128, *encoded.shape[-2:])
                if state is None
                else self.hidden_attention(state[0])
            )
            recurrent_state = (hidden, state[1]) if state else None
            state = self.recurrent(
                torch.cat([encoded, condition_features], dim=1),
                recurrent_state,
            )
        hidden, _ = state
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(hidden))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, hidden.mean((2, 3)), inputs, context)


class SelectiveStateSpace2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(channels, channels * 3, 1)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.decay = nn.Parameter(torch.full((channels,), -2.0))

    def forward(self, inputs: torch.Tensor, state: Optional[torch.Tensor]):
        value, gate, skip = self.input_projection(inputs).chunk(3, dim=1)
        value = torch.tanh(self.depthwise(value))
        decay = torch.sigmoid(self.decay)[None, :, None, None]
        if state is None:
            state = torch.zeros_like(value)
        state = decay * state + (1.0 - decay) * value
        output = torch.sigmoid(gate) * state + skip
        return self.output_projection(output), state


class VMRNN(BenchmarkBase):
    def __init__(
        self,
        input_steps: int,
        context_dim: int,
        spatial_context_channels: int,
        internal_size: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            input_steps,
            context_dim,
            spatial_context_channels,
            128,
            internal_size,
            dropout,
        )
        condition_channels = 2 + spatial_context_channels
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 128, 4, stride=4),
            nn.GELU(),
        )
        self.ssm = nn.ModuleList([SelectiveStateSpace2D(128) for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, inputs, spatial_prior, context, spatial_context=None, **_):
        frames, condition = self.prepare(inputs, spatial_prior, spatial_context)
        condition_features = self.condition_encoder(condition)
        states = [None] * len(self.ssm)
        output = condition_features
        for step in range(frames.size(1)):
            output = self.frame_encoder(frames[:, step]) + condition_features
            for index, layer in enumerate(self.ssm):
                residual = output
                output, states[index] = layer(output, states[index])
                output = F.gelu(output + residual)
        raw = (
            frames[:, -1, 0:1] + 0.4 * torch.tanh(self.decoder(output))
        ).clamp(0.0, 1.0)
        return self.format_output(raw, output.mean((2, 3)), inputs, context)


MODEL_REGISTRY = {
    "temporal_unet": TemporalUNet,
    "cnn_gru": CNNGRU,
    "convlstm": ConvLSTM,
    "convlstm_unet": ConvLSTMUNet,
    "predrnn": PredRNN,
    "predrnnv2": PredRNNv2,
    "simvpv2": SimVPv2,
    "earthformer_lite": EarthformerLite,
    "swinlstm": SwinLSTM,
    "vmrnn": VMRNN,
}


def build_benchmark_model(name: str, **model_config) -> BenchmarkBase:
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown benchmark model: {name}") from error
    return model_class(**model_config)
