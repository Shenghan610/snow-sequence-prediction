"""Thin adapters around pinned upstream spatiotemporal forecasting models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
VENDOR_COMMITS = {
    "simvpv2_official": "cb1afa3d4e23a7b6b8f24e06ddee1cd9378a21a9",
    "swinlstm_official": "4425bcfefbebfac85c9fc6c6659361b8154d5ca3",
    "vmrnn_official": "8e6cfb7da15cf2aae1fd8cadaf82f34d0f7de960",
}
OFFICIAL_BASELINE_SETTINGS = {
    "simvpv2_official": "SimVPv2 gSTA, hid_S=32, hid_T=256, N_S=4, N_T=6",
    "swinlstm_official": "SwinLSTM-B, patch=2, embed=128, depth=6",
    "vmrnn_official": "VMRNN-B, patch=4, embed=128, depth=6",
}


def _selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Pure PyTorch fallback matching the selective-scan reference recurrence."""

    del z
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    state_size = a.size(1)
    groups = b.size(1) if b.ndim == 4 else 1
    group_index = (
        torch.arange(channels, device=u.device) * groups // channels
    ).clamp_max(groups - 1)
    state = u.new_zeros(batch, channels, state_size, dtype=torch.float32)
    outputs = []
    a = a.float()
    chunk_size = 8
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk_delta = delta[:, :, start:end].float()
        chunk_b = (
            b[:, group_index, :, start:end].float()
            if b.ndim == 4
            else b[:, :, start:end]
            .float()[:, None]
            .expand(-1, channels, -1, -1)
        )
        chunk_c = (
            c[:, group_index, :, start:end].float()
            if c.ndim == 4
            else c[:, :, start:end]
            .float()[:, None]
            .expand(-1, channels, -1, -1)
        )
        transition = torch.exp(
            chunk_delta[:, :, None, :] * a[None, :, :, None]
        )
        input_term = (
            chunk_delta[:, :, None, :]
            * chunk_b
            * u[:, :, None, start:end].float()
        )
        prefix = transition.cumprod(dim=-1)
        states = prefix * (
            state[..., None]
            + (input_term / prefix.clamp_min(1e-30)).cumsum(dim=-1)
        )
        output = (states * chunk_c).sum(dim=2)
        if d is not None:
            output = output + d[None, :, None].float() * u[
                :, :, start:end
            ].float()
        outputs.append(output)
        state = states[..., -1]
    result = torch.cat(outputs, dim=-1)
    if return_last_state:
        return result, state
    return result


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class OfficialBaselineAdapter(nn.Module):
    def __init__(
        self,
        input_steps: int,
        input_channels: int,
        spatial_context_channels: int,
        image_size: int,
        condition_channels: int = 4,
    ) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.input_channels = input_channels
        self.spatial_context_channels = spatial_context_channels
        self.image_size = image_size
        condition_inputs = 2 + spatial_context_channels
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_inputs, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, condition_channels, kernel_size=1),
        )
        self.official_input_channels = input_channels + condition_channels

    def conditioned_sequence(
        self,
        inputs: torch.Tensor,
        spatial_prior: torch.Tensor,
        spatial_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, steps, _, height, width = inputs.shape
        condition = spatial_prior
        if spatial_context is not None:
            condition = torch.cat([condition, spatial_context], dim=1)
        condition = F.interpolate(
            condition,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        condition = self.condition_encoder(condition)
        repeated = condition[:, None].expand(batch, steps, -1, -1, -1)
        return torch.cat([inputs, repeated], dim=2)

    @staticmethod
    def format_output(
        raw_prediction: torch.Tensor,
        inputs: torch.Tensor,
        land_mask: Optional[torch.Tensor],
    ) -> dict:
        prediction = raw_prediction[:, 0:1].clamp(0.0, 1.0)
        last_map = inputs[:, -1, 0:1]
        if land_mask is None:
            mask = torch.ones_like(prediction)
        else:
            mask = land_mask.to(dtype=prediction.dtype)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        coverage = (prediction * mask).flatten(1).sum(dim=1) / denominator
        last_coverage = (last_map * mask).flatten(1).sum(dim=1) / denominator
        zero_energy = prediction.new_zeros(prediction.size(0), 1)
        return {
            "prediction": prediction,
            "coverage_prediction": coverage,
            "last_input_coverage": last_coverage,
            "coverage_delta": coverage - last_coverage,
            "energy": zero_energy,
            "manifold_coordinate": torch.stack(
                [coverage, coverage - last_coverage],
                dim=1,
            ),
            "auxiliary_predictions": [prediction],
            "update_norms": prediction.new_zeros(prediction.size(0), 0),
        }


class OfficialSimVPv2(OfficialBaselineAdapter):
    def __init__(self, *args, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        vendor = ROOT / "vendor" / "SimVPv2"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import openstl.modules  # noqa: F401

        module = _load_module(
            "official_simvp_model",
            vendor / "openstl" / "models" / "simvp_model.py",
        )
        SimVP_Model = module.SimVP_Model

        self.model = SimVP_Model(
            in_shape=(
                self.input_steps,
                self.official_input_channels,
                self.image_size,
                self.image_size,
            ),
            hid_S=32,
            hid_T=256,
            N_S=4,
            N_T=6,
            model_type="gSTA",
            drop=dropout,
            drop_path=dropout,
        )

    def forward(
        self,
        inputs,
        spatial_prior,
        context,
        spatial_context=None,
        land_mask=None,
        coverage_mask=None,
        **_,
    ):
        del context
        sequence = self.conditioned_sequence(inputs, spatial_prior, spatial_context)
        future_sequence = self.model(sequence)
        return self.format_output(
            future_sequence[:, 0],
            inputs,
            coverage_mask if coverage_mask is not None else land_mask,
        )


class OfficialSwinLSTM(OfficialBaselineAdapter):
    def __init__(self, *args, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        module = _load_module(
            "official_swinlstm_b",
            ROOT / "vendor" / "SwinLSTM" / "SwinLSTM_B.py",
        )
        self.model = module.SwinLSTM(
            img_size=self.image_size,
            patch_size=2,
            in_chans=self.official_input_channels,
            embed_dim=128,
            depths=[6],
            num_heads=[4],
            window_size=4,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=dropout,
        )

    def forward(
        self,
        inputs,
        spatial_prior,
        context,
        spatial_context=None,
        land_mask=None,
        coverage_mask=None,
        **_,
    ):
        del context
        sequence = self.conditioned_sequence(inputs, spatial_prior, spatial_context)
        states = [None]
        for step in range(sequence.size(1) - 1):
            _, states = self.model(sequence[:, step], states)
        prediction, _ = self.model(sequence[:, -1], states)
        return self.format_output(
            prediction,
            inputs,
            coverage_mask if coverage_mask is not None else land_mask,
        )


class OfficialVMRNN(OfficialBaselineAdapter):
    def __init__(
        self,
        *args,
        dropout: float = 0.1,
        patch_size: int = 4,
        embed_dim: int = 128,
        depths: int = 6,
        num_heads: int = 4,
        window_size: int = 4,
        model_image_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_image_size = model_image_size or self.image_size
        vendor = ROOT / "vendor" / "VMRNN"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import openstl.modules  # noqa: F401
        import openstl.modules.vmamba as vmamba

        if not hasattr(vmamba, "selective_scan_fn"):
            vmamba.selective_scan_fn = _selective_scan_reference

        module = _load_module(
            "official_vmrnn_model",
            vendor / "openstl" / "models" / "VMRNN_model.py",
        )
        VMRNN_B_Model = module.VMRNN_B_Model

        config = SimpleNamespace(
            in_shape=(
                self.input_steps,
                self.official_input_channels,
                self.model_image_size,
                self.model_image_size,
            ),
            patch_size=patch_size,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=dropout,
        )
        self.model = VMRNN_B_Model(config)

    def _recurrent_prediction(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden_state = None

        def recurrent_step(frame, *states):
            current_state = (states[0], states[1]) if states else None
            output, next_state = self.model.ST(frame, current_state)
            return output, *next_state

        output = None
        for step in range(sequence.size(1)):
            frame = sequence[:, step]
            flat_states = tuple(hidden_state) if hidden_state is not None else ()
            if self.training and torch.is_grad_enabled():
                result = checkpoint(
                    recurrent_step,
                    frame,
                    *flat_states,
                    use_reentrant=True,
                )
            else:
                result = recurrent_step(frame, *flat_states)
            output = result[0]
            hidden_state = (result[1], result[2])
        assert output is not None
        return output

    def forward(
        self,
        inputs,
        spatial_prior,
        context,
        spatial_context=None,
        land_mask=None,
        coverage_mask=None,
        **_,
    ):
        del context
        sequence = self.conditioned_sequence(inputs, spatial_prior, spatial_context)
        if self.model_image_size != sequence.size(-1):
            batch, steps, channels, _, _ = sequence.shape
            sequence = F.interpolate(
                sequence.reshape(batch * steps, channels, *sequence.shape[-2:]),
                size=(self.model_image_size, self.model_image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, steps, channels, self.model_image_size, self.model_image_size)
        prediction = self._recurrent_prediction(sequence)
        if prediction.shape[-2:] != inputs.shape[-2:]:
            prediction = F.interpolate(
                prediction,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.format_output(
            prediction,
            inputs,
            coverage_mask if coverage_mask is not None else land_mask,
        )


OFFICIAL_MODEL_REGISTRY = {
    "simvpv2_official": OfficialSimVPv2,
    "swinlstm_official": OfficialSwinLSTM,
    "vmrnn_official": OfficialVMRNN,
}


def build_official_model(name: str, **kwargs) -> OfficialBaselineAdapter:
    try:
        model_class = OFFICIAL_MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown official baseline: {name}") from error
    return model_class(**kwargs)
