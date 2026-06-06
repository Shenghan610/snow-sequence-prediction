import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from snow_prediction import (
    AliSnowDatasetRAM,
    DataDrivenSnowPredictor,
    load_model_state,
)


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = 21
    img_size: int = 256
    d_model: int = 160
    iterations: int = 8
    base_window: int = 7
    max_delta: float = 0.40
    hidden_dropout: float = 0.10
    feature_dropout: float = 0.0
    head_dropout: float = 0.10
    residual_gate_bias: float = -0.8


@dataclass
class PredictionResult:
    target_date: str
    input_start_date: str
    input_end_date: str
    predicted_coverage: float
    previous_coverage: float
    coverage_change: float
    residual_gate: float
    baseline_weights: list
    geotiff_path: Optional[str] = None
    png_path: Optional[str] = None
    metadata_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def next_date(date_text):
    date_obj = datetime.strptime(date_text, "%Y-%m-%d")
    return (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")


def save_prediction_outputs(
        prediction_map,
        reference_tif,
        output_dir,
        target_date,
        metadata
):
    os.makedirs(output_dir, exist_ok=True)
    prediction_map = np.asarray(prediction_map, dtype=np.float32)
    if prediction_map.ndim != 2:
        raise ValueError("prediction_map 必须是二维数组。")

    stem = f"SnowPrediction_{target_date}"
    geotiff_path = os.path.abspath(os.path.join(output_dir, f"{stem}.tif"))
    png_path = os.path.abspath(os.path.join(output_dir, f"{stem}.png"))
    metadata_path = os.path.abspath(os.path.join(output_dir, f"{stem}.json"))

    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        target_height = src.height
        target_width = src.width

    tensor = torch.from_numpy(prediction_map)[None, None]
    resized = F.interpolate(
        tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)

    profile.update(
        dtype="float32",
        count=1,
        nodata=None,
        compress="lzw"
    )
    with rasterio.open(geotiff_path, "w", **profile) as dst:
        dst.write(resized, 1)
        dst.set_band_description(1, "predicted_snow_cover")
        dst.update_tags(
            target_date=target_date,
            value_range="0-1",
            model="DataDrivenSnowPredictor"
        )

    plt.imsave(png_path, prediction_map, cmap="Blues", vmin=0.0, vmax=1.0)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return geotiff_path, png_path, metadata_path


class SnowInferenceService:
    def __init__(
            self,
            data_dir,
            external_feature_path,
            weights_path,
            device="auto",
            config=None
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.external_feature_path = os.path.abspath(external_feature_path)
        self.weights_path = os.path.abspath(weights_path)
        self.config = config or ModelConfig()
        self.device = self._select_device(device)

        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"未找到模型权重：{self.weights_path}")

        self.dataset = AliSnowDatasetRAM(
            data_dir=self.data_dir,
            seq_len=self.config.seq_len,
            target_size=(self.config.img_size, self.config.img_size),
            external_feature_path=self.external_feature_path
        )
        self.input_days = self.config.seq_len - 1
        if len(self.dataset.data_cache) < self.input_days:
            raise ValueError(f"至少需要 {self.input_days} 幅连续历史积雪影像。")

        self.model = DataDrivenSnowPredictor(
            in_channels=self.dataset.original_channels,
            d_model=self.config.d_model,
            iterations=self.config.iterations,
            season_dim=self.dataset.season_feature_dim,
            scalar_dim=self.dataset.scalar_feature_dim,
            base_window=self.config.base_window,
            max_delta=self.config.max_delta,
            hidden_dropout=self.config.hidden_dropout,
            feature_dropout=self.config.feature_dropout,
            head_dropout=self.config.head_dropout,
            residual_gate_bias=self.config.residual_gate_bias
        ).to(self.device)
        state = load_model_state(self.weights_path, self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @staticmethod
    def _select_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
        return selected

    def _build_latest_inputs(self, target_date):
        frames = self.dataset.data_cache[-self.input_days:]
        dates = self.dataset.date_list[-self.input_days:]
        input_seq = torch.stack(frames, dim=0)
        date_features = torch.stack(
            [self.dataset._date_features(date_text) for date_text in dates],
            dim=0
        )
        external_seq = torch.stack(
            [self.dataset.external_features.get(date_text) for date_text in dates],
            dim=0
        )
        season_features = torch.cat([date_features, external_seq], dim=1)
        scalar_features = torch.cat(
            [self.dataset._scalar_features(input_seq), external_seq],
            dim=1
        )
        target_season = torch.cat([
            self.dataset._date_features(target_date),
            self.dataset.external_features.get(target_date)
        ], dim=0)
        return (
            input_seq.unsqueeze(0),
            season_features.unsqueeze(0),
            scalar_features.unsqueeze(0),
            target_season.unsqueeze(0),
            self.dataset.spatial_prior.unsqueeze(0),
            dates
        )

    def predict(self, target_date=None, output_dir="predictions"):
        latest_date = self.dataset.date_list[-1]
        target_date = target_date or next_date(latest_date)
        datetime.strptime(target_date, "%Y-%m-%d")

        inputs = self._build_latest_inputs(target_date)
        input_seq, season_features, scalar_features, target_season, spatial_prior, dates = inputs
        input_seq = input_seq.to(self.device)
        season_features = season_features.to(self.device)
        scalar_features = scalar_features.to(self.device)
        target_season = target_season.to(self.device)
        spatial_prior = spatial_prior.to(self.device)

        with torch.inference_mode():
            prediction, _, _ = self.model(
                input_seq,
                spatial_prior,
                season_features,
                scalar_features,
                target_season
            )

        prediction_map = prediction[0, 0].detach().cpu().numpy()
        predicted_coverage = float(prediction_map.mean())
        previous_coverage = float(input_seq[0, -1, 0].mean())
        baseline_weights = (
            self.model.last_aux["baseline_weights"][0].detach().cpu().tolist()
        )
        residual_gate = float(
            self.model.last_aux["residual_gate"][0].detach().cpu()
        )

        result = PredictionResult(
            target_date=target_date,
            input_start_date=dates[0],
            input_end_date=dates[-1],
            predicted_coverage=predicted_coverage,
            previous_coverage=previous_coverage,
            coverage_change=predicted_coverage - previous_coverage,
            residual_gate=residual_gate,
            baseline_weights=baseline_weights
        )

        reference_tif = self.dataset.file_list[-1]
        metadata = result.to_dict()
        metadata["device"] = str(self.device)
        metadata["weights_path"] = self.weights_path
        geotiff_path, png_path, metadata_path = save_prediction_outputs(
            prediction_map=prediction_map,
            reference_tif=reference_tif,
            output_dir=output_dir,
            target_date=target_date,
            metadata=metadata
        )
        result.geotiff_path = geotiff_path
        result.png_path = png_path
        result.metadata_path = metadata_path

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
        return result
