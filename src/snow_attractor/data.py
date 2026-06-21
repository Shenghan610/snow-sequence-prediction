"""Leakage-aware snow-map sequence loading."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DATE_PATTERN = re.compile(r"SnowCover_(\d{4}-\d{2}-\d{2})\.tif$")
POWER_PARAMETERS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_LW_DWN",
)


class ExternalFeatureStore:
    def __init__(self, path: Path) -> None:
        self.names: list[str] = []
        self.raw: dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.names = [name for name in reader.fieldnames if name != "date"]
            for row in reader:
                self.raw[row["date"]] = np.asarray(
                    [float(row[name]) for name in self.names],
                    dtype=np.float32,
                )
        self.mean = np.zeros(len(self.names), dtype=np.float32)
        self.std = np.ones(len(self.names), dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.names)

    def fit(self, dates: list[str]) -> None:
        values = np.stack([self.raw[date] for date in dates if date in self.raw])
        self.mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std)

    def get(self, date: str) -> torch.Tensor:
        values = self.raw.get(date)
        if values is None:
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        return torch.from_numpy((values - self.mean) / self.std)


class SpatialFeatureStore:
    """Build georeferenced weather and terrain fields from the 5x5 point archive."""

    def __init__(
        self,
        directory: Path,
        image_size: int,
        reference_raster: Path,
    ) -> None:
        self.image_size = image_size
        self.dynamic_names = list(POWER_PARAMETERS)
        self.dynamic: dict[str, np.ndarray] = {}
        self.field_cache: dict[str, torch.Tensor] = {}

        points_path = directory / "external_grid_points.csv"
        raw_directory = directory / "power_grid_daily_raw"
        with open(points_path, "r", encoding="utf-8") as handle:
            point_rows = list(csv.DictReader(handle))
        grid_size = int(round(math.sqrt(len(point_rows))))
        if grid_size * grid_size != len(point_rows):
            raise ValueError("External point archive must form a square grid")
        self.grid_size = grid_size

        latitudes = np.asarray([float(row["latitude"]) for row in point_rows], dtype=np.float32)
        longitudes = np.asarray([float(row["longitude"]) for row in point_rows], dtype=np.float32)
        elevation = np.asarray([float(row["elevation_m"]) for row in point_rows], dtype=np.float32)
        latitude_grid = latitudes.reshape(grid_size, grid_size)
        longitude_grid = longitudes.reshape(grid_size, grid_size)
        elevation_grid = elevation.reshape(grid_size, grid_size)

        lat_spacing_m = np.gradient(latitude_grid[:, 0]) * 111_320.0
        lon_spacing_m = (
            np.gradient(longitude_grid[0, :])
            * 111_320.0
            * np.cos(np.deg2rad(latitude_grid.mean()))
        )
        dz_dy = np.gradient(elevation_grid, axis=0) / (lat_spacing_m[:, None] + 1e-6)
        dz_dx = np.gradient(elevation_grid, axis=1) / (lon_spacing_m[None, :] + 1e-6)
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(-dz_dx, dz_dy)
        terrain_radiation = np.cos(aspect) * np.sin(slope)
        terrain = np.stack(
            [
                elevation_grid,
                np.rad2deg(slope),
                np.sin(aspect),
                np.cos(aspect),
                terrain_radiation,
            ],
            axis=0,
        )
        terrain_mean = terrain.mean(axis=(1, 2), keepdims=True)
        terrain_std = terrain.std(axis=(1, 2), keepdims=True)
        terrain = (terrain - terrain_mean) / np.where(
            terrain_std < 1e-6,
            1.0,
            terrain_std,
        )
        with rasterio.open(reference_raster) as source:
            bounds = source.bounds
        longitude_centers = np.linspace(
            bounds.left,
            bounds.right,
            image_size,
            endpoint=False,
            dtype=np.float32,
        )
        longitude_centers += (bounds.right - bounds.left) / (2.0 * image_size)
        latitude_centers = np.linspace(
            bounds.top,
            bounds.bottom,
            image_size,
            endpoint=False,
            dtype=np.float32,
        )
        latitude_centers -= (bounds.top - bounds.bottom) / (2.0 * image_size)
        longitude_field, latitude_field = np.meshgrid(
            longitude_centers,
            latitude_centers,
        )
        longitude_field = (
            longitude_field - longitude_grid.mean()
        ) / longitude_grid.std().clip(min=1e-6)
        latitude_field = (
            latitude_field - latitude_grid.mean()
        ) / latitude_grid.std().clip(min=1e-6)
        coordinate_fields = torch.from_numpy(
            np.stack([longitude_field, latitude_field], axis=0).astype(np.float32)
        )
        self.static_fields = torch.cat(
            [coordinate_fields, self._resize_grid(terrain)],
            dim=0,
        )

        point_payloads: dict[int, dict] = {}
        for path in raw_directory.glob("power_point_*.json"):
            point_id = int(path.name.split("_")[2])
            point_payloads[point_id] = json.loads(path.read_text(encoding="utf-8"))
        if len(point_payloads) != len(point_rows):
            raise ValueError(
                f"Expected {len(point_rows)} POWER point files, found {len(point_payloads)}"
            )

        first_payload = point_payloads[0]["properties"]["parameter"]
        date_keys = sorted(first_payload[self.dynamic_names[0]])
        for date_key in date_keys:
            fields = np.empty(
                (len(self.dynamic_names), grid_size, grid_size),
                dtype=np.float32,
            )
            for parameter_index, parameter in enumerate(self.dynamic_names):
                values = []
                for point_id in range(len(point_rows)):
                    value = point_payloads[point_id]["properties"]["parameter"][parameter].get(
                        date_key,
                        -999.0,
                    )
                    values.append(float(value))
                grid = np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)
                invalid = ~np.isfinite(grid) | (grid <= -998.0)
                if invalid.any():
                    valid_mean = float(grid[~invalid].mean()) if (~invalid).any() else 0.0
                    grid[invalid] = valid_mean
                fields[parameter_index] = grid
            date = datetime.strptime(date_key, "%Y%m%d").strftime("%Y-%m-%d")
            self.dynamic[date] = fields

        self.dynamic_mean = np.zeros((len(self.dynamic_names), 1, 1), dtype=np.float32)
        self.dynamic_std = np.ones((len(self.dynamic_names), 1, 1), dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return int(self.static_fields.size(0) + len(self.dynamic_names) * 4)

    def _resize_grid(self, array: np.ndarray) -> torch.Tensor:
        # POWER rows run south-to-north; raster rows run north-to-south.
        tensor = torch.from_numpy(np.ascontiguousarray(array[:, ::-1, :])).unsqueeze(0)
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

    def fit(self, dates: list[str]) -> None:
        unique_dates = sorted({date for date in dates if date in self.dynamic})
        if not unique_dates:
            return
        values = np.stack([self.dynamic[date] for date in unique_dates], axis=0)
        self.dynamic_mean = values.mean(axis=(0, 2, 3), keepdims=False)[:, None, None]
        std = values.std(axis=(0, 2, 3), keepdims=False)[:, None, None]
        self.dynamic_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.field_cache.clear()

    def _daily_fields(self, date: str) -> torch.Tensor:
        cached = self.field_cache.get(date)
        if cached is not None:
            return cached
        values = self.dynamic.get(date)
        if values is None:
            fields = torch.zeros(
                len(self.dynamic_names),
                self.image_size,
                self.image_size,
            )
        else:
            normalized = (values - self.dynamic_mean) / self.dynamic_std
            fields = self._resize_grid(normalized.astype(np.float32))
        self.field_cache[date] = fields
        return fields

    def get(
        self,
        history_dates: list[str],
        target_date: str,
    ) -> torch.Tensor:
        history = torch.stack([self._daily_fields(date) for date in history_dates])
        target = self._daily_fields(target_date)
        return torch.cat(
            [
                self.static_fields,
                history[-1],
                history[-min(7, len(history)) :].mean(dim=0),
                target,
                target - history[-1],
            ],
            dim=0,
        )


class CausalSpatialFeatureStore:
    """History-only POWER fields with normalization fitted on the training fold."""

    def __init__(self, directory: Path, image_size: int) -> None:
        self.image_size = image_size
        self.dynamic_names = list(POWER_PARAMETERS)
        self.dynamic: dict[str, np.ndarray] = {}
        self.field_cache: dict[str, torch.Tensor] = {}
        points_path = directory / "external_grid_points.csv"
        raw_directory = directory / "power_grid_daily_raw"
        with open(points_path, "r", encoding="utf-8") as handle:
            point_rows = list(csv.DictReader(handle))
        grid_size = int(round(math.sqrt(len(point_rows))))
        if grid_size * grid_size != len(point_rows):
            raise ValueError("External point archive must form a square grid")
        self.grid_size = grid_size
        payloads = {}
        for path in raw_directory.glob("power_point_*.json"):
            point_id = int(path.name.split("_")[2])
            payloads[point_id] = json.loads(path.read_text(encoding="utf-8"))
        if len(payloads) != len(point_rows):
            raise ValueError(
                f"Expected {len(point_rows)} POWER point files, found {len(payloads)}"
            )
        first = payloads[0]["properties"]["parameter"]
        for date_key in sorted(first[self.dynamic_names[0]]):
            values = np.empty(
                (len(self.dynamic_names), grid_size, grid_size),
                dtype=np.float32,
            )
            for parameter_index, parameter in enumerate(self.dynamic_names):
                field = []
                for point_id in range(len(point_rows)):
                    value = payloads[point_id]["properties"]["parameter"][parameter].get(
                        date_key,
                        -999.0,
                    )
                    field.append(float(value))
                grid = np.asarray(field, dtype=np.float32).reshape(grid_size, grid_size)
                invalid = ~np.isfinite(grid) | (grid <= -998)
                if invalid.any():
                    grid[invalid] = float(grid[~invalid].mean()) if (~invalid).any() else 0.0
                values[parameter_index] = grid
            parsed = datetime.strptime(date_key, "%Y%m%d").strftime("%Y-%m-%d")
            self.dynamic[parsed] = values
        self.mean = np.zeros((len(self.dynamic_names), 1, 1), dtype=np.float32)
        self.std = np.ones((len(self.dynamic_names), 1, 1), dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.dynamic_names) * 2

    def fit(self, dates: list[str]) -> None:
        selected = [self.dynamic[item] for item in sorted(set(dates)) if item in self.dynamic]
        if not selected:
            return
        values = np.stack(selected)
        self.mean = values.mean(axis=(0, 2, 3))[:, None, None].astype(np.float32)
        std = values.std(axis=(0, 2, 3))[:, None, None].astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std)
        self.field_cache.clear()

    def normalization_state(self) -> dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_normalization_state(self, state: dict[str, list]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.field_cache.clear()

    def _field(self, date_value: str) -> torch.Tensor:
        cached = self.field_cache.get(date_value)
        if cached is not None:
            return cached
        values = self.dynamic.get(date_value)
        if values is None:
            result = torch.zeros(
                len(self.dynamic_names),
                self.image_size,
                self.image_size,
            )
        else:
            normalized = (values - self.mean) / self.std
            tensor = torch.from_numpy(
                np.ascontiguousarray(normalized[:, ::-1, :])
            ).unsqueeze(0)
            result = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
        self.field_cache[date_value] = result
        return result

    def get(self, history_dates: list[str]) -> torch.Tensor:
        history = torch.stack([self._field(item) for item in history_dates])
        return torch.cat(
            [
                history[-1],
                history[-min(7, len(history)) :].mean(dim=0),
            ],
            dim=0,
        )

    def get_global(self, history_dates: list[str]) -> torch.Tensor:
        return self.get(history_dates).mean(dim=(1, 2))


def resolve_snow_directory(data_root: Union[str, Path]) -> Path:
    root = Path(data_root)
    candidate = root / "snow_cover"
    snow_dir = candidate if candidate.is_dir() else root
    if not snow_dir.is_dir():
        raise FileNotFoundError(f"Snow data directory does not exist: {snow_dir}")
    return snow_dir


def discover_snow_files(data_root: Union[str, Path]) -> list[tuple[datetime, Path]]:
    snow_dir = resolve_snow_directory(data_root)
    dated_files: list[tuple[datetime, Path]] = []
    for path in snow_dir.glob("SnowCover_*.tif"):
        match = DATE_PATTERN.match(path.name)
        if match:
            dated_files.append((datetime.strptime(match.group(1), "%Y-%m-%d"), path))
    dated_files.sort(key=lambda item: item[0])
    if not dated_files:
        raise FileNotFoundError(f"No SnowCover_*.tif files found in {snow_dir}")
    return dated_files


def load_snow_tif(
    path: Union[str, Path],
    image_size: Union[int, None] = None,
) -> torch.Tensor:
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(array)
    if tensor.max() > 1.5:
        tensor = tensor / 100.0
    tensor = tensor.clamp(0.0, 1.0)
    if image_size is not None and tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


class SnowSequenceDataset(Dataset):
    """Construct only fully consecutive input/target windows."""

    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_length: int = 20,
        image_size: int = 128,
        cache_frames: bool = False,
        use_external_features: bool = True,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.cache_frames = cache_frames
        self.frame_cache: dict[int, torch.Tensor] = {}
        self.data_root = Path(data_root)
        self.dated_files = discover_snow_files(data_root)
        external_path = self.data_root / "external_features" / "external_daily_features.csv"
        external_directory = self.data_root / "external_features"
        self.external = (
            ExternalFeatureStore(external_path)
            if use_external_features and external_path.exists()
            else None
        )
        spatial_paths_exist = (
            (external_directory / "external_grid_points.csv").exists()
            and (external_directory / "power_grid_daily_raw").is_dir()
        )
        self.spatial_external = (
            SpatialFeatureStore(
                external_directory,
                image_size,
                self.dated_files[0][1],
            )
            if use_external_features and spatial_paths_exist
            else None
        )
        self.windows = self._build_continuous_windows()
        if not self.windows:
            raise ValueError("No continuous input/target windows are available")
        sample = self._load_frame(self.windows[0][0])
        self.input_channels = int(sample.shape[0])
        external_dim = self.external.feature_dim if self.external is not None else 0
        self.context_dim = self.sequence_length + (self.sequence_length - 1) + 37 + 4
        self.context_dim += external_dim * 4
        self.spatial_context_channels = (
            self.spatial_external.feature_dim if self.spatial_external is not None else 0
        )

    def _load_frame(self, frame_index: int) -> torch.Tensor:
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index]
        tensor = load_snow_tif(self.dated_files[frame_index][1], self.image_size)
        if self.cache_frames:
            self.frame_cache[frame_index] = tensor
        return tensor

    def _build_continuous_windows(self) -> list[tuple[int, int]]:
        total_frames = self.sequence_length + 1
        windows: list[tuple[int, int]] = []
        for start in range(len(self.dated_files) - total_frames + 1):
            end = start + total_frames
            dates = [item[0] for item in self.dated_files[start:end]]
            if all(
                dates[index + 1] - dates[index] == timedelta(days=1)
                for index in range(len(dates) - 1)
            ):
                windows.append((start, end))
        return windows

    def fit_external_normalization(self, sample_indices: list[int]) -> None:
        if self.external is None and self.spatial_external is None:
            return
        frame_indices: set[int] = set()
        for sample_index in sample_indices:
            start, end = self.windows[sample_index]
            frame_indices.update(range(start, end))
        dates = [
            self.dated_files[frame_index][0].strftime("%Y-%m-%d")
            for frame_index in sorted(frame_indices)
        ]
        if self.external is not None:
            self.external.fit(dates)
        if self.spatial_external is not None:
            self.spatial_external.fit(dates)

    @staticmethod
    def _calendar_features(date: datetime) -> torch.Tensor:
        angle = 2.0 * math.pi * date.timetuple().tm_yday / 365.25
        return torch.tensor(
            [
                math.sin(angle),
                math.cos(angle),
                math.sin(2.0 * angle),
                math.cos(2.0 * angle),
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _sequence_statistics(coverage: torch.Tensor) -> torch.Tensor:
        windows = (3, 5, 7, 10, 14, 20)
        values = [
            coverage[-1],
            coverage.mean(),
            coverage.std(unbiased=False),
            coverage.min(),
            coverage.max(),
            coverage[-1] - coverage[-2],
            coverage[-1] - coverage[0],
        ]
        for window in windows:
            chunk = coverage[-min(window, len(coverage)) :]
            x = torch.arange(len(chunk), dtype=coverage.dtype)
            x_centered = x - x.mean()
            slope = (
                (x_centered * (chunk - chunk.mean())).sum()
                / x_centered.square().sum().clamp_min(1e-6)
            )
            values.extend(
                [
                    chunk.mean(),
                    chunk.std(unbiased=False),
                    chunk.min(),
                    chunk.max(),
                    slope,
                ]
            )
        return torch.stack(values)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        start, end = self.windows[index]
        entries = self.dated_files[start:end]
        frames = torch.stack(
            [self._load_frame(frame_index) for frame_index in range(start, end)],
            dim=0,
        )
        inputs = frames[:-1]
        target = frames[-1, 0:1]
        first_band = inputs[:, 0:1]
        mean_prior = first_band.mean(dim=0)
        variance_prior = first_band.var(dim=0, unbiased=False)
        variance_prior = variance_prior / variance_prior.amax().clamp_min(1e-6)
        spatial_prior = torch.cat([mean_prior, variance_prior], dim=0)

        coverage = first_band.flatten(1).mean(dim=1)
        target_date = entries[-1][0]
        context_parts = [
            coverage,
            torch.diff(coverage),
            self._sequence_statistics(coverage),
            self._calendar_features(target_date),
        ]
        if self.external is not None:
            external_sequence = torch.stack(
                [
                    self.external.get(date.strftime("%Y-%m-%d"))
                    for date, _ in entries[:-1]
                ]
            )
            target_external = self.external.get(target_date.strftime("%Y-%m-%d"))
            context_parts.extend(
                [
                    external_sequence[-1],
                    external_sequence[-7:].mean(dim=0),
                    target_external,
                    target_external - external_sequence[-1],
                ]
            )
        context = torch.cat(context_parts)
        sample = {
            "inputs": inputs,
            "target": target,
            "spatial_prior": spatial_prior,
            "context": context,
            "target_date": target_date.strftime("%Y-%m-%d"),
        }
        if self.spatial_external is not None:
            history_dates = [date.strftime("%Y-%m-%d") for date, _ in entries[:-1]]
            sample["spatial_context"] = self.spatial_external.get(
                history_dates,
                target_date.strftime("%Y-%m-%d"),
            )
        return sample


class MaskedSnowSequenceDataset(Dataset):
    """QA-aware, history-only dataset backed by memory-mapped NumPy arrays."""

    def __init__(
        self,
        data_root: Union[str, Path],
        region_id: str,
        sequence_length: int = 20,
        image_size: int = 128,
        qa_policy: str = "strict",
        target_min_valid_fraction: float = 0.8,
    ) -> None:
        if qa_policy not in {"strict", "lenient"}:
            raise ValueError(f"Unsupported QA policy: {qa_policy}")
        self.data_root = Path(data_root)
        self.region_id = region_id
        self.sequence_length = sequence_length
        self.image_size = image_size
        processed = self.data_root / region_id / "processed"
        self.snow = np.load(processed / "snow.npy", mmap_mode="r")
        self.valid = np.load(
            processed / f"valid_{qa_policy}.npy",
            mmap_mode="r",
        )
        self.land_mask = np.load(processed / "land_mask.npy").astype(bool)
        self.terrain = np.load(processed / "terrain.npy").astype(np.float32)
        self.dates = [
            datetime.fromisoformat(item)
            for item in json.loads((processed / "dates.json").read_text(encoding="utf-8"))
        ]
        if self.snow.shape[1:] != (image_size, image_size):
            raise ValueError(
                f"Processed image size {self.snow.shape[1:]} does not match {image_size}"
            )
        external_directory = self.data_root / region_id / "external_features"
        self.weather = (
            CausalSpatialFeatureStore(external_directory, image_size)
            if (external_directory / "external_grid_points.csv").exists()
            else None
        )
        self.windows = self._build_windows(target_min_valid_fraction)
        if not self.windows:
            raise ValueError(f"No valid QA windows found for {region_id}")
        self.climatology = np.zeros(
            (366, image_size, image_size),
            dtype=np.float32,
        )
        self.climatology_ready = False
        self.input_channels = 3
        self.context_dim = sequence_length + (sequence_length - 1) + 37 + 4
        self.context_dim += self.weather.feature_dim if self.weather is not None else 0
        self.spatial_context_channels = 7 + (
            self.weather.feature_dim if self.weather is not None else 0
        )

    def _build_windows(self, minimum_fraction: float) -> list[tuple[int, int]]:
        windows = []
        total = self.sequence_length + 1
        denominator = max(int(self.land_mask.sum()), 1)
        for start in range(len(self.dates) - total + 1):
            end = start + total
            dates = self.dates[start:end]
            if not all(
                dates[index + 1] - dates[index] == timedelta(days=1)
                for index in range(total - 1)
            ):
                continue
            target_valid = self.valid[end - 1].astype(bool) & self.land_mask
            if target_valid.sum() / denominator >= minimum_fraction:
                windows.append((start, end))
        return windows

    def fit_external_normalization(self, sample_indices: list[int]) -> None:
        frame_indices: set[int] = set()
        for sample_index in sample_indices:
            start, end = self.windows[sample_index]
            frame_indices.update(range(start, end))
        sums = np.zeros_like(self.climatology, dtype=np.float64)
        counts = np.zeros_like(self.climatology, dtype=np.uint16)
        global_sum = np.zeros((self.image_size, self.image_size), dtype=np.float64)
        global_count = np.zeros((self.image_size, self.image_size), dtype=np.uint16)
        weather_dates = []
        for frame_index in sorted(frame_indices):
            current = self.snow[frame_index]
            mask = self.valid[frame_index].astype(bool) & self.land_mask
            day_index = self.dates[frame_index].timetuple().tm_yday - 1
            sums[day_index][mask] += current[mask]
            counts[day_index][mask] += 1
            global_sum[mask] += current[mask]
            global_count[mask] += 1
            weather_dates.append(self.dates[frame_index].strftime("%Y-%m-%d"))
        global_mean = np.divide(
            global_sum,
            global_count,
            out=np.zeros_like(global_sum, dtype=np.float64),
            where=global_count > 0,
        )
        for day_index in range(366):
            self.climatology[day_index] = np.divide(
                sums[day_index],
                counts[day_index],
                out=global_mean.copy(),
                where=counts[day_index] > 0,
            )
        self.climatology_ready = True
        if self.weather is not None:
            self.weather.fit(weather_dates)

    def normalization_state(self) -> dict:
        if not self.climatology_ready:
            raise RuntimeError("Dataset normalization has not been fitted")
        return {
            "climatology": self.climatology,
            "weather": (
                self.weather.normalization_state() if self.weather is not None else None
            ),
        }

    def load_normalization_state(self, state: dict) -> None:
        self.climatology = np.asarray(state["climatology"], dtype=np.float32)
        self.climatology_ready = True
        if self.weather is not None and state.get("weather") is not None:
            self.weather.load_normalization_state(state["weather"])

    def _fill_history(
        self,
        indices: range,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.climatology_ready:
            raise RuntimeError("Call fit_external_normalization before reading samples")
        frames = []
        masks = []
        ages = []
        previous = None
        previous_age = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for frame_index in indices:
            observed = self.valid[frame_index].astype(bool) & self.land_mask
            raw = np.asarray(self.snow[frame_index], dtype=np.float32)
            day_index = self.dates[frame_index].timetuple().tm_yday - 1
            fallback = self.climatology[day_index]
            if previous is None:
                filled = np.where(observed, raw, fallback)
                age = np.where(observed, 0.0, float(self.sequence_length))
            else:
                filled = np.where(observed, raw, previous)
                age = np.where(observed, 0.0, previous_age + 1.0)
            filled = np.where(self.land_mask, filled, 0.0).astype(np.float32)
            age = np.where(self.land_mask, age, 0.0).astype(np.float32)
            frames.append(filled)
            masks.append(observed.astype(np.float32))
            ages.append(np.clip(age / self.sequence_length, 0.0, 1.0))
            previous = filled
            previous_age = age
        return (
            torch.from_numpy(np.stack(frames)),
            torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(ages)),
        )

    @staticmethod
    def _calendar_features(date_value: datetime) -> torch.Tensor:
        return SnowSequenceDataset._calendar_features(date_value)

    @staticmethod
    def _sequence_statistics(coverage: torch.Tensor) -> torch.Tensor:
        return SnowSequenceDataset._sequence_statistics(coverage)

    def _spatial_static(self) -> torch.Tensor:
        height, width = self.land_mask.shape
        longitude = torch.linspace(-1.0, 1.0, width).repeat(height, 1)
        latitude = torch.linspace(1.0, -1.0, height)[:, None].repeat(1, width)
        terrain = torch.from_numpy(self.terrain.copy())
        terrain[0] = terrain[0] / 6000.0
        terrain[1] = terrain[1] / 90.0
        return torch.cat([longitude[None], latitude[None], terrain], dim=0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        start, end = self.windows[index]
        history_indices = range(start, end - 1)
        snow, observation_mask, observation_age = self._fill_history(history_indices)
        inputs = torch.stack([snow, observation_mask, observation_age], dim=1)
        land = torch.from_numpy(self.land_mask.copy()).float()[None]
        target_valid = (
            self.valid[end - 1].astype(bool) & self.land_mask
        )
        target_array = np.where(
            target_valid,
            self.snow[end - 1],
            0.0,
        ).astype(np.float32)
        target = torch.from_numpy(target_array)[None]
        target_mask = torch.from_numpy(target_valid.astype(np.float32))[None]
        coverage = (
            snow.flatten(1) * land.flatten()
        ).sum(dim=1) / land.sum().clamp_min(1.0)
        target_date = self.dates[end - 1]
        context_parts = [
            coverage,
            torch.diff(coverage),
            self._sequence_statistics(coverage),
            self._calendar_features(target_date),
        ]
        history_dates = [
            self.dates[frame_index].strftime("%Y-%m-%d")
            for frame_index in history_indices
        ]
        spatial_context = self._spatial_static()
        if self.weather is not None:
            context_parts.append(self.weather.get_global(history_dates))
            spatial_context = torch.cat(
                [spatial_context, self.weather.get(history_dates)],
                dim=0,
            )
        mean_prior = snow.mean(dim=0)
        variance_prior = snow.var(dim=0, unbiased=False)
        variance_prior = variance_prior / variance_prior.amax().clamp_min(1e-6)
        return {
            "inputs": inputs,
            "frames": snow[:, None],
            "observation_mask": observation_mask[:, None],
            "observation_age": observation_age[:, None],
            "target": target,
            "target_valid_mask": target_mask,
            "land_mask": land,
            "spatial_prior": torch.stack([mean_prior, variance_prior]),
            "context": torch.cat(context_parts),
            "spatial_context": spatial_context,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "region_id": self.region_id,
        }
