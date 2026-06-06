import csv
import glob
import math
import os
import re
from datetime import datetime

import numpy as np
import rasterio
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import SCALAR_FEATURE_DIM, SEASON_FEATURE_DIM


class ExternalFeatureStore:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.feature_names = []
        self.feature_dim = 0
        self.feature_by_date = {}

        if csv_path is None or not os.path.exists(csv_path):
            if csv_path:
                print(f"未找到外部特征文件：{csv_path}，将仅使用雪盖序列特征。")
            return

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.feature_names = [name for name in reader.fieldnames if name != "date"]
            for row in reader:
                values = []
                for name in self.feature_names:
                    try:
                        value = float(row[name])
                    except (TypeError, ValueError):
                        value = np.nan
                    values.append(value)
                rows.append((row["date"], values))

        if len(rows) == 0:
            self.feature_names = []
            return

        values = np.array([row[1] for row in rows], dtype=np.float32)
        col_mean = np.nanmean(values, axis=0)
        col_std = np.nanstd(values, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        col_std = np.where((np.isnan(col_std)) | (col_std < 1e-6), 1.0, col_std)
        values = np.where(np.isnan(values), col_mean[None, :], values)
        values = (values - col_mean[None, :]) / col_std[None, :]

        self.feature_dim = values.shape[1]
        self.zero_feature = torch.zeros(self.feature_dim, dtype=torch.float32)
        self.feature_by_date = {
            date_text: torch.tensor(values[index], dtype=torch.float32)
            for index, (date_text, _) in enumerate(rows)
        }
        print(f"外部气象/地形特征加载完成：{self.feature_dim} 个特征，{len(self.feature_by_date)} 天。")

    def get(self, date_text):
        if self.feature_dim == 0:
            return torch.empty(0, dtype=torch.float32)
        return self.feature_by_date.get(date_text, self.zero_feature).clone()


# ==========================================
# 2. 数据集：高清全局视野 + 真实数据先验
# ==========================================
class AliSnowDatasetRAM(Dataset):
    def __init__(self, data_dir, seq_len=21, target_size=(128, 128), external_feature_path=None):
        """
        target_size: 提升全局分辨率到 128x128，保留细节同时保留全局地理规律
        """
        self.seq_len = seq_len
        self.target_size = target_size
        self.transform = T.Resize(target_size, antialias=True)
        self.external_features = ExternalFeatureStore(external_feature_path)
        self.scalar_feature_dim = SCALAR_FEATURE_DIM + self.external_features.feature_dim
        self.season_feature_dim = SEASON_FEATURE_DIM + self.external_features.feature_dim

        search_path = os.path.join(data_dir, "SnowCover_*.tif")
        all_files = glob.glob(search_path)
        self.file_list = sorted(all_files, key=lambda x: self._extract_date(x))
        self.date_list = [self._extract_date(fpath) for fpath in self.file_list]

        if len(self.file_list) == 0:
            raise ValueError(f"错误：在 {data_dir} 未找到 TIF 文件！请检查路径。")

        print(f"找到 {len(self.file_list)} 个文件。正在预加载到内存 (Resolution: {target_size})...")

        self.data_cache = []
        for fpath in tqdm(self.file_list, desc="Loading Data"):
            with rasterio.open(fpath) as src:
                data = src.read()
                data = np.nan_to_num(data, nan=0.0)
                tensor = torch.from_numpy(data).float()
                tensor = self._normalize_tensor(tensor)
                tensor = self.transform(tensor)
                self.data_cache.append(tensor)

        self.original_channels = self.data_cache[0].shape[0]
        print(f"数据预加载完成！波段数: {self.original_channels}")
        print(f"标量时序特征维度: {self.scalar_feature_dim} | 目标日特征维度: {self.season_feature_dim}")
        print(f"模式: 预测未来 (输入前 {self.seq_len - 1} 天 -> 预测第 {self.seq_len} 天全局覆盖率)")

        # 核心：计算真实空间先验 (Mean & Variance Map)
        self._compute_spatial_priors()
        self._compute_calendar_priors()

    def _compute_spatial_priors(self):
        print("正在从高清历史数据中提取真实地理空间先验 (Mean & Variance)...")
        all_data = torch.stack(self.data_cache, dim=0)

        # all_data: (N, C, H, W). 将所有历史时刻和波段聚合成 2 个空间先验通道，
        # 避免多波段 TIF 导致 spatial_prior 变成 2*C 个通道。
        self.mean_map = torch.mean(all_data, dim=(0, 1)).unsqueeze(0)
        self.var_map = torch.var(all_data, dim=(0, 1), unbiased=False).unsqueeze(0)

        self.mean_map = self.mean_map / (self.mean_map.max() + 1e-6)
        self.var_map = self.var_map / (self.var_map.max() + 1e-6)

        self.spatial_prior = torch.cat([self.mean_map, self.var_map], dim=0)  # (2, H, W)
        print("全局空间先验提取完成！")

    def _compute_calendar_priors(self):
        print("正在提取按年排除的日序气候先验 (coverage climatology)...")
        self.coverage_by_date = {}
        self.coverage_by_doy = {}
        for date_text, frame in zip(self.date_list, self.data_cache):
            try:
                date_obj = datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError:
                continue
            coverage = float(frame[0].mean())
            self.coverage_by_date[date_text] = coverage
            self.coverage_by_doy.setdefault(date_obj.timetuple().tm_yday, []).append((date_obj.year, coverage))

        all_coverages = list(self.coverage_by_date.values())
        self.global_clim_mean = float(np.mean(all_coverages)) if all_coverages else 0.0
        self.global_clim_std = float(np.std(all_coverages)) if all_coverages else 0.0
        print("日序气候先验提取完成！")

    def _extract_date(self, filepath):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filepath)
        return match.group(1) if match else filepath

    def _normalize_tensor(self, tensor):
        # 使用固定尺度归一化，保留不同日期之间“整体雪多/雪少”的真实差异。
        tensor = torch.clamp(tensor, min=0.0)
        if tensor.max() > 1.5:
            tensor = tensor / 100.0
        return torch.clamp(tensor, 0.0, 1.0)

    def _calendar_prior_features(self, date_text):
        try:
            date_obj = datetime.strptime(date_text, "%Y-%m-%d")
            day_of_year = date_obj.timetuple().tm_yday
        except ValueError:
            return self.global_clim_mean, self.global_clim_std

        values = [
            coverage
            for year, coverage in self.coverage_by_doy.get(day_of_year, [])
            if year != date_obj.year
        ]
        if len(values) == 0:
            values = [coverage for _, coverage in self.coverage_by_doy.get(day_of_year, [])]
        if len(values) == 0:
            return self.global_clim_mean, self.global_clim_std
        return float(np.mean(values)), float(np.std(values))

    def _date_features(self, date_text):
        try:
            day_of_year = datetime.strptime(date_text, "%Y-%m-%d").timetuple().tm_yday
        except ValueError:
            day_of_year = 1
        angle = 2.0 * math.pi * day_of_year / 365.25
        semi_angle = 2.0 * angle
        clim_mean, clim_std = self._calendar_prior_features(date_text)
        return torch.tensor([
            math.sin(angle),
            math.cos(angle),
            math.sin(semi_angle),
            math.cos(semi_angle),
            clim_mean,
            clim_std
        ], dtype=torch.float32)

    @staticmethod
    def _rolling_stats(values, window=3):
        means, stds, mins, maxs = [], [], [], []
        for i in range(values.size(0)):
            start = max(0, i - window + 1)
            chunk = values[start:i + 1]
            means.append(chunk.mean())
            stds.append(chunk.std(unbiased=False))
            mins.append(chunk.min())
            maxs.append(chunk.max())
        return torch.stack(means), torch.stack(stds), torch.stack(mins), torch.stack(maxs)

    @classmethod
    def _build_scalar_features(cls, coverage, spatial_std, snow_ratio, strong_snow_ratio):
        coverage_delta = torch.zeros_like(coverage)
        coverage_delta[1:] = coverage[1:] - coverage[:-1]
        delta_abs = coverage_delta.abs()
        acceleration = torch.zeros_like(coverage)
        acceleration[2:] = coverage_delta[2:] - coverage_delta[1:-1]
        rolling_mean, rolling_std, rolling_min, rolling_max = cls._rolling_stats(coverage, window=3)
        return torch.stack([
            coverage,
            spatial_std,
            snow_ratio,
            strong_snow_ratio,
            coverage_delta,
            delta_abs,
            acceleration,
            rolling_mean,
            rolling_std,
            rolling_min,
            rolling_max
        ], dim=1)

    def _scalar_features(self, input_seq):
        first_band = input_seq[:, 0, :, :]
        flat = first_band.flatten(1)
        coverage = flat.mean(dim=1)
        spatial_std = flat.std(dim=1, unbiased=False)
        snow_ratio = (flat > 0.01).float().mean(dim=1)
        strong_snow_ratio = (flat > 0.10).float().mean(dim=1)
        return self._build_scalar_features(coverage, spatial_std, snow_ratio, strong_snow_ratio)

    def __len__(self):
        return max(0, len(self.file_list) - self.seq_len + 1)

    def __getitem__(self, idx):
        # 提取完整的高清时间序列，不进行任何破坏全局坐标的裁切
        window_frames = self.data_cache[idx: idx + self.seq_len]
        window_dates = self.date_list[idx: idx + self.seq_len]
        input_frames = window_frames[:-1]
        input_seq = torch.stack(input_frames, dim=0)  # (T, C, 128, 128)
        date_features = torch.stack([self._date_features(date_text) for date_text in window_dates[:-1]], dim=0)
        base_scalar_features = self._scalar_features(input_seq)
        external_seq = torch.stack([self.external_features.get(date_text) for date_text in window_dates[:-1]], dim=0)
        season_features = torch.cat([date_features, external_seq], dim=1)
        scalar_features = torch.cat([base_scalar_features, external_seq], dim=1)
        target_season = torch.cat([
            self._date_features(window_dates[-1]),
            self.external_features.get(window_dates[-1])
        ], dim=0)

        target_frame = window_frames[-1]
        target_map = target_frame[0:1, :, :]  # Predict the next-day snow heatmap.
        return input_seq, season_features, scalar_features, target_season, target_map, self.spatial_prior


# ==========================================
# 3. 融合先验信息的空间编码器
# ==========================================
