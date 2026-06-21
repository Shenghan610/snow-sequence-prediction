"""MOD10A1 quality masking and area-weighted raster preparation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

CALENDAR_DATE_TOKENS = (
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)"),
)
DAY_OF_YEAR_TOKEN = re.compile(r"(?:doy|A)(20\d{2})(\d{3})", re.IGNORECASE)


@dataclass(frozen=True)
class Region:
    region_id: str
    west: float
    south: float
    east: float
    north: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north

    def geojson(self) -> dict:
        coordinates = [
            [self.west, self.south],
            [self.east, self.south],
            [self.east, self.north],
            [self.west, self.north],
            [self.west, self.south],
        ]
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"region_id": self.region_id},
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                }
            ],
        }


REGIONS = {
    "ali": Region("ali", 78.3944, 29.6778, 86.1975, 35.7153),
    "tianshan": Region("tianshan", 80.0, 40.0, 88.0, 46.0),
}

MODIS_LAYERS = (
    "NDSI_Snow_Cover",
    "NDSI_Snow_Cover_Basic_QA",
    "NDSI_Snow_Cover_Algorithm_Flags_QA",
)


def daily_dates(start: date, end: date) -> list[date]:
    count = (end - start).days + 1
    return [start + timedelta(days=offset) for offset in range(count)]


def decode_mod10a1_masks(
    snow: np.ndarray,
    basic_qa: np.ndarray,
    algorithm_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict, lenient, and land masks for MOD10A1 Collection 6.1."""

    finite = np.isfinite(snow) & np.isfinite(basic_qa) & np.isfinite(algorithm_flags)
    snow_range = (snow >= 0) & (snow <= 100)
    flags = np.where(np.isfinite(algorithm_flags), algorithm_flags, 255).astype(
        np.uint8,
        copy=False,
    )
    inland_water = (flags & (1 << 0)) != 0
    low_visible = (flags & (1 << 1)) != 0
    temperature_height = (flags & (1 << 3)) != 0
    high_swir = (flags & (1 << 4)) != 0
    low_illumination = (flags & (1 << 7)) != 0

    land = finite & ~inland_water
    strict = (
        land
        & snow_range
        & (basic_qa <= 1)
        & ~low_visible
        & ~temperature_height
        & ~high_swir
        & ~low_illumination
    )
    lenient = land & snow_range & (basic_qa <= 2) & ~low_illumination
    return strict, lenient, land


def find_layer_files(directory: Path, layer: str) -> dict[date, Path]:
    matches: dict[date, Path] = {}
    for path in directory.rglob("*.tif"):
        lowered = path.name.lower()
        if layer.lower() not in lowered:
            continue
        if (
            layer == "NDSI_Snow_Cover"
            and (
                "basic_qa" in lowered
                or "algorithm_flags_qa" in lowered
            )
        ):
            continue
        parsed = parse_raster_date(path.name)
        if parsed is None:
            continue
        matches[parsed] = path
    return matches


def parse_raster_date(filename: str) -> date | None:
    for pattern in CALENDAR_DATE_TOKENS:
        token = pattern.search(filename)
        if token is not None:
            try:
                return date(
                    int(token.group(1)),
                    int(token.group(2)),
                    int(token.group(3)),
                )
            except ValueError:
                continue
    token = DAY_OF_YEAR_TOKEN.search(filename)
    if token is None:
        return None
    try:
        return date(int(token.group(1)), 1, 1) + timedelta(
            days=int(token.group(2)) - 1
        )
    except ValueError:
        return None


def projected_grid(
    region: Region,
    resolution: float = 500.0,
    crs: str = "EPSG:6933",
) -> tuple[CRS, rasterio.Affine, int, int]:
    dst_crs = CRS.from_string(crs)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326",
        dst_crs,
        *region.bbox,
        densify_pts=21,
    )
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = from_bounds(left, bottom, right, top, width, height)
    return dst_crs, transform, width, height


def read_to_grid(
    path: Path,
    dst_crs: CRS,
    dst_transform: rasterio.Affine,
    width: int,
    height: int,
) -> np.ndarray:
    destination = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        source_array = source.read(1).astype(np.float32)
        if source.nodata is not None:
            source_array[source_array == source.nodata] = np.nan
        reproject(
            source=source_array,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return destination


def area_weighted_resize(
    values: np.ndarray,
    valid: np.ndarray,
    source_transform: rasterio.Affine,
    source_crs: CRS,
    size: int = 128,
    minimum_valid_coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average valid 500 m cells into a fixed grid without treating gaps as zero."""

    height, width = values.shape
    left, bottom, right, top = rasterio.transform.array_bounds(
        height,
        width,
        source_transform,
    )
    destination_transform = from_bounds(left, bottom, right, top, size, size)
    numerator = np.zeros((size, size), dtype=np.float32)
    coverage = np.zeros((size, size), dtype=np.float32)
    source_weight = valid.astype(np.float32)
    source_value = np.where(valid, values, 0.0).astype(np.float32)
    for source, destination in (
        (source_value, numerator),
        (source_weight, coverage),
    ):
        reproject(
            source=source,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=destination_transform,
            dst_crs=source_crs,
            resampling=Resampling.average,
        )
    output_valid = coverage >= minimum_valid_coverage
    output = np.full((size, size), np.nan, dtype=np.float32)
    output[output_valid] = numerator[output_valid] / coverage[output_valid].clip(1e-6)
    return output, output_valid, coverage


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_windows(valid_dates: Iterable[date], sequence_length: int) -> list[date]:
    available = set(valid_dates)
    starts: list[date] = []
    for target in sorted(available):
        if all(
            target - timedelta(days=offset) in available
            for offset in range(sequence_length + 1)
        ):
            starts.append(target)
    return starts
