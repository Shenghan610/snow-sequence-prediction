"""Build fixed-grid masked arrays from AppEEARS MOD10A1 GeoTIFFs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from snow_attractor.qa import (  # noqa: E402
    REGIONS,
    area_weighted_resize,
    daily_dates,
    decode_mod10a1_masks,
    find_layer_files,
    projected_grid,
    read_to_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data_qa")
    parser.add_argument("--region", choices=tuple(REGIONS), required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    region = REGIONS[args.region]
    raw = root / region.region_id / "raw"
    output = root / region.region_id / "processed"
    output.mkdir(parents=True, exist_ok=True)

    layer_files = {
        "snow": find_layer_files(raw, "NDSI_Snow_Cover"),
        "basic": find_layer_files(raw, "NDSI_Snow_Cover_Basic_QA"),
        "flags": find_layer_files(raw, "NDSI_Snow_Cover_Algorithm_Flags_QA"),
    }
    dates = daily_dates(date.fromisoformat(args.start), date.fromisoformat(args.end))
    shape = (len(dates), args.size, args.size)
    snow = np.lib.format.open_memmap(
        output / "snow.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    strict = np.lib.format.open_memmap(
        output / "valid_strict.npy",
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    lenient = np.lib.format.open_memmap(
        output / "valid_lenient.npy",
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    land_counts = np.zeros((args.size, args.size), dtype=np.uint16)
    observed_counts = np.zeros((args.size, args.size), dtype=np.uint16)
    stats = []

    dst_crs, transform, width, height = projected_grid(region)
    for index, current_date in enumerate(dates):
        paths = {name: mapping.get(current_date) for name, mapping in layer_files.items()}
        if any(path is None for path in paths.values()):
            snow[index] = np.nan
            strict[index] = 0
            lenient[index] = 0
            stats.append(
                {
                    "date": current_date.isoformat(),
                    "available": False,
                    "strict_fraction": 0.0,
                    "lenient_fraction": 0.0,
                }
            )
            continue
        source_snow = read_to_grid(paths["snow"], dst_crs, transform, width, height)
        source_basic = read_to_grid(paths["basic"], dst_crs, transform, width, height)
        source_flags = read_to_grid(paths["flags"], dst_crs, transform, width, height)
        strict_mask, lenient_mask, land_mask = decode_mod10a1_masks(
            source_snow,
            source_basic,
            source_flags,
        )
        snow_small, strict_small, strict_coverage = area_weighted_resize(
            source_snow,
            strict_mask,
            transform,
            dst_crs,
            size=args.size,
        )
        _, lenient_small, _ = area_weighted_resize(
            source_snow,
            lenient_mask,
            transform,
            dst_crs,
            size=args.size,
        )
        _, land_small, land_coverage = area_weighted_resize(
            np.ones_like(source_snow),
            land_mask,
            transform,
            dst_crs,
            size=args.size,
        )
        snow[index] = snow_small / 100.0
        strict[index] = strict_small
        lenient[index] = lenient_small
        land_counts += land_small.astype(np.uint16)
        observed_counts += (land_coverage > 0).astype(np.uint16)
        denominator = max(int(land_small.sum()), 1)
        stats.append(
            {
                "date": current_date.isoformat(),
                "available": True,
                "strict_fraction": float(strict_small.sum() / denominator),
                "lenient_fraction": float(lenient_small.sum() / denominator),
                "mean_strict_coverage": float(strict_coverage[land_small].mean())
                if land_small.any()
                else 0.0,
            }
        )
        if (index + 1) % 50 == 0:
            snow.flush()
            strict.flush()
            lenient.flush()
            print(f"{region.region_id}: {index + 1}/{len(dates)}", flush=True)

    land_mask = land_counts >= np.maximum(observed_counts * 0.5, 1)
    np.save(output / "land_mask.npy", land_mask.astype(np.uint8))
    (output / "dates.json").write_text(
        json.dumps([item.isoformat() for item in dates], indent=2),
        encoding="utf-8",
    )
    (output / "qa_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    metadata = {
        "region_id": region.region_id,
        "bounds_wgs84": region.bbox,
        "source_product": "MOD10A1.061",
        "source_layers": {
            name: len(paths) for name, paths in layer_files.items()
        },
        "date_start": args.start,
        "date_end": args.end,
        "shape": shape,
        "projection": str(dst_crs),
        "source_resolution_m": 500,
        "minimum_cell_valid_coverage": 0.5,
        "target_min_valid_fraction": 0.8,
        "strict_qa": {
            "basic_qa": [0, 1],
            "excluded_algorithm_bits": [0, 1, 3, 4, 7],
        },
        "lenient_qa": {
            "basic_qa": [0, 1, 2],
            "excluded_algorithm_bits": [0, 7],
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    terrain_candidates = [
        path
        for path in raw.rglob("*.tif")
        if "srtmgl1_dem" in path.name.lower()
    ]
    if terrain_candidates:
        elevation = read_to_grid(
            terrain_candidates[0],
            dst_crs,
            transform,
            width,
            height,
        )
        left, bottom, right, top = rasterio.transform.array_bounds(
            height,
            width,
            transform,
        )
        small_transform = rasterio.transform.from_bounds(
            left,
            bottom,
            right,
            top,
            args.size,
            args.size,
        )
        elevation_small = np.full((args.size, args.size), np.nan, dtype=np.float32)
        reproject(
            source=elevation,
            destination=elevation_small,
            src_transform=transform,
            src_crs=dst_crs,
            src_nodata=np.nan,
            dst_transform=small_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        fill = float(np.nanmedian(elevation_small))
        elevation_small = np.nan_to_num(elevation_small, nan=fill)
        x_resolution = abs(small_transform.a)
        y_resolution = abs(small_transform.e)
        dz_dy, dz_dx = np.gradient(
            elevation_small,
            y_resolution,
            x_resolution,
        )
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(-dz_dx, dz_dy)
        terrain = np.stack(
            [
                elevation_small,
                np.rad2deg(slope),
                np.sin(aspect),
                np.cos(aspect),
                np.cos(aspect) * np.sin(slope),
            ],
            axis=0,
        ).astype(np.float32)
        np.save(output / "terrain.npy", terrain)


if __name__ == "__main__":
    main()
