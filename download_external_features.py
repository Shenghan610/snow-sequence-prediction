import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests


STUDY_REGION = {
    "west": 78.3944,
    "south": 29.6778,
    "east": 86.1975,
    "north": 35.7153,
}

START_DATE = "20200101"
END_DATE = "20241231"
GRID_SIZE = 5
OUTPUT_DIR = Path("ExternalClimateTerrain")
RAW_DIR = OUTPUT_DIR / "power_grid_daily_raw"
DAILY_CSV = OUTPUT_DIR / "external_daily_features.csv"
GRID_CSV = OUTPUT_DIR / "external_grid_points.csv"
TERRAIN_CSV = OUTPUT_DIR / "terrain_summary.csv"

POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_LW_DWN",
]


def make_grid(grid_size=GRID_SIZE):
    lons = np.linspace(STUDY_REGION["west"], STUDY_REGION["east"], grid_size)
    lats = np.linspace(STUDY_REGION["south"], STUDY_REGION["north"], grid_size)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def fetch_power_point(lat, lon, index):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"power_point_{index:02d}_{lat:.3f}_{lon:.3f}.json"
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    response = requests.get(POWER_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    time.sleep(0.4)
    return data


def terrain_from_elevation_grid(points):
    grid_size = int(math.sqrt(len(points)))
    lats = np.array([p["latitude"] for p in points], dtype=np.float64).reshape(grid_size, grid_size)
    lons = np.array([p["longitude"] for p in points], dtype=np.float64).reshape(grid_size, grid_size)
    elev = np.array([p["elevation_m"] for p in points], dtype=np.float64).reshape(grid_size, grid_size)

    lat_spacing_m = np.gradient(lats[:, 0]) * 111_320.0
    lon_spacing_m = np.gradient(lons[0, :]) * 111_320.0 * np.cos(np.deg2rad(lats.mean()))
    dz_dy = np.gradient(elev, axis=0) / (lat_spacing_m[:, None] + 1e-6)
    dz_dx = np.gradient(elev, axis=1) / (lon_spacing_m[None, :] + 1e-6)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.rad2deg(slope_rad)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    terrain_radiation_index = aspect_cos * np.sin(slope_rad)

    return {
        "elevation_mean": float(np.mean(elev)),
        "elevation_std": float(np.std(elev)),
        "elevation_min": float(np.min(elev)),
        "elevation_max": float(np.max(elev)),
        "slope_mean": float(np.mean(slope_deg)),
        "slope_std": float(np.std(slope_deg)),
        "aspect_sin_mean": float(np.mean(aspect_sin)),
        "aspect_cos_mean": float(np.mean(aspect_cos)),
        "terrain_radiation_index_mean": float(np.mean(terrain_radiation_index)),
    }


def aggregate_daily(point_payloads):
    all_dates = sorted(point_payloads[0]["properties"]["parameter"]["T2M"].keys())
    rows = []
    for date_key in all_dates:
        row = {"date": datetime.strptime(date_key, "%Y%m%d").strftime("%Y-%m-%d")}
        for param in POWER_PARAMETERS:
            values = []
            for payload in point_payloads:
                value = payload["properties"]["parameter"].get(param, {}).get(date_key)
                if value is None or value == -999:
                    continue
                values.append(float(value))
            if len(values) == 0:
                mean_value = float("nan")
                std_value = float("nan")
            else:
                arr = np.array(values, dtype=np.float64)
                mean_value = float(np.mean(arr))
                std_value = float(np.std(arr))
            row[f"{param}_mean"] = mean_value
            row[f"{param}_std"] = std_value

        row["T2M_range_mean"] = row["T2M_MAX_mean"] - row["T2M_MIN_mean"]
        row["melt_degree_mean"] = max(row["T2M_mean"], 0.0)
        row["cold_degree_mean"] = max(-row["T2M_mean"], 0.0)
        row["snowfall_proxy_mean"] = row["PRECTOTCORR_mean"] if row["T2M_mean"] <= 0.5 else 0.0
        row["rainfall_proxy_mean"] = row["PRECTOTCORR_mean"] if row["T2M_mean"] > 0.5 else 0.0
        row["wind_solar_interaction"] = row["WS10M_mean"] * row["ALLSKY_SFC_SW_DWN_mean"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    point_payloads = []
    point_rows = []
    print(f"Downloading NASA POWER daily data for {len(grid)} grid points...")
    for index, (lat, lon) in enumerate(grid):
        payload = fetch_power_point(lat, lon, index)
        elevation = payload.get("geometry", {}).get("coordinates", [lon, lat, float("nan")])[2]
        point_rows.append({
            "point_id": index,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": float(elevation),
        })
        point_payloads.append(payload)
        print(f"  point {index + 1:02d}/{len(grid)} lat={lat:.3f} lon={lon:.3f} elev={float(elevation):.1f}m")

    daily_rows = aggregate_daily(point_payloads)
    terrain = terrain_from_elevation_grid(point_rows)
    for row in daily_rows:
        row.update(terrain)

    write_csv(GRID_CSV, point_rows)
    write_csv(DAILY_CSV, daily_rows)
    write_csv(TERRAIN_CSV, [terrain])
    print(f"Wrote {DAILY_CSV} with {len(daily_rows)} daily rows and {len(daily_rows[0]) - 1} features.")
    print(f"Wrote {GRID_CSV}")
    print(f"Wrote {TERRAIN_CSV}")


if __name__ == "__main__":
    main()
