"""Download NASA POWER daily fields on a 5x5 grid for both study regions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from snow_attractor.data import POWER_PARAMETERS  # noqa: E402
from snow_attractor.qa import REGIONS, sha256_file  # noqa: E402

POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data_qa")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=tuple(REGIONS),
        default=list(REGIONS),
        help="Region ids to download. Use --regions ali for v7.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned region/date/point counts without writing files.",
    )
    return parser.parse_args()


def selected_regions(region_ids: Iterable[str]):
    return [REGIONS[item] for item in region_ids]


def planned_regions(region_ids: Iterable[str], start: str, end: str) -> list[dict]:
    return [
        {
            "region_id": region.region_id,
            "start": start,
            "end": end,
            "grid_points": 25,
            "parameters": list(POWER_PARAMETERS),
        }
        for region in selected_regions(region_ids)
    ]


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(planned_regions(args.regions, args.start, args.end), indent=2))
        return
    root = Path(args.root)
    for region in selected_regions(args.regions):
        directory = root / region.region_id / "external_features"
        raw = directory / "power_grid_daily_raw"
        raw.mkdir(parents=True, exist_ok=True)
        latitudes = np.linspace(region.south, region.north, 5)
        longitudes = np.linspace(region.west, region.east, 5)
        rows = []
        manifest = []
        point_id = 0
        for latitude in latitudes:
            for longitude in longitudes:
                destination = raw / (
                    f"power_point_{point_id:02d}_{latitude:.3f}_{longitude:.3f}.json"
                )
                params = {
                    "parameters": ",".join(POWER_PARAMETERS),
                    "community": "AG",
                    "longitude": f"{longitude:.6f}",
                    "latitude": f"{latitude:.6f}",
                    "start": args.start,
                    "end": args.end,
                    "format": "JSON",
                }
                if not destination.exists():
                    for attempt in range(1, 6):
                        try:
                            response = requests.get(
                                POWER_API,
                                params=params,
                                timeout=120,
                            )
                            response.raise_for_status()
                            payload = response.json()
                            temporary = destination.with_suffix(".json.part")
                            temporary.write_text(
                                json.dumps(payload),
                                encoding="utf-8",
                            )
                            os.replace(temporary, destination)
                            break
                        except (requests.RequestException, ValueError) as error:
                            print(
                                f"[{region.region_id}] point={point_id:02d} "
                                f"attempt={attempt}/5 failed: {error}",
                                flush=True,
                            )
                            if attempt == 5:
                                raise
                            time.sleep(2**attempt)
                payload = json.loads(destination.read_text(encoding="utf-8"))
                elevation = payload.get("geometry", {}).get("coordinates", [0, 0, 0])[2]
                rows.append(
                    {
                        "point_id": point_id,
                        "latitude": f"{latitude:.6f}",
                        "longitude": f"{longitude:.6f}",
                        "elevation_m": elevation,
                    }
                )
                manifest.append(
                    {
                        "point_id": point_id,
                        "path": str(destination.resolve()),
                        "sha256": sha256_file(destination),
                    }
                )
                print(
                    f"[{region.region_id}] point={point_id + 1:02d}/25 "
                    f"lat={latitude:.3f} lon={longitude:.3f}",
                    flush=True,
                )
                point_id += 1
        directory.mkdir(parents=True, exist_ok=True)
        with open(
            directory / "external_grid_points.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (directory / "power_manifest.json").write_text(
            json.dumps(
                {
                    "region_id": region.region_id,
                    "start": args.start,
                    "end": args.end,
                    "parameters": list(POWER_PARAMETERS),
                    "files": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{region.region_id}] completed: {len(manifest)} files",
            flush=True,
        )


if __name__ == "__main__":
    main()
