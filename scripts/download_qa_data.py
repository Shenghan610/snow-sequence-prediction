"""Submit and download AppEEARS MOD10A1/SRTM tasks for the QA experiment."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from snow_attractor.qa import MODIS_LAYERS, REGIONS, sha256_file  # noqa: E402

API = "https://appeears.earthdatacloud.nasa.gov/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("submit", "status", "download", "all"),
    )
    parser.add_argument("--root", default="data_qa")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=tuple(REGIONS),
        default=list(REGIONS),
        help="Region ids to submit. Use --regions ali for v7.",
    )
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned task names and payloads without logging in or writing files.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--file-attempts",
        type=int,
        default=20,
        help="Retries per individual bundle file before deferring it to a later pass.",
    )
    parser.add_argument(
        "--download-idle-sleep",
        type=int,
        default=120,
        help="Seconds to sleep between download passes when files are still missing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent file downloads. Use >1 only for already completed tasks.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP(S)_PROXY/ALL_PROXY environment variables for this process.",
    )
    return parser.parse_args()


def selected_regions(region_ids: Iterable[str]):
    return [REGIONS[item] for item in region_ids]


def modis_payload(region, year: int) -> dict:
    task_name = f"snow-qa-{region.region_id}-{year}"
    return {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": f"01-01-{year}",
                    "endDate": f"12-31-{year}",
                }
            ],
            "layers": [
                {"product": "MOD10A1.061", "layer": layer}
                for layer in MODIS_LAYERS
            ],
            "geo": region.geojson(),
            "output": {
                "format": {"type": "geotiff", "filename_date": "calendar"},
                "projection": "geographic",
            },
        },
    }


def terrain_payload(region) -> dict:
    task_name = f"terrain-srtm-{region.region_id}"
    return {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [{"startDate": "02-11-2000", "endDate": "02-22-2000"}],
            "layers": [
                {"product": "SRTMGL1_NC.003", "layer": "SRTMGL1_DEM"}
            ],
            "geo": region.geojson(),
            "output": {
                "format": {"type": "geotiff", "filename_date": "calendar"},
                "projection": "geographic",
            },
        },
    }


def planned_tasks(region_ids: Iterable[str], start_year: int, end_year: int) -> list[dict]:
    tasks = []
    for region in selected_regions(region_ids):
        for year in range(start_year, end_year + 1):
            tasks.append(
                {
                    "task_name": f"snow-qa-{region.region_id}-{year}",
                    "kind": "modis",
                    "region_id": region.region_id,
                    "year": year,
                    "request": modis_payload(region, year),
                }
            )
        tasks.append(
            {
                "task_name": f"terrain-srtm-{region.region_id}",
                "kind": "terrain",
                "region_id": region.region_id,
                "request": terrain_payload(region),
            }
        )
    return tasks


def load_manifest(root: Path) -> dict:
    path = root / "appeears_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 2, "tasks": [], "downloads": [], "events": []}


def save_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "appeears_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def record_event(
    root: Path,
    manifest: dict,
    event: str,
    **details,
) -> None:
    manifest.setdefault("events", []).append(
        {
            "event": event,
            "time": datetime.now(timezone.utc).isoformat(),
            **details,
        }
    )
    save_manifest(root, manifest)


def api_request(
    method: str,
    url: str,
    *,
    attempts: int = 5,
    **kwargs,
) -> requests.Response:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("Unreachable retry state")


def token() -> str:
    existing = os.environ.get("APPEEARS_TOKEN")
    if existing:
        return existing
    username = os.environ.get("EARTHDATA_USERNAME") or input("Earthdata username: ")
    password = os.environ.get("EARTHDATA_PASSWORD") or getpass.getpass(
        "Earthdata password: "
    )
    response = api_request(
        "POST",
        f"{API}/login",
        auth=(username, password),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["token"]


def request_headers(bearer: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}


def task_exists(manifest: dict, task_name: str) -> bool:
    return any(
        item["task_name"] == task_name and item.get("status") != "failed"
        for item in manifest["tasks"]
    )


def failed_task(manifest: dict, task_name: str) -> dict | None:
    return next(
        (
            item
            for item in manifest["tasks"]
            if item["task_name"] == task_name and item.get("status") == "failed"
        ),
        None,
    )


def submit_tasks(
    root: Path,
    bearer: str,
    manifest: dict,
    *,
    region_ids: Iterable[str],
    start_year: int,
    end_year: int,
) -> None:
    headers = request_headers(bearer)
    for region in selected_regions(region_ids):
        for year in range(start_year, end_year + 1):
            task_name = f"snow-qa-{region.region_id}-{year}"
            if task_exists(manifest, task_name):
                continue
            payload = modis_payload(region, year)
            try:
                response = api_request(
                    "POST",
                    f"{API}/task",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
            except requests.RequestException as error:
                record_event(
                    root,
                    manifest,
                    "submission_failed",
                    task_name=task_name,
                    error=str(error),
                    request=payload,
                    attempts=5,
                )
                raise
            item = response.json()
            retry = failed_task(manifest, task_name)
            task_record = {
                "task_id": item["task_id"],
                "task_name": task_name,
                "kind": "modis",
                "region_id": region.region_id,
                "year": year,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "request": payload,
                "status": "submitted",
                "attempts": int(retry.get("attempts", 0)) + 1 if retry else 1,
            }
            if retry is None:
                manifest["tasks"].append(task_record)
            else:
                retry.clear()
                retry.update(task_record)
            save_manifest(root, manifest)
            record_event(
                root,
                manifest,
                "submitted",
                task_name=task_name,
                task_id=item["task_id"],
            )

        task_name = f"terrain-srtm-{region.region_id}"
        if task_exists(manifest, task_name):
            continue
        payload = terrain_payload(region)
        try:
            response = api_request(
                "POST",
                f"{API}/task",
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.RequestException as error:
            record_event(
                root,
                manifest,
                "submission_failed",
                task_name=task_name,
                error=str(error),
                request=payload,
                attempts=5,
            )
            raise
        item = response.json()
        retry = failed_task(manifest, task_name)
        task_record = {
            "task_id": item["task_id"],
            "task_name": task_name,
            "kind": "terrain",
            "region_id": region.region_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "request": payload,
            "status": "submitted",
            "attempts": int(retry.get("attempts", 0)) + 1 if retry else 1,
        }
        if retry is None:
            manifest["tasks"].append(task_record)
        else:
            retry.clear()
            retry.update(task_record)
        save_manifest(root, manifest)
        record_event(
            root,
            manifest,
            "submitted",
            task_name=task_name,
            task_id=item["task_id"],
        )


def refresh_status(root: Path, bearer: str, manifest: dict) -> bool:
    headers = request_headers(bearer)
    complete = True
    for item in manifest["tasks"]:
        if item.get("status") == "done":
            continue
        response = api_request(
            "GET",
            f"{API}/status/{item['task_id']}",
            headers=headers,
            timeout=60,
            allow_redirects=False,
        )
        if response.status_code == 303:
            status = "done"
        else:
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status", payload.get("status_type", "processing"))
            item["progress"] = payload.get("progress")
        item["status"] = status
        item["updated_at"] = datetime.now(timezone.utc).isoformat()
        complete &= status in {"done", "failed"}
        if status == "failed":
            record_event(
                root,
                manifest,
                "task_failed",
                task_name=item["task_name"],
                task_id=item["task_id"],
                response=payload,
            )
    save_manifest(root, manifest)
    return complete


def download_completed(
    root: Path,
    bearer: str,
    manifest: dict,
    *,
    file_attempts: int = 20,
) -> dict[str, int]:
    headers = request_headers(bearer)
    known = {item["file_id"] for item in manifest["downloads"]}
    downloaded = 0
    failed = 0
    pending = 0
    for task in manifest["tasks"]:
        if task.get("status") != "done":
            continue
        try:
            response = api_request(
                "GET",
                f"{API}/bundle/{task['task_id']}",
                headers=headers,
                timeout=120,
            )
        except requests.RequestException as error:
            failed += 1
            record_event(
                root,
                manifest,
                "bundle_list_failed",
                task_id=task["task_id"],
                task_name=task["task_name"],
                error=str(error),
            )
            continue
        files = response.json().get("files", [])
        for item in files:
            if item["file_id"] in known:
                continue
            pending += 1
            destination = (
                root
                / task["region_id"]
                / "raw"
                / task["task_name"]
                / item["file_name"]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with api_request(
                    "GET",
                    f"{API}/bundle/{task['task_id']}/{item['file_id']}",
                    headers=headers,
                    stream=True,
                    timeout=(30, 300),
                    attempts=file_attempts,
                ) as file_response:
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    with open(temporary, "wb") as handle:
                        for chunk in file_response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    temporary.replace(destination)
            except requests.RequestException as error:
                failed += 1
                record_event(
                    root,
                    manifest,
                    "download_failed",
                    task_name=task["task_name"],
                    task_id=task["task_id"],
                    file_id=item["file_id"],
                    file_name=item["file_name"],
                    error=str(error),
                    attempts=file_attempts,
                )
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.unlink(missing_ok=True)
                continue
            manifest["downloads"].append(
                {
                    "task_id": task["task_id"],
                    "file_id": item["file_id"],
                    "file_name": item["file_name"],
                    "path": str(destination.resolve()),
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            known.add(item["file_id"])
            downloaded += 1
            if len(manifest["downloads"]) % 25 == 0:
                save_manifest(root, manifest)
        record_event(
            root,
            manifest,
            "task_bundle_downloaded",
            task_id=task["task_id"],
            task_name=task["task_name"],
            total_downloads=len(manifest["downloads"]),
            pass_downloaded=downloaded,
            pass_failed=failed,
            pass_pending=pending,
        )
    save_manifest(root, manifest)
    return {"downloaded": downloaded, "failed": failed, "pending": pending}


def _download_one_file(
    root: Path,
    bearer: str,
    task: dict,
    item: dict,
    *,
    file_attempts: int,
) -> tuple[str, dict | None, str | None]:
    headers = request_headers(bearer)
    destination = (
        root
        / task["region_id"]
        / "raw"
        / task["task_name"]
        / item["file_name"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        return (
            "downloaded",
            {
                "task_id": task["task_id"],
                "file_id": item["file_id"],
                "file_name": item["file_name"],
                "path": str(destination.resolve()),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "reconciled_existing_file": True,
            },
            None,
        )
    try:
        with api_request(
            "GET",
            f"{API}/bundle/{task['task_id']}/{item['file_id']}",
            headers=headers,
            stream=True,
            timeout=(30, 300),
            attempts=file_attempts,
        ) as file_response:
            with open(temporary, "wb") as handle:
                for chunk in file_response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(destination)
    except requests.RequestException as error:
        temporary.unlink(missing_ok=True)
        return ("failed", None, str(error))
    return (
        "downloaded",
        {
            "task_id": task["task_id"],
            "file_id": item["file_id"],
            "file_name": item["file_name"],
            "path": str(destination.resolve()),
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
        None,
    )


def download_completed_parallel(
    root: Path,
    bearer: str,
    manifest: dict,
    *,
    file_attempts: int = 20,
    workers: int = 8,
) -> dict[str, int]:
    headers = request_headers(bearer)
    known = {item["file_id"] for item in manifest["downloads"]}
    pending_items: list[tuple[dict, dict]] = []
    failed = 0
    for task in manifest["tasks"]:
        if task.get("status") != "done":
            continue
        try:
            response = api_request(
                "GET",
                f"{API}/bundle/{task['task_id']}",
                headers=headers,
                timeout=120,
            )
        except requests.RequestException as error:
            failed += 1
            record_event(
                root,
                manifest,
                "bundle_list_failed",
                task_id=task["task_id"],
                task_name=task["task_name"],
                error=str(error),
            )
            continue
        for item in response.json().get("files", []):
            if item["file_id"] not in known:
                pending_items.append((task, item))
    if not pending_items:
        save_manifest(root, manifest)
        return {"downloaded": 0, "failed": failed, "pending": 0}

    downloaded = 0
    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _download_one_file,
                root,
                bearer,
                task,
                item,
                file_attempts=file_attempts,
            )
            for task, item in pending_items
        ]
        for future in as_completed(futures):
            status, record, error = future.result()
            if status == "downloaded" and record is not None:
                if record["file_id"] not in known:
                    manifest["downloads"].append(record)
                    known.add(record["file_id"])
                    downloaded += 1
            else:
                failed += 1
                record_event(
                    root,
                    manifest,
                    "parallel_download_failed",
                    error=error,
                )
            if downloaded and downloaded % 25 == 0:
                print(f"downloaded={downloaded}/{len(pending_items)}", flush=True)
                save_manifest(root, manifest)
    record_event(
        root,
        manifest,
        "parallel_download_pass",
        downloaded=downloaded,
        failed=failed,
        pending=len(pending_items),
        workers=workers,
    )
    save_manifest(root, manifest)
    return {"downloaded": downloaded, "failed": failed, "pending": len(pending_items)}


def main() -> None:
    args = parse_args()
    if args.no_proxy:
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            os.environ.pop(key, None)
    root = Path(args.root)
    if args.dry_run:
        print(
            json.dumps(
                planned_tasks(args.regions, args.start_year, args.end_year),
                indent=2,
            )
        )
        return
    manifest = load_manifest(root)
    bearer = token()
    if args.command in {"submit", "all"}:
        submit_tasks(
            root,
            bearer,
            manifest,
            region_ids=args.regions,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    if args.command == "status":
        refresh_status(root, bearer, manifest)
    if args.command == "download":
        refresh_status(root, bearer, manifest)
        while True:
            if args.workers > 1:
                stats = download_completed_parallel(
                    root,
                    bearer,
                    manifest,
                    file_attempts=args.file_attempts,
                    workers=args.workers,
                )
            else:
                stats = download_completed(
                    root,
                    bearer,
                    manifest,
                    file_attempts=args.file_attempts,
                )
            if stats["pending"] == 0 and stats["failed"] == 0:
                break
            if stats["downloaded"] == 0:
                break
            time.sleep(args.download_idle_sleep)
    if args.command == "all":
        while not refresh_status(root, bearer, manifest):
            if args.workers > 1:
                download_completed_parallel(
                    root,
                    bearer,
                    manifest,
                    file_attempts=args.file_attempts,
                    workers=args.workers,
                )
            else:
                download_completed(root, bearer, manifest, file_attempts=args.file_attempts)
            time.sleep(args.poll_seconds)
        while True:
            if args.workers > 1:
                stats = download_completed_parallel(
                    root,
                    bearer,
                    manifest,
                    file_attempts=args.file_attempts,
                    workers=args.workers,
                )
            else:
                stats = download_completed(
                    root,
                    bearer,
                    manifest,
                    file_attempts=args.file_attempts,
                )
            if stats["pending"] == 0 and stats["failed"] == 0:
                break
            record_event(root, manifest, "download_pass_incomplete", **stats)
            time.sleep(args.download_idle_sleep)


if __name__ == "__main__":
    main()
