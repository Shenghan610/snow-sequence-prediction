"""Print the latest QA experiment progress, metrics, errors, and ETA."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qa_experiment.yaml"


def latest_history(training_root: Path) -> dict | None:
    histories = sorted(
        training_root.rglob("history.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not histories:
        return None
    path = histories[0]
    with open(path, "r", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {"path": str(path.resolve()), **rows[-1]}


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifact_root = Path(config["project"]["artifacts_dir"])
    data_root = Path(config["data"]["root"])
    logs = artifact_root / "pipeline_logs"
    manifest_path = data_root / "appeears_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"tasks": [], "downloads": []}
    )
    completed = []
    for path in logs.glob("train_*.done.json"):
        completed.append(json.loads(path.read_text(encoding="utf-8")))
    formal_done = len(
        list((artifact_root / "training").glob("*/*/seed_*/fold_*/summary.json"))
    )
    duration = median([float(item["seconds"]) for item in completed])
    remaining_jobs = max(33 - formal_done, 0)
    eta_hours = (
        remaining_jobs * duration / 3600.0
        if duration is not None
        else None
    )
    failures = []
    for path in sorted(logs.glob("*.failed.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        failures.append(
            {
                "task": payload["task"],
                "returncode": payload["returncode"],
                "stderr": payload["stderr"],
            }
        )
    tasks = manifest.get("tasks", [])
    report = {
        "reported_at": datetime.now().astimezone().isoformat(),
        "stage": (
            "waiting_for_earthdata_credentials"
            if not tasks
            else "data_download"
            if any(item.get("status") != "done" for item in tasks)
            else "preprocessing_or_training"
        ),
        "power_files": len(list(data_root.rglob("power_point_*.json"))),
        "appeears_tasks": {
            "done": sum(item.get("status") == "done" for item in tasks),
            "total": len(tasks),
        },
        "appeears_downloads": len(manifest.get("downloads", [])),
        "processed_regions": {
            region_id: (
                data_root / region_id / "processed" / "metadata.json"
            ).exists()
            for region_id in ("ali", "tianshan")
        },
        "formal_training": {
            "done": formal_done,
            "total": 33,
            "development_runs": len(
                list(
                    (
                        artifact_root
                        / "training"
                        / "proposed"
                    ).glob("*/seed_42/development/summary.json")
                )
            ),
            "estimated_remaining_hours": eta_hours,
        },
        "latest_metrics": latest_history(artifact_root / "training"),
        "failures": failures,
    }
    (artifact_root / "progress_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
