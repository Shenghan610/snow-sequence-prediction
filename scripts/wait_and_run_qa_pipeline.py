"""Wait for AppEEARS downloads, then launch the full resumable pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data_qa")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=168)
    return parser.parse_args()


def downloads_ready(root: Path) -> tuple[bool, dict]:
    manifest_path = root / "appeears_manifest.json"
    if not manifest_path.exists():
        return False, {"reason": "manifest_missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    events = manifest.get("events", [])
    bundle_events = {
        item.get("task_id")
        for item in events
        if item.get("event") == "task_bundle_downloaded"
    }
    raw_ready = {
        region_id: any((root / region_id / "raw").rglob("*.tif"))
        for region_id in ("ali", "tianshan")
    }
    failed = [item["task_name"] for item in tasks if item.get("status") == "failed"]
    state = {
        "tasks": len(tasks),
        "done": sum(item.get("status") == "done" for item in tasks),
        "failed": failed,
        "bundle_events": len(bundle_events),
        "downloads": len(manifest.get("downloads", [])),
        "raw_ready": raw_ready,
        "partial_files": len(list(root.rglob("*.part"))),
    }
    ready = (
        len(tasks) == 12
        and state["done"] == 12
        and not failed
        and len(bundle_events) == 12
        and all(raw_ready.values())
        and state["partial_files"] == 0
    )
    return ready, state


def main() -> None:
    args = parse_args()
    data_root = Path(args.root).resolve()
    artifact_root = ROOT / "artifacts" / "qa_experiment"
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "watcher_status.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    next_report = 0.0
    while time.monotonic() < deadline:
        ready, state = downloads_ready(data_root)
        payload = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "ready": ready,
            **state,
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if ready:
            break
        if state.get("failed"):
            raise RuntimeError(f"AppEEARS tasks failed: {state['failed']}")
        if time.monotonic() >= next_report:
            subprocess.run(
                [sys.executable, "scripts/report_qa_progress.py"],
                cwd=ROOT,
                check=False,
            )
            next_report = time.monotonic() + 1800
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError("Timed out waiting for AppEEARS downloads")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_qa_pipeline.py",
            "all",
            "--config",
            "configs/qa_experiment.yaml",
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


if __name__ == "__main__":
    main()
