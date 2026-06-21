"""Capture code, package, hardware, and split metadata for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qa_experiment.yaml"
TRACKED_FILES = (
    "configs/qa_experiment.yaml",
    "run_qa_experiment.py",
    "scripts/download_qa_data.py",
    "scripts/download_power_regions.py",
    "scripts/preprocess_qa_dataset.py",
    "scripts/run_qa_pipeline.py",
    "scripts/analyze_qa_results.py",
    "src/snow_attractor/qa.py",
    "src/snow_attractor/data.py",
    "src/snow_attractor/model.py",
    "src/snow_attractor/losses.py",
    "src/snow_attractor/evaluation.py",
    "src/snow_attractor/official_baselines.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    destination = Path(config["project"]["artifacts_dir"]) / "reproducibility"
    destination.mkdir(parents=True, exist_ok=True)
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        }
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "splits": config["splits"],
        "data": config["data"],
        "training": config["training"],
        "vendor_commits": {
            name: git_commit(ROOT / "vendor" / directory)
            for name, directory in {
                "SimVPv2": "SimVPv2",
                "SwinLSTM": "SwinLSTM",
                "VMRNN": "VMRNN",
            }.items()
        },
        "code_sha256": {
            relative: sha256(ROOT / relative)
            for relative in TRACKED_FILES
        },
    }
    data_root = Path(config["data"]["root"])
    metadata["processed_metadata"] = {}
    for region_id in (config["data"]["region_id"], config["data"]["external_region_id"]):
        path = data_root / region_id / "processed" / "metadata.json"
        if path.exists():
            metadata["processed_metadata"][region_id] = json.loads(
                path.read_text(encoding="utf-8")
            )
    (destination / "environment.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    (destination / "pip_freeze.txt").write_text(
        freeze.stdout,
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
