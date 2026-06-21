"""Resumable orchestration for the full QA-aware experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CONFIG = ROOT / "configs" / "qa_experiment.yaml"
VARIANTS = ("residual_only", "dominance_open_gate", "warmstart_dominance")
FOLDS = ("fold_1", "fold_2", "fold_3")
SEEDS = (17, 42, 73)
OFFICIAL_MODELS = (
    "simvpv2_official",
    "swinlstm_official",
    "vmrnn_official",
)
SELECTION_SCORE_RULE = (
    "coverage_r2 - coverage_mae - pixel_mae - pixel_rmse + pixel_ssim "
    "+ snow_iou + snow_f1"
)
EXPERIMENT_REVISION = "dominance_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "status",
            "preprocess",
            "smoke",
            "development",
            "train",
            "evaluate",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def artifacts(config: dict) -> Path:
    return Path(config["project"]["artifacts_dir"])


def validation_selection_score(metrics: dict) -> float:
    return (
        float(metrics["coverage_r2"])
        - float(metrics["coverage_mae"])
        - float(metrics["pixel_mae"])
        - float(metrics["pixel_rmse"])
        + float(metrics["pixel_ssim"])
        + float(metrics["snow_iou"])
        + float(metrics["snow_f1"])
    )


def run_logged(
    command: list[str],
    log_dir: Path,
    task_name: str,
    *,
    force: bool = False,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    done_path = log_dir / f"{task_name}.done.json"
    if done_path.exists() and not force:
        print(f"skip completed task: {task_name}", flush=True)
        return
    stdout_path = log_dir / f"{task_name}.stdout.log"
    stderr_path = log_dir / f"{task_name}.stderr.log"
    started = time.perf_counter()
    print(f"start task: {task_name}", flush=True)
    with open(stdout_path, "a", encoding="utf-8") as stdout, open(
        stderr_path,
        "a",
        encoding="utf-8",
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    payload = {
        "task": task_name,
        "command": command,
        "returncode": process.returncode,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    if process.returncode != 0:
        (log_dir / f"{task_name}.failed.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(process.returncode, command)
    done_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed_path = log_dir / f"{task_name}.failed.json"
    failed_path.unlink(missing_ok=True)
    print(f"completed task: {task_name}", flush=True)


def require_raw_data(config: dict) -> None:
    root = Path(config["data"]["root"])
    manifest_path = root / "appeears_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "AppEEARS manifest is missing. Run scripts/download_qa_data.py all first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    failed = [item["task_name"] for item in tasks if item.get("status") == "failed"]
    incomplete = [item["task_name"] for item in tasks if item.get("status") != "done"]
    if failed:
        raise RuntimeError(f"AppEEARS tasks failed: {failed}")
    if len(tasks) != 12 or incomplete:
        raise RuntimeError(
            f"Expected 12 completed AppEEARS tasks; found {len(tasks)}, "
            f"incomplete={incomplete}"
        )
    for region_id in ("ali", "tianshan"):
        raw = root / region_id / "raw"
        if not raw.exists() or not any(raw.rglob("*.tif")):
            raise RuntimeError(f"No downloaded GeoTIFFs found for {region_id}")


def processed_ready(config: dict, region_id: str) -> bool:
    processed = Path(config["data"]["root"]) / region_id / "processed"
    required = (
        "snow.npy",
        "valid_strict.npy",
        "valid_lenient.npy",
        "land_mask.npy",
        "terrain.npy",
        "dates.json",
        "qa_stats.json",
        "metadata.json",
    )
    return all((processed / name).exists() for name in required)


def preprocess(config: dict, force: bool) -> None:
    require_raw_data(config)
    logs = artifacts(config) / "pipeline_logs"
    for region_id in ("ali", "tianshan"):
        if processed_ready(config, region_id) and not force:
            print(f"skip processed region: {region_id}", flush=True)
            continue
        run_logged(
            [
                str(PYTHON),
                "scripts/preprocess_qa_dataset.py",
                "--root",
                str(config["data"]["root"]),
                "--region",
                region_id,
                "--size",
                str(config["data"]["image_size"]),
            ],
            logs,
            f"preprocess_{region_id}",
            force=force,
        )
    run_logged(
        [str(PYTHON), "scripts/capture_qa_environment.py"],
        logs,
        "capture_environment",
        force=True,
    )


def require_processed(config: dict) -> None:
    missing = [
        region_id
        for region_id in ("ali", "tianshan")
        if not processed_ready(config, region_id)
    ]
    if missing:
        raise RuntimeError(f"Processed QA data missing for: {missing}")


def training_command(
    config_path: Path,
    model: str,
    fold: str,
    seed: int,
    variant: str = "dominance",
    max_epochs: int | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        "run_qa_experiment.py",
        "--config",
        str(config_path),
        "--mode",
        "train",
        "--model",
        model,
        "--fold",
        fold,
        "--seed",
        str(seed),
        "--variant",
        variant,
    ]
    if max_epochs is not None:
        command.extend(["--max-epochs", str(max_epochs)])
    return command


def smoke(config: dict, config_path: Path, force: bool) -> None:
    require_processed(config)
    logs = artifacts(config) / "pipeline_logs"
    for model in (
        "proposed",
        "convlstm",
        *OFFICIAL_MODELS,
        "nearest_memory",
        "no_memory",
    ):
        run_logged(
            [
                str(PYTHON),
                "run_qa_experiment.py",
                "--config",
                str(config_path),
                "--mode",
                "smoke",
                "--model",
                model,
            ],
            logs,
            f"smoke_{model}",
            force=force,
        )


def development(config: dict, config_path: Path, force: bool) -> str:
    require_processed(config)
    logs = artifacts(config) / "pipeline_logs"
    summaries = []
    run_logged(
        training_command(
            config_path,
            "convlstm",
            "development",
            42,
            "dominance",
        ),
        logs,
        f"development_convlstm_seed42_{EXPERIMENT_REVISION}",
        force=force,
    )
    for variant in VARIANTS:
        run_logged(
            training_command(
                config_path,
                "proposed",
                "development",
                42,
                variant,
            ),
            logs,
            f"development_{variant}_{EXPERIMENT_REVISION}",
            force=force,
        )
        summary_path = (
            artifacts(config)
            / "training"
            / "proposed"
            / variant
            / "seed_42"
            / "development"
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["validation"]
        score = validation_selection_score(metrics)
        summaries.append({"variant": variant, "score": score, "metrics": metrics})
    eligible = [
        item
        for item in summaries
        if item["metrics"].get("backbone_win_fraction", 1.0) < 0.5
    ]
    selected_pool = eligible or summaries
    selected = max(selected_pool, key=lambda item: item["score"])["variant"]
    selection = {
        "selected_variant": selected,
        "rule": SELECTION_SCORE_RULE,
        "revision": EXPERIMENT_REVISION,
        "eligibility_rule": "prefer backbone_win_fraction < 0.5 when available",
        "candidates": summaries,
    }
    path = artifacts(config) / "selected_variant.json"
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selected


def selected_variant(config: dict) -> str:
    path = artifacts(config) / "selected_variant.json"
    if not path.exists():
        raise RuntimeError("Development selection is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = {item["variant"] for item in payload.get("candidates", [])}
    if (
        candidates != set(VARIANTS)
        or payload.get("rule") != SELECTION_SCORE_RULE
        or payload.get("revision") != EXPERIMENT_REVISION
    ):
        raise RuntimeError(
            "Development selection is stale. Run the development stage again."
        )
    return payload["selected_variant"]


def formal_jobs(variant: str) -> list[tuple[str, str, int, str]]:
    jobs = []
    for fold in FOLDS:
        for seed in SEEDS:
            jobs.append(("convlstm", fold, seed, "dominance"))
            jobs.append(("proposed", fold, seed, variant))
        for model in OFFICIAL_MODELS:
            jobs.append((model, fold, 42, "dominance"))
        jobs.append(("nearest_memory", fold, 42, "dominance"))
        jobs.append(("no_memory", fold, 42, "dominance"))
    return jobs


def training_summary_path(
    config: dict,
    model: str,
    fold: str,
    seed: int,
    variant: str,
) -> Path:
    suffix = variant if model == "proposed" else "default"
    return (
        artifacts(config)
        / "training"
        / model
        / suffix
        / f"seed_{seed}"
        / fold
        / "summary.json"
    )


def fold1_gate(config: dict, variant: str) -> dict:
    rows = []
    for model, job_variant in (("proposed", variant), ("convlstm", "dominance")):
        for seed in SEEDS:
            path = training_summary_path(config, model, "fold_1", seed, job_variant)
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "metrics": summary["validation"],
                }
            )

    def mean(model: str, metric: str) -> float:
        values = [
            row["metrics"][metric]
            for row in rows
            if row["model"] == model
        ]
        return sum(values) / len(values)

    proposed = {metric: mean("proposed", metric) for metric in (
        "coverage_r2",
        "coverage_mae",
        "pixel_mae",
        "pixel_ssim",
        "snow_iou",
        "snow_f1",
    )}
    convlstm = {metric: mean("convlstm", metric) for metric in proposed}
    direction_wins = {
        "pixel_mae": sum(
            1
            for seed in SEEDS
            if next(
                row["metrics"]["pixel_mae"]
                for row in rows
                if row["model"] == "proposed" and row["seed"] == seed
            )
            < next(
                row["metrics"]["pixel_mae"]
                for row in rows
                if row["model"] == "convlstm" and row["seed"] == seed
            )
        ),
        "pixel_ssim": sum(
            1
            for seed in SEEDS
            if next(
                row["metrics"]["pixel_ssim"]
                for row in rows
                if row["model"] == "proposed" and row["seed"] == seed
            )
            > next(
                row["metrics"]["pixel_ssim"]
                for row in rows
                if row["model"] == "convlstm" and row["seed"] == seed
            )
        ),
        "snow_iou": sum(
            1
            for seed in SEEDS
            if next(
                row["metrics"]["snow_iou"]
                for row in rows
                if row["model"] == "proposed" and row["seed"] == seed
            )
            > next(
                row["metrics"]["snow_iou"]
                for row in rows
                if row["model"] == "convlstm" and row["seed"] == seed
            )
        ),
        "snow_f1": sum(
            1
            for seed in SEEDS
            if next(
                row["metrics"]["snow_f1"]
                for row in rows
                if row["model"] == "proposed" and row["seed"] == seed
            )
            > next(
                row["metrics"]["snow_f1"]
                for row in rows
                if row["model"] == "convlstm" and row["seed"] == seed
            )
        ),
    }
    passed = (
        proposed["pixel_mae"] < convlstm["pixel_mae"]
        and proposed["pixel_ssim"] > convlstm["pixel_ssim"]
        and proposed["snow_iou"] > convlstm["snow_iou"]
        and proposed["snow_f1"] > convlstm["snow_f1"]
        and proposed["coverage_r2"] >= convlstm["coverage_r2"] - 0.01
        and proposed["coverage_mae"] <= convlstm["coverage_mae"] + 0.001
        and sum(value >= 2 for value in direction_wins.values()) >= 3
    )
    payload = {
        "passed": passed,
        "variant": variant,
        "revision": EXPERIMENT_REVISION,
        "proposed_mean": proposed,
        "convlstm_mean": convlstm,
        "direction_wins": direction_wins,
        "criteria": (
            "Proposed mean improves pixel_mae, pixel_ssim, snow_iou, snow_f1; "
            "coverage_r2 and coverage_mae do not materially regress; at least "
            "three of four main directions win in >=2 seeds"
        ),
    }
    path = artifacts(config) / "fold1_gate.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_formal(config: dict, config_path: Path, force: bool) -> bool:
    require_processed(config)
    variant = selected_variant(config)
    logs = artifacts(config) / "pipeline_logs"
    model_max_epochs = config.get("training", {}).get("model_max_epochs", {})

    fold1_core = [
        (model, fold, seed, job_variant)
        for model, fold, seed, job_variant in formal_jobs(variant)
        if fold == "fold_1" and model in {"convlstm", "proposed"}
    ]
    remaining = [
        job
        for job in formal_jobs(variant)
        if job not in fold1_core
    ]

    for model, fold, seed, job_variant in fold1_core:
        task = f"train_{model}_{job_variant}_{fold}_seed{seed}_{EXPERIMENT_REVISION}"
        run_logged(
            training_command(
                config_path,
                model,
                fold,
                seed,
                job_variant,
                max_epochs=model_max_epochs.get(model),
            ),
            logs,
            task,
            force=force,
        )
    gate = fold1_gate(config, variant)
    print(f"fold1 gate passed: {gate['passed']}", flush=True)
    if not gate["passed"]:
        print("stop formal matrix because fold_1 gate did not pass", flush=True)
        return False

    for model, fold, seed, job_variant in remaining:
        task = f"train_{model}_{job_variant}_{fold}_seed{seed}_{EXPERIMENT_REVISION}"
        run_logged(
            training_command(
                config_path,
                model,
                fold,
                seed,
                job_variant,
                max_epochs=model_max_epochs.get(model),
            ),
            logs,
            task,
            force=force,
        )
    return True


def checkpoint_path(
    config: dict,
    model: str,
    fold: str,
    seed: int,
    variant: str,
) -> Path:
    suffix = variant if model == "proposed" else "default"
    return (
        artifacts(config)
        / "training"
        / model
        / suffix
        / f"seed_{seed}"
        / fold
        / "best_model.pt"
    )


def evaluate(config: dict, config_path: Path, force: bool) -> None:
    variant = selected_variant(config)
    logs = artifacts(config) / "pipeline_logs"
    jobs = formal_jobs(variant)
    for model, fold, seed, job_variant in jobs:
        checkpoint = checkpoint_path(
            config,
            model,
            fold,
            seed,
            job_variant,
        )
        if not checkpoint.exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
        evaluations = ["validation"]
        if fold == "fold_3":
            evaluations.extend(["frozen_test", "external"])
        for evaluation_name in evaluations:
            task = (
                f"evaluate_{model}_{job_variant}_{fold}_seed{seed}_"
                f"{evaluation_name}"
            )
            run_logged(
                [
                    str(PYTHON),
                    "run_qa_experiment.py",
                    "--config",
                    str(config_path),
                    "--mode",
                    "evaluate",
                    "--model",
                    model,
                    "--checkpoint",
                    str(checkpoint),
                    "--evaluation",
                    evaluation_name,
                ],
                logs,
                task,
                force=force,
            )


def analyze(config: dict, config_path: Path, force: bool) -> None:
    run_logged(
        [
            str(PYTHON),
            "scripts/analyze_qa_results.py",
            "--config",
            str(config_path),
        ],
        artifacts(config) / "pipeline_logs",
        "analyze_qa_results",
        force=force,
    )


def status(config: dict) -> dict:
    root = Path(config["data"]["root"])
    manifest_path = root / "appeears_manifest.json"
    tasks = []
    downloads = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = manifest.get("tasks", [])
        downloads = manifest.get("downloads", [])
    logs = artifacts(config) / "pipeline_logs"
    failed = sorted(path.stem.replace(".failed", "") for path in logs.glob("*.failed.json"))
    done = list(logs.glob("*.done.json"))
    variant_path = artifacts(config) / "selected_variant.json"
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "power_files": len(list(root.rglob("power_point_*.json"))),
        "appeears_tasks": {
            "total": len(tasks),
            "done": sum(item.get("status") == "done" for item in tasks),
            "failed": sum(item.get("status") == "failed" for item in tasks),
        },
        "appeears_downloads": len(downloads),
        "processed": {
            region_id: processed_ready(config, region_id)
            for region_id in ("ali", "tianshan")
        },
        "selected_variant": (
            json.loads(variant_path.read_text(encoding="utf-8"))["selected_variant"]
            if variant_path.exists()
            else None
        ),
        "pipeline_tasks_completed": len(done),
        "pipeline_failures": failed,
        "formal_training_summaries": len(
            list((artifacts(config) / "training").glob("*/*/seed_*/fold_*/summary.json"))
        ),
    }
    artifacts(config).mkdir(parents=True, exist_ok=True)
    (artifacts(config) / "pipeline_status.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "status":
        print(json.dumps(status(config), indent=2))
        return
    if args.stage in {"preprocess", "all"}:
        preprocess(config, args.force)
    if args.stage in {"smoke", "all"}:
        smoke(config, config_path, args.force)
    if args.stage in {"development", "all"}:
        development(config, config_path, args.force)
    training_completed = True
    if args.stage in {"train", "all"}:
        training_completed = train_formal(config, config_path, args.force)
    if args.stage in {"evaluate", "all"} and training_completed:
        evaluate(config, config_path, args.force)
    if args.stage in {"analyze", "all"} and training_completed:
        analyze(config, config_path, args.force)
    print(json.dumps(status(config), indent=2))


if __name__ == "__main__":
    main()
