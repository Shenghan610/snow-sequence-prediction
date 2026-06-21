"""Paired temporal bootstrap analysis for QA-aware experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOWER_IS_BETTER = (
    "coverage_mae",
    "coverage_rmse",
    "pixel_mae",
    "pixel_rmse",
)
HIGHER_IS_BETTER = ("coverage_r2", "pixel_ssim", "snow_iou", "snow_f1")
METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER
PRIMARY_METRICS = ("coverage_r2", "pixel_mae", "pixel_ssim", "snow_f1")
FOLDS = ("fold_1", "fold_2", "fold_3")
SEEDS = (17, 42, 73)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qa_experiment.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    def parse_value(value: str) -> str | float:
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, "r", encoding="utf-8-sig") as handle:
        return [
            {
                key: parse_value(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def aggregate_seed_rows(paths: list[Path]) -> list[dict]:
    collections = [read_rows(path) for path in paths]
    date_sets = [{row["date"] for row in rows} for rows in collections]
    common = sorted(set.intersection(*date_sets))
    indexed = [
        {row["date"]: row for row in rows}
        for rows in collections
    ]
    aggregate = []
    for current_date in common:
        rows = [items[current_date] for items in indexed]
        aggregate.append(
            {
                "date": current_date,
                **{
                    key: float(np.mean([row[key] for row in rows]))
                    for key in rows[0]
                    if key != "date"
                },
            }
        )
    return aggregate


def coverage_r2(rows: list[dict], indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(rows))
    target = np.asarray([rows[index]["target_coverage"] for index in indices])
    prediction = np.asarray(
        [rows[index]["predicted_coverage"] for index in indices]
    )
    denominator = np.square(target - target.mean()).sum()
    return float(1.0 - np.square(prediction - target).sum() / max(denominator, 1e-8))


def metric_value(rows: list[dict], metric: str, indices=None) -> float:
    if metric == "coverage_r2":
        return coverage_r2(rows, indices)
    if indices is None:
        indices = np.arange(len(rows))
    selected = [rows[index] for index in indices]
    if metric == "coverage_mae":
        return float(
            np.mean(
                [
                    abs(row["predicted_coverage"] - row["target_coverage"])
                    for row in selected
                ]
            )
        )
    if metric == "coverage_rmse":
        return float(
            np.sqrt(
                np.mean(
                    [
                        (row["predicted_coverage"] - row["target_coverage"]) ** 2
                        for row in selected
                    ]
                )
            )
        )
    if metric == "pixel_mae":
        denominator = sum(row["valid_pixel_count"] for row in selected)
        return float(
            sum(row["pixel_absolute_sum"] for row in selected)
            / max(denominator, 1.0)
        )
    if metric == "pixel_rmse":
        denominator = sum(row["valid_pixel_count"] for row in selected)
        return float(
            np.sqrt(
                sum(row["pixel_squared_sum"] for row in selected)
                / max(denominator, 1.0)
            )
        )
    if metric in {"snow_iou", "snow_f1"}:
        true_positive = sum(row["snow_true_positive"] for row in selected)
        false_positive = sum(row["snow_false_positive"] for row in selected)
        false_negative = sum(row["snow_false_negative"] for row in selected)
        if metric == "snow_iou":
            return float(
                true_positive
                / max(true_positive + false_positive + false_negative, 1.0)
            )
        return float(
            2.0
            * true_positive
            / max(2.0 * true_positive + false_positive + false_negative, 1.0)
        )
    return float(np.mean([rows[index][metric] for index in indices]))


def oriented_improvement(
    candidate: list[dict],
    baseline: list[dict],
    metric: str,
    indices=None,
) -> float:
    candidate_value = metric_value(candidate, metric, indices)
    baseline_value = metric_value(baseline, metric, indices)
    if metric in LOWER_IS_BETTER:
        return baseline_value - candidate_value
    return candidate_value - baseline_value


def moving_block_indices(
    length: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(np.ceil(length / block_size))
    starts = rng.integers(0, length, size=blocks)
    indices = np.concatenate(
        [
            (np.arange(start, start + block_size) % length)
            for start in starts
        ]
    )
    return indices[:length]


def paired_bootstrap(
    candidate: list[dict],
    baseline: list[dict],
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    if [row["date"] for row in candidate] != [row["date"] for row in baseline]:
        raise ValueError("Candidate and baseline dates do not match")
    observed = oriented_improvement(candidate, baseline, metric)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = moving_block_indices(len(candidate), 7, rng)
        samples[replicate] = oriented_improvement(
            candidate,
            baseline,
            metric,
            indices,
        )
    p_value = float((1 + np.sum(samples <= 0)) / (replicates + 1))
    return {
        "candidate": metric_value(candidate, metric),
        "baseline": metric_value(baseline, metric),
        "oriented_improvement": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_value_one_sided": p_value,
    }


def holm_adjust(results: dict[str, dict]) -> None:
    ordered = sorted(results, key=lambda key: results[key]["p_value_one_sided"])
    count = len(ordered)
    running = 0.0
    for rank, metric in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * results[metric]["p_value_one_sided"])
        running = max(running, adjusted)
        results[metric]["p_value_holm"] = running


def evaluation_path(
    artifacts: Path,
    model: str,
    variant: str,
    seed: int,
    fold: str,
    evaluation: str,
    region_id: str,
) -> Path:
    suffix = variant if model == "proposed" else "default"
    return (
        artifacts
        / "training"
        / model
        / suffix
        / f"seed_{seed}"
        / fold
        / "evaluation"
        / evaluation
        / region_id
        / "per_sample.csv"
    )


def comparison(
    artifacts: Path,
    variant: str,
    fold: str,
    evaluation: str,
    region_id: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    proposed = aggregate_seed_rows(
        [
            evaluation_path(
                artifacts,
                "proposed",
                variant,
                seed,
                fold,
                evaluation,
                region_id,
            )
            for seed in SEEDS
        ]
    )
    convlstm = aggregate_seed_rows(
        [
            evaluation_path(
                artifacts,
                "convlstm",
                "dominance",
                seed,
                fold,
                evaluation,
                region_id,
            )
            for seed in SEEDS
        ]
    )
    results = {
        metric: paired_bootstrap(
            proposed,
            convlstm,
            metric,
            replicates,
            rng,
        )
        for metric in METRICS
    }
    holm_adjust(results)
    return {
        "fold": fold,
        "evaluation": evaluation,
        "region_id": region_id,
        "samples": len(proposed),
        "metrics": results,
    }


def verdict(comparisons: dict[str, dict]) -> dict:
    validation = [comparisons[f"validation_{fold}"] for fold in FOLDS]
    mean_improvement = {
        metric: float(
            np.mean(
                [
                    item["metrics"][metric]["oriented_improvement"]
                    for item in validation
                ]
            )
        )
        for metric in METRICS
    }
    fold_consistency = {
        metric: sum(
            item["metrics"][metric]["oriented_improvement"] > 0
            for item in validation
        )
        for metric in PRIMARY_METRICS
    }
    frozen_positive = sum(
        comparisons["frozen_test"]["metrics"][metric]["oriented_improvement"] > 0
        for metric in METRICS
    )
    external_positive = sum(
        comparisons["external"]["metrics"][metric]["oriented_improvement"] > 0
        for metric in METRICS
    )
    conditions = {
        "all_mean_metrics_better": all(
            improvement > 0 for improvement in mean_improvement.values()
        ),
        "primary_metrics_positive_in_at_least_two_folds": all(
            count >= 2 for count in fold_consistency.values()
        ),
        "frozen_2024_overall_direction": frozen_positive >= 6,
        "external_tianshan_overall_direction": external_positive >= 6,
    }
    return {
        "claim_supported": all(conditions.values()),
        "conditions": conditions,
        "mean_oriented_improvement": mean_improvement,
        "primary_fold_positive_counts": fold_consistency,
        "frozen_positive_metrics": frozen_positive,
        "external_positive_metrics": external_positive,
        "overall_direction_rule": "At least 6 of 8 metrics improve.",
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    artifacts = Path(config["project"]["artifacts_dir"])
    selection = json.loads(
        (artifacts / "selected_variant.json").read_text(encoding="utf-8")
    )
    variant = selection["selected_variant"]
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for fold in FOLDS:
        comparisons[f"validation_{fold}"] = comparison(
            artifacts,
            variant,
            fold,
            "validation",
            config["data"]["region_id"],
            args.bootstrap_replicates,
            rng,
        )
    comparisons["frozen_test"] = comparison(
        artifacts,
        variant,
        "fold_3",
        "frozen_test",
        config["data"]["region_id"],
        args.bootstrap_replicates,
        rng,
    )
    comparisons["external"] = comparison(
        artifacts,
        variant,
        "fold_3",
        "external",
        config["data"]["external_region_id"],
        args.bootstrap_replicates,
        rng,
    )
    output = {
        "selected_variant": variant,
        "bootstrap": {
            "type": "paired circular moving block bootstrap",
            "block_days": 7,
            "replicates": args.bootstrap_replicates,
            "multiple_testing": "Holm within each evaluation",
        },
        "comparisons": comparisons,
        "dominance_verdict": verdict(comparisons),
    }
    destination = artifacts / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "qa_comparison.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["dominance_verdict"], indent=2))


if __name__ == "__main__":
    main()
