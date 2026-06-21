"""Train or evaluate one QA-aware experiment job."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from snow_attractor.data import MaskedSnowSequenceDataset  # noqa: E402
from snow_attractor.benchmark_models import build_benchmark_model  # noqa: E402
from snow_attractor.evaluation import forecast_metrics  # noqa: E402
from snow_attractor.hopfield_cann import HopfieldCANNForecastNet  # noqa: E402
from snow_attractor.losses import AttractorEnergyLoss  # noqa: E402
from snow_attractor.model import AttractorEnergyUNet  # noqa: E402
from snow_attractor.official_baselines import (  # noqa: E402
    OFFICIAL_BASELINE_SETTINGS,
    VENDOR_COMMITS,
    build_official_model,
)
from snow_attractor.training import load_config, set_seed  # noqa: E402
from snow_attractor.transformer_cann import TransformerCANNLyapunovNet  # noqa: E402
from train import batch_to_device, evaluate_loader, write_csv  # noqa: E402

MODEL_NAMES = (
    "proposed",
    "hopfield_cann",
    "hopfield_no_episode",
    "hopfield_no_prototypes",
    "hopfield_no_cann",
    "hopfield_2d_cann",
    "convlstm",
    "convlstm_unet",
    "predrnn",
    "predrnnv2",
    "simvpv2",
    "simvpv2_official",
    "swinlstm",
    "swinlstm_official",
    "vmrnn",
    "vmrnn_official",
    "nearest_memory",
    "no_memory",
    "transformer_cann_lyapunov",
    "transformer_cann_lyapunov_boundary",
    "transformer_no_lyapunov",
    "transformer_no_cann",
    "transformer_no_spatial",
    "transformer_no_temporal",
)
HOPFIELD_MODEL_NAMES = {
    "hopfield_cann",
    "hopfield_no_episode",
    "hopfield_no_prototypes",
    "hopfield_no_cann",
    "hopfield_2d_cann",
}
VARIANTS = (
    "residual_only",
    "dominance",
    "guarded",
    "dominance_open_gate",
    "warmstart_dominance",
)
SELECTION_SCORE_RULE = (
    "coverage_r2 - coverage_mae - pixel_mae - pixel_rmse + pixel_ssim "
    "+ snow_iou + snow_f1"
)


def official_baseline_settings(config: dict) -> dict:
    settings = dict(OFFICIAL_BASELINE_SETTINGS)
    model_config = config.get("model", {})
    if any(key in model_config for key in ("vmrnn_patch_size", "vmrnn_embed_dim", "vmrnn_depths")):
        settings["vmrnn_official"] = (
            "VMRNN reduced, "
            f"patch={int(model_config.get('vmrnn_patch_size', 4))}, "
            f"embed={int(model_config.get('vmrnn_embed_dim', 128))}, "
            f"depth={int(model_config.get('vmrnn_depths', 6))}, "
            f"heads={int(model_config.get('vmrnn_num_heads', 4))}, "
            f"model_size={int(model_config.get('vmrnn_model_image_size', 128))}"
        )
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qa_experiment.yaml")
    parser.add_argument("--mode", choices=("train", "evaluate", "smoke"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", choices=("development", "fold_1", "fold_2", "fold_3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=VARIANTS, default="dominance")
    parser.add_argument("--checkpoint")
    parser.add_argument("--evaluation", choices=("validation", "frozen_test", "external"))
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


def target_date(dataset: MaskedSnowSequenceDataset, index: int) -> str:
    _, end = dataset.windows[index]
    return dataset.dates[end - 1].strftime("%Y-%m-%d")


def indices_between(
    dataset: MaskedSnowSequenceDataset,
    start: str,
    end: str,
) -> list[int]:
    return [
        index
        for index in range(len(dataset))
        if start <= target_date(dataset, index) <= end
    ]


def split_indices(dataset: MaskedSnowSequenceDataset, split: dict) -> tuple[list[int], list[int]]:
    train = indices_between(dataset, split["train_start"], split["train_end"])
    validation = indices_between(
        dataset,
        split["validation_start"],
        split["validation_end"],
    )
    if not train or not validation:
        raise ValueError(
            f"Empty split: train={len(train)}, validation={len(validation)}"
        )
    return train, validation


def prepare_dataset(config: dict, region_id: str) -> MaskedSnowSequenceDataset:
    data = config["data"]
    return MaskedSnowSequenceDataset(
        data_root=data["root"],
        region_id=region_id,
        sequence_length=int(data["sequence_length"]),
        image_size=int(data["image_size"]),
        qa_policy=data["qa_policy"],
        target_min_valid_fraction=float(data["target_min_valid_fraction"]),
    )


def dynamic_model_config(config: dict, dataset: MaskedSnowSequenceDataset) -> dict:
    model_config = copy.deepcopy(config["model"])
    model_config.update(
        {
            "input_steps": dataset.sequence_length,
            "input_channels": dataset.input_channels,
            "context_dim": dataset.context_dim,
            "spatial_context_channels": dataset.spatial_context_channels,
        }
    )
    return model_config


def filter_constructor_kwargs(cls: type, kwargs: dict) -> dict:
    parameters = inspect.signature(cls.__init__).parameters
    accepted = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in accepted}


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


def convlstm_checkpoint_path(config: dict, fold: str, seed: int) -> Path:
    return (
        Path(config["project"]["artifacts_dir"])
        / "training"
        / "convlstm"
        / "default"
        / f"seed_{seed}"
        / fold
        / "best_model.pt"
    )


def build_model(
    name: str,
    config: dict,
    dataset: MaskedSnowSequenceDataset,
    variant: str,
) -> torch.nn.Module:
    model_config = dynamic_model_config(config, dataset)
    if name in HOPFIELD_MODEL_NAMES:
        if name == "hopfield_no_episode":
            model_config["use_hopfield_episode"] = False
        elif name == "hopfield_no_prototypes":
            model_config["use_hopfield_prototypes"] = False
        elif name == "hopfield_no_cann":
            model_config["use_continuous_attractor"] = False
        elif name == "hopfield_2d_cann":
            model_config["use_season_coordinate"] = False
        return HopfieldCANNForecastNet(
            **filter_constructor_kwargs(HopfieldCANNForecastNet, model_config)
        )
    if name in {"proposed", "nearest_memory", "no_memory"}:
        if name == "nearest_memory":
            model_config["attractor_mode"] = "nearest"
        elif name == "no_memory":
            model_config["attractor_mode"] = "none"
            model_config["attractor_iterations"] = 0
        if variant == "guarded":
            model_config["residual_gate_init"] = -4.0
        elif variant in {"dominance_open_gate", "warmstart_dominance"}:
            model_config["residual_gate_init"] = -1.5
        return AttractorEnergyUNet(
            **filter_constructor_kwargs(AttractorEnergyUNet, model_config)
        )
    if name == "transformer_cann_lyapunov":
        return TransformerCANNLyapunovNet(
            **filter_constructor_kwargs(TransformerCANNLyapunovNet, model_config)
        )
    if name == "transformer_cann_lyapunov_boundary":
        model_config["enable_boundary_refiner"] = True
        model_config["coverage_calibration"] = True
        return TransformerCANNLyapunovNet(
            **filter_constructor_kwargs(TransformerCANNLyapunovNet, model_config)
        )
    if name in {
        "transformer_no_lyapunov",
        "transformer_no_cann",
        "transformer_no_spatial",
        "transformer_no_temporal",
    }:
        if name == "transformer_no_cann":
            model_config["attractor_iterations"] = 0
        elif name == "transformer_no_spatial":
            model_config["spatial_layers"] = 0
        elif name == "transformer_no_temporal":
            model_config["temporal_layers"] = 0
        return TransformerCANNLyapunovNet(
            **filter_constructor_kwargs(TransformerCANNLyapunovNet, model_config)
        )
    benchmark_names = {
        "convlstm": "convlstm",
        "convlstm_unet": "convlstm_unet",
        "predrnn": "predrnn",
        "predrnnv2": "predrnnv2",
        "simvpv2": "simvpv2",
        "swinlstm": "swinlstm",
        "vmrnn": "vmrnn",
    }
    if name in benchmark_names:
        model_config["input_channels"] = 1
        return build_benchmark_model(
            benchmark_names[name],
            **model_config,
        )
    official_kwargs = {}
    if name == "vmrnn_official":
        official_kwargs = {
            "patch_size": int(model_config.get("vmrnn_patch_size", 4)),
            "embed_dim": int(model_config.get("vmrnn_embed_dim", 128)),
            "depths": int(model_config.get("vmrnn_depths", 6)),
            "num_heads": int(model_config.get("vmrnn_num_heads", 4)),
            "window_size": int(model_config.get("vmrnn_window_size", 4)),
            "model_image_size": int(model_config.get("vmrnn_model_image_size", config["data"]["image_size"])),
        }
    return build_official_model(
        name,
        input_steps=dataset.sequence_length,
        input_channels=dataset.input_channels,
        spatial_context_channels=dataset.spatial_context_channels,
        image_size=int(config["data"]["image_size"]),
        dropout=float(model_config["dropout"]),
        **official_kwargs,
    )


def build_criterion(name: str, config: dict, variant: str) -> AttractorEnergyLoss:
    loss_config = copy.deepcopy(config["loss"])
    if variant == "residual_only":
        loss_config["dominance_weight"] = 0.0
    elif variant in {"guarded", "dominance_open_gate", "warmstart_dominance"}:
        loss_config["dominance_weight"] = 2.0
    attractor_models = {
        "proposed",
        "nearest_memory",
        "no_memory",
        "transformer_cann_lyapunov",
        "transformer_cann_lyapunov_boundary",
        "transformer_no_lyapunov",
        "transformer_no_cann",
        "transformer_no_spatial",
        "transformer_no_temporal",
        *HOPFIELD_MODEL_NAMES,
    }
    if name == "transformer_no_lyapunov":
        loss_config["lyapunov_weight"] = 0.0
        loss_config["energy_monotonicity_weight"] = 0.0
    if name == "transformer_no_cann":
        loss_config["lyapunov_weight"] = 0.0
        loss_config["energy_monotonicity_weight"] = 0.0
    if name not in attractor_models:
        for key in (
            "energy_monotonicity_weight",
            "coordinate_coverage_weight",
            "coordinate_change_weight",
            "manifold_distance_weight",
            "manifold_first_order_weight",
            "manifold_second_order_weight",
            "noise_consistency_weight",
            "dominance_weight",
        ):
            loss_config[key] = 0.0
    return AttractorEnergyLoss(**loss_config)


def run_directory(
    config: dict,
    name: str,
    fold: str,
    seed: int,
    variant: str,
) -> Path:
    suffix = variant if name == "proposed" or name in HOPFIELD_MODEL_NAMES else "default"
    return (
        Path(config["project"]["artifacts_dir"])
        / "training"
        / name
        / suffix
        / f"seed_{seed}"
        / fold
    )


def backbone_parameter_prefixes() -> tuple[str, str]:
    return ("temporal_encoder.", "backbone_decoder.")


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    prefixes = backbone_parameter_prefixes()
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = trainable


def load_convlstm_backbone(
    model: torch.nn.Module,
    config: dict,
    fold: str,
    seed: int,
    device: torch.device,
) -> bool:
    path = convlstm_checkpoint_path(config, fold, seed)
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source = checkpoint["model_state"]
    current = model.state_dict()
    prefixes = backbone_parameter_prefixes()
    copied = {}
    for key, value in source.items():
        if key.startswith(prefixes) and key in current and current[key].shape == value.shape:
            copied[key] = value
    if not copied:
        return False
    current.update(copied)
    model.load_state_dict(current)
    return True


def optimizer_parameter_groups(
    model: torch.nn.Module,
    learning_rate: float,
    variant: str,
) -> list[dict]:
    if variant != "warmstart_dominance":
        return [{"params": list(model.parameters()), "lr": learning_rate}]
    prefixes = backbone_parameter_prefixes()
    backbone = []
    rest = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            backbone.append(parameter)
        else:
            rest.append(parameter)
    groups = [{"params": rest, "lr": learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": learning_rate * 0.2})
    return groups


def save_normalization(path: Path, state: dict) -> None:
    weather = state.get("weather")
    np.savez_compressed(
        path,
        climatology=state["climatology"],
        weather_mean=np.asarray(weather["mean"], dtype=np.float32)
        if weather is not None
        else np.empty(0, dtype=np.float32),
        weather_std=np.asarray(weather["std"], dtype=np.float32)
        if weather is not None
        else np.empty(0, dtype=np.float32),
    )


def load_normalization(path: Path) -> dict:
    payload = np.load(path)
    weather = None
    if payload["weather_mean"].size:
        weather = {
            "mean": payload["weather_mean"].tolist(),
            "std": payload["weather_std"].tolist(),
        }
    return {"climatology": payload["climatology"], "weather": weather}


def model_forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["inputs"],
        batch["spatial_prior"],
        batch["context"],
        batch.get("spatial_context"),
        land_mask=batch.get("land_mask"),
        coverage_mask=batch.get("target_valid_mask"),
    )


def train_job(args: argparse.Namespace, config: dict) -> dict:
    if args.fold is None:
        raise ValueError("--fold is required for training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = prepare_dataset(config, config["data"]["region_id"])
    train_indices, validation_indices = split_indices(
        dataset,
        config["splits"][args.fold],
    )
    dataset.fit_external_normalization(train_indices)
    set_seed(args.seed)
    model = build_model(args.model, config, dataset, args.variant).to(device)
    warmstart_loaded = False
    warmstart_freeze_epochs = 0
    if args.model == "proposed" and args.variant == "warmstart_dominance":
        warmstart_loaded = load_convlstm_backbone(
            model,
            config,
            args.fold,
            args.seed,
            device,
        )
        warmstart_freeze_epochs = int(
            config["training"].get("warmstart_freeze_epochs", 5)
        )
        if warmstart_loaded and warmstart_freeze_epochs > 0:
            set_backbone_trainable(model, False)
    criterion = build_criterion(args.model, config, args.variant)
    physical_batch = int(
        config["training"]["physical_batch_size"][args.model]
    )
    effective_batch = int(config["training"]["effective_batch_size"])
    accumulation_steps = max(1, math.ceil(effective_batch / physical_batch))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=physical_batch,
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=physical_batch,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            float(config["training"]["learning_rate"]),
            args.variant,
        ),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epochs = args.max_epochs or int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    run_dir = run_directory(
        config,
        args.model,
        args.fold,
        args.seed,
        args.variant,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = run_dir / "normalization.npz"
    save_normalization(normalization_path, dataset.normalization_state())
    history = []
    best_score = -float("inf")
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if (
            warmstart_loaded
            and warmstart_freeze_epochs > 0
            and epoch == warmstart_freeze_epochs + 1
        ):
            set_backbone_trainable(model, True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = batch_to_device(raw_batch, device)
            group_start = ((step - 1) // accumulation_steps) * accumulation_steps
            group_size = min(
                accumulation_steps,
                len(train_loader) - group_start,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model_forward(model, batch)
                losses = criterion(
                    outputs,
                    batch["target"],
                    valid_mask=batch["target_valid_mask"],
                    land_mask=batch["land_mask"],
                )
                scaled_loss = losses["total"] / group_size
            scaler.scale(scaled_loss).backward()
            running_loss += float(losses["total"].detach())
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_loader(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
        )
        score = validation_selection_score(validation)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "joint_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"val_{key}": validation[key]
                for key in (
                    "coverage_r2",
                    "coverage_mae",
                    "coverage_rmse",
                    "pixel_mae",
                    "pixel_rmse",
                    "pixel_ssim",
                    "snow_iou",
                    "snow_f1",
                )
            },
        }
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        if score > best_score + 1e-4:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "model_config": dynamic_model_config(config, dataset),
                    "variant": args.variant,
                    "fold": args.fold,
                    "seed": args.seed,
                    "normalization_path": str(normalization_path.resolve()),
                    "vendor_commits": VENDOR_COMMITS,
                    "official_baseline_settings": official_baseline_settings(config),
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "selection_score_rule": SELECTION_SCORE_RULE,
                    "warmstart_loaded": warmstart_loaded,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        print(
            f"model={args.model} variant={args.variant} fold={args.fold} "
            f"seed={args.seed} epoch={epoch:03d} score={score:.5f} "
            f"R2={validation['coverage_r2']:.4f} "
            f"MAE={validation['pixel_mae']:.5f} "
            f"SSIM={validation['pixel_ssim']:.4f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break
    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate_loader(
        model,
        validation_loader,
        criterion,
        device,
        amp_enabled,
    )
    summary = {
        "model": args.model,
        "variant": args.variant,
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": checkpoint["best_epoch"],
        "best_score": checkpoint.get("best_score"),
        "selection_score_rule": checkpoint.get(
            "selection_score_rule",
            SELECTION_SCORE_RULE,
        ),
        "warmstart_loaded": checkpoint.get("warmstart_loaded", warmstart_loaded),
        "training_seconds": time.perf_counter() - started,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_dates": [
            target_date(dataset, train_indices[0]),
            target_date(dataset, train_indices[-1]),
        ],
        "validation_dates": [
            target_date(dataset, validation_indices[0]),
            target_date(dataset, validation_indices[-1]),
        ],
        "validation": validation,
        "vendor_commits": VENDOR_COMMITS,
        "official_baseline_settings": official_baseline_settings(config),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def export_predictions(
    model: torch.nn.Module,
    dataset: MaskedSnowSequenceDataset,
    indices: list[int],
    criterion: AttractorEnergyLoss,
    device: torch.device,
    batch_size: int,
    output: Path,
) -> dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []
    prediction_maps = []
    target_maps = []
    mask_maps = []
    cursor = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        outputs = model_forward(model, batch)
        for local_index in range(batch["target"].size(0)):
            prediction = outputs["prediction"][local_index : local_index + 1].float().cpu()
            target = batch["target"][local_index : local_index + 1].float().cpu()
            mask = batch["target_valid_mask"][local_index : local_index + 1].float().cpu()
            metrics = forecast_metrics(prediction, target, mask)
            valid_count = mask.sum().clamp_min(1.0)
            predicted_coverage = float((prediction * mask).sum() / valid_count)
            target_coverage = float((target * mask).sum() / valid_count)
            error = prediction - target
            valid_boolean = mask.bool()
            predicted_snow = prediction >= 0.10
            target_snow = target >= 0.10
            rows.append(
                {
                    "date": target_date(dataset, indices[cursor]),
                    "predicted_coverage": predicted_coverage,
                    "target_coverage": target_coverage,
                    "valid_pixel_count": float(valid_count),
                    "pixel_absolute_sum": float((error.abs() * mask).sum()),
                    "pixel_squared_sum": float((error.square() * mask).sum()),
                    "snow_true_positive": float(
                        (predicted_snow & target_snow & valid_boolean).sum()
                    ),
                    "snow_false_positive": float(
                        (predicted_snow & ~target_snow & valid_boolean).sum()
                    ),
                    "snow_false_negative": float(
                        (~predicted_snow & target_snow & valid_boolean).sum()
                    ),
                    **metrics,
                }
            )
            prediction_maps.append(prediction.numpy().astype(np.float16))
            target_maps.append(target.numpy().astype(np.float16))
            mask_maps.append(mask.numpy().astype(np.uint8))
            cursor += 1
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_sample.csv", rows)
    np.savez_compressed(
        output / "maps.npz",
        prediction=np.concatenate(prediction_maps),
        target=np.concatenate(target_maps),
        valid_mask=np.concatenate(mask_maps),
        dates=np.asarray([row["date"] for row in rows]),
    )
    summary = {
        "samples": len(rows),
        "metrics": forecast_metrics(
            torch.cat(
                [torch.from_numpy(item.astype(np.float32)) for item in prediction_maps]
            ),
            torch.cat(
                [torch.from_numpy(item.astype(np.float32)) for item in target_maps]
            ),
            torch.cat(
                [torch.from_numpy(item.astype(np.float32)) for item in mask_maps]
            ),
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate_job(args: argparse.Namespace, config: dict) -> dict:
    if not args.checkpoint or not args.evaluation:
        raise ValueError("--checkpoint and --evaluation are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    region_id = (
        config["data"]["external_region_id"]
        if args.evaluation == "external"
        else config["data"]["region_id"]
    )
    dataset = prepare_dataset(config, region_id)
    dataset.load_normalization_state(
        load_normalization(Path(checkpoint["normalization_path"]))
    )
    model = build_model(
        checkpoint["model_name"],
        config,
        dataset,
        checkpoint.get("variant", "dominance"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if args.evaluation == "validation":
        split = config["splits"][checkpoint["fold"]]
        indices = indices_between(
            dataset,
            split["validation_start"],
            split["validation_end"],
        )
    else:
        frozen = config["splits"]["frozen_test"]
        indices = indices_between(dataset, frozen["start"], frozen["end"])
    criterion = build_criterion(
        checkpoint["model_name"],
        config,
        checkpoint.get("variant", "dominance"),
    )
    output = (
        checkpoint_path.parent
        / "evaluation"
        / args.evaluation
        / region_id
    )
    return export_predictions(
        model,
        dataset,
        indices,
        criterion,
        device,
        int(config["training"]["physical_batch_size"][checkpoint["model_name"]]),
        output,
    )


def smoke_job(args: argparse.Namespace, config: dict) -> None:
    dataset = prepare_dataset(config, config["data"]["region_id"])
    dataset.fit_external_normalization(list(range(min(32, len(dataset)))))
    model = build_model(args.model, config, dataset, args.variant)
    criterion = build_criterion(args.model, config, args.variant)
    batch = next(iter(DataLoader(Subset(dataset, [0]), batch_size=1)))
    outputs = model_forward(model, batch)
    losses = criterion(
        outputs,
        batch["target"],
        valid_mask=batch["target_valid_mask"],
        land_mask=batch["land_mask"],
    )
    losses["total"].backward()
    print(
        json.dumps(
            {
                "model": args.model,
                "prediction_shape": list(outputs["prediction"].shape),
                "loss": float(losses["total"].detach()),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    if args.mode == "train":
        print(json.dumps(train_job(args, config), indent=2))
    elif args.mode == "evaluate":
        print(json.dumps(evaluate_job(args, config), indent=2))
    else:
        smoke_job(args, config)


if __name__ == "__main__":
    main()
