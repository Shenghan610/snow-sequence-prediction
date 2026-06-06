import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import ModelEMA, load_model_state, parse_args, set_seed
from .data import AliSnowDatasetRAM
from .evaluation import (
    baseline_anchor_loss,
    evaluate_and_visualize,
    evaluate_baselines,
    evaluate_metrics,
    snow_heatmap_loss,
)
from .model import DataDrivenSnowPredictor


def run_training(args=None):
    args = args or parse_args()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "Ali_SnowData")
    external_feature_path = os.path.join(
        project_dir,
        "ExternalClimateTerrain",
        "external_daily_features.csv"
    )
    best_model_path = os.path.join(
        project_dir,
        "best_highres_snow_heatmap_model.pth"
    )
    last_model_path = os.path.join(
        project_dir,
        "highres_snow_heatmap_model.pth"
    )
    figure_path = os.path.join(project_dir, "snow_heatmap_prediction.png")

    # --- 核心回归参数 ---
    SEQ_LEN = args.seq_len
    IMG_SIZE = (args.img_size, args.img_size)
    EPOCHS = args.epochs
    LR = args.lr
    ITERATIONS = args.iterations
    LAMBDA_ENERGY = args.energy_weight
    BASE_WINDOW = args.base_window
    HIGH_SNOW_WEIGHT = args.high_snow_weight
    CHANGE_WEIGHT = args.change_weight
    EARLY_STOP_PATIENCE = args.patience
    BATCH_SIZE = args.batch_size

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Config: d_model={args.d_model}, iterations={args.iterations}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, hidden_dropout={args.hidden_dropout}, "
        f"feature_dropout={args.feature_dropout}, head_dropout={args.head_dropout}, "
        f"max_delta={args.max_delta}, residual_l1={args.residual_l1_weight}, "
        f"residual_gate_bias={args.residual_gate_bias}, ema={not args.no_ema}"
    )
    print(f"计算设备: {device}")

    try:
        # 直接加载全局数据集
        dataset = AliSnowDatasetRAM(
            data_dir=data_path,
            seq_len=SEQ_LEN,
            target_size=IMG_SIZE,
            external_feature_path=external_feature_path
        )

        if len(dataset) < 2:
            raise ValueError(f"有效样本数只有 {len(dataset)}，无法划分训练集和验证集。请增加 TIF 文件数量或减小 SEQ_LEN。")

        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=train_generator
        )
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"数据划分: 训练样本 {train_size} 个，验证样本 {val_size} 个")
        evaluate_baselines(val_dataloader)

        model = DataDrivenSnowPredictor(
            in_channels=dataset.original_channels,
            d_model=args.d_model,
            iterations=ITERATIONS,
            season_dim=dataset.season_feature_dim,
            scalar_dim=dataset.scalar_feature_dim,
            base_window=BASE_WINDOW,
            max_delta=args.max_delta,
            hidden_dropout=args.hidden_dropout,
            feature_dropout=args.feature_dropout,
            head_dropout=args.head_dropout,
            residual_gate_bias=args.residual_gate_bias
        ).to(device)
        ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1e-6
        )

        loss_history = []
        val_r2_history = []
        best_val_r2 = -float("inf")
        epochs_without_improvement = 0
        history_path = os.path.join(project_dir, "training_history.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "energy_loss",
                "anchor_loss",
                "heatmap_delta_l1",
                "mean_residual_gate",
                "mean_abs_delta",
                "val_pixel_mae",
                "val_pixel_mse",
                "val_coverage_r2",
                "best_coverage_r2",
                "lr",
                "eval_model"
            ])

        print(f"\n开始高清全局视野下的时空预测训练 (Epochs={EPOCHS})...")

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            epoch_energy_loss = 0.0
            epoch_anchor_loss = 0.0
            epoch_residual_l1 = 0.0
            epoch_residual_gate = 0.0
            epoch_abs_delta = 0.0

            loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

            for inputs, season_features, scalar_features, target_season, targets, spatial_prior in loop:
                inputs = inputs.to(device)
                season_features = season_features.to(device)
                scalar_features = scalar_features.to(device)
                target_season = target_season.to(device)
                targets = targets.to(device)
                spatial_prior = spatial_prior.to(device)

                optimizer.zero_grad()

                preds, energy_traj, _ = model(inputs, spatial_prior, season_features, scalar_features, target_season)
                raw_mse = F.mse_loss(preds, targets)
                loss_main = snow_heatmap_loss(
                    preds,
                    targets,
                    scalar_features,
                    high_snow_weight=HIGH_SNOW_WEIGHT,
                    change_weight=CHANGE_WEIGHT
                )

                e_diff = energy_traj[:, 1:] - energy_traj[:, :-1]
                loss_energy = F.relu(e_diff).mean()
                loss_anchor = baseline_anchor_loss(
                    preds,
                    scalar_features,
                    season_features,
                    target_season,
                    base_window=BASE_WINDOW
                )
                residual_delta = model.last_aux["residual_delta"]
                residual_gate = model.last_aux["residual_gate"]
                loss_residual_l1 = residual_delta.abs().mean()

                loss = (
                    loss_main
                    + LAMBDA_ENERGY * loss_energy
                    + args.anchor_weight * loss_anchor
                    + args.residual_l1_weight * loss_residual_l1
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

                epoch_loss += loss.item()
                epoch_energy_loss += loss_energy.item()
                epoch_anchor_loss += loss_anchor.item()
                epoch_residual_l1 += loss_residual_l1.item()
                epoch_residual_gate += residual_gate.detach().mean().item()
                epoch_abs_delta += residual_delta.detach().abs().mean().item()
                loop.set_postfix(
                    mse=raw_mse.item(),
                    weighted=loss_main.item(),
                    anchor=loss_anchor.item(),
                    gate=residual_gate.detach().mean().item(),
                    delta=residual_delta.detach().abs().mean().item()
                )

            avg_loss = epoch_loss / len(train_dataloader)
            avg_e_loss = epoch_energy_loss / len(train_dataloader)
            avg_anchor_loss = epoch_anchor_loss / len(train_dataloader)
            avg_residual_l1 = epoch_residual_l1 / len(train_dataloader)
            avg_residual_gate = epoch_residual_gate / len(train_dataloader)
            avg_abs_delta = epoch_abs_delta / len(train_dataloader)

            loss_history.append(avg_loss)

            eval_model = ema.module if ema is not None else model
            val_pixel_mae, val_pixel_mse, val_coverage_r2 = evaluate_metrics(eval_model, val_dataloader, device)
            val_r2_history.append(val_coverage_r2)
            scheduler.step(val_coverage_r2)

            if val_coverage_r2 > best_val_r2 + args.min_delta_r2:
                best_val_r2 = val_coverage_r2
                epochs_without_improvement = 0
                torch.save(eval_model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]["lr"]
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    f"{avg_loss:.8f}",
                    f"{avg_e_loss:.8f}",
                    f"{avg_anchor_loss:.8f}",
                    f"{avg_residual_l1:.8f}",
                    f"{avg_residual_gate:.8f}",
                    f"{avg_abs_delta:.8f}",
                    f"{val_pixel_mae:.8f}",
                    f"{val_pixel_mse:.10f}",
                    f"{val_coverage_r2:.8f}",
                    f"{best_val_r2:.8f}",
                    f"{current_lr:.8g}",
                    "ema" if ema is not None else "raw"
                ])

            if (epoch + 1) % 10 == 0:
                print(
                    f"   [Info] Epoch {epoch + 1} | Train Loss: {avg_loss:.5f} | "
                    f"Val Pixel MAE: {val_pixel_mae:.5f} | Val Pixel MSE: {val_pixel_mse:.6f} | "
                    f"Val Coverage R2: {val_coverage_r2:.4f} | Best R2: {best_val_r2:.4f} | "
                    f"Gate: {avg_residual_gate:.3f} | |Delta|: {avg_abs_delta:.4f}"
                )

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"验证集 R2 连续 {EARLY_STOP_PATIENCE} 个 epoch 没有提升，提前停止训练。")
                break

        print(f"\n训练完成！最佳验证集 R2: {best_val_r2:.4f}")
        torch.save(model.state_dict(), last_model_path)
        model.load_state_dict(load_model_state(best_model_path, device))

        # 使用验证集评估并绘制图表，避免只看训练集表现
        evaluate_and_visualize(
            model,
            val_dataloader,
            device,
            loss_history,
            val_r2_history,
            output_path=figure_path,
            show_plot=args.show_plot
        )

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback

        traceback.print_exc()
