import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .model import DataDrivenSnowPredictor


def regression_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    r2 = 1 - (np.sum((targets - preds) ** 2) / (np.sum((targets - np.mean(targets)) ** 2) + 1e-8))
    return mae, mse, r2


def heatmap_metrics(pred_maps, target_maps):
    pred_maps = np.array(pred_maps)
    target_maps = np.array(target_maps)
    pixel_mae = np.mean(np.abs(pred_maps - target_maps))
    pixel_mse = np.mean((pred_maps - target_maps) ** 2)
    pred_coverage = pred_maps.reshape(pred_maps.shape[0], -1).mean(axis=1)
    target_coverage = target_maps.reshape(target_maps.shape[0], -1).mean(axis=1)
    coverage_mae, coverage_mse, coverage_r2 = regression_metrics(pred_coverage, target_coverage)
    return pixel_mae, pixel_mse, coverage_mae, coverage_mse, coverage_r2


def snow_heatmap_loss(pred_maps, target_maps, scalar_features, high_snow_weight=3.0, change_weight=2.0):
    target_coverage = target_maps.flatten(1).mean(dim=1)
    pred_coverage = pred_maps.flatten(1).mean(dim=1)
    last_coverage = scalar_features[:, -1, 0]
    target_change = (target_coverage - last_coverage).abs()
    snow_weights = high_snow_weight * torch.clamp(target_maps / 0.15, max=1.5)
    change_weights = change_weight * torch.clamp(target_change / 0.05, max=2.0).view(-1, 1, 1, 1)
    weights = 1.0 + snow_weights + change_weights

    squared_error = (pred_maps - target_maps) ** 2
    smooth_error = F.smooth_l1_loss(pred_maps, target_maps, reduction='none', beta=0.03)
    pixel_loss = (weights * (0.75 * squared_error + 0.25 * smooth_error)).mean()
    coverage_loss = F.smooth_l1_loss(pred_coverage, target_coverage, beta=0.03)
    return pixel_loss + 0.20 * coverage_loss


def baseline_anchor_loss(preds, scalar_features, season_features, target_season, base_window=7):
    candidates = DataDrivenSnowPredictor._baseline_candidates(
        scalar_features,
        season_features=season_features,
        target_season=target_season,
        base_window=base_window
    )
    anomaly_persistence = candidates[:, 4].detach()
    pred_coverage = preds.flatten(1).mean(dim=1)
    return F.smooth_l1_loss(pred_coverage, anomaly_persistence, beta=0.04)


def weighted_mse_loss(preds, targets, high_snow_weight=4.0):
    weights = 1.0 + high_snow_weight * torch.clamp(targets / 0.10, max=2.0)
    return (weights * (preds - targets) ** 2).mean()


def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, season_features, scalar_features, target_season, targets, spatial_prior in dataloader:
            inputs = inputs.to(device)
            season_features = season_features.to(device)
            scalar_features = scalar_features.to(device)
            target_season = target_season.to(device)
            targets = targets.to(device)
            spatial_prior = spatial_prior.to(device)

            preds, _, _ = model(inputs, spatial_prior, season_features, scalar_features, target_season)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) == 0:
        return float("inf"), float("inf"), -float("inf")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    pixel_mae, pixel_mse, _, _, coverage_r2 = heatmap_metrics(all_preds, all_targets)
    return pixel_mae, pixel_mse, coverage_r2


def evaluate_baselines(dataloader):
    all_targets = []
    last_day_preds, mean7_preds, trend7_preds = [], [], []
    clim_preds, anomaly_preds, mean_seq_preds = [], [], []

    for inputs, season_features, scalar_features, target_season, targets, _ in dataloader:
        all_targets.extend(targets.flatten(1).mean(dim=1).numpy())
        candidates = DataDrivenSnowPredictor._baseline_candidates(
            scalar_features,
            season_features=season_features,
            target_season=target_season,
            base_window=7
        )
        last_day_preds.extend(candidates[:, 0].numpy())
        mean7_preds.extend(candidates[:, 1].numpy())
        trend7_preds.extend(candidates[:, 2].numpy())
        clim_preds.extend(candidates[:, 3].numpy())
        anomaly_preds.extend(candidates[:, 4].numpy())
        mean_seq_preds.extend(scalar_features[:, :, 0].mean(dim=1).numpy())

    last_mae, last_mse, last_r2 = regression_metrics(last_day_preds, all_targets)
    mean7_mae, mean7_mse, mean7_r2 = regression_metrics(mean7_preds, all_targets)
    trend7_mae, trend7_mse, trend7_r2 = regression_metrics(trend7_preds, all_targets)
    clim_mae, clim_mse, clim_r2 = regression_metrics(clim_preds, all_targets)
    anomaly_mae, anomaly_mse, anomaly_r2 = regression_metrics(anomaly_preds, all_targets)
    mean_mae, mean_mse, mean_r2 = regression_metrics(mean_seq_preds, all_targets)
    print(f"Baseline-前一天: MAE={last_mae:.6f}, MSE={last_mse:.6f}, R2={last_r2:.4f}")
    print(f"Baseline-最近7天均值: MAE={mean7_mae:.6f}, MSE={mean7_mse:.6f}, R2={mean7_r2:.4f}")
    print(f"Baseline-7天趋势: MAE={trend7_mae:.6f}, MSE={trend7_mse:.6f}, R2={trend7_r2:.4f}")
    print(f"Baseline-日序气候均值: MAE={clim_mae:.6f}, MSE={clim_mse:.6f}, R2={clim_r2:.4f}")
    print(f"Baseline-气候异常延续: MAE={anomaly_mae:.6f}, MSE={anomaly_mse:.6f}, R2={anomaly_r2:.4f}")
    print(f"Baseline-序列均值: MAE={mean_mae:.6f}, MSE={mean_mse:.6f}, R2={mean_r2:.4f}")


def evaluate_and_visualize(model, dataloader, device, loss_history, val_r2_history, output_path=None, show_plot=False):
    model.eval()
    all_preds, all_targets, all_mean5_baseline = [], [], []
    sample_energy = None
    sample_last_map, sample_target_map, sample_pred_map = None, None, None

    print("\n正在进行模型评估...")

    if len(dataloader) == 0:
        print("验证集为空，跳过评估。")
        return

    with torch.no_grad():
        for i, (inputs, season_features, scalar_features, target_season, targets, spatial_prior) in enumerate(dataloader):
            inputs = inputs.to(device)
            season_features = season_features.to(device)
            scalar_features = scalar_features.to(device)
            target_season = target_season.to(device)
            targets = targets.to(device)
            spatial_prior = spatial_prior.to(device)

            preds, energies, _ = model(inputs, spatial_prior, season_features, scalar_features, target_season)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_mean5_baseline.extend(scalar_features[:, -5:, 0].mean(dim=1).cpu().numpy())

            if i == 0:
                sample_energy = energies.cpu().numpy()
                sample_last_map = inputs[0, -1, 0].detach().cpu().numpy()
                sample_target_map = targets[0, 0].detach().cpu().numpy()
                sample_pred_map = preds[0, 0].detach().cpu().numpy()

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 1, 0, 0))
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 1, 0, 0))
    all_mean5_baseline = np.array(all_mean5_baseline)

    if len(all_preds) == 0:
        print("没有可评估的样本，跳过可视化。")
        return

    mae, mse, coverage_mae, coverage_mse, r2 = heatmap_metrics(all_preds, all_targets)
    all_pred_coverage = all_preds.reshape(all_preds.shape[0], -1).mean(axis=1)
    all_target_coverage = all_targets.reshape(all_targets.shape[0], -1).mean(axis=1)
    print(
        f"\nHeatmap metrics: Pixel MAE={mae:.6f}, Pixel MSE={mse:.6f}, "
        f"Coverage MAE={coverage_mae:.6f}, Coverage MSE={coverage_mse:.6f}, Coverage R2={r2:.4f}"
    )

    print(f"\n最终评估指标: MAE={mae:.6f}, MSE={mse:.6f}, R2 Score={r2:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(f'Next-Day Snow Heatmap Predictor (Coverage R2={r2:.3f})', fontsize=16)

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(loss_history, label='Weighted Train Loss', color='purple')
    ax1.set_title('1. Training Loss & Validation R2', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1_r2 = ax1.twinx()
    ax1_r2.plot(val_r2_history, label='Val Coverage R2', color='tab:green', linestyle='--')
    ax1_r2.set_ylabel('Coverage R2')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    ax1.grid(True, alpha=0.5)

    ax2 = fig.add_subplot(2, 4, 2)
    idx = np.random.choice(len(all_pred_coverage), size=min(500, len(all_pred_coverage)), replace=False)
    ax2.scatter(all_target_coverage[idx], all_pred_coverage[idx], alpha=0.6, c='tab:blue', s=30)
    lims = [0, max(all_target_coverage.max(), all_pred_coverage.max())]
    ax2.plot(lims, lims, 'k--', alpha=0.75)
    ax2.set_title('2. True vs Predicted Coverage Mean', fontsize=12)

    ax3 = fig.add_subplot(2, 4, 3)
    subset = min(50, len(all_target_coverage))
    ax3.plot(all_target_coverage[:subset], 'k-', label='True')
    ax3.plot(all_pred_coverage[:subset], 'g--', label='Pred')
    ax3.plot(all_mean5_baseline[:subset], color='tab:blue', linestyle=':', label='Mean-5 Baseline')
    ax3.axhline(0.3, color='r', linestyle=':', label='Passable Threshold')
    ax3.set_title('3. Prediction Sequence', fontsize=12)
    ax3.legend()

    ax4 = fig.add_subplot(2, 4, 4)
    if sample_last_map is not None:
        im4 = ax4.imshow(sample_last_map, cmap='viridis', vmin=0.0, vmax=1.0)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title('4. Last Input Snow Map', fontsize=12)
    ax4.axis('off')

    ax5 = fig.add_subplot(2, 4, 5)
    if sample_target_map is not None:
        im5 = ax5.imshow(sample_target_map, cmap='viridis', vmin=0.0, vmax=1.0)
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_title('5. True Next-Day Heatmap', fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 4, 6)
    if sample_pred_map is not None:
        im6 = ax6.imshow(sample_pred_map, cmap='viridis', vmin=0.0, vmax=1.0)
        fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_title('6. Predicted Next-Day Heatmap', fontsize=12)
    ax6.axis('off')

    ax7 = fig.add_subplot(2, 4, 7)
    if sample_pred_map is not None and sample_target_map is not None:
        im7 = ax7.imshow(np.abs(sample_pred_map - sample_target_map), cmap='magma', vmin=0.0, vmax=0.5)
        fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    ax7.set_title('7. Absolute Error Map', fontsize=12)
    ax7.axis('off')

    ax8 = fig.add_subplot(2, 4, 8)
    if sample_energy is not None:
        steps = np.arange(sample_energy.shape[1])
        for k in range(min(5, sample_energy.shape[0])):
            ax8.plot(steps, sample_energy[k], marker='o', label=f'Sample {k}')
    ax8.set_title('8. Internal Energy Minimization', fontsize=12)
    ax8.set_xlabel('Thinking Steps')
    ax8.set_ylabel('Energy')
    ax8.grid(True)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"评估图已保存: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ==========================================
# 7. 主程序
# ==========================================
