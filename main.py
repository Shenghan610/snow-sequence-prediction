"""Training entry point and backward-compatible public imports."""

from snow_prediction import (
    SCALAR_FEATURE_DIM,
    SEASON_FEATURE_DIM,
    AliSnowDatasetRAM,
    DataDrivenSnowPredictor,
    ExternalFeatureStore,
    ModelEMA,
    PriorDynamicalSelfAttention,
    PriorSpatialEncoder,
    baseline_anchor_loss,
    evaluate_and_visualize,
    evaluate_baselines,
    evaluate_metrics,
    heatmap_metrics,
    load_model_state,
    parse_args,
    regression_metrics,
    run_training,
    set_seed,
    snow_heatmap_loss,
    weighted_mse_loss,
)

__all__ = [
    "SCALAR_FEATURE_DIM",
    "SEASON_FEATURE_DIM",
    "AliSnowDatasetRAM",
    "DataDrivenSnowPredictor",
    "ExternalFeatureStore",
    "ModelEMA",
    "PriorDynamicalSelfAttention",
    "PriorSpatialEncoder",
    "baseline_anchor_loss",
    "evaluate_and_visualize",
    "evaluate_baselines",
    "evaluate_metrics",
    "heatmap_metrics",
    "load_model_state",
    "parse_args",
    "regression_metrics",
    "run_training",
    "set_seed",
    "snow_heatmap_loss",
    "weighted_mse_loss",
]


if __name__ == "__main__":
    run_training()
