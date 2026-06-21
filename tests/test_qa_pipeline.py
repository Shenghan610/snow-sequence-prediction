import json
from datetime import date, timedelta

import numpy as np
import torch

from snow_attractor.data import CausalSpatialFeatureStore, MaskedSnowSequenceDataset
from snow_attractor.losses import AttractorEnergyLoss
from snow_attractor.model import AttractorEnergyUNet
from snow_attractor.qa import decode_mod10a1_masks, parse_raster_date


def test_mod10a1_strict_mask_and_filename_dates():
    snow = np.full((1, 7), 50.0, dtype=np.float32)
    basic = np.asarray([[0, 1, 2, 0, 0, 0, 0]], dtype=np.float32)
    flags = np.asarray(
        [[0, 0, 0, 1 << 0, 1 << 1, 1 << 4, 1 << 7]],
        dtype=np.float32,
    )
    strict, lenient, land = decode_mod10a1_masks(snow, basic, flags)

    assert strict.tolist() == [[True, True, False, False, False, False, False]]
    assert lenient.tolist() == [[True, True, True, False, True, True, False]]
    assert land.tolist() == [[True, True, True, False, True, True, True]]
    assert parse_raster_date("layer_2024_02_29.tif") == date(2024, 2, 29)
    assert parse_raster_date("MOD10A1.A2024060.tile.tif") == date(2024, 2, 29)
    assert parse_raster_date("layer_20240229.tif") == date(2024, 2, 29)


def _write_processed_region(root, region_id="ali", days=12, size=8):
    processed = root / region_id / "processed"
    processed.mkdir(parents=True)
    dates = [date(2020, 1, 1) + timedelta(days=offset) for offset in range(days)]
    snow = np.stack(
        [
            np.full((size, size), offset / 20.0, dtype=np.float32)
            for offset in range(days)
        ]
    )
    valid = np.ones_like(snow, dtype=np.uint8)
    valid[0, 0, 0] = 0
    np.save(processed / "snow.npy", snow)
    np.save(processed / "valid_strict.npy", valid)
    np.save(processed / "valid_lenient.npy", valid)
    np.save(processed / "land_mask.npy", np.ones((size, size), dtype=np.uint8))
    np.save(processed / "terrain.npy", np.zeros((5, size, size), dtype=np.float32))
    (processed / "dates.json").write_text(
        json.dumps([item.isoformat() for item in dates]),
        encoding="utf-8",
    )


def test_masked_dataset_keeps_target_unfilled_and_uses_history_only(tmp_path):
    _write_processed_region(tmp_path)
    dataset = MaskedSnowSequenceDataset(
        tmp_path,
        "ali",
        sequence_length=4,
        image_size=8,
        target_min_valid_fraction=0.5,
    )
    dataset.fit_external_normalization([0, 1, 2])
    sample = dataset[0]

    assert sample["inputs"].shape == (4, 3, 8, 8)
    assert sample["target"].shape == (1, 8, 8)
    assert sample["target_valid_mask"].shape == (1, 8, 8)
    assert sample["observation_mask"][0, 0, 0, 0] == 0
    assert sample["observation_age"][0, 0, 0, 0] == 1
    assert torch.isfinite(sample["inputs"]).all()


def test_causal_power_store_excludes_future_date(tmp_path):
    directory = tmp_path / "external_features"
    raw = directory / "power_grid_daily_raw"
    raw.mkdir(parents=True)
    rows = ["point_id,latitude,longitude,elevation_m"]
    parameters = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "T2MDEW",
        "RH2M",
        "PRECTOTCORR",
        "WS10M",
        "ALLSKY_SFC_SW_DWN",
        "ALLSKY_SFC_LW_DWN",
    ]
    for point_id in range(4):
        rows.append(f"{point_id},{point_id // 2},{point_id % 2},1000")
        payload = {
            "properties": {
                "parameter": {
                    name: {
                        "20200101": float(point_id),
                        "20200102": float(point_id + 1),
                        "20200103": 9999.0,
                    }
                    for name in parameters
                }
            }
        }
        (raw / f"power_point_{point_id:02d}_0_0.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    (directory / "external_grid_points.csv").write_text(
        "\n".join(rows),
        encoding="utf-8",
    )
    store = CausalSpatialFeatureStore(directory, image_size=8)
    store.fit(["2020-01-01", "2020-01-02"])
    features = store.get(["2020-01-01", "2020-01-02"])

    assert features.shape == (18, 8, 8)
    assert float(features.abs().max()) < 10.0


def test_shared_residual_starts_at_backbone_and_coverage_is_map_derived():
    model = AttractorEnergyUNet(
        input_steps=4,
        input_channels=3,
        context_dim=4,
        base_channels=8,
        bottleneck_channels=32,
        attractor_iterations=1,
        attention_heads=4,
        internal_size=32,
        temporal_fusion_mode="gated_bottleneck",
        temporal_hidden_channels=16,
        shared_backbone_residual=True,
        dropout=0.0,
    )
    land = torch.zeros(2, 1, 32, 32)
    land[:, :, :, :16] = 1
    outputs = model(
        torch.rand(2, 4, 3, 32, 32),
        torch.rand(2, 2, 32, 32),
        torch.rand(2, 4),
        land_mask=land,
    )

    assert torch.allclose(
        outputs["prediction"],
        outputs["backbone_prediction"].clamp(0.0, 1.0),
        atol=1e-6,
    )
    expected = (outputs["prediction"] * land).flatten(1).sum(1) / land.sum(
        dim=(1, 2, 3)
    )
    assert torch.allclose(outputs["coverage_prediction"], expected, atol=1e-6)


def test_dominance_penalty_only_applies_when_hybrid_is_worse():
    target = torch.zeros(2, 1, 4, 4)
    prediction = torch.stack(
        [torch.zeros(1, 4, 4), torch.ones(1, 4, 4)]
    )
    backbone = torch.stack(
        [torch.ones(1, 4, 4), torch.zeros(1, 4, 4)]
    )
    outputs = {
        "prediction": prediction,
        "backbone_prediction": backbone,
        "coverage_prediction": prediction.flatten(1).mean(1),
        "last_input_coverage": torch.zeros(2),
        "coverage_delta": prediction.flatten(1).mean(1),
        "energy": torch.zeros(2, 1),
        "manifold_coordinate": torch.zeros(2, 2),
        "update_norms": torch.zeros(2, 0),
        "auxiliary_predictions": [],
    }
    criterion = AttractorEnergyLoss(
        final_l1_weight=0.0,
        final_mse_weight=0.0,
        coverage_l1_weight=0.0,
        coverage_mse_weight=0.0,
        dominance_weight=1.0,
    )
    losses = criterion(outputs, target, valid_mask=torch.ones_like(target))

    assert torch.allclose(losses["dominance"], torch.tensor(1.5))
