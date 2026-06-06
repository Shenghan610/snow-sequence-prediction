import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from inference import (
    ModelConfig,
    SnowInferenceService,
    next_date,
    save_prediction_outputs
)
from snow_prediction import DataDrivenSnowPredictor


class InferenceUnitTests(unittest.TestCase):
    def test_next_date_handles_year_boundary(self):
        self.assertEqual(next_date("2024-12-31"), "2025-01-01")

    def test_model_forward_returns_bounded_heatmap(self):
        model = DataDrivenSnowPredictor(
            in_channels=1,
            d_model=32,
            iterations=2,
            season_dim=6,
            scalar_dim=11,
            base_window=3,
            hidden_dropout=0.0,
            feature_dropout=0.0,
            head_dropout=0.0
        ).eval()
        inputs = torch.rand(2, 4, 1, 32, 32)
        spatial_prior = torch.rand(2, 2, 32, 32)
        season_features = torch.rand(2, 4, 6)
        scalar_features = torch.rand(2, 4, 11)
        target_season = torch.rand(2, 6)

        with torch.inference_mode():
            prediction, energies, _ = model(
                inputs,
                spatial_prior,
                season_features,
                scalar_features,
                target_season
            )

        self.assertEqual(tuple(prediction.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(energies.shape), (2, 3))
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)

    def test_prediction_outputs_preserve_reference_geography(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "reference.tif")
            profile = {
                "driver": "GTiff",
                "height": 20,
                "width": 30,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": from_origin(78.0, 36.0, 0.01, 0.01)
            }
            with rasterio.open(reference_path, "w", **profile) as dst:
                dst.write(np.zeros((20, 30), dtype=np.uint8), 1)

            paths = save_prediction_outputs(
                prediction_map=np.full((8, 8), 0.25, dtype=np.float32),
                reference_tif=reference_path,
                output_dir=temp_dir,
                target_date="2025-01-01",
                metadata={"test": True}
            )

            for path in paths:
                self.assertTrue(os.path.isfile(path))
            with rasterio.open(paths[0]) as result:
                self.assertEqual((result.height, result.width), (20, 30))
                self.assertEqual(result.crs.to_string(), "EPSG:4326")
                self.assertAlmostEqual(float(result.read(1).mean()), 0.25, places=5)


@unittest.skipUnless(
    os.getenv("RUN_MODEL_INTEGRATION_TEST") == "1",
    "设置 RUN_MODEL_INTEGRATION_TEST=1 后运行真实权重集成测试。"
)
class RealModelIntegrationTests(unittest.TestCase):
    def test_best_weight_prediction(self):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        service = SnowInferenceService(
            data_dir=os.path.join(project_dir, "Ali_SnowData"),
            external_feature_path=os.path.join(
                project_dir,
                "ExternalClimateTerrain",
                "external_daily_features.csv"
            ),
            weights_path=os.path.join(
                project_dir,
                "best_highres_snow_heatmap_model.pth"
            ),
            device="auto",
            config=ModelConfig()
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result = service.predict(output_dir=output_dir)
            datetime.strptime(result.target_date, "%Y-%m-%d")
            self.assertGreaterEqual(result.predicted_coverage, 0.0)
            self.assertLessEqual(result.predicted_coverage, 1.0)
            self.assertTrue(os.path.isfile(result.geotiff_path))
            self.assertTrue(os.path.isfile(result.png_path))
            self.assertTrue(os.path.isfile(result.metadata_path))


if __name__ == "__main__":
    unittest.main()
