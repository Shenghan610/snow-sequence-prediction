import argparse
import json
import os

from inference import SnowInferenceService


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="部署并运行积雪热力图预测模型。")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(script_dir, "Ali_SnowData")
    )
    parser.add_argument(
        "--external-features",
        default=os.path.join(
            script_dir,
            "ExternalClimateTerrain",
            "external_daily_features.csv"
        )
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(script_dir, "best_highres_snow_heatmap_model.pth")
    )
    parser.add_argument("--target-date", default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "predictions")
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    service = SnowInferenceService(
        data_dir=args.data_dir,
        external_feature_path=args.external_features,
        weights_path=args.weights,
        device=args.device
    )
    result = service.predict(
        target_date=args.target_date,
        output_dir=args.output_dir
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
