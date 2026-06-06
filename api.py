import os
from functools import lru_cache
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:
    raise RuntimeError(
        "API 部署需要 FastAPI。请先执行：pip install -r requirements-deploy.txt"
    ) from exc

from inference import SnowInferenceService


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(
    title="Snow Sequence Prediction API",
    version="1.0.0",
    description="使用最佳模型权重预测下一日积雪覆盖热力图。"
)


@lru_cache(maxsize=1)
def get_service():
    return SnowInferenceService(
        data_dir=os.getenv(
            "SNOW_DATA_DIR",
            os.path.join(SCRIPT_DIR, "Ali_SnowData")
        ),
        external_feature_path=os.getenv(
            "SNOW_EXTERNAL_FEATURES",
            os.path.join(
                SCRIPT_DIR,
                "ExternalClimateTerrain",
                "external_daily_features.csv"
            )
        ),
        weights_path=os.getenv(
            "SNOW_MODEL_WEIGHTS",
            os.path.join(SCRIPT_DIR, "best_highres_snow_heatmap_model.pth")
        ),
        device=os.getenv("SNOW_DEVICE", "auto")
    )


@app.get("/health")
def health():
    service = get_service()
    return {
        "status": "ok",
        "device": str(service.device),
        "weights": service.weights_path,
        "latest_input_date": service.dataset.date_list[-1]
    }


@app.post("/predict")
def predict(target_date: Optional[str] = None):
    try:
        result = get_service().predict(
            target_date=target_date,
            output_dir=os.getenv(
                "SNOW_OUTPUT_DIR",
                os.path.join(SCRIPT_DIR, "predictions")
            )
        )
        return result.to_dict()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
