"""
Phase 3 — FastAPI prediction service.

Endpoints:
  GET  /health          — liveness check
  GET  /model/info      — loaded model metadata
  POST /predict         — single-flow prediction
  POST /predict/batch   — batch prediction
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict as predictor


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor._Models.get()   # pre-load all models before accepting requests
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NetGuardAI",
    description=(
        "AI-powered Network Intrusion Detection System. "
        "Runs a supervised classifier (XGBoost) and an LSTM Autoencoder "
        "in parallel on every network flow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FlowRequest(BaseModel):
    features: Union[dict[str, float], list[float]] = Field(
        ...,
        description=(
            "Network flow features as a named dict (key = column name) "
            "or an ordered list of 47 float values."
        ),
        examples=[{"Destination Port": 80.0, "Flow Duration": 1234.0}],
    )


class BatchFlowRequest(BaseModel):
    flows: list[Union[dict[str, float], list[float]]] = Field(
        ..., description="List of flow feature dicts or ordered lists.",
    )


class PredictionResponse(BaseModel):
    label: str              = Field(..., description="Predicted class (BENIGN or attack type)")
    is_attack: bool         = Field(..., description="True if traffic is classified as an attack")
    confidence: float       = Field(..., description="Classifier probability for the predicted class")
    anomaly_score: float    = Field(..., description="Autoencoder reconstruction MSE")
    is_anomaly: bool        = Field(..., description="True if anomaly_score exceeds the threshold")
    anomaly_threshold: float = Field(..., description="Current anomaly detection threshold (p95)")


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    binary_model: str
    multi_model: str
    n_features: int
    feature_cols: list[str]
    n_classes: int
    classes: list[str]
    anomaly_threshold: float
    device: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info():
    return predictor.model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(body: FlowRequest):
    try:
        return predictor.predict(body.features)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(body: BatchFlowRequest):
    if not body.flows:
        raise HTTPException(status_code=422, detail="flows list must not be empty")
    try:
        results = predictor.predict_batch(body.flows)
        return {"count": len(results), "results": results}
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
