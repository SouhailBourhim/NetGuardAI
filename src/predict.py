"""
Phase 3 — Unified inference: supervised classifier + LSTM Autoencoder.

Both pipelines run on every flow. The supervised model classifies known
attack types; the autoencoder flags anomalies by reconstruction error.

Models are loaded directly from data/processed/ (exported from MLflow
artifacts) to avoid mlflow.sklearn.load_model issues on Python 3.14.

Import-order note: XGBoost models must be unpickled BEFORE torch is
imported. Importing torch first causes a segfault on Python 3.14 +
XGBoost 3.x due to a memory allocator conflict. _Models.__init__ loads
all sklearn/XGBoost artifacts first, then imports torch lazily.
"""

import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

WINDOW_SIZE = 50


# ---------------------------------------------------------------------------
# Model cache — loaded once on first access
# ---------------------------------------------------------------------------

class _Models:
    _instance: "_Models | None" = None

    def __init__(self):
        # ── Step 1: load XGBoost/sklearn artifacts BEFORE importing torch ──
        with open(PROCESSED / "model_binary.pkl", "rb") as f:
            self.binary_model = pickle.load(f)
        with open(PROCESSED / "model_multi.pkl", "rb") as f:
            self.multi_model = pickle.load(f)
        with open(PROCESSED / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(PROCESSED / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(PROCESSED / "feature_cols.pkl", "rb") as f:
            self.feature_cols = pickle.load(f)
        with open(PROCESSED / "multi_class_names.pkl", "rb") as f:
            self.multi_class_names = pickle.load(f)
        self.threshold = float(np.load(PROCESSED / "autoencoder_threshold.npy")[0])

        # ── Step 2: now safe to import and use torch ──
        import torch
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.autoencoder = torch.load(
            PROCESSED / "model_autoencoder.pth",
            map_location=self.device,
            weights_only=False,
        )
        self.autoencoder.eval()

        self.binary_model_name = type(self.binary_model).__name__
        self.multi_model_name  = type(self.multi_model).__name__

        log.info(
            "Models loaded — binary=%s  multi=%s  device=%s  threshold=%.4f",
            self.binary_model_name, self.multi_model_name,
            self.device, self.threshold,
        )

    @classmethod
    def get(cls) -> "_Models":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_vector(features: Union[dict, list]) -> list[float]:
    m = _Models.get()
    if isinstance(features, dict):
        try:
            return [float(features[col]) for col in m.feature_cols]
        except KeyError as missing:
            raise KeyError(f"Feature not found in input: {missing}")
    return [float(v) for v in features]


def _scale_batch(vecs: list[list[float]]) -> np.ndarray:
    m = _Models.get()
    return m.scaler.transform(np.array(vecs, dtype=np.float32))


def _reconstruction_error(X_scaled: np.ndarray) -> np.ndarray:
    """Per-flow MSE via tiled-window autoencoder inference."""
    import torch
    m = _Models.get()
    errors = np.empty(len(X_scaled), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(X_scaled)):
            window = np.tile(X_scaled[i], (WINDOW_SIZE, 1))[np.newaxis]
            t = torch.tensor(window, dtype=torch.float32).to(m.device)
            errors[i] = float(((m.autoencoder(t) - t) ** 2).mean().item())
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(features: Union[dict, list]) -> dict:
    """
    Run both detection pipelines on a single network flow.

    Args:
        features: Feature dict (keyed by column name) or ordered list of
                  len(feature_cols) float values.

    Returns:
        dict with keys: label, is_attack, confidence,
                        anomaly_score, is_anomaly, anomaly_threshold
    """
    m = _Models.get()
    X = _scale_batch([_to_vector(features)])

    is_attack  = bool(m.binary_model.predict(X)[0])
    probs      = m.binary_model.predict_proba(X)[0]
    confidence = float(probs[1] if is_attack else probs[0])
    label      = (
        m.multi_class_names[int(m.multi_model.predict(X)[0])]
        if is_attack else "BENIGN"
    )
    score = float(_reconstruction_error(X)[0])

    return {
        "label":             label,
        "is_attack":         is_attack,
        "confidence":        round(confidence, 4),
        "anomaly_score":     round(score, 6),
        "is_anomaly":        bool(score > m.threshold),
        "anomaly_threshold": round(m.threshold, 6),
    }


def predict_batch(feature_list: list[Union[dict, list]]) -> list[dict]:
    """Run both detection pipelines on a batch of network flows."""
    m = _Models.get()
    X = _scale_batch([_to_vector(f) for f in feature_list])

    binary_preds = m.binary_model.predict(X)
    binary_probs = m.binary_model.predict_proba(X)
    multi_preds  = m.multi_model.predict(X)
    scores       = _reconstruction_error(X)

    return [
        {
            "label":             (
                m.multi_class_names[int(ci)] if bool(ia) else "BENIGN"
            ),
            "is_attack":         bool(ia),
            "confidence":        round(float(pr[1] if bool(ia) else pr[0]), 4),
            "anomaly_score":     round(float(sc), 6),
            "is_anomaly":        bool(float(sc) > m.threshold),
            "anomaly_threshold": round(m.threshold, 6),
        }
        for ia, pr, ci, sc in zip(binary_preds, binary_probs, multi_preds, scores)
    ]


def model_info() -> dict:
    m = _Models.get()
    return {
        "binary_model":      m.binary_model_name,
        "multi_model":       m.multi_model_name,
        "n_features":        len(m.feature_cols),
        "feature_cols":      m.feature_cols,
        "n_classes":         len(m.multi_class_names),
        "classes":           m.multi_class_names,
        "anomaly_threshold": m.threshold,
        "device":            str(m.device),
    }
