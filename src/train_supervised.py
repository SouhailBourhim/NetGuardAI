"""
Phase 2 — supervised training: Logistic Regression, Random Forest, XGBoost.
Trains each model for both binary and multi-class classification.
All runs tracked in MLflow under experiments netguardai_binary / netguardai_multiclass.
"""

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MLFLOW_URI = str(ROOT / "mlruns")

# Cap multi-class SMOTE output to avoid excessive memory/time on large arrays
MAX_MULTICLASS_TRAIN = 500_000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_processed():
    data = {}
    for name in [
        "X_train", "X_test",
        "y_train_binary", "y_test_binary",
        "X_train_multi", "y_train_multi", "y_test_multi",
    ]:
        data[name] = np.load(PROCESSED / f"{name}.npy")
        log.info("Loaded %-20s  shape=%s", name, data[name].shape)
    with open(PROCESSED / "label_encoder.pkl", "rb") as f:
        data["label_encoder"] = pickle.load(f)
    return data


def stratified_subsample(X, y, max_n):
    if len(X) <= max_n:
        return X, y
    log.info("Subsampling %d → %d (stratified)", len(X), max_n)
    return resample(X, y, n_samples=max_n, stratify=y, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, title):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        title=title, ylabel="True label", xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, mode):
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if mode == "binary" and y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(model_name, model, X_train, y_train, X_test, y_test,
                   class_names, mode, experiment_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({"model": model_name, "mode": mode, "phase": "2"})
        mlflow.log_params({k: str(v) for k, v in model.get_params().items()})

        log.info("[%s | %s] Training on %d samples …", mode, model_name, len(X_train))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob, mode)
        mlflow.log_metrics(metrics)
        log.info("[%s | %s] %s", mode, model_name,
                 {k: round(v, 4) for k, v in metrics.items()})

        report = classification_report(
            y_test, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0,
        )
        mlflow.log_text(report, "classification_report.txt")

        cm = confusion_matrix(y_test, y_pred)
        fig = plot_confusion_matrix(cm, class_names, f"{model_name} ({mode})")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.sklearn.log_model(model, artifact_path="model")
        log.info("[%s | %s] Done.", mode, model_name)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="saga",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(skip_binary: bool = False):
    mlflow.set_tracking_uri(MLFLOW_URI)
    data = load_processed()
    le = data["label_encoder"]

    # ── Binary ──────────────────────────────────────────────────────────────
    if not skip_binary:
        log.info("=== BINARY CLASSIFICATION ===")
        binary_names = ["BENIGN", "ATTACK"]
        for name, model in build_models().items():
            run_experiment(
                model_name=name, model=model,
                X_train=data["X_train"], y_train=data["y_train_binary"],
                X_test=data["X_test"],   y_test=data["y_test_binary"],
                class_names=binary_names, mode="binary",
                experiment_name="netguardai_binary",
            )

    # ── Multi-class ──────────────────────────────────────────────────────────
    log.info("=== MULTI-CLASS CLASSIFICATION ===")
    X_tr_m, y_tr_m = stratified_subsample(
        data["X_train_multi"], data["y_train_multi"], MAX_MULTICLASS_TRAIN
    )

    # Re-map labels to contiguous 0..N-1 based on classes present in train.
    # Some rare classes (e.g. Heartbleed, DoS GoldenEye) fall entirely in the
    # test split due to temporal ordering, so they can't be trained on.
    train_class_idx = sorted(np.unique(y_tr_m).tolist())
    missing = [le.classes_[i] for i in range(len(le.classes_)) if i not in train_class_idx]
    if missing:
        log.warning("Classes absent from train split (excluded from test eval): %s", missing)
    remap = {old: new for new, old in enumerate(train_class_idx)}
    y_tr_m_r = np.array([remap[c] for c in y_tr_m])
    test_known = np.isin(data["y_test_multi"], train_class_idx)
    y_test_m_r = np.array([remap[c] for c in data["y_test_multi"][test_known]])
    X_test_m   = data["X_test"][test_known]
    multi_names = [le.classes_[i] for i in train_class_idx]
    log.info("Multi-class train classes: %d  |  test samples after filter: %d",
             len(multi_names), len(y_test_m_r))

    # Persist class metadata for inference
    with open(PROCESSED / "multi_class_names.pkl", "wb") as f:
        pickle.dump(multi_names, f)
    with open(PROCESSED / "train_class_idx.pkl", "wb") as f:
        pickle.dump(train_class_idx, f)

    for name, model in build_models().items():
        if name == "xgboost":
            model.set_params(objective="multi:softprob", eval_metric="mlogloss",
                             num_class=len(multi_names))
        run_experiment(
            model_name=name, model=model,
            X_train=X_tr_m,    y_train=y_tr_m_r,
            X_test=X_test_m,   y_test=y_test_m_r,
            class_names=multi_names, mode="multi",
            experiment_name="netguardai_multiclass",
        )

    log.info("Phase 2 supervised training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-binary", action="store_true")
    args = parser.parse_args()
    run(skip_binary=args.skip_binary)
