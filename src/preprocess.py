"""
Phase 1 preprocessing pipeline for CICIDS2017.

Outputs saved to data/processed/:
  X_train.npy, X_test.npy
  y_train_binary.npy,  y_test_binary.npy
  X_train_multi.npy,   y_train_multi.npy,  y_test_multi.npy
  X_train_scaled.npy,  y_train_binary_raw.npy  (for autoencoder)
  scaler.pkl, label_encoder.pkl, feature_cols.pkl
"""

import glob
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "MachineLearningCVE"
PROCESSED_DIR = ROOT / "data" / "processed"

CORR_THRESHOLD = 0.95
TEST_FRAC = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load & concatenate
# ---------------------------------------------------------------------------
def load_data(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    log.info("Loading %d CSV files …", len(csv_files))
    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        log.info("  %-65s  rows=%d", Path(path).name, len(df))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined shape: %s", combined.shape)
    return combined


# ---------------------------------------------------------------------------
# 2-5. Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    before = len(df)
    df = df.drop_duplicates()
    log.info("Dropped %d duplicate rows", before - len(df))

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    log.info("Dropped %d rows with NaN/inf values", before - len(df))

    return df.reset_index(drop=True)


def drop_low_variance(df: pd.DataFrame, feature_cols: list) -> list:
    variances = df[feature_cols].var()
    constant = variances[variances == 0].index.tolist()
    if constant:
        log.info("Dropping %d constant features: %s", len(constant), constant)
    return [c for c in feature_cols if c not in constant]


def drop_highly_correlated(df: pd.DataFrame, feature_cols: list, threshold: float) -> list:
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    log.info(
        "Dropping %d highly-correlated features (threshold=%.2f): %s",
        len(to_drop),
        threshold,
        to_drop,
    )
    return [c for c in feature_cols if c not in to_drop]


# ---------------------------------------------------------------------------
# 6. Encode labels
# ---------------------------------------------------------------------------
def encode_labels(series: pd.Series):
    binary = (series != "BENIGN").astype(int).values
    le = LabelEncoder()
    multi = le.fit_transform(series.values)
    return binary, multi, le


# ---------------------------------------------------------------------------
# 7. Temporal split (CSV order preserved — no shuffle)
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, test_frac: float):
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    log.info("Train size: %d  |  Test size: %d", len(train), len(test))
    return train, test


# ---------------------------------------------------------------------------
# 8. Scale (fit on train only)
# ---------------------------------------------------------------------------
def scale(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# 9. SMOTE (train only)
# ---------------------------------------------------------------------------
def apply_smote(X: np.ndarray, y: np.ndarray, label: str = "") -> tuple:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    log.info("Class distribution before SMOTE %s: %s", label, counts)
    min_count = min(counts.values())
    k = min(5, min_count - 1) if min_count > 1 else 1
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    log.info("After SMOTE %s: %d → %d samples", label, len(y), len(y_res))
    return X_res, y_res


# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
def save_artifacts(processed_dir: Path, artifacts: dict):
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        path = processed_dir / name
        if name.endswith(".npy"):
            np.save(str(path), obj)
        else:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        log.info("Saved %-35s  shape=%s", name, getattr(obj, "shape", "—"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    df = load_data(RAW_DIR)
    df = clean(df)

    feature_cols = [c for c in df.columns if c != "Label"]
    feature_cols = drop_low_variance(df, feature_cols)
    feature_cols = drop_highly_correlated(df, feature_cols, CORR_THRESHOLD)
    log.info("Features retained after filtering: %d", len(feature_cols))

    binary_labels, multi_labels, le = encode_labels(df["Label"])
    df["_binary"] = binary_labels
    df["_multi"] = multi_labels

    train_df, test_df = temporal_split(df, TEST_FRAC)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)
    y_train_binary = train_df["_binary"].values
    y_test_binary = test_df["_binary"].values
    y_train_multi = train_df["_multi"].values
    y_test_multi = test_df["_multi"].values

    X_train_s, X_test_s, scaler = scale(X_train_raw, X_test_raw)

    # SMOTE variants for supervised classifiers
    X_train_bin_sm, y_train_bin_sm = apply_smote(X_train_s, y_train_binary, "(binary)")
    X_train_multi_sm, y_train_multi_sm = apply_smote(X_train_s, y_train_multi, "(multi)")

    save_artifacts(
        PROCESSED_DIR,
        {
            # binary classifier inputs (SMOTE applied)
            "X_train.npy": X_train_bin_sm,
            "X_test.npy": X_test_s,
            "y_train_binary.npy": y_train_bin_sm,
            "y_test_binary.npy": y_test_binary,
            # multi-class classifier inputs (SMOTE applied)
            "X_train_multi.npy": X_train_multi_sm,
            "y_train_multi.npy": y_train_multi_sm,
            "y_test_multi.npy": y_test_multi,
            # unaugmented scaled train (for LSTM autoencoder — uses benign only)
            "X_train_scaled.npy": X_train_s,
            "y_train_binary_raw.npy": y_train_binary,
            # persisted pipeline objects
            "scaler.pkl": scaler,
            "label_encoder.pkl": le,
            "feature_cols.pkl": feature_cols,
        },
    )
    log.info("Preprocessing complete. Artifacts saved to %s", PROCESSED_DIR)


if __name__ == "__main__":
    run()
