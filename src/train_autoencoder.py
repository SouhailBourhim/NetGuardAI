"""
Phase 2 — LSTM Autoencoder for zero-day anomaly detection.
Trains exclusively on benign traffic; high reconstruction error at inference = anomaly.

Windowing: consecutive flows are grouped into non-overlapping windows of WINDOW_SIZE.
The model learns to reconstruct normal window patterns; attack windows produce high MSE.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MLFLOW_URI = str(ROOT / "mlruns")

WINDOW_SIZE = 50
HIDDEN_SIZE = 64
N_LAYERS = 2
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
THRESHOLD_PERCENTILE = 95
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers):
        super().__init__()
        dropout = 0.2 if n_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden_size, n_layers,
                               batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers,
                               batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        seq_len = x.shape[1]
        _, (h, c) = self.encoder(x)
        # Seed decoder with last encoder hidden state repeated across timesteps
        dec_in = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.output_layer(dec_out)  # (batch, seq_len, n_features)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(X: np.ndarray, window_size: int) -> np.ndarray:
    """Non-overlapping windows; truncates the last partial window."""
    n = (len(X) // window_size) * window_size
    return X[:n].reshape(-1, window_size, X.shape[1])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def reconstruction_errors(model, loader, device) -> np.ndarray:
    """Per-window mean squared reconstruction error."""
    model.eval()
    errors = []
    for (x,) in loader:
        x = x.to(device)
        mse = ((model(x) - x) ** 2).mean(dim=(1, 2))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load data
    X_train_scaled = np.load(PROCESSED / "X_train_scaled.npy").astype(np.float32)
    y_train_raw    = np.load(PROCESSED / "y_train_binary_raw.npy")
    X_test         = np.load(PROCESSED / "X_test.npy").astype(np.float32)
    y_test         = np.load(PROCESSED / "y_test_binary.npy")
    n_features     = X_train_scaled.shape[1]
    log.info("n_features=%d  train=%d  test=%d", n_features, len(X_train_scaled), len(X_test))

    # Benign-only training data
    X_benign = X_train_scaled[y_train_raw == 0]
    log.info("Benign train samples: %d", len(X_benign))

    # Windows
    X_win = make_windows(X_benign, WINDOW_SIZE)
    log.info("Train windows: %s", X_win.shape)

    val_split     = int(len(X_win) * 0.9)
    X_win_train   = torch.tensor(X_win[:val_split])
    X_win_val     = torch.tensor(X_win[val_split:])
    train_loader  = DataLoader(TensorDataset(X_win_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(TensorDataset(X_win_val),   batch_size=BATCH_SIZE)

    # Model / optimiser
    model     = LSTMAutoencoder(n_features, HIDDEN_SIZE, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("netguardai_autoencoder")

    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({
            "window_size": WINDOW_SIZE, "hidden_size": HIDDEN_SIZE,
            "n_layers": N_LAYERS, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "n_features": n_features, "benign_train_samples": len(X_benign),
            "device": str(device),
        })

        # Training loop
        best_val_loss, best_state = float("inf"), None
        train_losses, val_losses  = [], []

        for epoch in range(1, EPOCHS + 1):
            t_loss   = train_epoch(model, train_loader, optimizer, criterion, device)
            val_errs = reconstruction_errors(model, val_loader, device)
            v_loss   = float(val_errs.mean())
            scheduler.step(v_loss)

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss}, step=epoch)
            log.info("Epoch %2d/%d  train=%.6f  val=%.6f", epoch, EPOCHS, t_loss, v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        # Anomaly threshold from benign validation errors
        threshold = float(np.percentile(val_errs, THRESHOLD_PERCENTILE))
        mlflow.log_metric("anomaly_threshold", threshold)
        log.info("Anomaly threshold (p%d): %.6f", THRESHOLD_PERCENTILE, threshold)

        # Training curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, EPOCHS + 1), train_losses, label="train")
        ax.plot(range(1, EPOCHS + 1), val_losses,   label="val")
        ax.set(xlabel="Epoch", ylabel="MSE Loss",
               title="LSTM Autoencoder — Training Curve")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, "training_curve.png")
        plt.close(fig)

        # ── Evaluate on test set ────────────────────────────────────────────
        X_test_win = make_windows(X_test, WINDOW_SIZE)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test_win)), batch_size=BATCH_SIZE)
        test_errs   = reconstruction_errors(model, test_loader, device)

        n_win = len(X_test_win)
        y_true_flow  = y_test[: n_win * WINDOW_SIZE]           # trim to full windows
        y_pred_win   = (test_errs > threshold).astype(int)
        y_pred_flow  = np.repeat(y_pred_win, WINDOW_SIZE)       # broadcast to per-flow
        flow_scores  = np.repeat(test_errs, WINDOW_SIZE)        # continuous score per flow

        # Window-level label: 1 if any flow in window is an attack
        y_win_true = y_true_flow.reshape(n_win, WINDOW_SIZE).max(axis=1)

        metrics = {
            "test_accuracy":       accuracy_score(y_true_flow, y_pred_flow),
            "test_f1_weighted":    f1_score(y_true_flow, y_pred_flow, average="weighted", zero_division=0),
            "test_precision":      precision_score(y_true_flow, y_pred_flow, average="binary", zero_division=0),
            "test_recall":         recall_score(y_true_flow, y_pred_flow, average="binary", zero_division=0),
            "test_roc_auc":        roc_auc_score(y_true_flow, flow_scores),
        }
        mlflow.log_metrics(metrics)
        log.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        report = classification_report(
            y_true_flow, y_pred_flow,
            target_names=["BENIGN", "ANOMALY"], zero_division=0,
        )
        mlflow.log_text(report, "classification_report.txt")

        # Reconstruction error distribution
        benign_errs = test_errs[y_win_true == 0]
        attack_errs = test_errs[y_win_true == 1]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(benign_errs, bins=100, alpha=0.6, label="Benign",  color="steelblue", density=True)
        ax.hist(attack_errs, bins=100, alpha=0.6, label="Attack",  color="tomato",    density=True)
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"Threshold (p{THRESHOLD_PERCENTILE})")
        ax.set(xlabel="Reconstruction Error (MSE)", ylabel="Density",
               title="Reconstruction Error Distribution — Test Set")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, "reconstruction_error_dist.png")
        plt.close(fig)

        # Save threshold for inference
        np.save(str(PROCESSED / "autoencoder_threshold.npy"), np.array([threshold]))
        mlflow.log_artifact(str(PROCESSED / "autoencoder_threshold.npy"))

        mlflow.pytorch.log_model(model, artifact_path="model")
        log.info("Autoencoder training complete.")


if __name__ == "__main__":
    run()
