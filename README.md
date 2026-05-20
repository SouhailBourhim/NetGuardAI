# NetGuardAI

An end-to-end AI-powered Network Intrusion Detection System built on the CICIDS2017 dataset.

## Architecture

Two detection pipelines run in tandem:

- **Supervised** — Logistic Regression, Random Forest, XGBoost classify known attack types
- **Unsupervised** — LSTM Autoencoder detects zero-day anomalies via reconstruction error

Experiments are tracked with MLflow, predictions served via FastAPI, and everything visualised in a Streamlit dashboard — fully containerised with Docker.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — 8 days of network traffic, ~2.83M flows, 78 features, 15 classes (BENIGN + 14 attack types).

Place the pre-extracted CSVs under `data/raw/MachineLearningCVE/`.

## Results

### Supervised classifiers

| Model | Binary F1 | Binary ROC-AUC | Multi-class F1 |
|---|---|---|---|
| XGBoost ⭐ | **0.9915** | **0.9999** | **0.9963** |
| Random Forest | 0.9806 | 0.9981 | 0.9919 |
| Logistic Regression | 0.8762 | 0.9666 | 0.8529 |

XGBoost is registered as `netguardai_best_binary` → **Production** in the MLflow Model Registry.

> Multi-class evaluation covers 13 of 15 classes. `DoS GoldenEye` and `Heartbleed` fall entirely in the test split due to temporal ordering and cannot be trained on.

### LSTM Autoencoder (zero-day detection)

Trained exclusively on benign traffic. Reconstruction error above the p95 threshold triggers an anomaly alert.

| Metric | Score |
|---|---|
| F1 (weighted) | 0.8872 |
| Precision | 0.8214 |
| Recall | 0.8421 |
| ROC-AUC | 0.9247 |

## Project structure

```
├── data/
│   ├── raw/MachineLearningCVE/   # CICIDS2017 CSVs (not committed)
│   └── processed/                # cleaned arrays & pipeline objects
├── notebooks/
│   ├── 01_eda.ipynb              # class distribution, correlation heatmap, feature importance
│   ├── 02_preprocessing.ipynb   # preprocessing walkthrough & artifact inspection
│   ├── 03_supervised_training.ipynb  # model comparison, confusion matrices, ROC-AUC
│   └── 04_autoencoder.ipynb     # training curve, error distribution, threshold sweep
├── src/
│   ├── preprocess.py             # load, clean, split, scale, SMOTE → data/processed/
│   ├── train_supervised.py       # LR / RF / XGBoost, both binary & multi-class, MLflow
│   ├── train_autoencoder.py      # LSTM Autoencoder trained on benign-only traffic, MLflow
│   ├── predict.py                # unified inference (supervised + autoencoder)
│   └── promote_best_model.py     # promotes best binary run to MLflow Production
├── api/
│   └── main.py                   # FastAPI prediction service
├── dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
├── mlruns/                       # MLflow tracking (not committed)
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Phase 1 — Preprocessing:**
```bash
python src/preprocess.py
```

**Phase 2 — Training:**
```bash
python src/train_supervised.py        # binary + multi-class (LR, RF, XGBoost)
python src/train_autoencoder.py       # LSTM Autoencoder
python src/promote_best_model.py      # promote best binary model to Production
```

**Notebooks** (use the `Python 3 (NetGuardAI)` kernel):
```bash
jupyter notebook
```

**Start all services with Docker:**
```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Preprocessing | ✅ Done | Load, clean, temporal split, StandardScaler, SMOTE |
| 2 — Training | ✅ Done | Supervised (LR/RF/XGBoost) + LSTM Autoencoder, MLflow tracking |
| 3 — Serving | 🔜 Next | FastAPI inference endpoint + model promotion |
| 4 — Dashboard | 🔜 | Streamlit real-time monitoring |
