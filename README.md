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

## Project structure

```
├── data/
│   ├── raw/MachineLearningCVE/   # CICIDS2017 CSVs (not committed)
│   └── processed/                # cleaned arrays & pipeline objects
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   └── 02_preprocessing.ipynb   # preprocessing walkthrough
├── src/
│   ├── preprocess.py             # Phase 1 — clean, split, scale, SMOTE
│   ├── train_supervised.py       # Phase 2 — LR / RF / XGBoost + MLflow
│   ├── train_autoencoder.py      # Phase 2 — LSTM Autoencoder + MLflow
│   ├── predict.py                # Phase 3 — unified inference
│   └── promote_best_model.py     # Phase 3 — MLflow model promotion
├── api/
│   └── main.py                   # FastAPI prediction service
├── dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
├── mlruns/                       # MLflow tracking (not committed)
├── Dockerfile.api
├── Dockerfile.dashboard
└── docker-compose.yml
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run preprocessing:**
```bash
python src/preprocess.py
```

**Run EDA notebook:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Start all services with Docker:**
```bash
docker-compose up --build
```

Services:
| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Preprocessing | ✅ Done | Load, clean, split, scale, SMOTE |
| 2 — Training | 🔜 Next | Supervised + Autoencoder, MLflow tracking |
| 3 — Serving | 🔜 | FastAPI inference + model promotion |
| 4 — Dashboard | 🔜 | Streamlit real-time monitoring |
