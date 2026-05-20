"""
Phase 3 — promotes the best supervised binary classifier to MLflow 'Production'.
Ranks all finished runs in netguardai_binary by f1_weighted, registers the winner,
and archives any previously promoted version.
"""

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI      = str(ROOT / "mlruns")
EXPERIMENT_NAME = "netguardai_binary"
RANK_METRIC     = "f1_weighted"
REGISTERED_NAME = "netguardai_best_binary"


def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run train_supervised.py first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{RANK_METRIC} DESC"],
        max_results=20,
    )
    if not runs:
        raise RuntimeError("No finished runs found in experiment.")

    best   = runs[0]
    score  = best.data.metrics.get(RANK_METRIC, 0.0)
    name   = best.data.tags.get("model", best.info.run_id)
    model_uri = f"runs:/{best.info.run_id}/model"

    log.info("Best run: %-25s  %s=%.4f", name, RANK_METRIC, score)
    log.info("Model URI: %s", model_uri)

    # Register
    mv = mlflow.register_model(model_uri, REGISTERED_NAME)
    log.info("Registered '%s' version %s", REGISTERED_NAME, mv.version)

    # Promote to Production (archives existing Production versions)
    client.transition_model_version_stage(
        name=REGISTERED_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info("'%s' v%s → Production", REGISTERED_NAME, mv.version)

    # Summary table
    print(f"\nRanking — {EXPERIMENT_NAME} by {RANK_METRIC}:")
    print(f"  {'#':<4} {'Model':<25} {RANK_METRIC}")
    print(f"  {'-'*4} {'-'*25} {'-'*10}")
    for i, r in enumerate(runs, 1):
        n  = r.data.tags.get("model", r.info.run_id)
        s  = r.data.metrics.get(RANK_METRIC, 0.0)
        tag = "  ← Production" if i == 1 else ""
        print(f"  {i:<4} {n:<25} {s:.4f}{tag}")


if __name__ == "__main__":
    run()
