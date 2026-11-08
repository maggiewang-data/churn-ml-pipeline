"""Experiment tracking with MLflow.

Every training run records what went in (data fingerprint, model choice,
business assumptions) and what came out (metrics, threshold, the model
itself). That answers the questions people ask months after deployment:
"which model is live, what data was it trained on, and why did we pick it?"

View runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db     ->  http://127.0.0.1:5000
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import ROOT

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_LOCATION = (ROOT / "mlruns").as_uri()
EXPERIMENT = "churn-model"


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data, so a run can be tied to the exact data it saw."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(row_hashes).hexdigest()[:12]


def log_training_run(metrics: dict, data_hash: str, model_path: Path,
                     extra_files: list[Path]) -> str | None:
    """Log one training run. Returns the MLflow run id, or None if MLflow is not installed."""
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed; skipping experiment tracking.")
        return None

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{metrics['model']}-{metrics['trained_at']}") as run:
        mlflow.log_params({
            "model": metrics["model"],
            "threshold": metrics["threshold"],
            "n_train": metrics["n_train"],
            "data_hash": data_hash,
            **{f"assume_{k}": v for k, v in metrics["business_assumptions"].items()},
        })
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics["test"].items()})
        for name, res in metrics["cv_comparison"].items():
            mlflow.log_metric(f"cv_pr_auc_{name}", res["pr_auc_mean"])
        mlflow.log_artifact(str(model_path))
        for f in extra_files:
            if f.exists():
                mlflow.log_artifact(str(f))
        return run.info.run_id
