"""
core.mlops.tracking — MLflow experiment tracking helpers.

Replaces the copy-pasted `mlflow.start_run() / log_params() / log_metrics() /
register_model()` boilerplate that appears in 7+ repos across this portfolio.

Usage:
    from core.mlops.tracking import log_training_run, promote_best_model

    run_id = log_training_run(
        model=lgbm_model,
        model_name="lightgbm",
        params={"n_estimators": 200, "learning_rate": 0.05},
        metrics={"auc": 0.94, "f1": 0.87},
        artifact_paths={"shap_plot": "outputs/shap.png"},
        tags={"task_type": "classification", "dataset": "credit_fraud"},
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from core.config import settings

logger = logging.getLogger(__name__)

# ── MLflow initialisation ──────────────────────────────────────────────────────


def _init_mlflow() -> None:
    """Set tracking URI and experiment name from settings. Idempotent."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


# ── Core logging helper ────────────────────────────────────────────────────────


def log_training_run(
    model: Any,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    register: bool = True,
) -> str:
    """
    Log a single model training run to MLflow and optionally register it.

    Args:
        model:          Trained model object (sklearn, LightGBM, XGBoost, or PyTorch).
        model_name:     Display name for the model (e.g. "lightgbm", "xgboost").
        params:         Hyperparameters to log.
        metrics:        Evaluation metrics to log (floats only).
        artifact_paths: Dict of {label: local_file_path} to upload as artifacts.
        tags:           Extra run tags (e.g. task_type, dataset_name).
        register:       If True, register the model in the MLflow Model Registry.

    Returns:
        The MLflow run_id string.
    """
    _init_mlflow()

    with mlflow.start_run(run_name=model_name) as run:
        # Tags
        mlflow.set_tag("model_name", model_name)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))

        # Params and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model — detect framework automatically
        _log_model_by_type(model, model_name)

        # Artifacts
        if artifact_paths:
            for label, path in artifact_paths.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=label)
                else:
                    logger.warning("Artifact not found, skipping: %s", path)

        run_id = run.info.run_id

    # Register in Model Registry
    if register:
        try:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{model_name}",
                name=model_name,
            )
            logger.info("Registered model '%s' from run %s", model_name, run_id)
        except Exception as exc:
            logger.warning("Model registration failed (non-fatal): %s", exc)

    return run_id


def _log_model_by_type(model: Any, artifact_name: str) -> None:
    """Detect model type and call the appropriate mlflow.*.log_model()."""
    try:
        import lightgbm as lgb  # type: ignore

        if isinstance(model, lgb.Booster) or hasattr(model, "booster_"):
            mlflow.lightgbm.log_model(model, artifact_name)
            return
    except ImportError:
        pass

    try:
        import xgboost as xgb  # type: ignore

        if isinstance(model, xgb.XGBModel):
            mlflow.xgboost.log_model(model, artifact_name)
            return
    except ImportError:
        pass

    # Fallback: sklearn-compatible (RandomForest, LogisticRegression, etc.)
    try:
        mlflow.sklearn.log_model(model, artifact_name)
    except Exception:
        # Last resort: pickle
        import pickle
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            mlflow.log_artifact(f.name, artifact_path=artifact_name)


# ── Model Registry helpers ─────────────────────────────────────────────────────


def get_best_run(
    metric: str = "auc",
    ascending: bool = False,
) -> dict[str, Any] | None:
    """
    Return the run with the best value for `metric` in the current experiment.

    Args:
        metric:    MLflow metric key to sort by.
        ascending: If True, lower is better (e.g. for RMSE).

    Returns:
        Dict with keys: run_id, params, metrics, tags. None if no runs found.
    """
    _init_mlflow()
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if not runs:
        return None

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "params": best.data.params,
        "metrics": best.data.metrics,
        "tags": best.data.tags,
    }


def promote_to_production(model_name: str, version: str) -> None:
    """
    Transition a registered model version to the Production stage in MLflow.

    Also archives all other Production versions of that model.
    """
    _init_mlflow()
    client = MlflowClient()

    # Archive existing Production versions
    for mv in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived",
        )
        logger.info("Archived model '%s' v%s", model_name, mv.version)

    # Promote the new version
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
    )
    logger.info("Promoted model '%s' v%s to Production", model_name, version)


def log_comparison_table(results: list[dict[str, Any]]) -> None:
    """
    Log a multi-model comparison as a single MLflow run with a metrics table.

    `results` is a list of dicts like:
        [{"model": "lightgbm", "auc": 0.94, "f1": 0.87, "train_time_s": 12.3}, ...]

    Each model gets its own metric logged with a model-namespaced key, e.g.
        lightgbm_auc, xgboost_auc, ...
    """
    _init_mlflow()
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.set_tag("run_type", "comparison")
        for result in results:
            model_label = result.get("model", "unknown")
            for key, val in result.items():
                if key == "model":
                    continue
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"{model_label}_{key}", val)
