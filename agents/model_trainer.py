"""
agents.model_trainer — Parallel Model Training Sub-Agents node.

Spawns one sub-agent per model using asyncio.gather() for true parallel execution.
Each sub-agent trains its model, computes metrics, and writes its result to
state["model_results"][model_name].

Models trained (based on task type from Orchestrator):
  Classification: LightGBM, XGBoost, RandomForest, MLP, LogisticRegression
  Regression:     LightGBM, XGBoost, RandomForest, MLP, LinearRegression

All training is done locally (not in E2B) — the sandbox is only for generated code.
Model objects are stored in state for the Evaluator Agent to access.
"""

from __future__ import annotations

import asyncio
import logging
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
)

from agents.state import AgentState, ModelResult
from core.mlops.tracking import log_training_run, log_comparison_table
from core.config import settings

logger = logging.getLogger(__name__)

# ── Model factories ────────────────────────────────────────────────────────────


def _make_model(model_name: str, task_type: str, is_imbalanced: bool) -> Any:
    """Return a freshly instantiated model for the given name."""
    scale_pos = 10 if is_imbalanced and task_type == "classification" else 1

    if model_name == "lightgbm":
        import lightgbm as lgb

        if task_type == "classification":
            return lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=42,
                verbose=-1,
            )
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        if task_type == "classification":
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        kwargs = dict(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        if task_type == "classification":
            kwargs["class_weight"] = "balanced" if is_imbalanced else None
            return RandomForestClassifier(**kwargs)
        return RandomForestRegressor(**kwargs)

    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        kwargs = dict(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )
        if task_type == "classification":
            return MLPClassifier(**kwargs)
        return MLPRegressor(**kwargs)

    if model_name in ("logistic_regression", "linear_regression"):
        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced" if is_imbalanced else None,
                random_state=42,
            )
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)

    raise ValueError(f"Unknown model: {model_name}")


def _compute_metrics(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, task_type: str
) -> dict[str, float]:
    """Compute evaluation metrics appropriate for the task type."""
    metrics = {}

    if task_type == "classification":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        metrics["f1"] = round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
        )
        metrics["precision"] = round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
        )
        metrics["recall"] = round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
        )
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics["auc"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
            else:
                metrics["auc"] = round(
                    roc_auc_score(y_test, y_proba, multi_class="ovr"), 4
                )
        except Exception:
            metrics["auc"] = 0.0
    else:
        y_pred = model.predict(X_test)
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        metrics["r2"] = round(r2_score(y_test, y_pred), 4)
        try:
            metrics["mape"] = round(mean_absolute_percentage_error(y_test, y_pred), 4)
        except Exception:
            metrics["mape"] = 0.0

    return metrics


async def _train_single_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    is_imbalanced: bool,
    provider: str,
    api_key: str,
    model_llm_name: str,
    feature_names: list[str],
) -> tuple[str, ModelResult]:
    """Train a single model asynchronously. Returns (model_name, ModelResult)."""
    loop = asyncio.get_event_loop()

    def _train():
        start = time.time()
        tracemalloc.start()

        model = _make_model(model_name, task_type, is_imbalanced)
        model.fit(X_train, y_train)

        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = _compute_metrics(model, X_test, y_test, task_type)

        # Feature importance
        fi = {}
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(feature_names, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            fi = dict(zip(feature_names, coef.tolist()))

        # Log to MLflow
        try:
            run_id = log_training_run(
                model=model,
                model_name=model_name,
                params={"task_type": task_type, "is_imbalanced": is_imbalanced},
                metrics=metrics,
                tags={"agent": "model_trainer", "provider": provider},
                register=False,
            )
        except Exception:
            run_id = ""

        return ModelResult(
            model_name=model_name,
            model_object=model,
            params={"task_type": task_type},
            metrics=metrics,
            feature_importance=fi,
            train_time_seconds=round(elapsed, 2),
            memory_mb=round(peak_mem / 1024 / 1024, 2),
            mlflow_run_id=run_id,
            error=None,
        )

    try:
        result = await loop.run_in_executor(None, _train)
        return model_name, result
    except Exception as exc:
        logger.error("Training %s failed: %s", model_name, exc, exc_info=True)
        return model_name, ModelResult(
            model_name=model_name,
            model_object=None,
            params={},
            metrics={},
            feature_importance={},
            train_time_seconds=0.0,
            memory_mb=0.0,
            mlflow_run_id="",
            error=str(exc),
        )


def run_model_trainer(state: AgentState) -> dict:
    """
    LangGraph node: train all models in parallel with asyncio.gather().

    Reads:  state["feature_result"], state["dataset_profile"]
    Writes: state["model_results"], state["current_step"], state["logs"]
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))

    try:
        plan = state.get("_orchestrator_plan") or {}
        profile = state.get("dataset_profile") or {}
        feature_result = state.get("feature_result") or {}

        task_type = profile.get("task_type", "classification")
        target_col = profile.get("target_column", "")
        is_imbalanced = profile.get("is_imbalanced", False)
        suggested_models = plan.get(
            "suggested_models", ["lightgbm", "xgboost", "random_forest"]
        )

        logs.append(
            f"[{timestamp}] MODEL TRAINER — Training {len(suggested_models)} models in parallel: {suggested_models}"
        )

        # Load the processed CSV
        csv_path = feature_result.get("transformed_csv_path") or state.get(
            "csv_path", ""
        )
        df = pd.read_csv(csv_path)

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
            )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode target if classification
        le = None
        if task_type == "classification" and y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Drop any remaining non-numeric columns (shouldn't happen after FE, but safety net)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = X.drop(columns=non_numeric)
            logger.warning(
                "Dropped non-numeric columns before training: %s", non_numeric
            )

        feature_names = X.columns.tolist()
        X_arr = X.values
        y_arr = np.array(y)

        # Train/test split
        if task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42
            )

        logs.append(
            f"[{timestamp}] MODEL TRAINER — Train: {len(X_train)}, Test: {len(X_test)} rows"
        )

        # Run all models in parallel
        async def _train_all():
            tasks = [
                _train_single_model(
                    model_name=m,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    task_type=task_type,
                    is_imbalanced=is_imbalanced,
                    provider=state["provider"],
                    api_key=state["api_key"],
                    model_llm_name=state["model_name"],
                    feature_names=feature_names,
                )
                for m in suggested_models
            ]
            return await asyncio.gather(*tasks)

        results_list = asyncio.run(_train_all())
        model_results = {name: result for name, result in results_list}

        # Log results
        for name, result in model_results.items():
            if result.get("error"):
                logs.append(
                    f"[{timestamp}] MODEL TRAINER — {name.upper()}: FAILED — {result['error']}"
                )
            else:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in result["metrics"].items()
                )
                logs.append(
                    f"[{timestamp}] MODEL TRAINER — {name.upper()}: {metrics_str} ({result['train_time_seconds']}s)"
                )

        # Log comparison to MLflow
        try:
            comparison_rows = [
                {
                    "model": name,
                    **result["metrics"],
                    "train_time_s": result["train_time_seconds"],
                }
                for name, result in model_results.items()
                if not result.get("error")
            ]
            log_comparison_table(comparison_rows)
        except Exception:
            pass

        logs.append(f"[{timestamp}] MODEL TRAINER — Done.")

        return {
            "model_results": model_results,
            "current_step": "evaluator",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Model Trainer failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] MODEL TRAINER — ERROR: {exc}")
        return {
            "error": f"Model Trainer failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
