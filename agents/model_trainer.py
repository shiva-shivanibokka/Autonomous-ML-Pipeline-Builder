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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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

# Above this many training rows we skip cross-validation to keep runs snappy.
# ponytail: fixed threshold; make it configurable if large datasets become common.
_CV_ROW_CAP = 50_000


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a leakage-safe preprocessing transformer.

    Numeric   → median impute + standardize.
    Categorical → most-frequent impute + one-hot (unknown categories ignored).

    Fit happens inside each model's Pipeline on the training fold ONLY, so the
    test set never influences imputation statistics, scaling, or encoding.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# ── Model factories ────────────────────────────────────────────────────────────


def _make_model(model_name: str, task_type: str, scale_pos: float) -> Any:
    """Return a freshly instantiated estimator. `scale_pos` weights the positive class."""
    is_imbalanced = scale_pos > 1

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
                # use_label_encoder was removed in xgboost 2.x — passing it now errors.
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


def _cross_validate(pipe: Pipeline, X_train, y_train, task_type: str) -> tuple[float, float, str]:
    """Leakage-free CV score (prep is re-fit inside each fold). Skips huge datasets."""
    if len(X_train) > _CV_ROW_CAP:
        return 0.0, 0.0, "skipped(n>cap)"
    if task_type == "classification":
        scorer = "f1_weighted"
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        scorer = "r2"
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipe, X_train, y_train, cv=splitter, scoring=scorer, n_jobs=1)
        return round(float(scores.mean()), 4), round(float(scores.std()), 4), scorer
    except Exception as exc:
        logger.warning("CV failed for a model (non-fatal): %s", exc)
        return 0.0, 0.0, f"failed:{scorer}"


async def _train_single_model(
    model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    scale_pos: float,
    provider: str,
) -> tuple[str, ModelResult]:
    """Train one model (prep+estimator Pipeline) asynchronously. Returns (name, ModelResult)."""
    loop = asyncio.get_event_loop()

    def _train():
        start = time.time()
        tracemalloc.start()

        # Each model gets its OWN cloned preprocessor so the fit is per-pipeline.
        from sklearn.base import clone

        estimator = _make_model(model_name, task_type, scale_pos)
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", estimator)])

        # Cross-validate BEFORE the final fit (prep re-fit inside every fold → no leakage).
        cv_mean, cv_std, cv_metric = _cross_validate(pipe, X_train, y_train, task_type)

        pipe.fit(X_train, y_train)

        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = _compute_metrics(pipe, X_test, y_test, task_type)

        # Feature importance mapped onto the transformed feature names.
        fi = {}
        try:
            feat_names = pipe.named_steps["prep"].get_feature_names_out().tolist()
            model = pipe.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                fi = dict(zip(feat_names, model.feature_importances_.tolist()))
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                fi = dict(zip(feat_names, np.ravel(coef).tolist()))
        except Exception:
            fi = {}

        # Log to MLflow
        try:
            run_id = log_training_run(
                model=pipe,
                model_name=model_name,
                params={"task_type": task_type, "scale_pos_weight": scale_pos},
                metrics={**metrics, "cv_mean": cv_mean, "cv_std": cv_std},
                tags={"agent": "model_trainer", "provider": provider},
                register=False,
            )
        except Exception:
            run_id = ""

        return ModelResult(
            model_name=model_name,
            model_object=pipe,
            params={"task_type": task_type},
            metrics=metrics,
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_metric=cv_metric,
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
            cv_mean=0.0,
            cv_std=0.0,
            cv_metric="",
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

        # Capture the RAW input schema (before any preprocessing) so the served
        # model knows exactly what columns/types to expect at inference time.
        feature_schema = [
            {"name": str(c), "dtype": str(X[c].dtype)} for c in X.columns
        ]

        # Encode target if classification
        if task_type == "classification" and y.dtype == object:
            y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

        y_arr = np.asarray(y)

        # Compute a real positive-class weight for imbalanced binary problems
        # (replaces the old hardcoded 10). 1.0 = no reweighting.
        scale_pos = 1.0
        if task_type == "classification":
            classes, counts = np.unique(y_arr, return_counts=True)
            if len(classes) == 2:
                neg, pos = counts.max(), counts.min()
                scale_pos = round(float(neg) / float(pos), 3) if pos else 1.0

        # Preprocessing (imputation/scaling/encoding) is fit INSIDE each model's
        # Pipeline on the training fold only — never on the test set. No leakage.
        preprocessor = _build_preprocessor(X)

        # Train/test holdout split — DataFrames preserved so the ColumnTransformer
        # can address columns by name.
        stratify = y_arr if (task_type == "classification" and len(np.unique(y_arr)) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=42, stratify=stratify
        )

        logs.append(
            f"[{timestamp}] MODEL TRAINER — Train: {len(X_train)}, Test: {len(X_test)} rows"
            + (f" | scale_pos_weight={scale_pos}" if scale_pos != 1.0 else "")
        )

        # Run all models in parallel
        async def _train_all():
            tasks = [
                _train_single_model(
                    model_name=m,
                    preprocessor=preprocessor,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    task_type=task_type,
                    scale_pos=scale_pos,
                    provider=state["provider"],
                )
                for m in suggested_models
            ]
            return await asyncio.gather(*tasks)

        results_list = asyncio.run(_train_all())
        model_results = {name: result for name, result in results_list}

        # Bounded sample of the HELD-OUT test set for SHAP (raw features → the
        # winning Pipeline transforms it downstream). Explains on unseen data.
        shap_sample = X_test.head(200).to_dict("records")

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
                cv_str = (
                    f" | CV {result['cv_metric']}={result['cv_mean']}±{result['cv_std']}"
                    if result.get("cv_metric") and not result["cv_metric"].startswith("skipped")
                    else ""
                )
                logs.append(
                    f"[{timestamp}] MODEL TRAINER — {name.upper()}: {metrics_str}{cv_str} ({result['train_time_seconds']}s)"
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
            "feature_schema": feature_schema,
            "shap_sample": shap_sample,
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
