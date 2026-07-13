"""
agents.evaluator — Evaluator Agent node.

Compares all model training results, selects the best model with written
justification, runs SHAP explainability on the winner, and checks for
demographic bias on any sensitive features present.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.state import AgentState, EvaluationResult
from core.llm_utils import build_system_prompt, extract_content, safe_parse
from core.providers import get_llm

logger = logging.getLogger(__name__)


SENSITIVE_FEATURES = {
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age",
    "nationality",
    "religion",
    "disability",
    "marital_status",
}


# Metrics where a LOWER value is better (everything else: higher is better).
_LOWER_IS_BETTER = {"rmse", "mape"}


def _select_winner(
    model_results: dict[str, Any], primary_metric: str
) -> tuple[str | None, list[str]]:
    """
    Deterministically rank models by the primary metric (no LLM in the loop).

    Selection is pure argmax/argmin on a number — the LLM only narrates the
    decision afterwards. Failed models are excluded. Ties fall back to the
    leakage-free cross-validation mean.
    """
    valid = {
        n: r
        for n, r in model_results.items()
        if not r.get("error") and r.get("model_object") is not None
    }
    if not valid:
        return None, []

    lower_better = primary_metric in _LOWER_IS_BETTER

    def sort_key(name: str) -> tuple[float, float]:
        r = valid[name]
        metrics = r.get("metrics", {})
        primary = metrics.get(primary_metric)
        # Orient so that "bigger key = better" regardless of metric direction.
        primary_key = (
            (-float(primary) if lower_better else float(primary))
            if primary is not None
            else -1e18
        )
        return (primary_key, float(r.get("cv_mean", 0.0)))

    ranking = sorted(valid, key=sort_key, reverse=True)
    return ranking[0], ranking


# The LLM writes a human-readable justification for the ALREADY-decided winner.
JUSTIFICATION_PROMPT = build_system_prompt(
    role="a senior ML engineer explaining a model selection decision",
    context=(
        "The winning model has ALREADY been chosen deterministically by the primary "
        "metric. Your job is only to write a clear 2-3 sentence justification for a "
        "stakeholder: reference the metric values and, where relevant, the trade-off "
        "against training time or interpretability. Return JSON with a single key "
        "'justification' (string)."
    ),
)


class _Justification(BaseModel):
    justification: str = Field(description="2-3 sentence explanation")


def _run_shap(
    pipeline: Any,
    X_test: pd.DataFrame,
    output_dir: str,
) -> str | None:
    """
    Run SHAP on the winning Pipeline over HELD-OUT test rows.

    The pipeline's preprocessing step transforms the raw test features first, then
    SHAP explains the model step on those transformed features — so the plot
    reflects generalisation behaviour, not memorised training data.
    """
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        Xt = prep.transform(X_test)
        if hasattr(Xt, "toarray"):  # sparse from OneHotEncoder
            Xt = Xt.toarray()
        feature_names = list(prep.get_feature_names_out())

        if hasattr(model, "feature_importances_"):
            # Tree-based models — exact, fast.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt[:200])
        else:
            # Linear/neural — KernelExplainer with a small background sample.
            background = shap.sample(Xt, min(50, len(Xt)))
            predict_fn = (
                model.predict_proba if hasattr(model, "predict_proba") else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(Xt[:50])

        # Multi-class → take the positive/second class slice.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            Xt[: len(shap_values)],
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return plot_path

    except Exception as exc:
        logger.warning("SHAP explainability failed (non-fatal): %s", exc)
        return None


def _check_bias(
    df: pd.DataFrame,
    target_col: str,
    winner_name: str,
    model: Any,
    feature_names: list[str],
) -> list[str]:
    """Flag potential bias if sensitive demographic features are in the dataset."""
    warnings = []
    present_sensitive = [col for col in df.columns if col.lower() in SENSITIVE_FEATURES]
    if present_sensitive:
        warnings.append(
            f"Sensitive features detected: {present_sensitive}. "
            f"Evaluate {winner_name} for disparate impact before deployment. "
            f"Consider fairness metrics: demographic parity, equalized odds."
        )
    return warnings


def run_evaluator(state: AgentState) -> dict:
    """
    LangGraph node: pick the best model, run SHAP, flag bias.

    Reads:  state["model_results"], state["dataset_profile"], state["feature_result"]
    Writes: state["evaluation_result"], state["current_step"], state["logs"]
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(
        f"[{timestamp}] EVALUATOR — Comparing {len(state.get('model_results', {}))} models..."
    )

    try:
        model_results = state.get("model_results", {})
        profile = state.get("dataset_profile") or {}
        plan = state.get("_orchestrator_plan") or {}

        task_type = profile.get("task_type", "classification")
        target_col = profile.get("target_column", "")
        primary_metric = plan.get(
            "primary_metric", "auc" if task_type == "classification" else "rmse"
        )

        # Build comparison table (includes leakage-free CV columns)
        comparison_table = []
        for name, result in model_results.items():
            row = {
                "model": name,
                **result.get("metrics", {}),
                "cv_mean": result.get("cv_mean", 0.0),
                "cv_std": result.get("cv_std", 0.0),
                "train_time_s": result.get("train_time_seconds", 0),
                "memory_mb": result.get("memory_mb", 0),
                "failed": bool(result.get("error")),
            }
            comparison_table.append(row)

        # ── Deterministic winner selection (no LLM decides the number) ──────────
        winner_name, ranking = _select_winner(model_results, primary_metric)
        if winner_name is None:
            raise ValueError("No model trained successfully; cannot evaluate.")

        winner_result = model_results.get(winner_name, {})
        winner_metric_val = winner_result.get("metrics", {}).get(primary_metric)
        logs.append(
            f"[{timestamp}] EVALUATOR — Winner: {winner_name} "
            f"({primary_metric}={winner_metric_val})"
        )

        # ── LLM writes the justification narrative only (non-fatal) ─────────────
        justification = (
            f"{winner_name} was selected as it achieved the best {primary_metric} "
            f"({winner_metric_val}) among {len(ranking)} candidate models."
        )
        try:
            llm = get_llm(
                provider=state["provider"],
                api_key=state["api_key"],
                model=state["model_name"],
            )
            table_str = "\n".join(
                f"  {row['model']}: "
                + ", ".join(f"{k}={v}" for k, v in row.items() if k != "model")
                for row in comparison_table
            )
            resp = llm.invoke(
                [
                    SystemMessage(content=JUSTIFICATION_PROMPT),
                    HumanMessage(
                        content=(
                            f"Task: {task_type}. Primary metric: {primary_metric}.\n"
                            f"Chosen winner: {winner_name}.\n\n"
                            f"Model results:\n{table_str}\n\n"
                            "Return JSON: {\"justification\": \"...\"}"
                        )
                    ),
                ]
            )
            parsed = safe_parse(extract_content(resp), _Justification)
            if parsed and parsed.justification.strip():
                justification = parsed.justification.strip()
        except Exception as exc:
            logger.warning("Justification LLM call failed (non-fatal): %s", exc)

        logs.append(f"[{timestamp}] EVALUATOR — Justification: {justification}")

        # ── SHAP on held-out test rows + persist the winning pipeline ──────────
        shap_plot_path = ""
        bias_warnings: list[str] = []
        winner_pipe = winner_result.get("model_object")
        out_dir = state.get("output_dir", "outputs")

        if winner_pipe is not None:
            os.makedirs(out_dir, exist_ok=True)

            # Persist the full pipeline (prep + model) so the generated API is runnable.
            try:
                import json

                import joblib

                joblib.dump(winner_pipe, os.path.join(out_dir, "model.pkl"))
                with open(
                    os.path.join(out_dir, "feature_schema.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(state.get("feature_schema", []), f, indent=2)
                logs.append(
                    f"[{timestamp}] EVALUATOR — Saved model.pkl + feature_schema.json"
                )
            except Exception as exc:
                logger.warning("Model persistence failed (non-fatal): %s", exc)

            # SHAP over the held-out test sample.
            sample = state.get("shap_sample") or []
            if sample:
                logs.append(
                    f"[{timestamp}] EVALUATOR — Running SHAP on {winner_name} (test set)..."
                )
                try:
                    X_test = pd.DataFrame(sample)
                    cols = [f["name"] for f in state.get("feature_schema", [])]
                    keep = [c for c in cols if c in X_test.columns]
                    if keep:
                        X_test = X_test[keep]
                    shap_plot_path = _run_shap(winner_pipe, X_test, out_dir) or ""

                    bias_warnings = _check_bias(
                        X_test, target_col, winner_name, winner_pipe, keep
                    )
                    for w in bias_warnings:
                        logs.append(f"[{timestamp}] EVALUATOR — BIAS WARNING: {w}")
                except Exception as exc:
                    logger.warning("SHAP/bias step failed (non-fatal): %s", exc)

        logs.append(f"[{timestamp}] EVALUATOR — Done.")

        evaluation_result: EvaluationResult = {
            "winner_model": winner_name,
            "ranking": ranking,
            "justification": justification,
            "primary_metric": primary_metric,
            "shap_plot_path": shap_plot_path,
            "bias_warnings": bias_warnings,
            "comparison_table": comparison_table,
        }

        return {
            "evaluation_result": evaluation_result,
            "current_step": "code_generator",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Evaluator failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] EVALUATOR — ERROR: {exc}")
        return {
            "error": f"Evaluator failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
