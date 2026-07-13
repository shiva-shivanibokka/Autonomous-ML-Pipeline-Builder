"""
agents.data_analyst — Data Analyst Agent node.

Profiles the uploaded CSV dataset and produces a structured DatasetProfile.
This is pure Python — no LLM needed for EDA. The agent computes statistics
directly with pandas, then asks the LLM to recommend preprocessing steps
in natural language.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.state import AgentState, DatasetProfile
from core.llm_utils import build_system_prompt, extract_content, safe_parse
from core.providers import get_llm

logger = logging.getLogger(__name__)


class PreprocessingRecommendations(BaseModel):
    recommendations: list[str] = Field(
        description="List of specific preprocessing steps to apply, in order"
    )
    notes: str = Field(description="Any important observations about the dataset")


SYSTEM_PROMPT = build_system_prompt(
    role="a senior data scientist specialising in data quality and ML preprocessing",
    context=(
        "Given a dataset profile, recommend specific, actionable preprocessing steps. "
        "Consider: missing value imputation strategy, encoding for categorical features, "
        "scaling for numeric features, class imbalance handling (SMOTE, class_weight), "
        "outlier treatment, date feature decomposition. "
        "Be specific about WHICH columns need WHICH treatment. "
        "Order recommendations from most to least critical."
    ),
)


def _profile_dataframe(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics on a DataFrame without any LLM."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Detect datetime columns stored as strings
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except Exception:
            pass

    missing_pct = {
        col: round(df[col].isna().mean() * 100, 2)
        for col in df.columns
        if df[col].isna().any()
    }

    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]

    # Simple outlier detection: IQR method
    outlier_cols = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
        if n_outliers > len(df) * 0.01:  # >1% outliers
            outlier_cols.append(col)

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_pct": missing_pct,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "outlier_cols": outlier_cols,
    }


def _detect_imbalance(df: pd.DataFrame, target_col: str) -> tuple[bool, float | None]:
    """Check if a classification target is imbalanced (minority/majority < 0.2)."""
    if target_col not in df.columns:
        return False, None
    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return False, None
    ratio = counts.min() / counts.max()
    return ratio < 0.2, round(ratio, 4)


def run_data_analyst(state: AgentState) -> dict:
    """
    LangGraph node: profile the dataset and recommend preprocessing.

    Reads:  state["csv_path"], state["dataset_profile"]["target_column"]
    Writes: state["dataset_profile"] (enriched), state["current_step"], state["logs"]
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(f"[{timestamp}] DATA ANALYST — Profiling dataset...")

    try:
        csv_path = state["csv_path"]
        df = pd.read_csv(csv_path)

        profile_data = _profile_dataframe(df)
        target_col = (state.get("dataset_profile") or {}).get("target_column", "")

        # Class imbalance check
        task_type = (state.get("dataset_profile") or {}).get(
            "task_type", "classification"
        )
        is_imbalanced, class_ratio = False, None
        if task_type == "classification" and target_col:
            is_imbalanced, class_ratio = _detect_imbalance(df, target_col)

        logs.append(
            f"[{timestamp}] DATA ANALYST — Shape: {profile_data['n_rows']} rows × {profile_data['n_cols']} cols"
        )
        logs.append(
            f"[{timestamp}] DATA ANALYST — Missing values in {len(profile_data['missing_pct'])} columns"
        )
        if is_imbalanced:
            logs.append(
                f"[{timestamp}] DATA ANALYST — Class imbalance detected (ratio={class_ratio})"
            )

        # Ask LLM for preprocessing recommendations
        llm = get_llm(
            provider=state["provider"],
            api_key=state["api_key"],
            model=state["model_name"],
        )

        user_prompt = (
            f"Dataset profile:\n"
            f"  - Shape: {profile_data['n_rows']} rows × {profile_data['n_cols']} cols\n"
            f"  - Target column: '{target_col}'\n"
            f"  - Task type: {task_type}\n"
            f"  - Numeric columns ({len(profile_data['numeric_cols'])}): {profile_data['numeric_cols'][:10]}\n"
            f"  - Categorical columns ({len(profile_data['categorical_cols'])}): {profile_data['categorical_cols'][:10]}\n"
            f"  - Datetime columns: {profile_data['datetime_cols']}\n"
            f"  - Missing values: {profile_data['missing_pct']}\n"
            f"  - High-cardinality categoricals: {profile_data['high_cardinality_cols']}\n"
            f"  - Outlier columns: {profile_data['outlier_cols']}\n"
            f"  - Class imbalanced: {is_imbalanced} (minority/majority ratio: {class_ratio})\n\n"
            "Return JSON with keys: recommendations (list[str]), notes (str)."
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = extract_content(response)
        recs = safe_parse(raw, PreprocessingRecommendations)

        recommendations = (
            recs.recommendations
            if recs
            else [
                "Impute missing numeric values with median",
                "Impute missing categorical values with mode",
                "One-hot encode low-cardinality categoricals",
                "Target-encode high-cardinality categoricals",
                "Scale numeric features with StandardScaler",
            ]
        )
        notes = recs.notes if recs else ""

        for rec in recommendations[:5]:  # Log first 5
            logs.append(f"[{timestamp}] DATA ANALYST — Rec: {rec}")
        logs.append(f"[{timestamp}] DATA ANALYST — Done.")

        dataset_profile: DatasetProfile = {
            **profile_data,
            "target_column": target_col,
            "task_type": task_type,
            "is_imbalanced": is_imbalanced,
            "class_ratio": class_ratio,
            "recommended_preprocessing": recommendations,
            "notes": notes,
        }

        return {
            "dataset_profile": dataset_profile,
            "current_step": "feature_engineer",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Data Analyst failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] DATA ANALYST — ERROR: {exc}")
        return {
            "error": f"Data Analyst failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
