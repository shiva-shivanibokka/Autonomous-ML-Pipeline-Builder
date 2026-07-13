"""
agents.feature_engineer — Feature Engineer Agent node.

Generates a complete preprocessing + feature engineering Python script
based on the DatasetProfile from the Data Analyst Agent.

The generated code is executed in an E2B sandbox (or subprocess fallback).
If execution fails, the self-correction loop in sandbox/executor.py retries.

The output is a cleaned, transformed CSV written to a known path in the sandbox.
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, FeatureEngineeringResult
from core.llm_utils import build_system_prompt, extract_content, strip_fences
from core.providers import get_codegen_llm
from core.rag import retrieve_context

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = build_system_prompt(
    role="an expert ML engineer who writes production-quality feature engineering code",
    context=(
        "You will write a complete Python preprocessing script that applies only "
        "LEAKAGE-SAFE, structural transforms. Imputation, scaling, and encoding are "
        "handled downstream inside a scikit-learn Pipeline that is fit on the training "
        "split only — so you must NOT do them here (doing them on the full dataset "
        "would leak test information).\n\n"
        "The script must:\n"
        "1. Load the CSV from the path variable `INPUT_CSV_PATH`\n"
        "2. Apply the leakage-safe transforms below\n"
        "3. Save the transformed DataFrame to `OUTPUT_CSV_PATH`\n"
        "4. Print a summary: number of rows, columns, and any new features created\n\n"
        "DO (safe — computed per-row, independent of other rows):\n"
        "- Drop constant columns, exact-duplicate columns, and obvious ID columns\n"
        "- Decompose datetime columns into year, month, day, dayofweek, is_weekend\n"
        "- Create per-row derived features (ratios, differences between two columns)\n"
        "- Keep categorical columns AS-IS (strings) — do not encode them\n"
        "- Keep missing values AS-IS — do not impute them\n"
        "- Keep the target column unchanged\n\n"
        "DO NOT (these leak or are handled downstream):\n"
        "- Do NOT impute missing values (SimpleImputer runs downstream)\n"
        "- Do NOT scale/standardize numeric features (StandardScaler runs downstream)\n"
        "- Do NOT one-hot / label / target encode (OneHotEncoder runs downstream)\n"
        "- Do NOT apply SMOTE or resampling\n\n"
        "- Use only: pandas, numpy (already installed)\n"
        "- Print CREATED_FEATURES: [list] and DROPPED_FEATURES: [list] to stdout\n"
        "- Output ONLY the Python code, no explanation.\n"
    ),
)


def _build_feature_engineering_prompt(state: AgentState) -> str:
    profile = state.get("dataset_profile") or {}

    # Retrieve grounding from the ML-knowledge base (RAG).
    query = (
        f"feature engineering preprocessing for {profile.get('task_type', 'classification')} "
        f"with categorical columns, missing values, datetime features; avoid data leakage; "
        f"class imbalance {profile.get('is_imbalanced', False)}"
    )
    grounding = retrieve_context(query, k=3)
    grounding_block = (
        f"Relevant best-practices from the knowledge base — follow these:\n\n{grounding}\n\n"
        if grounding
        else ""
    )

    return (
        grounding_block
        + f"Dataset profile:\n"
        f"  - Target column: '{profile.get('target_column', 'unknown')}'\n"
        f"  - Task type: {profile.get('task_type', 'classification')}\n"
        f"  - Numeric columns: {profile.get('numeric_cols', [])}\n"
        f"  - Categorical columns: {profile.get('categorical_cols', [])}\n"
        f"  - Datetime columns: {profile.get('datetime_cols', [])}\n"
        f"  - High-cardinality categoricals: {profile.get('high_cardinality_cols', [])}\n"
        f"  - Missing value columns: {list(profile.get('missing_pct', {}).keys())}\n"
        f"  - Outlier columns: {profile.get('outlier_cols', [])}\n"
        f"  - Class imbalanced: {profile.get('is_imbalanced', False)}\n\n"
        f"Preprocessing recommendations:\n"
        + "\n".join(
            f"  {i + 1}. {rec}"
            for i, rec in enumerate(profile.get("recommended_preprocessing", []))
        )
        + "\n\n"
        "Write a complete Python script that:\n"
        "1. Defines INPUT_CSV_PATH = '/data/input.csv'\n"
        "2. Defines OUTPUT_CSV_PATH = '/data/processed.csv'\n"
        "3. Loads, preprocesses, and saves the data\n"
        "4. Prints CREATED_FEATURES: [...] and DROPPED_FEATURES: [...]\n\n"
        "Return ONLY the Python code."
    )


def run_feature_engineer(state: AgentState) -> dict:
    """
    LangGraph node: generate and execute preprocessing code.

    Reads:  state["dataset_profile"], state["csv_path"]
    Writes: state["feature_result"], state["current_step"], state["logs"]
    """
    from sandbox.executor import execute_with_retry

    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(f"[{timestamp}] FEATURE ENGINEER — Retrieving ML best-practices (RAG)...")
    logs.append(f"[{timestamp}] FEATURE ENGINEER — Generating preprocessing code...")

    try:
        llm = get_codegen_llm(
            provider=state["provider"],
            api_key=state["api_key"],
        )

        user_prompt = _build_feature_engineering_prompt(state)

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_code = strip_fences(extract_content(response))

        logs.append(
            f"[{timestamp}] FEATURE ENGINEER — Code generated ({len(raw_code)} chars). Executing in sandbox..."
        )

        # Execute in E2B sandbox with self-correction
        result = execute_with_retry(
            code=raw_code,
            csv_path=state["csv_path"],
            llm=llm,
            max_retries=3,
        )

        if not result["success"]:
            raise RuntimeError(
                f"Feature engineering failed after {result['attempts']} attempts. "
                f"Last error: {result['last_error']}"
            )

        logs.append(
            f"[{timestamp}] FEATURE ENGINEER — Execution succeeded in {result['attempts']} attempt(s)."
        )

        # Parse created/dropped features from stdout
        stdout = result.get("stdout", "")
        created = _parse_list_from_stdout(stdout, "CREATED_FEATURES")
        dropped = _parse_list_from_stdout(stdout, "DROPPED_FEATURES")

        if created:
            logs.append(f"[{timestamp}] FEATURE ENGINEER — Created features: {created}")
        if dropped:
            logs.append(f"[{timestamp}] FEATURE ENGINEER — Dropped: {dropped}")

        logs.append(f"[{timestamp}] FEATURE ENGINEER — Done.")

        feature_result: FeatureEngineeringResult = {
            "preprocessing_code": result["final_code"],
            "new_features_created": created,
            "dropped_columns": dropped,
            "leakage_warnings": [],
            "transformed_csv_path": result.get(
                "output_csv_path", "/data/processed.csv"
            ),
        }

        return {
            "feature_result": feature_result,
            "current_step": "model_trainer",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Feature Engineer failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] FEATURE ENGINEER — ERROR: {exc}")
        return {
            "error": f"Feature Engineer failed: {exc}",
            "status": "failed",
            "logs": logs,
        }


def _parse_list_from_stdout(stdout: str, key: str) -> list[str]:
    """Parse a list printed to stdout like: CREATED_FEATURES: ['a', 'b', 'c']"""
    import ast

    for line in stdout.splitlines():
        if line.startswith(f"{key}:"):
            try:
                raw = line.split(":", 1)[1].strip()
                return ast.literal_eval(raw)
            except Exception:
                pass
    return []
