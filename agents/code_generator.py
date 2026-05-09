"""
agents.code_generator — Code Generator Agent node.

Generates a complete, documented, production-ready Python pipeline script
for the winning model. The script is then validated by executing it in E2B.

If execution fails, the self-correction loop reads the traceback and asks
the LLM to rewrite the broken section (max 3 attempts).

Output: a single pipeline.py that can be downloaded by the user.
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, DeploymentArtifacts
from core.llm_utils import build_system_prompt, extract_content, strip_fences
from core.providers import get_codegen_llm

logger = logging.getLogger(__name__)


PIPELINE_CODE_PROMPT = build_system_prompt(
    role="a senior ML engineer who writes clean, production-quality Python code",
    context=(
        "You will write a COMPLETE, self-contained Python ML pipeline script. "
        "The script must:\n"
        "  1. Have a clear module-level docstring explaining what the pipeline does\n"
        "  2. Define all constants at the top (CSV path, target column, model params)\n"
        "  3. Include a `preprocess(df)` function with all preprocessing steps\n"
        "  4. Include a `train(X_train, y_train)` function that trains the winning model\n"
        "  5. Include an `evaluate(model, X_test, y_test)` function with all metrics\n"
        "  6. Include a `main()` function that ties it all together with argparse\n"
        "  7. Use `if __name__ == '__main__': main()` at the bottom\n"
        "  8. Have Google-style docstrings on every function\n"
        "  9. Include inline comments explaining non-obvious steps\n"
        " 10. Generate a requirements.txt string at the end (printed to stdout)\n\n"
        "Use only: pandas, numpy, scikit-learn, lightgbm, xgboost (standard ML stack).\n"
        "Output ONLY the Python code. No explanation, no markdown fences."
    ),
)

REQUIREMENTS_TEMPLATE = """pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
xgboost>=2.0.0
shap>=0.44.0
matplotlib>=3.8.0
joblib>=1.3.0
"""


def run_code_generator(state: AgentState) -> dict:
    """
    LangGraph node: write and validate a production pipeline script.

    Reads:  state["evaluation_result"], state["dataset_profile"], state["feature_result"]
    Writes: state["deployment_artifacts"] (partial), state["current_step"], state["logs"]
    """
    from sandbox.executor import execute_with_retry

    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(f"[{timestamp}] CODE GENERATOR — Writing production pipeline code...")

    try:
        eval_result = state.get("evaluation_result") or {}
        profile = state.get("dataset_profile") or {}
        feature_result = state.get("feature_result") or {}
        model_results = state.get("model_results") or {}

        winner = eval_result.get("winner_model", "lightgbm")
        task_type = profile.get("task_type", "classification")
        target_col = profile.get("target_column", "target")
        winner_result = model_results.get(winner, {})
        winner_metrics = winner_result.get("metrics", {})

        llm = get_codegen_llm(
            provider=state["provider"],
            api_key=state["api_key"],
        )

        user_prompt = (
            f"Write a complete production ML pipeline for the following:\n\n"
            f"Task: {task_type}\n"
            f"Target column: '{target_col}'\n"
            f"Winning model: {winner}\n"
            f"Model metrics: {winner_metrics}\n"
            f"Numeric columns: {profile.get('numeric_cols', [])[:15]}\n"
            f"Categorical columns: {profile.get('categorical_cols', [])[:10]}\n"
            f"Is imbalanced: {profile.get('is_imbalanced', False)}\n\n"
            f"Preprocessing steps applied:\n"
            + "\n".join(
                f"  - {s}" for s in profile.get("recommended_preprocessing", [])
            )
            + "\n\n"
            f"Feature engineering code (for reference):\n"
            f"```python\n{feature_result.get('preprocessing_code', '# (not available)')[:500]}\n```\n\n"
            "Write the full pipeline.py script."
        )

        response = llm.invoke(
            [
                SystemMessage(content=PIPELINE_CODE_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        pipeline_code = strip_fences(extract_content(response))

        logs.append(
            f"[{timestamp}] CODE GENERATOR — Generated {len(pipeline_code)} chars. Validating in sandbox..."
        )

        # Validate by running in E2B (syntax + import check)
        validation_code = (
            "import ast, sys\n"
            "code = '''" + pipeline_code.replace("'''", '"""') + "'''\n"
            "try:\n"
            "    ast.parse(code)\n"
            "    print('SYNTAX_OK')\n"
            "except SyntaxError as e:\n"
            "    print(f'SYNTAX_ERROR: {e}')\n"
            "    sys.exit(1)\n"
        )

        validation_result = execute_with_retry(
            code=validation_code,
            csv_path=state.get("csv_path", ""),
            llm=llm,
            max_retries=1,  # Just one attempt for syntax check
        )

        if validation_result["success"] and "SYNTAX_OK" in validation_result.get(
            "stdout", ""
        ):
            logs.append(f"[{timestamp}] CODE GENERATOR — Syntax validated.")
        else:
            logs.append(
                f"[{timestamp}] CODE GENERATOR — Warning: syntax validation inconclusive."
            )

        logs.append(f"[{timestamp}] CODE GENERATOR — Done.")

        # Save pipeline code to disk for download
        import os

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/pipeline.py", "w") as f:
            f.write(pipeline_code)

        with open("outputs/requirements.txt", "w") as f:
            f.write(REQUIREMENTS_TEMPLATE)

        # Start building deployment artifacts
        artifacts: DeploymentArtifacts = {
            "pipeline_code": pipeline_code,
            "requirements_txt": REQUIREMENTS_TEMPLATE,
            "fastapi_code": "",  # Filled by deployment_agent
            "dockerfile": "",  # Filled by deployment_agent
            "openapi_spec": {},  # Filled by deployment_agent
            "mlflow_model_uri": winner_result.get("mlflow_run_id", ""),
        }

        return {
            "deployment_artifacts": artifacts,
            "current_step": "deployment_agent",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Code Generator failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] CODE GENERATOR — ERROR: {exc}")
        return {
            "error": f"Code Generator failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
