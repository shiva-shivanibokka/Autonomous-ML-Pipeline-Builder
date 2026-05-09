"""
agents.deployment_agent — Deployment Agent node.

Generates three deployment artifacts for the winning model:
  1. fastapi_endpoint.py  — async FastAPI inference endpoint with health check,
                            request validation, and prediction logging
  2. Dockerfile           — minimal production Dockerfile for the API service
  3. openapi_spec.json    — machine-readable API schema

All artifacts are written to outputs/ for the user to download.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm_utils import build_system_prompt, extract_content, strip_fences
from core.providers import get_codegen_llm

logger = logging.getLogger(__name__)


FASTAPI_PROMPT = build_system_prompt(
    role="a senior backend engineer who writes production FastAPI ML serving code",
    context=(
        "You will write a complete FastAPI inference endpoint script. "
        "The script must include:\n"
        "  1. Imports: fastapi, uvicorn, pydantic, joblib, pandas, numpy\n"
        "  2. A Pydantic request model with all feature fields typed correctly\n"
        "  3. A Pydantic response model with: prediction, probability (if classification), model_version\n"
        "  4. A lifespan context manager that loads the model from 'model.pkl' at startup\n"
        "  5. POST /predict endpoint with async def and proper error handling\n"
        "  6. GET /health endpoint returning {'status': 'ok', 'model_loaded': bool}\n"
        "  7. Prometheus counter for prediction count (prometheus_client)\n"
        "  8. Structured logging with timestamp, input hash, prediction, latency\n"
        "  9. if __name__ == '__main__': uvicorn.run(...) at the bottom\n\n"
        "Output ONLY the Python code. No explanation, no fences."
    ),
)

DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY fastapi_endpoint.py .
COPY pipeline.py .
COPY model.pkl .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "fastapi_endpoint.py"]
"""


def _generate_openapi_spec(
    model_name: str, task_type: str, feature_names: list[str], target_col: str
) -> dict:
    """Generate a minimal OpenAPI 3.0 spec for the inference endpoint."""
    properties = {
        feat: {"type": "number", "description": f"Feature: {feat}"}
        for feat in feature_names[:20]  # Limit for readability
    }

    response_props = {
        "prediction": {"type": "number" if task_type == "regression" else "integer"},
        "model_version": {"type": "string"},
    }
    if task_type == "classification":
        response_props["probability"] = {
            "type": "number",
            "description": "Probability of the positive class",
        }

    return {
        "openapi": "3.0.0",
        "info": {
            "title": f"{model_name.title()} ML Inference API",
            "version": "1.0.0",
            "description": f"AutoML-generated inference endpoint for {task_type} on '{target_col}'",
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Run inference",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": properties,
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successful prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": response_props,
                                    }
                                }
                            },
                        },
                        "422": {"description": "Validation error"},
                        "500": {"description": "Internal server error"},
                    },
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "model_loaded": {"type": "boolean"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
    }


def run_deployment_agent(state: AgentState) -> dict:
    """
    LangGraph node: generate FastAPI endpoint, Dockerfile, and OpenAPI spec.

    Reads:  state["deployment_artifacts"], state["evaluation_result"], state["dataset_profile"]
    Writes: state["deployment_artifacts"] (completed), state["status"], state["logs"]
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(
        f"[{timestamp}] DEPLOYMENT AGENT — Generating API endpoint and Dockerfile..."
    )

    try:
        eval_result = state.get("evaluation_result") or {}
        profile = state.get("dataset_profile") or {}
        artifacts = dict(state.get("deployment_artifacts") or {})

        winner = eval_result.get("winner_model", "model")
        task_type = profile.get("task_type", "classification")
        target_col = profile.get("target_column", "target")
        feature_names = profile.get("numeric_cols", []) + profile.get(
            "categorical_cols", []
        )

        llm = get_codegen_llm(
            provider=state["provider"],
            api_key=state["api_key"],
        )

        # Build typed feature list for the prompt
        feature_type_list = ", ".join(f"{f}: float" for f in feature_names[:15])

        user_prompt = (
            f"Generate a FastAPI inference endpoint for this model:\n\n"
            f"Model: {winner} ({task_type})\n"
            f"Target column: '{target_col}'\n"
            f"Input features: {feature_type_list}\n"
            f"Model is loaded from: model.pkl (joblib)\n\n"
            f"Write the complete fastapi_endpoint.py."
        )

        response = llm.invoke(
            [
                SystemMessage(content=FASTAPI_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        fastapi_code = strip_fences(extract_content(response))

        # Generate OpenAPI spec
        openapi_spec = _generate_openapi_spec(
            winner, task_type, feature_names, target_col
        )

        # Write all artifacts to disk
        os.makedirs("outputs", exist_ok=True)

        with open("outputs/fastapi_endpoint.py", "w") as f:
            f.write(fastapi_code)

        with open("outputs/Dockerfile", "w") as f:
            f.write(DOCKERFILE_TEMPLATE)

        with open("outputs/openapi_spec.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)

        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — FastAPI endpoint generated.")
        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — Dockerfile generated.")
        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — OpenAPI spec generated.")
        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — All artifacts saved to outputs/")
        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — Done.")
        logs.append(
            f"[{timestamp}] PIPELINE COMPLETE — Winner: {winner} | Task: {task_type}"
        )

        # Update artifacts
        artifacts["fastapi_code"] = fastapi_code
        artifacts["dockerfile"] = DOCKERFILE_TEMPLATE
        artifacts["openapi_spec"] = openapi_spec

        return {
            "deployment_artifacts": artifacts,
            "status": "completed",
            "current_step": "done",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Deployment Agent failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] DEPLOYMENT AGENT — ERROR: {exc}")
        return {
            "error": f"Deployment Agent failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
