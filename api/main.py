"""
api.main — FastAPI backend for the Autonomous ML Pipeline Builder.

Endpoints:
    POST /upload                        — Upload a CSV, get an opaque upload_id + preview
    POST /pipeline/run                  — Start a pipeline run (async, returns pipeline_id)
    GET  /pipeline/{id}/status          — Poll pipeline status
    GET  /pipeline/{id}/result          — Get full pipeline result
    GET  /pipeline/{id}/logs            — Get log lines (supports ?offset=)
    GET  /pipeline/{id}/artifacts/{f}   — Download a generated artifact for that run
    GET  /health                        — Health check

Security notes:
    - Callers never pass filesystem paths. /upload issues an opaque upload_id; the
      server resolves it to a path inside UPLOAD_DIR, so a client cannot make the
      pipeline read arbitrary server files.
    - API keys supplied for LLM calls are never stored in the run record or returned.
    - Artifacts are namespaced per pipeline_id so concurrent runs never collide.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    HealthResponse,
    PipelineRequest,
    PipelineResultResponse,
    PipelineStatusResponse,
    UploadResponse,
)
from core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loudly on boot if production is misconfigured (unsafe exec, open CORS).
    settings.validate_for_production()
    yield


app = FastAPI(
    title="Autonomous ML Pipeline Builder API",
    version="1.0.0",
    description=(
        "Upload a CSV, describe your ML problem, and watch AI agents "
        "build, evaluate, and deploy a complete ML pipeline."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── In-memory state store (replaced by a persistent store in Phase C) ─────────
_pipeline_states: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# Uploads and artifacts live under the working directory. UPLOAD_DIR holds raw
# CSVs keyed by upload_id; ARTIFACTS_ROOT/<pipeline_id> holds generated files.
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ARTIFACTS_ROOT = Path("outputs")
ARTIFACTS_ROOT.mkdir(exist_ok=True)

_ID_RE = re.compile(r"^[a-f0-9]{32}$")  # uuid4().hex — the only shape we accept
_MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024

_ALLOWED_ARTIFACTS = {
    "pipeline.py",
    "requirements.txt",
    "fastapi_endpoint.py",
    "Dockerfile",
    "openapi_spec.json",
    "shap_summary.png",
    "model.pkl",
    "feature_schema.json",
}


def _resolve_upload(upload_id: str) -> Path:
    """Map an opaque upload_id back to its CSV path, rejecting anything unsafe."""
    if not _ID_RE.match(upload_id):
        raise HTTPException(status_code=400, detail="Malformed upload_id")
    path = UPLOAD_DIR / f"{upload_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    return path


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()


# ── Upload ────────────────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadResponse, tags=["Pipeline"])
async def upload_csv(file: UploadFile = File(...)):
    """Store an uploaded CSV under a server-issued id and return a preview."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )

    upload_id = uuid.uuid4().hex
    csv_path = UPLOAD_DIR / f"{upload_id}.csv"
    csv_path.write_bytes(content)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        csv_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=df.columns.tolist(),
        preview=df.head(5).fillna("").to_dict(orient="records"),
    )


# ── Pipeline run (async) ───────────────────────────────────────────────────────


def _run_pipeline_sync(pipeline_id: str, csv_path: str, request: PipelineRequest) -> None:
    """Run pipeline in a thread pool — called by asyncio executor."""
    from pipeline.runner import run_pipeline

    _pipeline_states[pipeline_id]["status"] = "running"
    try:
        final_state = run_pipeline(
            csv_path=csv_path,
            business_problem=request.business_problem,
            provider=request.provider,
            api_key=request.api_key,
            model_name=request.model_name,
            pipeline_id=pipeline_id,
        )
        # Never retain the caller's API key in the stored record.
        final_state.pop("api_key", None)
        _pipeline_states[pipeline_id].update(
            {
                "status": final_state.get("status", "completed"),
                "state": final_state,
                "error": final_state.get("error"),
            }
        )
    except Exception as exc:
        logger.error("Pipeline %s crashed: %s", pipeline_id, exc, exc_info=True)
        _pipeline_states[pipeline_id].update({"status": "failed", "error": str(exc)})


@app.post("/pipeline/run", tags=["Pipeline"])
async def run_pipeline_endpoint(request: PipelineRequest):
    """Start a pipeline run. Returns pipeline_id for polling."""
    csv_path = _resolve_upload(request.upload_id)

    pipeline_id = uuid.uuid4().hex
    _pipeline_states[pipeline_id] = {"status": "pending", "state": None, "error": None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_sync, pipeline_id, str(csv_path), request)

    return {"pipeline_id": pipeline_id, "status": "pending"}


# ── Status polling ────────────────────────────────────────────────────────────


@app.get(
    "/pipeline/{pipeline_id}/status",
    response_model=PipelineStatusResponse,
    tags=["Pipeline"],
)
async def get_pipeline_status(pipeline_id: str):
    """Poll the current status of a running pipeline."""
    if pipeline_id not in _pipeline_states:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    record = _pipeline_states[pipeline_id]
    state = record.get("state") or {}
    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        status=record["status"],
        current_step=state.get("current_step", "pending"),
        error=record.get("error"),
        log_count=len(state.get("logs", [])),
    )


# ── Result ────────────────────────────────────────────────────────────────────


@app.get(
    "/pipeline/{pipeline_id}/result",
    response_model=PipelineResultResponse,
    tags=["Pipeline"],
)
async def get_pipeline_result(pipeline_id: str):
    """Get the full result of a completed pipeline."""
    if pipeline_id not in _pipeline_states:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    record = _pipeline_states[pipeline_id]
    if record["status"] in ("pending", "running"):
        raise HTTPException(status_code=202, detail="Pipeline still running")

    state = record.get("state") or {}
    eval_result = state.get("evaluation_result") or {}
    winner = eval_result.get("winner_model", "")
    model_results = state.get("model_results") or {}
    winner_metrics = (model_results.get(winner) or {}).get("metrics", {})
    artifacts = state.get("deployment_artifacts") or {}

    return PipelineResultResponse(
        pipeline_id=pipeline_id,
        status=record["status"],
        winner_model=winner,
        primary_metric=eval_result.get("primary_metric"),
        metrics=winner_metrics,
        justification=eval_result.get("justification"),
        bias_warnings=eval_result.get("bias_warnings", []),
        comparison_table=eval_result.get("comparison_table", []),
        has_shap_plot=bool(eval_result.get("shap_plot_path")),
        has_pipeline_code=bool(artifacts.get("pipeline_code")),
        has_fastapi_endpoint=bool(artifacts.get("fastapi_code")),
        has_dockerfile=bool(artifacts.get("dockerfile")),
        logs=state.get("logs", []),
    )


# ── Logs ──────────────────────────────────────────────────────────────────────


@app.get("/pipeline/{pipeline_id}/logs", tags=["Pipeline"])
async def get_pipeline_logs(pipeline_id: str, offset: int = 0):
    """Get log lines from a pipeline run, starting at offset."""
    if pipeline_id not in _pipeline_states:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    state = _pipeline_states[pipeline_id].get("state") or {}
    all_logs = state.get("logs", [])
    return {"logs": all_logs[max(offset, 0):], "total": len(all_logs)}


# ── Artifacts download ────────────────────────────────────────────────────────


@app.get("/pipeline/{pipeline_id}/artifacts/{filename}", tags=["Artifacts"])
async def download_artifact(pipeline_id: str, filename: str):
    """Download a generated artifact for a specific run."""
    if not _ID_RE.match(pipeline_id):
        raise HTTPException(status_code=400, detail="Malformed pipeline_id")
    if filename not in _ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=400, detail=f"Unknown artifact: {filename}")
    path = ARTIFACTS_ROOT / pipeline_id / filename
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Artifact not yet generated: {filename}"
        )
    return FileResponse(path, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
    )
