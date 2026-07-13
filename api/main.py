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
    GET  /metrics                       — Prometheus metrics

Security notes:
    - Callers never pass filesystem paths. /upload issues an opaque upload_id; the
      server resolves it to a path inside UPLOAD_DIR, so a client cannot make the
      pipeline read arbitrary server files.
    - API keys supplied for LLM calls are never stored in the run record or returned.
    - Artifacts are namespaced per pipeline_id so concurrent runs never collide.

Persistence:
    - Run state lives in a SQLite-backed store (survives restarts, TTL-swept), not a
      process dict, so it neither leaks memory nor vanishes on redeploy.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api.schemas import (
    HealthResponse,
    PipelineRequest,
    PipelineResultResponse,
    PipelineStatusResponse,
    UploadResponse,
)
from core.config import settings
from core.logging_config import configure_logging, get_logger
from core.store import RunStore

configure_logging()
log = get_logger("api")


# ── Prometheus metrics ────────────────────────────────────────────────────────
RUNS_STARTED = Counter("pipeline_runs_started_total", "Pipeline runs started")
RUNS_COMPLETED = Counter("pipeline_runs_completed_total", "Pipeline runs completed")
RUNS_FAILED = Counter("pipeline_runs_failed_total", "Pipeline runs failed")
REQ_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)


# ── App setup ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loudly on boot if production is misconfigured (unsafe exec, open CORS).
    settings.validate_for_production()
    log.info("api_startup", app_env=settings.app_env, backend=settings.execution_backend)
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


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQ_LATENCY.labels(request.method, path).observe(elapsed)
    log.info(
        "request",
        method=request.method,
        path=path,
        status=response.status_code,
        latency_ms=round(elapsed * 1000, 1),
    )
    return response


# ── Storage & execution ───────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
_store = RunStore(db_path=DATA_DIR / "runs.db")

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


def _now() -> float:
    return time.time()


def _resolve_upload(upload_id: str) -> Path:
    """Map an opaque upload_id back to its CSV path, rejecting anything unsafe."""
    if not _ID_RE.match(upload_id):
        raise HTTPException(status_code=400, detail="Malformed upload_id")
    path = UPLOAD_DIR / f"{upload_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    return path


def _safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """A JSON-serialisable snapshot of pipeline state (no keys, no model objects)."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k in ("api_key", "shap_sample"):
            continue
        if k == "model_results":
            out[k] = {
                name: {kk: vv for kk, vv in (r or {}).items() if kk != "model_object"}
                for name, r in (v or {}).items()
            }
        else:
            out[k] = v
    return out


# ── Health & metrics ──────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()


@app.get("/metrics", tags=["System"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    """Run pipeline in a worker thread, persisting live progress to the store."""
    from pipeline.runner import run_pipeline_streaming

    RUNS_STARTED.inc()
    _store.update(pipeline_id, _now(), status="running")

    def _on_update(state: dict) -> None:
        _store.update(
            pipeline_id,
            _now(),
            status=state.get("status", "running"),
            state=_safe_state(state),
            error=state.get("error"),
        )

    try:
        final_state = run_pipeline_streaming(
            csv_path=csv_path,
            business_problem=request.business_problem,
            provider=request.provider,
            api_key=request.api_key,
            model_name=request.model_name,
            pipeline_id=pipeline_id,
            on_update=_on_update,
        )
        final_state.pop("api_key", None)
        status = final_state.get("status", "completed")
        _store.update(
            pipeline_id,
            _now(),
            status=status,
            state=_safe_state(final_state),
            error=final_state.get("error"),
        )
        (RUNS_FAILED if status == "failed" else RUNS_COMPLETED).inc()
    except Exception as exc:
        RUNS_FAILED.inc()
        log.error("pipeline_crashed", pipeline_id=pipeline_id, error=str(exc))
        _store.update(pipeline_id, _now(), status="failed", error=str(exc))


@app.post("/pipeline/run", tags=["Pipeline"])
async def run_pipeline_endpoint(request: PipelineRequest):
    """Start a pipeline run. Returns pipeline_id for polling."""
    csv_path = _resolve_upload(request.upload_id)

    pipeline_id = uuid.uuid4().hex
    _store.create(pipeline_id, _now())

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_sync, pipeline_id, str(csv_path), request)

    return {"pipeline_id": pipeline_id, "status": "pending"}


# ── Status polling ────────────────────────────────────────────────────────────


def _require_record(pipeline_id: str) -> dict:
    record = _store.get(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return record


@app.get(
    "/pipeline/{pipeline_id}/status",
    response_model=PipelineStatusResponse,
    tags=["Pipeline"],
)
async def get_pipeline_status(pipeline_id: str):
    """Poll the current status of a running pipeline."""
    record = _require_record(pipeline_id)
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
    record = _require_record(pipeline_id)
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
    record = _require_record(pipeline_id)
    state = record.get("state") or {}
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
