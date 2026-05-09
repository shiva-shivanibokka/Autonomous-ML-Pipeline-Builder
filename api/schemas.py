"""
api.schemas — Pydantic v2 request and response models for the FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Request body for POST /pipeline/run."""

    csv_path: str = Field(description="Local path to the uploaded CSV file")
    business_problem: str = Field(
        description="Plain English description of the ML task",
        min_length=10,
        max_length=2000,
    )
    provider: str = Field(default="anthropic", description="LLM provider")
    api_key: str = Field(default="", description="API key (optional if set in env)")
    model_name: str = Field(
        default="", description="Model name (uses provider default if empty)"
    )


class PipelineStatusResponse(BaseModel):
    """Response for GET /pipeline/{pipeline_id}/status."""

    pipeline_id: str
    status: str  # "running" | "completed" | "failed"
    current_step: str
    error: Optional[str] = None
    log_count: int


class PipelineResultResponse(BaseModel):
    """Response for GET /pipeline/{pipeline_id}/result."""

    pipeline_id: str
    status: str
    winner_model: Optional[str] = None
    primary_metric: Optional[str] = None
    metrics: Optional[dict[str, float]] = None
    justification: Optional[str] = None
    bias_warnings: list[str] = []
    comparison_table: list[dict[str, Any]] = []
    has_shap_plot: bool = False
    has_pipeline_code: bool = False
    has_fastapi_endpoint: bool = False
    has_dockerfile: bool = False
    logs: list[str] = []


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    """Response for POST /upload."""

    csv_path: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[str]
    preview: list[dict[str, Any]]  # First 5 rows as list of dicts
