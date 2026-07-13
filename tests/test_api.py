"""
tests.test_api — API contract + security tests for the FastAPI backend.

These prove the Phase A hardening: callers cannot make the pipeline read
arbitrary server files, and artifacts are scoped per run.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import api.main

    return TestClient(api.main.app)


def _csv_bytes() -> bytes:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    return df.to_csv(index=False).encode()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_returns_opaque_id_not_path(client):
    r = client.post("/upload", files={"file": ("data.csv", _csv_bytes(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    # The response must not leak a filesystem path — only an opaque id.
    assert "csv_path" not in body
    assert len(body["upload_id"]) == 32
    assert body["n_rows"] == 3


def test_run_rejects_arbitrary_path_as_upload_id(client):
    """The old hole: passing a server path. Must be rejected, never read."""
    r = client.post(
        "/pipeline/run",
        json={"upload_id": "/etc/passwd", "business_problem": "leak the passwd file"},
    )
    assert r.status_code == 400  # malformed upload_id, not a 200 that reads the file


def test_run_rejects_unknown_upload_id(client):
    r = client.post(
        "/pipeline/run",
        json={"upload_id": "0" * 32, "business_problem": "predict the y column here"},
    )
    assert r.status_code == 404  # well-formed but never issued


def test_non_csv_upload_rejected(client):
    r = client.post("/upload", files={"file": ("evil.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 400


def test_artifact_route_rejects_unknown_filename(client):
    r = client.get(f"/pipeline/{'a' * 32}/artifacts/passwd")
    assert r.status_code == 400


def test_metrics_endpoint_exposes_prometheus(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "pipeline_runs_started_total" in r.text


def test_status_unknown_pipeline_is_404(client):
    r = client.get(f"/pipeline/{'b' * 32}/status")
    assert r.status_code == 404
