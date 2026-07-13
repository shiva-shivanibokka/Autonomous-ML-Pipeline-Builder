# Production image for the FastAPI backend — targets Google Cloud Run.
# Cloud Run injects $PORT (8080) and provides secrets/env at deploy time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    APP_ENV=production

WORKDIR /app

# Build tools for lightgbm/xgboost wheels + curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only (no tests, web/, notebooks — see .dockerignore).
COPY core/ core/
COPY agents/ agents/
COPY pipeline/ pipeline/
COPY sandbox/ sandbox/
COPY api/ api/
COPY knowledge/ knowledge/

# Run as a non-root user; pre-create the writable dirs it needs.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p outputs uploads data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run runs its own health probes; this HEALTHCHECK helps other platforms.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f "http://localhost:$PORT/health" || exit 1

# Shell form so $PORT expands. One worker: state is per-instance (SQLite + in-proc
# executor); Cloud Run scales by adding instances, not workers.
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
