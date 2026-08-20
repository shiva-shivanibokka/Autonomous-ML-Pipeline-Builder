# Autonomous ML Pipeline Builder

**Upload a CSV, describe your problem in plain English, and a team of AI agents builds, evaluates, and packages a complete, deployable ML pipeline — live.**

[![CI](https://github.com/shiva-shivanibokka/Autonomous-ML-Pipeline-Builder/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/Autonomous-ML-Pipeline-Builder/actions/workflows/ci.yml)
![Backend](https://img.shields.io/badge/backend-FastAPI%20%C2%B7%20LangGraph-0b7285)
![Frontend](https://img.shields.io/badge/frontend-Next.js%20%C2%B7%20Vercel-black)
![License](https://img.shields.io/badge/license-MIT-blue)

> ### Recruiter TL;DR
> - **What it is:** a full-stack AI system that turns a raw CSV + a plain-English goal into a trained, SHAP-explained, deployment-ready ML model — built end to end by a 7-agent LangGraph crew, with a Next.js console that streams every agent's work in real time.
> - **Hardest problem solved:** eliminating train/test **data leakage** across the whole flow (preprocessing is a scikit-learn `Pipeline` fit on the training fold only) *and* shipping a **runnable** artifact — the full pipeline serializes to `model.pkl`, so the generated FastAPI service predicts on raw input with zero training/serving skew.
> - **Production concerns are addressed, not gestured at:** 78 automated tests (CI-green), structured JSON logging + Prometheus metrics, security hardening (no arbitrary file reads, sandboxed code execution, keys never stored), a Docker image built on every push, and a Cloud Run + Vercel deployment path with a documented runbook.

A LangGraph crew of seven agents plans the approach, profiles the data, engineers leakage-safe features, trains and cross-validates several models in parallel, explains the winner with SHAP, and emits a runnable FastAPI + Docker inference bundle. A Next.js console streams every agent's work in real time.

> **The console is published as a replay of one real run** — the backend needs an LLM key per run and a paid sandbox, so it is not hosted. Everything you see there came out of an actual pipeline run; see [The hosted console is a replay](#the-hosted-console-is-a-replay). To run it for real, [`docker compose up`](#run-it-locally) gives you the whole stack.

### What one real run produced

Telco customer churn, 7,043 rows × 21 columns, `claude-sonnet-5`, ~2 minutes end to end:

| | |
|---|---|
| Models trained & cross-validated | `lightgbm`, `xgboost`, `random_forest`, `logistic_regression` |
| Winner (chosen by argmax, not by asking the model) | **`random_forest`** — AUC **0.836** vs xgboost 0.8328, lightgbm 0.8322 |
| Fairness | flagged `gender` as a sensitive feature before deployment |
| Emitted | `model.pkl`, `pipeline.py`, `fastapi_endpoint.py`, `Dockerfile`, `requirements.txt`, `openapi_spec.json`, `feature_schema.json`, SHAP plot |
| Verified after the fact | `model.pkl` loads as `Pipeline(['prep','model'])` and predicts on **raw, untouched CSV rows** — no manual preprocessing, no training/serving skew |

The self-correction loop earned its keep in that run: the Feature Engineer's first sandboxed attempt failed, the traceback went back to the model, and the retry succeeded. It's in the log stream.

---

## Why this project is interesting

Most early-career ML projects are a notebook with a model that was never deployed and quietly leaks test data into training. This one is built the other way around:

- **No data leakage.** Preprocessing is a scikit-learn `Pipeline` fit on the training fold only, validated with k-fold cross-validation. ([ADR 0002](docs/adr/0002-leakage-safe-pipeline.md))
- **The output actually runs.** The winning pipeline (preprocessing + model) is serialized to `model.pkl`; the generated FastAPI service loads it and predicts on raw input — no training/serving skew.
- **Built to deploy properly.** Non-root container for Cloud Run — built and smoke-imported in CI on every push — static frontend for Vercel, secrets via Secret Manager, health checks, structured logs, and a Prometheus `/metrics` endpoint.
- **Retrieval-grounded agents.** The code-generating agents retrieve from a curated ML best-practices knowledge base (RAG) instead of relying on parametric memory alone.
- **Hardened.** No arbitrary file reads, generated code never runs on the host in production (isolated E2B sandbox), API keys are never stored, CORS is locked down, and the app refuses to boot if production is misconfigured.

## Architecture

```mermaid
flowchart LR
  subgraph Vercel["Vercel — Next.js console"]
    UI["Upload · live agent rail · results"]
  end
  subgraph CloudRun["Google Cloud Run — FastAPI"]
    API["REST API + Prometheus /metrics"]
    STORE[("SQLite run store")]
    subgraph Graph["LangGraph pipeline"]
      O[Orchestrator] --> DA[Data Analyst] --> FE[Feature Engineer]
      FE --> MT[Model Trainer] --> EV[Evaluator] --> CG[Code Generator] --> DEP[Deployment Agent]
    end
    RAG[("ML knowledge base — RAG")]
  end
  E2B["E2B sandbox\n(runs generated code)"]
  LLM["LLM provider\n(Anthropic / OpenAI / Groq)"]

  UI -->|"HTTPS"| API
  API --> Graph
  API --- STORE
  FE -. retrieve .-> RAG
  CG -. retrieve .-> RAG
  FE -->|generated code| E2B
  O & DA & EV --> LLM
```

**Why this shape?** The UI and the pipeline have opposite runtime profiles — static assets updated often vs. multi-minute stateful jobs that spawn sandboxed work. Serverless functions can't host the latter (they time out and don't hold state), so the backend is a long-lived Cloud Run container and the frontend is a Vercel static app; they talk over HTTPS with a locked-down CORS contract. Full reasoning in [ADR 0001](docs/adr/0001-frontend-backend-split.md).

## The seven agents

| # | Agent | What it does |
|---|-------|--------------|
| 1 | **Orchestrator** | Detects task type (classification / regression / time series), picks the models and primary metric. |
| 2 | **Data Analyst** | Profiles the dataset — dtypes, missing values, class imbalance, outliers. |
| 3 | **Feature Engineer** | Generates & runs leakage-safe structural preprocessing in a sandbox; self-corrects on failure. |
| 4 | **Model Trainer** | Trains 3–5 models **in parallel** (`asyncio`), each as a leakage-safe pipeline with k-fold CV. |
| 5 | **Evaluator** | Deterministically selects the winner, runs SHAP on the held-out test set, flags fairness risks, persists `model.pkl`. |
| 6 | **Code Generator** | Writes a clean, documented `pipeline.py`. |
| 7 | **Deployment Agent** | Emits a FastAPI inference endpoint, Dockerfile, and OpenAPI spec. |

A **self-correction loop** wraps sandboxed execution: when generated code fails, the traceback is fed back to the LLM to fix it (up to 3 attempts).

## Skills demonstrated

Real capabilities exercised in this repo (mapped to how they're usually named), with where to find each:

| Competency | In this repo |
|---|---|
| **LLM application development · RAG · agentic systems** | 7-node LangGraph orchestration (`agents/`), TF-IDF/embedding retrieval grounding (`core/rag/`), self-correction loop (`sandbox/`) |
| **Production ML / MLOps** | Serving decoupled from training, full-pipeline persistence (`model.pkl`), MLflow tracking (`core/mlops/`) |
| **Data engineering / ETL** | Raw CSV → profiled → leakage-safe feature pipeline → model-ready (`agents/data_analyst.py`, `agents/model_trainer.py`) |
| **RESTful API design** | FastAPI service with upload/run/status/logs/result/artifacts endpoints (`api/`) |
| **Asynchronous / concurrent systems** | Parallel model training via `asyncio`, background run workers via a thread pool |
| **System design & architecture** | Two ADRs documenting the split and the leakage fix (`docs/adr/`) |
| **Observability & monitoring** | Structured JSON logging (structlog), health check, Prometheus `/metrics` |
| **Application security** | Server-issued upload IDs (no arbitrary file read), sandboxed execution, no stored keys, CORS allowlist, fail-loud prod boot |
| **Containerization & CI/CD** | Non-root Dockerfile built in CI, GitHub Actions (lint + test + image build + frontend build) |
| **Cloud-native deployment (GCP Cloud Run · Vercel)** | Deploy script + runbook (`deploy/`) — configured, not yet live |
| **Automated testing** | 78 pytest tests incl. security, ML-correctness, RAG-quality, and regression checks (`tests/`) |
| **Frontend engineering** | Next.js (App Router) + TypeScript console with live streaming (`web/`) |

## Tech stack

**Backend** — Python 3.11 · FastAPI · LangGraph · LangChain · scikit-learn · LightGBM · XGBoost · SHAP · MLflow · E2B · Pydantic v2 · Prometheus · structlog
**Frontend** — Next.js (App Router) · TypeScript · Tailwind CSS
**Infra** — Docker · Google Cloud Run · Vercel · GitHub Actions

Exact pinned versions are in [`requirements.txt`](requirements.txt) and [`web/package.json`](web/package.json).

## Run it locally

Two ways. Docker is the short one.

### Option A — the whole stack, one command (recommended)

**Prerequisites:** Docker Desktop, and an LLM API key (Anthropic / OpenAI / Groq).

```bash
cp .env.example .env          # add your API key
docker compose up --build
```

> **If a key in `.env` seems to be ignored, it is.** An environment variable
> already set in your shell — or a Windows user-level variable from another
> project — beats the file, which is standard but silent, and the symptom is a
> `401` that looks exactly like a bad key. The app now prints
> `Ignoring .env for ANTHROPIC_API_KEY …` when that happens. Unset the stale
> variable and it will pick up the file.

| | |
|---|---|
| App | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

Open the app, drop in a CSV, describe the goal, watch the agents run. `Ctrl+C`
to stop, `docker compose down -v` to also discard the volumes.

You need no Python or Node toolchain on your machine, and generated code runs in
the API container's own subprocess sandbox rather than on your host — which is
why `ALLOW_LOCAL_EXEC` defaults to true in `docker-compose.yml` and nowhere else.

### Option B — run the pieces directly

**Prerequisites:** Python 3.11, Node 20+, and an LLM API key (Anthropic / OpenAI / Groq).

**Backend**

```bash
cp .env.example .env            # add your API key(s)
pip install -r requirements-dev.txt

# Local dev: run generated code in a subprocess sandbox (E2B not required locally).
export ALLOW_LOCAL_EXEC=true EXECUTION_BACKEND=subprocess
uvicorn api.main:app --reload --port 8000
# Interactive API docs: http://localhost:8000/docs
```

> Security note: `ALLOW_LOCAL_EXEC=true` runs LLM-generated code on your machine and is for **local dev only**. In production the app requires an [E2B](https://e2b.dev) sandbox key and refuses to boot otherwise.

**Frontend**

```bash
cd web
cp .env.local.example .env.local   # defaults to http://localhost:8000
npm install
npm run dev                        # http://localhost:3000
```

Open http://localhost:3000, drop in a CSV, describe the goal, and watch the agents run.

## Usage (API)

The frontend drives these; you can also call them directly:

```bash
# 1. Upload a CSV — returns an opaque upload_id (never a filesystem path)
curl -F "file=@data.csv" http://localhost:8000/upload

# 2. Start a run
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"upload_id":"<id>","business_problem":"Predict the target column",
       "provider":"anthropic","api_key":"sk-...","model_name":""}'

# 3. Poll status / stream logs, then fetch results and artifacts
curl http://localhost:8000/pipeline/<pipeline_id>/status
curl http://localhost:8000/pipeline/<pipeline_id>/logs
curl -O http://localhost:8000/pipeline/<pipeline_id>/artifacts/model.pkl
```

## Testing

```bash
pip install -r requirements-dev.txt
ALLOW_LOCAL_EXEC=true pytest -q          # 78 tests (1 skips without the E2B extra)
ruff check .                             # lint
```

CI runs the same lint + tests (with coverage), builds the production Docker image, and builds the frontend on every push and PR ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Coverage spans the API security contract (arbitrary-path rejection, key redaction), the leakage-safe training + persistence path, deterministic model selection, RAG retrieval quality, and the run store.

## Project structure

```
agents/        # the seven LangGraph nodes
pipeline/      # graph wiring + run entry points (blocking, streaming)
core/          # config, LLM providers, MLflow tracking, RAG retriever, run store
sandbox/       # E2B / subprocess execution with a self-correction loop
api/            # FastAPI app + schemas
knowledge/     # ML best-practices corpus (RAG source)
web/           # Next.js frontend (deploys to Vercel)
web/public/demo/  # the recorded run the hosted console replays
deploy/        # Cloud Run deploy script + runbook
scripts/       # record_demo_run.py - captures a real run for the replay
docs/adr/      # architecture decision records
tests/         # pytest suite
```

## The hosted console is a replay

The backend is a long-lived container that needs an LLM key per run and a paid
sandbox, so it is not hosted anywhere (see [Deploying it yourself](#deploying-it-yourself)).
The console still is: it replays **one real recorded run** from static files,
with a banner giving the recording date, the dataset, the model, and the real
duration it was compressed from.

Every log line, metric, SHAP plot and downloadable file on that page came out of
an actual run. Nothing is simulated, no pipeline is executing while you watch,
and the page says so plainly — a replay that pretended otherwise would be worth
less than no demo at all.

The recording committed here is the Telco churn run described [at the top of
this file](#what-one-real-run-produced): 44 log lines, 9 progress frames, 122
seconds compressed to 30, and the eight artifacts it emitted — including the
7.4 MB `model.pkl`, which you can download from the page and load yourself.

### Record a run

```bash
# The key can live in .env instead of the command line.
export ALLOW_LOCAL_EXEC=true EXECUTION_BACKEND=subprocess
python scripts/record_demo_run.py   --csv data/your_dataset.csv   --problem "Predict which customers churn. The target column is Churn."   --provider anthropic
```

`ALLOW_LOCAL_EXEC` is required because the Feature Engineer runs generated code;
the recorder checks for it up front rather than failing after two paid LLM
calls. Set `E2B_API_KEY` instead to use the isolated cloud sandbox.

That runs the real pipeline and writes `web/public/demo/run.json` (progress
timeline + full log stream + final result) plus `web/public/demo/artifacts/`
(the `pipeline.py`, `model.pkl`, `Dockerfile`, SHAP plot and the rest it
produced). Commit both.

Three things the recorder will not do: record a failed run, write a recording
that contains your API key (it scans the payload and every artifact first), or
copy an artifact over 20 MB into a static bundle.

### Publish it

Import the repo on Vercel as a **git-connected project** — not a CLI one-off, so
every push to `main` redeploys:

| Setting | Value |
|---|---|
| Root Directory | `web` |
| `NEXT_PUBLIC_DEMO_MODE` | `1` |
| `NEXT_PUBLIC_API_BASE_URL` | leave unset |

With `NEXT_PUBLIC_DEMO_MODE=1` the console autoplays the recording instead of
reaching for an API that is not there. Locally it stays off, and the replay is a
button next to the live controls.

If a recording is ever missing, the page says so rather than inventing one.

## Deploying it yourself

The backend is **not hosted here**, and that is a decision rather than an omission: it needs an LLM key per run, a paid sandbox, ~2 GiB of memory, and multi-minute jobs that no free serverless tier will hold. A public instance would either bill the author for every visitor's training run or ask strangers to paste their own keys into someone else's server.

If you have the accounts, the path is fully configured and nothing here needs writing — only running.

### What it costs you

| Piece | Where | What you need |
|---|---|---|
| Backend | Google Cloud Run | A GCP project with billing enabled. Card required; Always Free covers a demo's usage. |
| Sandbox | [E2B](https://e2b.dev) | An API key. **Production refuses to boot without one** — it will not run model-written code on your server. |
| LLM | Anthropic / OpenAI / Groq | Callers supply their own key per request; server-side keys are optional fallbacks. Groq has a free tier. |
| Frontend | Vercel | Free Hobby account, no card. |

### 1. Backend → Cloud Run

```bash
# One-time: create the secrets the runbook lists (at minimum E2B_API_KEY)
PROJECT_ID=my-project \
FRONTEND_ORIGIN=https://your-app.vercel.app \
  ./deploy/deploy-cloudrun.sh
```

The script builds the `Dockerfile` via Cloud Build and deploys it non-root, mounting **only the secrets that exist** — so a missing optional `OPENAI_API_KEY` no longer fails the whole deploy. It prints the service URL when done.

Two things happen automatically that are worth knowing about:

- **`APP_ENV=production` makes the app fail loudly on a bad config** — it refuses to start with host code execution enabled, or with `ALLOWED_ORIGINS=*`. Misconfiguration is a crash on boot, not a quiet security hole.
- **`.gcloudignore` keeps the upload small.** `gcloud` does not read `.dockerignore`, and without this file it would ship ~370 MB of frontend `node_modules` to Cloud Build for an image that never uses them.

Full runbook — secrets, IAM, configuration reference, rollback — in **[deploy/README.md](deploy/README.md)**.

### 2. Frontend → Vercel

Import the repo as a git-connected project so every push redeploys:

| Setting | Value |
|---|---|
| Root Directory | `web` |
| `NEXT_PUBLIC_API_BASE_URL` | your Cloud Run URL |
| `NEXT_PUBLIC_DEMO_MODE` | `0` (or unset) |

### 3. Close the loop

Set the backend's `ALLOWED_ORIGINS` to the exact Vercel URL and redeploy the service. CORS is an explicit allowlist in production — a wildcard is refused at boot.

### Not using Cloud Run?

The backend is a plain container that reads `$PORT`, so anything that runs a long-lived Docker image works — Fly, Render, Railway, a VM behind nginx. Two requirements are non-negotiable regardless of host: **at least ~2 GiB of memory** (LightGBM + XGBoost + SHAP), and **a request timeout long enough for a multi-minute job**, or the run must be polled rather than awaited — which is what `/pipeline/{id}/status` already does.

### Don't want to host the backend at all?

Publish the frontend alone in replay mode. It costs nothing, never expires, and still shows the whole system working — see [The hosted console is a replay](#the-hosted-console-is-a-replay).

## Roadmap / known limitations

- **State durability:** the run store (SQLite) and artifacts are per-instance and reset on a Cloud Run cold start — fine for a demo. The `RunStore` interface is intentionally small to swap in Postgres/Redis + object storage.
- **Cross-validation** is capped by a row threshold to keep demo runs fast; larger datasets skip CV and say so in the log.
- **Hyperparameters** are fixed per model — an Optuna tuning pass is a natural next addition.
- **The E2B sandbox path has no live test.** CI runs the subprocess backend, so it
  verifies the security gate, the pinned SDK's call surface, and how a failed
  execution is rendered — but nothing in CI actually talks to a sandbox. Running one
  pipeline with a real E2B key is the remaining gap.
- **Cold starts.** A Cloud Run instance that has scaled to zero pays ~10-20s of
  container start before the first request, and the run store resets with it.

## License

MIT — see [LICENSE](LICENSE).
