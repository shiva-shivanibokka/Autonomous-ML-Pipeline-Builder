# Deployment

Two independent deployables:

| Component | Platform | Why |
|---|---|---|
| **Backend** (`Dockerfile`, FastAPI + agents) | **Google Cloud Run** | Long-running (multi-minute) training jobs, in-process state, and sandbox execution don't fit serverless functions. Cloud Run runs the container, scales to zero when idle (cheap), and gives a clean secrets/IAM story. |
| **Frontend** (`web/`, Next.js) | **Vercel** | Native Next.js host, global CDN, preview deploys per PR. |

The two talk over HTTPS; the frontend reads the backend URL from `NEXT_PUBLIC_API_BASE_URL`, and the backend restricts CORS to the frontend origin via `ALLOWED_ORIGINS`.

---

## Backend → Cloud Run

### 1. One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
```

### 2. Create secrets (never put keys in env files or code)

```bash
printf 'sk-ant-...' | gcloud secrets create ANTHROPIC_API_KEY --data-file=-
printf 'e2b_...'    | gcloud secrets create E2B_API_KEY        --data-file=-
printf 'sk-...'     | gcloud secrets create OPENAI_API_KEY     --data-file=-   # optional
```

Grant the Cloud Run runtime service account access:

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
for S in ANTHROPIC_API_KEY E2B_API_KEY OPENAI_API_KEY; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

> **Why E2B is required in production:** the backend refuses to boot with `APP_ENV=production` unless a real E2B sandbox key is set. Generated code then runs in an isolated cloud sandbox instead of on the server. Host execution (`ALLOW_LOCAL_EXEC`) is a local-dev-only escape hatch and stays off in prod. Get a key at https://e2b.dev.

### 3. Deploy

```bash
PROJECT_ID=YOUR_PROJECT_ID FRONTEND_ORIGIN=https://your-app.vercel.app \
  ./deploy/deploy-cloudrun.sh
```

The script builds `Dockerfile` via Cloud Build and deploys, wiring secrets and setting `APP_ENV=production`, `EXECUTION_BACKEND=e2b`, and `ALLOWED_ORIGINS` to your Vercel URL. It prints the service URL when done.

### Configuration reference

| Var | Value in prod | Notes |
|---|---|---|
| `APP_ENV` | `production` | Startup fails fast if misconfigured (see `core/config.py`) |
| `EXECUTION_BACKEND` | `e2b` | Required in prod |
| `ALLOWED_ORIGINS` | your Vercel URL | CORS allowlist; `*` is rejected in prod |
| `ANTHROPIC_API_KEY` / `E2B_API_KEY` / `OPENAI_API_KEY` | Secret Manager | Mounted, never baked into the image |
| `MAX_UPLOAD_MB` | `50` (default) | Upload size cap |

### Rollback

Cloud Run keeps every revision. Roll back by shifting traffic:

```bash
gcloud run services update-traffic ml-pipeline-api --region us-central1 --to-revisions PREVIOUS_REVISION=100
```

### Note on state

Run state (SQLite) and artifacts live on the instance's ephemeral filesystem, so they reset on a cold start and are per-instance. That's fine for a demo and keeps costs at zero when idle. To make runs durable across restarts/instances, point the store at a managed DB (the `RunStore` interface is intentionally small) and artifacts at object storage.

---

## Frontend → Vercel

1. Import the repo in Vercel and set the **Root Directory** to `web`.
2. Add an environment variable: `NEXT_PUBLIC_API_BASE_URL = https://<your-cloud-run-url>`.
3. Deploy. Vercel auto-detects Next.js; every push gets a preview URL, `main` promotes to production.

After the first backend deploy, copy the Cloud Run URL into `ALLOWED_ORIGINS` (backend) and `NEXT_PUBLIC_API_BASE_URL` (frontend) so the two trust each other.
