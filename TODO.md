# TODO

What's left after the production overhaul. The app is complete and CI-green; these
are deployment steps (need your accounts) and optional enhancements.

## Deploy (needs your accounts — I can't do these for you)

- [ ] **Backend → Cloud Run.** Create a GCP project, get an [E2B key](https://e2b.dev)
      (production refuses to boot without it), add secrets to Secret Manager, then run:
      `PROJECT_ID=… FRONTEND_ORIGIN=https://<your-vercel-url> ./deploy/deploy-cloudrun.sh`
      Full steps: [deploy/README.md](deploy/README.md).
- [ ] **Frontend → Vercel.** Import the repo, set **Root Directory = `web`**, add env
      var `NEXT_PUBLIC_API_BASE_URL = <Cloud Run URL>`, deploy.
- [ ] **Close the loop.** Put the Cloud Run URL in the backend's `ALLOWED_ORIGINS`
      and the frontend's `NEXT_PUBLIC_API_BASE_URL` so the two trust each other.

## Placeholders to fill

- [ ] `README.md` — add the live demo URLs (top of file).
- [ ] `web/app/page.tsx` — the header `view source ↗` link points at `https://github.com`;
      set it to this repo's URL.

## Verify end to end

- [ ] Run one full happy-path pipeline with a real LLM key (upload → run → winner →
      SHAP → download `model.pkl`). Everything below the API layer is unit-tested, but a
      live run confirms the LLM-dependent agents end to end.
- [ ] Build the Docker image locally or in CI (`docker build -t ml-api .`) — it's been
      validated by review + a working local `uvicorn`, not an actual image build
      (Docker isn't installed in the dev environment used to build this).

## Optional enhancements (nice-to-have, not required)

- [ ] **Hyperparameter tuning** — a small Optuna / RandomizedSearchCV pass per model.
- [ ] **Durable state** — move `RunStore` (SQLite, per-instance) to Postgres/Redis and
      artifacts to object storage, so runs survive Cloud Run cold starts / multiple
      instances. The store interface is intentionally small to make this localized.
- [ ] **CI housekeeping** — GitHub is warning that `actions/checkout@v4` /
      `setup-python@v5` / `setup-node@v4` target Node 20 (auto-forced to Node 24). Bump
      to newer action majors when available. Non-blocking.
- [ ] **Docker build in CI** — add a job that builds the image (no push) to catch
      Dockerfile regressions. Skipped for now to keep CI fast.
