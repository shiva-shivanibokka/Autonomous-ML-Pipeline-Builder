# TODO

What's left. The app is complete and CI-green; the remaining items need accounts
that only the repo owner has, plus a few optional enhancements.

## Deploy (needs accounts)

- [ ] **Backend → Cloud Run.** Create a GCP project, get an [E2B key](https://e2b.dev)
      (production refuses to boot without one), add secrets to Secret Manager, then run:
      `PROJECT_ID=… FRONTEND_ORIGIN=https://<vercel-url> ./deploy/deploy-cloudrun.sh`
      Full steps: [deploy/README.md](deploy/README.md).
- [ ] **Frontend → Vercel.** Import the repo, set **Root Directory = `web`**, add env
      var `NEXT_PUBLIC_API_BASE_URL = <Cloud Run URL>`, deploy.
- [ ] **Close the loop.** Put the Vercel URL in the backend's `ALLOWED_ORIGINS`
      and the Cloud Run URL in the frontend's `NEXT_PUBLIC_API_BASE_URL` so the two
      trust each other.
- [ ] `README.md` — add the live demo URLs at the top once they exist.

## Verify end to end

- [ ] Run one full happy-path pipeline against a real LLM key (upload → run → winner →
      SHAP → download `model.pkl`). Everything below the API layer is unit-tested, but
      only a live run exercises the LLM-dependent agents together.
- [ ] Run one pipeline with `EXECUTION_BACKEND=e2b` and a real E2B key. CI now checks
      that the installed E2B package exposes the API `sandbox/executor.py` calls, and
      that a failed execution is rendered rather than iterated — but no test actually
      talks to a sandbox.

## Optional enhancements

- [ ] **Hyperparameter tuning** — a small Optuna / RandomizedSearchCV pass per model.
- [ ] **Durable state** — move `RunStore` (SQLite, per-instance) to Postgres/Redis and
      artifacts to object storage, so runs survive Cloud Run cold starts and scale past
      one instance. The store interface is intentionally small to keep this localized.
- [ ] **CI action versions** — `actions/checkout@v4` / `setup-python@v5` / `setup-node@v4`
      target Node 20 and get auto-forced to Node 24. Bump when newer majors land.
      Non-blocking.

## Done

- [x] **Docker image is built in CI** (`docker` job) and the app is imported inside it.
      It was previously validated by review only, never actually built.
- [x] **`view source` link** in `web/app/page.tsx` now points at this repository
      instead of `https://github.com`.
- [x] **`web/.env.local.example` ships.** `web/.gitignore` had `.env*`, which excluded
      it, so the README's `cp .env.local.example .env.local` failed on a fresh clone.
- [x] **E2B pin corrected** to a version whose API matches the code (see the audit
      notes in `requirements.txt` and `sandbox/executor.py`).
