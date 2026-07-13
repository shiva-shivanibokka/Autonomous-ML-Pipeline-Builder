# ADR 0001 — Split the frontend (Vercel) from the backend (Cloud Run)

- **Status:** Accepted
- **Date:** 2026-07

## Context

The app has two very different runtime profiles:

1. A **UI** — static assets + client-side interactivity, updated frequently.
2. A **pipeline backend** — multi-minute ML training jobs, an in-process thread
   pool, live per-run state, SHAP, and sandboxed execution of generated code.

The original build used a single Gradio process for both, deployed as one
container. The requirement is now a frontend on Vercel.

## Decision

Split into two independently deployed services:

- **Frontend → Vercel** (Next.js). Native host, global CDN, preview deploy per PR.
- **Backend → Google Cloud Run** (container). They communicate over HTTPS; the
  frontend reads the backend URL from `NEXT_PUBLIC_API_BASE_URL`, and the backend
  restricts CORS to the frontend origin.

Serverless functions (Vercel/Lambda) were rejected for the backend: training runs
exceed typical function timeouts, and the pipeline holds in-process state and spawns
work that doesn't fit a request/response function. Cloud Run runs the long-lived
container, scales to zero when idle (near-zero cost), and gives a clean
secrets/IAM story.

## Consequences

- **Positive:** each side scales and deploys on its own; the UI stack (React/Next)
  is a stronger, more legible artifact than Gradio; secrets stay server-side.
- **Cost:** two deploy targets and an explicit CORS/URL contract to keep in sync.
- **At 10× load:** Cloud Run adds instances. Because run state is per-instance
  (see ADR 0002's note on the store), scaling out means moving the `RunStore` to a
  shared backend (Postgres/Redis) and artifacts to object storage — the store
  interface is intentionally small to make that a localized change.
