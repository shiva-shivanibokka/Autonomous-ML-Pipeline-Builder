#!/usr/bin/env bash
# Deploy the FastAPI backend to Google Cloud Run from the repo's Dockerfile.
#
# Prereqs (one-time): see deploy/README.md — gcloud CLI, an authenticated project,
# enabled APIs, and secrets created in Secret Manager.
#
# Usage:
#   PROJECT_ID=my-proj FRONTEND_ORIGIN=https://my-app.vercel.app ./deploy/deploy-cloudrun.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ml-pipeline-api}"
FRONTEND_ORIGIN="${FRONTEND_ORIGIN:?set FRONTEND_ORIGIN (your Vercel URL, e.g. https://app.vercel.app)}"

echo "Deploying $SERVICE to $REGION in $PROJECT_ID ..."

# Mount only the secrets that exist. deploy/README.md marks OPENAI_API_KEY (and
# GROQ_API_KEY) optional, but --set-secrets fails the whole deploy when it names
# a secret that was never created — so following the runbook exactly used to
# break here. Callers supply their own key per request anyway; these are
# server-side fallbacks.
SECRETS=""
for NAME in ANTHROPIC_API_KEY OPENAI_API_KEY GROQ_API_KEY E2B_API_KEY; do
  if gcloud secrets describe "$NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
    SECRETS="${SECRETS:+$SECRETS,}${NAME}=${NAME}:latest"
  else
    echo "  note: secret $NAME not found — skipping"
  fi
done

if [ -z "$SECRETS" ]; then
  echo "error: no provider secrets found in $PROJECT_ID." >&2
  echo "       Create at least E2B_API_KEY — production refuses to boot without it." >&2
  echo "       See deploy/README.md step 2." >&2
  exit 1
fi

# Builds the Dockerfile via Cloud Build, then deploys. Secrets are mounted as env
# vars from Secret Manager; non-secret config is passed with --set-env-vars.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 4 \
  --max-instances 3 \
  --set-env-vars "APP_ENV=production,EXECUTION_BACKEND=e2b,ALLOWED_ORIGINS=${FRONTEND_ORIGIN},MLFLOW_TRACKING_URI=file:./mlruns" \
  --set-secrets "$SECRETS"

echo
echo "Deployed. Service URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
  --format 'value(status.url)'
echo
echo "Next: set NEXT_PUBLIC_API_BASE_URL to that URL in your Vercel project, and redeploy the frontend."
