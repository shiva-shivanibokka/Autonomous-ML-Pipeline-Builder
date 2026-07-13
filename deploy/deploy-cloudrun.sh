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
  --set-secrets "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,E2B_API_KEY=E2B_API_KEY:latest"

echo
echo "Deployed. Service URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
  --format 'value(status.url)'
echo
echo "Next: set NEXT_PUBLIC_API_BASE_URL to that URL in your Vercel project, and redeploy the frontend."
