# Model serving & deployment

- Serialize the ENTIRE pipeline (preprocessing + model) with joblib, not just the
  estimator. Inference then runs on raw input with no manual preprocessing to keep in
  sync — the #1 source of training/serving skew.
- Validate request payloads with a typed schema (Pydantic) so bad input fails with a
  clear 422 rather than a confusing 500 deep in prediction.
- Expose a health endpoint that reports whether the model actually loaded, so the
  platform can gate traffic until the service is ready.
- Emit prediction count, latency, and error-rate metrics; log an input hash (not raw
  PII) with each prediction for debuggability.
- Pin dependency versions and load the model once at startup (not per request).
- Return the model version with every prediction so you can trace which model produced
  a given output after a rollout.
