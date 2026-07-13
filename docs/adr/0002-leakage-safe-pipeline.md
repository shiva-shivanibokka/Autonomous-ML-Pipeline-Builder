# ADR 0002 — Preprocessing lives in a leakage-safe sklearn Pipeline

- **Status:** Accepted
- **Date:** 2026-07

## Context

Originally the feature-engineer agent generated a preprocessing script that
imputed, scaled, and encoded the **entire dataset**, wrote a transformed CSV, and
the trainer split train/test *afterwards*. That leaks: scalers and encoders were
fit on rows that later became the test set, inflating every metric. Separately,
the "deployment" artifacts referenced a `model.pkl` that was never actually
written, so the generated API couldn't run.

## Decision

- Move imputation, scaling, and encoding into a scikit-learn `ColumnTransformer`
  that is composed with each estimator as a `Pipeline` and **fit on the training
  fold only** (holdout split + k-fold `cross_val_score`, which re-fits per fold).
- The feature-engineer agent now performs only leakage-safe, per-row structural
  transforms (datetime decomposition, dropping id/constant columns, derived
  ratios). It no longer fits any statistics.
- Persist the **whole fitted Pipeline** (preprocessing + model) as `model.pkl`
  plus a `feature_schema.json`, so the generated FastAPI/Docker bundle predicts on
  raw input with no separate preprocessing step to drift out of sync.
- Winner selection is deterministic (argmax on the primary metric); the LLM writes
  only the justification narrative.

## Consequences

- **Positive:** metrics reflect generalization; cross-validation gives error bars;
  the exported artifact is genuinely runnable; training/serving skew is designed
  out because one object holds both steps.
- **Trade-off:** the agent has less latitude to invent bespoke encodings, and CV
  adds compute (capped by a row threshold to keep demo runs fast).
