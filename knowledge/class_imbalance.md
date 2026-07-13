# Handling class imbalance

When one class dominates (e.g. fraud, churn, rare disease), accuracy is misleading —
a model predicting the majority class scores high while being useless.

- Evaluate with metrics that respect imbalance: AUC, F1, precision/recall, PR-AUC.
  Avoid raw accuracy as the primary metric.
- Reweight the loss instead of resampling when possible: `class_weight="balanced"`
  (sklearn) or `scale_pos_weight = n_negative / n_positive` (LightGBM/XGBoost).
- If resampling, do it INSIDE cross-validation folds (train only), never before the
  split — otherwise synthetic/duplicated rows leak into the test set.
- Stratify the train/test split and the CV folds on the target so every fold keeps the
  class ratio.
- Consider the decision threshold: the default 0.5 is rarely optimal under imbalance;
  tune it against the business cost of false positives vs false negatives.
