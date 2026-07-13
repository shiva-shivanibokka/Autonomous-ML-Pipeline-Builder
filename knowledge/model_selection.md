# Model selection & evaluation

- Compare models on a held-out test set AND cross-validation (report mean ± std). A
  single split can be lucky; CV shows stability.
- Pick the primary metric from the problem: AUC/F1 for imbalanced classification, RMSE
  or R² for regression, MAPE for demand/forecasting where relative error matters.
- Prefer the simplest model within ~1-2% of the best. A logistic/linear model that
  nearly matches a gradient-boosted ensemble is often the better production choice:
  faster inference, easier to explain, fewer failure modes.
- Gradient-boosted trees (LightGBM, XGBoost) are strong tabular baselines; always
  include them. RandomForest is a robust, low-tuning fallback.
- Keep model selection deterministic (argmax on the chosen metric). Use an LLM to
  explain the decision, not to make it.
- Log every run (params, metrics, artifacts) to an experiment tracker for
  reproducibility and comparison over time.
