# Preprocessing best practices

- Fit imputers, scalers, and encoders on the TRAINING split only, then transform
  train and test with the fitted objects. Fitting on the full dataset leaks test
  distribution information (means, category vocabularies) into training.
- Prefer median imputation for numeric columns (robust to outliers) and
  most-frequent imputation for categoricals.
- Standardize numeric features (zero mean, unit variance) for distance- and
  gradient-based models (logistic/linear regression, SVM, MLP). Tree models
  (LightGBM, XGBoost, RandomForest) do not require scaling.
- One-hot encode low-cardinality categoricals; set `handle_unknown="ignore"` so
  categories unseen at training time do not crash inference.
- Cap one-hot cardinality (e.g. top-N categories) to avoid an explosion of sparse
  columns on high-cardinality fields.
- Wrap the whole preprocessing + model as a single scikit-learn `Pipeline` so the
  exact transforms are reproduced at inference from raw input.
