# Feature engineering

Prefer leakage-safe, per-row transforms that don't depend on other rows or the target:

- Datetime columns: extract year, month, day, day-of-week, is_weekend, hour. These are
  computed per row and never leak.
- Numeric interactions: ratios and differences between related columns (e.g.
  amount / balance, price - cost) often carry more signal than the raw columns.
- Drop columns that carry no signal: constant columns, exact duplicates, and identifier
  columns (row ids, UUIDs) that the model would only memorize.
- Log-transform strongly right-skewed positive numeric features to stabilize variance.
- Leave imputation, scaling, and categorical encoding to a downstream scikit-learn
  `Pipeline` fit on the training fold — doing them here, on the full dataset, leaks.
- Keep engineered features interpretable; you will need to explain them (and reproduce
  them at inference) later.
