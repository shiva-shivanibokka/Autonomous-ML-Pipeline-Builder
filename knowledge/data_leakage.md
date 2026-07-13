# Avoiding data leakage

Data leakage is when information unavailable at prediction time influences training,
producing metrics that are too optimistic and a model that underperforms in production.

Common sources and fixes:
- Preprocessing fit on the full dataset before the train/test split. Fix: fit only on
  training data (use a `Pipeline` + `cross_val_score`, which re-fits per fold).
- Target leakage: a feature that is a proxy for, or computed from, the target (e.g. a
  column populated only after the outcome is known). Drop such features.
- Time leakage in temporal data: using future rows to predict the past. Split by time,
  not randomly, and build lag features only from past observations.
- Duplicate or near-duplicate rows spanning train and test. Deduplicate first.
- Group leakage: rows from the same entity (user, patient) in both train and test.
  Use grouped splitting when entities repeat.
