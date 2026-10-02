"""Download the benchmark datasets from OpenML into eval_sop/data/ (git-ignored).

Datasets: binary and multiclass tasks from the OpenML-CC18 suite, 500-3,200
rows, so the full LLM arm fits a local-model budget. Each CSV keeps the
OpenML default target, renamed to nothing: the column name is preserved.
"""
import json
from pathlib import Path

import openml

# (OpenML dataset id, short name). All are members of OpenML-CC18 (suite 99).
DATASETS = [
    (31, "credit-g"),
    (37, "diabetes"),
    (1464, "blood-transfusion"),
    (1480, "ilpd"),
    (1494, "qsar-biodeg"),
    (1510, "wdbc"),
    (40994, "climate-model-simulation-crashes"),
    (1067, "kc1"),
    (29, "credit-approval"),
    (54, "vehicle"),
    (188, "eucalyptus"),
    (40982, "steel-plates-fault"),
    (23, "cmc"),
    (15, "breast-w"),
    (3, "kr-vs-kp"),
]

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
meta = []
for did, name in DATASETS:
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    X, y, cat, cols = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    df = X.copy()
    target = ds.default_target_attribute
    df[target] = y
    p = out / f"{did}_{name}.csv"
    df.to_csv(p, index=False)
    meta.append({"openml_id": did, "name": name, "version": ds.version, "target": target,
                 "n_rows": len(df), "n_features": X.shape[1], "n_classes": int(y.nunique()),
                 "n_categorical": int(sum(cat)), "pct_missing_cells": round(float(X.isna().mean().mean()) * 100, 2),
                 "csv": p.name})
    print(meta[-1], flush=True)
(Path(__file__).parent / "datasets.json").write_text(json.dumps(meta, indent=2))
