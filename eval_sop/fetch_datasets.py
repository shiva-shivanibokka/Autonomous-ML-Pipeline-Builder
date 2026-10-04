"""Download the benchmark datasets from OpenML into eval_sop/data/ (git-ignored).

Datasets: binary and multiclass tasks from the OpenML-CC18 suite, 500-3,200
rows, so the full LLM arm fits a local-model budget. Each CSV keeps the
OpenML default target, renamed to nothing: the column name is preserved.
"""
import json
import sys
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

# Robustness set, added after the main results (review request): every other
# OpenML-CC18 dataset with 500-3,200 rows and <= 60 features. Selection rule
# for both sets is stated in RESULTS.md; this list was not pre-registered.
ROBUSTNESS = [
    (23381, "dresses-sales"),
    (1063, "kc2"),
    (6332, "cylinder-bands"),
    (11, "balance-scale"),
    (469, "analcatdata_dmft"),
    (50, "tic-tac-toe"),
    (307, "vowel"),
    (1068, "pc1"),
    (1462, "banknote-authentication"),
    (1049, "pc4"),
    (1050, "pc3"),
    (40975, "car"),
    (18, "mfeat-morphological"),
    (22, "mfeat-zernike"),
    (40984, "segment"),
]

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
robust = "--robustness" in sys.argv
meta = []
for did, name in (ROBUSTNESS if robust else DATASETS):
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
(Path(__file__).parent / ("datasets_robustness.json" if robust else "datasets.json")).write_text(
    json.dumps(meta, indent=2))
