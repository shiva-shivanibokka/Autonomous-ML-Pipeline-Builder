"""
Leakage audit of the LLM-generated feature-engineering code.

The feature engineer's script runs on the WHOLE CSV, before the trainer makes
its train/test split. The prompt asks for per-row ("leakage-safe") transforms
only, but nothing checks it (leakage_warnings is hard-coded to []). This audit
checks it empirically. For each dataset's final generated script (taken from
the sys_llm runs in results/runs.jsonl):

  full   run the script on the full CSV (what the system does)
  train  run it on the seed-0 TRAINING rows only
  perm   run it on the full CSV with the target column randomly permuted

and compares outputs:
  * cross-row dependence: the train rows of `full` differ from `train` (same
    rows, same order) => a feature of a training row depends on other rows,
    including test rows. Split into "schema" (different columns kept/created,
    e.g. a constant-column drop decided on all rows) and "values".
  * target dependence: any non-target column differs between `full` and `perm`
    => the features are computed from the label (target leakage).
  * row changes: the script dropped or added rows (breaks the row alignment
    the trainer's split relies on).

All scripts run through the system's own sandbox.executor._execute_subprocess:
a throwaway directory and an environment with no API keys.

Usage: python -m eval_sop.leakage_audit  -> results/leakage_audit.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in list(os.environ):
    if _k.endswith(("_API_KEY", "_TOKEN")):
        del os.environ[_k]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from eval_sop.bench import encode_y, split  # noqa: E402
from sandbox.executor import _execute_subprocess  # noqa: E402


def run(code: str, csv: Path) -> tuple[pd.DataFrame | None, str]:
    r = _execute_subprocess(code, str(csv), 120)
    if not r["success"] or not r["output_csv_path"]:
        return None, (r.get("stderr") or "no output")[-300:]
    return pd.read_csv(r["output_csv_path"]), ""


def frames_differ(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    bad = []
    for c in cols:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.append(c)
        elif not (x.astype(str) == y.astype(str)).all():
            bad.append(c)
    return bad


def audit_one(ds: dict, code: str, workdir: Path) -> dict:
    target = ds["target"]
    df = pd.read_csv(HERE / "data" / ds["csv"])
    y = encode_y(df[target])
    idx = np.arange(len(df))
    tr_idx, _, _, _ = split(idx, y, 0)
    tr_idx = np.sort(tr_idx)

    workdir.mkdir(parents=True, exist_ok=True)
    p_full, p_train, p_perm = workdir / "full.csv", workdir / "train.csv", workdir / "perm.csv"
    df.to_csv(p_full, index=False)
    df.iloc[tr_idx].to_csv(p_train, index=False)
    perm = df.copy()
    perm[target] = np.random.RandomState(0).permutation(perm[target].to_numpy())
    perm.to_csv(p_perm, index=False)

    out = {"openml_id": ds["openml_id"], "dataset": ds["name"]}
    full, e1 = run(code, p_full)
    train, e2 = run(code, p_train)
    permd, e3 = run(code, p_perm)
    if full is None or train is None or permd is None:
        out.update({"status": "script_failed", "errors": [e1, e2, e3]})
        return out

    out["rows_full_in_out"] = [len(df), len(full)]
    out["rows_train_in_out"] = [len(tr_idx), len(train)]
    out["row_change"] = len(full) != len(df) or len(train) != len(tr_idx)
    out["target_kept"] = target in full.columns
    cols_full, cols_train = list(full.columns), list(train.columns)
    out["schema_differs"] = set(cols_full) != set(cols_train)
    out["cols_only_full"] = sorted(set(cols_full) - set(cols_train))
    out["cols_only_train"] = sorted(set(cols_train) - set(cols_full))

    common = [c for c in cols_full if c in cols_train and c != target]
    if out["row_change"]:
        out["value_diff_cols"] = None  # rows not alignable
    else:
        out["value_diff_cols"] = frames_differ(full.iloc[tr_idx], train, common)

    # Target dependence (same rows, permuted label).
    if len(permd) == len(full) and set(permd.columns) == set(full.columns):
        feats = [c for c in full.columns if c != target]
        out["target_dependent_cols"] = frames_differ(full, permd, feats)
    else:
        out["target_dependent_cols"] = ["<schema or rows changed under label permutation>"]

    out["cross_row_dependence"] = bool(out["schema_differs"] or (out["value_diff_cols"] or []))
    out["target_leakage"] = bool(out["target_dependent_cols"])
    out["status"] = "ok"
    return out


def main():
    meta = {d["openml_id"]: d for d in json.loads((HERE / "datasets.json").read_text())}
    runs = [json.loads(x) for f in sorted((HERE / "results").glob("runs*.jsonl"))
            for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    codes = {}
    for r in runs:
        if r["arm"] == "sys_llm" and r.get("seed") == 0 and r.get("fe_code"):
            codes[r["openml_id"]] = r["fe_code"]
    results = []
    for did, code in sorted(codes.items()):
        res = audit_one(meta[did], code, HERE / "work" / "leak" / str(did))
        results.append(res)
        print(json.dumps(res), flush=True)
    (HERE / "results" / "leakage_audit.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
