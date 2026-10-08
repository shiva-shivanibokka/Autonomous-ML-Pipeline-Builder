"""Positive/negative controls for the leakage audit: does it flag known-leaky scripts?

Usage: python -m eval_sop.leakage_audit_controls -> results/leakage_audit_controls.json
"""
import json
from pathlib import Path

from eval_sop.leakage_audit import audit_one

HERE = Path(__file__).resolve().parent
HEAD = "import pandas as pd\nINPUT_CSV_PATH = '/data/input.csv'\nOUTPUT_CSV_PATH = '/data/processed.csv'\ndf = pd.read_csv(INPUT_CSV_PATH)\n"
TAIL = "df.to_csv(OUTPUT_CSV_PATH, index=False)\n"
CONTROLS = {
    # expected: no flags
    "per_row_ratio (safe)": "df['r'] = df['plas'] / (df['mass'] + 1)\n",
    # expected: cross-row dependence (statistic over all rows, incl. test)
    "global_standardize (leaky)": "df['plas'] = (df['plas'] - df['plas'].mean()) / df['plas'].std()\n",
    # expected: cross-row AND target dependence
    "target_mean_encode (leaky)": "df['age_te'] = df.groupby('age')['class'].transform(lambda s: (s == 'tested_positive').mean())\n",
    # expected: row change
    "drop_duplicates (row change)": "df = df.drop_duplicates(subset=['preg'])\n",
}
ds = next(d for d in json.loads((HERE / "datasets.json").read_text()) if d["openml_id"] == 37)
out = []
for name, body in CONTROLS.items():
    r = audit_one(ds, HEAD + body + TAIL, HERE / "work" / "leak_controls" / name.split()[0])
    r["control"] = name
    out.append(r)
    print(name, {k: r.get(k) for k in ("cross_row_dependence", "target_leakage", "row_change", "value_diff_cols", "target_dependent_cols")}, flush=True)
(HERE / "results" / "leakage_audit_controls.json").write_text(json.dumps(out, indent=2))
