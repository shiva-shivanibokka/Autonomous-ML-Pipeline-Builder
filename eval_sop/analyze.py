"""
Summarise results/runs*.jsonl into results/summary.json and results/summary.md.

Unit of analysis for tests and CIs: the DATASET (per-dataset mean over seeds),
because seeds within a dataset are not independent. Paired comparisons use the
Wilcoxon signed-rank test (two-sided, scipy, zero-differences dropped by the
"wilcox" rule) and a percentile bootstrap (10,000 resamples of datasets,
RandomState(0)) for the mean paired difference.

Two treatments of failed runs are reported:
  completed  a dataset enters a pairwise comparison only if both arms
             completed every seed on it
  imputed    a failed run scores AUC 0.5 (a constant predictor), as in AMLB
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS = ["logreg", "lgbm", "xgb", "flaml11", "flaml", "sys_fixed", "sys_fixed_trainw", "sys_llm"]
# Two pre-declared families; Holm is applied within each family and failure mode.
LLM_PAIRS = [("sys_llm", "sys_fixed"), ("sys_llm", "flaml11"), ("sys_llm", "flaml")]
NONLLM_PAIRS = [("sys_fixed", "flaml"), ("sys_fixed", "flaml11"), ("sys_fixed", "lgbm"),
                ("sys_fixed", "xgb"), ("sys_fixed", "logreg"), ("flaml", "lgbm")]
# Sensitivity check, reported on its own (not part of the corrected family).
EXTRA_PAIRS = [("sys_fixed_trainw", "sys_fixed")]
PAIRS = LLM_PAIRS + NONLLM_PAIRS + EXTRA_PAIRS


def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rs = np.random.RandomState(seed)
    means = d[rs.randint(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    import sys

    global RES
    if len(sys.argv) > 1:  # optional results directory, e.g. eval_sop/results/robustness
        RES = Path(sys.argv[1]).resolve()
    runs = [json.loads(x) for f in sorted(RES.glob("runs*.jsonl"))
            for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Last record wins for a duplicated (arm, dataset, seed).
    key = {}
    for r in runs:
        key[(r["arm"], r["openml_id"], r["seed"])] = r
    runs = list(key.values())
    arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]
    dsets = sorted({r["openml_id"] for r in runs})
    names = {r["openml_id"]: r["dataset"] for r in runs}
    seeds = sorted({r["seed"] for r in runs})

    auc = defaultdict(dict)       # (arm, ds) -> {seed: auc or None}
    for r in runs:
        auc[(r["arm"], r["openml_id"])][r["seed"]] = r["auc"] if r.get("ok") else None

    def ds_mean(arm, ds, impute):
        """impute: False = failures excluded (None if any seed failed); True / "const" =
        a failed run scores 0.5; "fixed" = a failed run falls back to sys_fixed's AUC
        on the same dataset and seed (what a user gets by rerunning without the LLM)."""
        vals = auc.get((arm, ds), {})
        xs = []
        for s in seeds:
            v = vals.get(s)
            if v is None:
                if not impute:
                    return None
                v = auc.get(("sys_fixed", ds), {}).get(s) if impute == "fixed" else 0.5
                if v is None:
                    return None
            xs.append(v)
        return float(np.mean(xs)) if xs else None

    summary = {"n_datasets": len(dsets), "seeds": seeds, "arms": arms}

    # Per-dataset table.
    table = []
    for ds in dsets:
        row = {"openml_id": ds, "dataset": names[ds]}
        for a in arms:
            vals = [v for v in auc.get((a, ds), {}).values() if v is not None]
            nfail = sum(1 for v in auc.get((a, ds), {}).values() if v is None)
            row[a] = {"mean": float(np.mean(vals)) if vals else None,
                      "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                      "n_ok": len(vals), "n_fail": nfail}
        table.append(row)
    summary["per_dataset"] = table

    # Average rank (imputed means, so every arm has a value on every dataset).
    M = np.array([[ds_mean(a, ds, True) for a in arms] for ds in dsets], float)
    ranks = np.vstack([rankdata(-row) for row in M])
    summary["avg_rank_imputed"] = {a: float(ranks[:, i].mean()) for i, a in enumerate(arms)}
    if "sys_llm" in arms:  # ranks without the LLM arm, as quoted for the non-LLM comparison
        idx = [i for i, a in enumerate(arms) if a != "sys_llm"]
        rk = np.vstack([rankdata(-row) for row in M[:, idx]])
        summary["avg_rank_non_llm"] = {arms[i]: float(rk[:, j].mean()) for j, i in enumerate(idx)}
    MF = np.array([[ds_mean(a, ds, "fixed") if a == "sys_llm" else ds_mean(a, ds, True) for a in arms] for ds in dsets], float)
    summary["mean_auc_fallback_fixed"] = {a: float(MF[:, i].mean()) for i, a in enumerate(arms)}
    ok_ds = [ds for ds in dsets if all(ds_mean(a, ds, False) is not None for a in arms)]
    summary["datasets_all_arms_completed"] = ok_ds
    summary["mean_auc_failures_excluded"] = {
        a: float(np.mean([ds_mean(a, ds, False) for ds in ok_ds])) if ok_ds else None for a in arms}
    summary["mean_auc_imputed"] = {a: float(M[:, i].mean()) for i, a in enumerate(arms)}

    # Pairwise tests.
    tests = []
    for a, b in PAIRS:
        if a not in arms or b not in arms:
            continue
        modes = ("completed", "imputed", "fallback_fixed") if a == "sys_llm" else ("completed", "imputed")
        for mode in modes:
            imp = {"completed": False, "imputed": True, "fallback_fixed": "fixed"}[mode]
            pairs = [(ds_mean(a, ds, imp), ds_mean(b, ds, imp)) for ds in dsets]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            d = np.array([x - y for x, y in pairs])
            try:
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            lo, hi = boot_ci(d)
            tests.append({"a": a, "b": b, "mode": mode, "family": ("llm" if (a, b) in LLM_PAIRS else "sensitivity" if (a, b) in EXTRA_PAIRS else "non_llm"),
                          "n_datasets": len(d),
                          "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                          "ci95": [lo, hi], "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                          "ties": int((d == 0).sum()), "wilcoxon_p": p})
    # Holm-Bonferroni within each (family, failure mode).
    for key in {(t["family"], t["mode"]) for t in tests}:
        fam = sorted([t for t in tests if (t["family"], t["mode"]) == key and t["wilcoxon_p"] == t["wilcoxon_p"]],
                     key=lambda t: t["wilcoxon_p"])
        m, run_max = len(fam), 0.0
        for i, t in enumerate(fam):
            run_max = max(run_max, min(1.0, (m - i) * t["wilcoxon_p"]))
            t["holm_p"] = run_max
    for t in tests:
        t.setdefault("holm_p", None)
    summary["paired_tests"] = tests

    # Failures and wall time per arm.
    per_arm = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        walls = [r["wall_s"] for r in rs]
        per_arm[a] = {"runs": len(rs), "failed": sum(not r.get("ok") for r in rs),
                      "wall_median_s": float(np.median(walls)), "wall_mean_s": float(np.mean(walls)),
                      "wall_total_s": float(np.sum(walls)),
                      "errors": Counter((r.get("error") or "")[:90] for r in rs if not r.get("ok")).most_common(5)}
    # Candidate models that errored inside a completed system run (item 2: the
    # XGBoost target-coding bug). A run still "succeeds" with fewer candidates.
    cand = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a and r.get("ok") and r.get("models")]
        per = {}
        for r in rs:
            for name, m in r["models"].items():
                if m.get("error"):
                    per.setdefault(name, {"runs": 0, "datasets": set()})
                    per[name]["runs"] += 1
                    per[name]["datasets"].add(r["dataset"])
        if per:
            cand[a] = {k: {"runs": v["runs"], "datasets": sorted(v["datasets"])} for k, v in per.items()}
    summary["candidate_model_failures"] = cand
    summary["per_arm"] = per_arm

    # LLM-arm diagnostics.
    llm = [r for r in runs if r["arm"] == "sys_llm"]
    if llm:
        s0 = [r for r in llm if r["seed"] == seeds[0]]
        attempts = [a for r in s0 for a in [r.get("fe_attempts") or []]]
        summary["llm"] = {
            "model": llm[0].get("llm_model"),
            "llm_calls_total": int(sum(r.get("llm_calls", 0) for r in llm)),
            "llm_seconds_total": float(sum(r.get("llm_seconds", 0) for r in llm)),
            "planned_metric": dict(Counter((r.get("plan") or {}).get("primary_metric") for r in s0)),
            "planned_models": dict(Counter(",".join((r.get("plan") or {}).get("suggested_models") or []) for r in s0)),
            "winner": dict(Counter(r.get("winner") for r in llm if r.get("ok"))),
            "fe_attempts_per_dataset": [len(a) for a in attempts],
            "fe_first_try_success": sum(1 for a in attempts if a and a[0]["returncode_ok"] and a[0]["wrote_output"]),
            "fe_exit0_but_no_output_attempts": sum(1 for a in attempts for x in a if x["returncode_ok"] and not x["wrote_output"]),
            "fe_datasets_with_exit0_no_output": sum(1 for a in attempts if any(x["returncode_ok"] and not x["wrote_output"] for x in a)),
            "fe_failed_datasets": sum(1 for r in s0 if not r.get("ok") and "Feature Engineer" in (r.get("error") or "")),
            "orchestrator_failed": sum(1 for r in s0 if "Orchestrator" in (r.get("error") or "")),
            "same_split_as_baselines": dict(Counter(r.get("same_split_as_baselines") for r in llm if r.get("ok"))),
            "datasets_with_seed0_record": len(s0),
        }
    sf = [r for r in runs if r["arm"] == "sys_fixed" and r.get("ok")]
    if sf:
        summary["sys_fixed_winner"] = dict(Counter(r.get("winner") for r in sf))

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown.
    L = []
    L.append(f"Datasets: {len(dsets)}; seeds: {seeds}; metric: test ROC AUC (macro OvR for multiclass).\n")
    L.append("| dataset | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for row in table:
        cells = []
        for a in arms:
            c = row[a]
            if c["mean"] is None:
                cells.append(f"FAIL x{c['n_fail']}")
            else:
                sd = f" ± {c['std']:.3f}" if c["std"] is not None else ""
                fl = f" ({c['n_fail']} fail)" if c["n_fail"] else ""
                cells.append(f"{c['mean']:.3f}{sd}{fl}")
        L.append(f"| {row['dataset']} ({row['openml_id']}) | " + " | ".join(cells) + " |")
    L.append("| **mean AUC (fail = 0.5)** | " + " | ".join(f"{summary['mean_auc_imputed'][a]:.4f}" for a in arms) + " |")
    L.append("| **mean AUC, sys_llm failures -> sys_fixed fallback** | " + " | ".join(f"{summary['mean_auc_fallback_fixed'][a]:.4f}" for a in arms) + " |")
    L.append(f"| **mean AUC, failures excluded ({len(ok_ds)} datasets where every arm completed every seed)** | "
             + " | ".join(f"{summary['mean_auc_failures_excluded'][a]:.4f}" if ok_ds else "-" for a in arms) + " |")
    L.append(f"| **average rank among all {len(arms)} arms (1 = best; failed run = 0.5)** | "
             + " | ".join(f"{summary['avg_rank_imputed'][a]:.2f}" for a in arms) + " |")
    if "avg_rank_non_llm" in summary:
        L.append("| **average rank among the non-LLM arms only** | "
                 + " | ".join(f"{summary['avg_rank_non_llm'][a]:.2f}" if a in summary["avg_rank_non_llm"] else "-"
                              for a in arms) + " |")
    if "candidate_model_failures" in summary and summary["candidate_model_failures"]:
        L.append("| candidate models that errored inside completed runs | "
                 + " | ".join(
                     ("; ".join(f"{k} on {len(v['datasets'])} ds ({v['runs']} runs)"
                                for k, v in summary["candidate_model_failures"][a].items())
                      if a in summary["candidate_model_failures"] else "none")
                     for a in arms) + " |")
    L.append("| failed runs | " + " | ".join(f"{per_arm[a]['failed']}/{per_arm[a]['runs']}" for a in arms) + " |")
    L.append("| median wall time per run (s) | " + " | ".join(f"{per_arm[a]['wall_median_s']:.1f}" for a in arms) + " |")
    L.append("")
    L.append("Failure modes: completed = datasets where both arms completed every seed; imputed = failed run scores 0.5; "
             "fallback_fixed = failed sys_llm run takes sys_fixed's score. Holm is within family (llm / non_llm) and mode.")
    L.append("")
    L.append("| comparison | failures | n datasets | mean ΔAUC | 95% bootstrap CI | W/L/T | Wilcoxon p | Holm p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in tests:
        L.append(f"| {t['a']} − {t['b']} | {t['mode']} | {t['n_datasets']} | {t['mean_diff']:+.4f} | "
                 f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {t['wins']}/{t['losses']}/{t['ties']} | {t['wilcoxon_p']:.3g} | "
                 + (f"{t['holm_p']:.3g} |" if t["holm_p"] is not None else "- |"))
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # the table uses "Δ"; Windows consoles default to cp1252
    print("\n".join(L))


if __name__ == "__main__":
    main()
