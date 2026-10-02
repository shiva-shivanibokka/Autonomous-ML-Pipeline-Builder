"""
Benchmark harness: this system vs simple baselines vs FLAML on OpenML-CC18 tasks.

Arms (all see the same stratified 80/20 split of the same CSV for a seed):
  logreg      sklearn LogisticRegression behind the system's own preprocessor
  lgbm        LightGBM with library defaults behind the same preprocessor
  xgb         XGBoost with library defaults behind the same preprocessor
  flaml       FLAML AutoML, --flaml-budget seconds, metric = ROC AUC (OvR if multiclass)
  sys_llm     this system's LangGraph agents: orchestrator -> data analyst ->
              feature engineer (LLM code, run in a stripped subprocess) ->
              model trainer -> evaluator. LLM = local Ollama model.
  sys_fixed   the same trainer + evaluator nodes with every LLM step removed:
              fixed plan (lightgbm, xgboost, random_forest; select on CV AUC),
              no generated feature engineering, raw CSV.

Reported metric: test ROC AUC (binary) / macro one-vs-rest ROC AUC (multiclass),
computed by this harness from predict_proba on the held-out 20%, once per run.

Usage (from the repo root):
  python -m eval_sop.bench --arms logreg lgbm flaml sys_fixed sys_llm --seeds 0 1 2
Results append to eval_sop/results/runs.jsonl (one JSON object per arm x dataset x seed).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

# The generated code must never see a key, and neither should anything else here.
for _k in list(os.environ):
    if _k.endswith(("_API_KEY", "_TOKEN")):
        del os.environ[_k]
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./eval_sop/work/mlruns")
# The benchmark machine is shared and CPU-saturated; LightGBM/XGBoost/OpenMP at
# "all cores" oversubscribe badly (one default LightGBM fit took 339 s). Every
# arm runs with the same 1-thread-per-library cap (BENCH_THREADS) so wall
# times are comparable; 1 thread was the fastest setting measured here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_v] = os.environ.get("BENCH_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK = HERE / "work"
RESULTS = HERE / "results"


# ── helpers ───────────────────────────────────────────────────────────────────


def load(ds: dict) -> pd.DataFrame:
    return pd.read_csv(DATA / ds["csv"])


def encode_y(y: pd.Series) -> np.ndarray:
    return LabelEncoder().fit_transform(y.astype(str) if y.dtype == object else y)


def split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def score(y_test, proba) -> dict:
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        auc = roc_auc_score(y_test, proba[:, 1])
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    pred = np.asarray(proba).argmax(axis=1)
    return {"auc": float(auc), "balanced_accuracy": float(balanced_accuracy_score(y_test, pred))}


# ── baseline arms ─────────────────────────────────────────────────────────────


def arm_sklearn(df, target, seed, which):
    from agents.model_trainer import _build_preprocessor

    X = df.drop(columns=[target])
    y = encode_y(df[target])
    Xtr, Xte, ytr, yte = split(X, y, seed)
    if which == "logreg":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(max_iter=1000, random_state=seed)
    elif which == "xgb":
        from xgboost import XGBClassifier

        est = XGBClassifier(random_state=seed, n_jobs=int(os.environ["OMP_NUM_THREADS"]), verbosity=0)
    else:
        import lightgbm as lgb

        est = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    pipe = Pipeline([("prep", _build_preprocessor(Xtr)), ("model", est)])
    pipe.fit(Xtr, ytr)
    return score(yte, pipe.predict_proba(Xte)), {}


def arm_flaml(df, target, seed, budget):
    from flaml import AutoML

    X = df.drop(columns=[target]).copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = encode_y(df[target])
    Xtr, Xte, ytr, yte = split(X, y, seed)
    automl = AutoML()
    metric = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"
    automl.fit(Xtr, ytr, task="classification", metric=metric, time_budget=budget,
               seed=seed, verbose=0, log_file_name="", n_jobs=int(os.environ["OMP_NUM_THREADS"]))
    return score(yte, automl.predict_proba(Xte)), {
        "flaml_best_estimator": automl.best_estimator,
        "flaml_budget_s": budget,
    }


# ── system arms ───────────────────────────────────────────────────────────────


class _NoLLM:
    def invoke(self, *_a, **_k):
        raise RuntimeError("LLM disabled in this arm")


def _system_patches(planner, codegen):
    """Patches shared by both system arms (no MLflow, no SHAP, no narrative LLM)."""
    import pipeline.graph as g

    ps = [
        patch("agents.model_trainer.log_training_run", return_value=""),
        patch("agents.model_trainer.log_comparison_table"),
        # SHAP plotting is presentation only and slow (KernelExplainer); it does
        # not influence the model, the selection or the score.
        patch("agents.evaluator._run_shap", return_value=None),
        # The evaluator's LLM only writes a justification for an already-chosen
        # winner; disabled so it cannot cost LLM calls. Its fallback text is used.
        patch("agents.evaluator.get_llm", return_value=_NoLLM()),
        patch.object(g, "run_code_generator", lambda st: {}),
        patch.object(g, "run_deployment_agent", lambda st: {}),
    ]
    if planner is not None:
        ps += [
            patch("agents.orchestrator.get_llm", return_value=planner),
            patch("agents.data_analyst.get_llm", return_value=planner),
            patch("agents.feature_engineer.get_codegen_llm", return_value=codegen),
        ]
    return ps


class _Attempts:
    """Wraps the subprocess executor to record every attempt (incl. silent no-output ones)."""

    def __init__(self, real):
        self.real = real
        self.log: list[dict] = []

    def __call__(self, code, csv_path, timeout):
        t0 = time.time()
        r = self.real(code, csv_path, timeout)
        self.log.append({
            "returncode_ok": bool(r["success"]),
            "wrote_output": bool(r.get("output_csv_path")),
            "stderr_tail": (r.get("stderr") or "")[-400:],
            "seconds": round(time.time() - t0, 2),
        })
        return r


def _winner_score(final, target, seed):
    """Re-derive the system's own test split and score its winning pipeline."""
    ev = final["evaluation_result"]
    winner = ev["winner_model"]
    pipe = final["model_results"][winner]["model_object"]
    fr = final.get("feature_result") or {}
    path = fr.get("transformed_csv_path") or final["csv_path"]
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = encode_y(df[target])
    Xtr, Xte, ytr, yte = split(X, y, seed)
    return score(yte, pipe.predict_proba(Xte)), len(df), y


def arm_system(ds, df, seed, planner, codegen, use_llm):
    import pipeline.graph as g
    import sandbox.executor as ex
    from agents.evaluator import run_evaluator
    from agents.model_trainer import run_model_trainer

    target = ds["target"]
    wdir = WORK / ("sys_llm" if use_llm else "sys_fixed") / f"{ds['openml_id']}"
    wdir.mkdir(parents=True, exist_ok=True)
    csv = wdir / "input.csv"  # stable path => identical prompts across seeds
    shutil.copyfile(DATA / ds["csv"], csv)
    out_dir = wdir / f"out_seed{seed}"

    attempts = _Attempts(ex._execute_subprocess)
    extra: dict = {}
    ps = _system_patches(planner if use_llm else None, codegen)
    ps.append(patch("sandbox.executor._execute_subprocess", attempts))
    ps.append(patch("sandbox.executor.settings"))
    started = [p.start() for p in ps]
    try:
        s = started[-1]
        s.execution_backend = "subprocess"
        s.e2b_api_key = ""
        s.allow_local_exec = True
        s.sandbox_timeout_seconds = 120
        state = {
            "csv_path": str(csv),
            "business_problem": (
                f"Predict the '{target}' column of the '{ds['name']}' dataset. "
                f"This is a classification problem."
            ),
            "provider": "ollama-local",
            "api_key": "",
            "model_name": getattr(planner, "model", ""),
            "pipeline_id": f"{ds['openml_id']}-{seed}",
            "output_dir": str(out_dir),
            "random_seed": seed,
            "status": "running",
            "model_results": {},
            "logs": [],
        }
        if use_llm:
            final = g.build_graph().invoke(state)
        else:
            state.update({
                "dataset_profile": {"task_type": "classification", "target_column": target},
                "orchestrator_plan": {
                    "task_type": "classification",
                    "primary_metric": "auc",
                    "target_column": target,
                    "suggested_models": ["lightgbm", "xgboost", "random_forest"],
                },
                "feature_result": {},
            })
            state.update(run_model_trainer(state))
            if not state.get("error"):
                state.update(run_evaluator(state))
            final = state
    finally:
        for p in ps:
            p.stop()

    extra["fe_attempts"] = attempts.log
    extra["plan"] = final.get("orchestrator_plan")
    extra["logs_tail"] = (final.get("logs") or [])[-40:]
    fr = final.get("feature_result") or {}
    if fr:
        extra["fe_code"] = fr.get("preprocessing_code")
        extra["fe_created"] = fr.get("new_features_created")
        extra["fe_dropped"] = fr.get("dropped_columns")
    if final.get("error"):
        extra["error"] = final["error"]
        return None, extra
    ev = final["evaluation_result"]
    extra.update({
        "winner": ev["winner_model"],
        "ranking": ev["ranking"],
        "primary_metric": ev["primary_metric"],
        "selection_basis": ev.get("selection_basis"),
        "models": {
            n: {"cv_mean": r.get("cv_mean"), "cv_metric": r.get("cv_metric"),
                "holdout": r.get("metrics"), "error": r.get("error")}
            for n, r in final["model_results"].items()
        },
    })
    sc, n_rows, y_sys = _winner_score(final, target, seed)
    y_raw = encode_y(df[target])
    extra["processed_rows"] = n_rows
    extra["same_split_as_baselines"] = bool(n_rows == len(df) and np.array_equal(y_sys, y_raw))
    return sc, extra


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["logreg", "lgbm", "xgb", "flaml", "sys_fixed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--datasets", nargs="*", type=int, default=None, help="OpenML ids")
    ap.add_argument("--flaml-budget", type=int, default=60)
    ap.add_argument("--flaml-arm-name", default="flaml",
                    help="record FLAML runs under this arm name (e.g. flaml11 for an 11 s budget)")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint, e.g. Ollama or https://api.groq.com/openai/v1")
    ap.add_argument("--llm-model", "--llm", dest="llm", default="qwen2.5:7b",
                    help="planner model (orchestrator, data analyst)")
    ap.add_argument("--codegen-model", default=None,
                    help="feature-engineering/self-correction model (default: same as --llm-model)")
    ap.add_argument("--llm-key-file", default=None,
                    help="dotenv file to read the API key from, in-process (never printed or exported)")
    ap.add_argument("--llm-key-name", default=None, help="variable name inside --llm-key-file")
    ap.add_argument("--out", default=str(RESULTS / "runs.jsonl"))
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--cache", default=str(RESULTS / "llm_cache.jsonl"))
    args = ap.parse_args()

    sys.path.insert(0, str(HERE.parent))
    os.chdir(HERE.parent)
    RESULTS.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    meta = json.loads((HERE / "datasets.json").read_text())
    if args.datasets:
        meta = [d for d in meta if d["openml_id"] in args.datasets]

    from eval_sop.llm import CachedChat

    stats: dict = {}
    cache = Path(args.cache)
    # Mirrors core.providers: planner temperature 0 / 4096 tokens,
    # codegen temperature 0.1 / 8192 tokens.
    key = ""
    if args.llm_key_file and args.llm_key_name:
        from dotenv import dotenv_values

        key = dotenv_values(args.llm_key_file).get(args.llm_key_name) or ""
    planner = CachedChat(args.llm, 0.0, 4096, cache, "planner", stats, args.llm_base_url, key)
    codegen = CachedChat(args.codegen_model or args.llm, 0.1, 8192, cache, "codegen", stats,
                         args.llm_base_url, key)
    del key

    done = set()
    out = Path(args.out)
    if args.skip_done and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["openml_id"], r["seed"]))

    import flaml
    import lightgbm
    import sklearn
    import xgboost

    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                "flaml": flaml.__version__, "pandas": pd.__version__, "numpy": np.__version__}

    for ds in meta:
        df = load(ds)
        for seed in args.seeds:
            for arm in args.arms:
                if ((args.flaml_arm_name if arm == "flaml" else arm), ds["openml_id"], seed) in done:
                    continue
                before = dict(stats)
                t0 = time.time()
                rec_arm = args.flaml_arm_name if arm == "flaml" else arm
                rec = {"arm": rec_arm, "openml_id": ds["openml_id"], "dataset": ds["name"], "seed": seed}
                try:
                    if arm in ("logreg", "lgbm", "xgb"):
                        sc, extra = arm_sklearn(df, ds["target"], seed, arm)
                    elif arm == "flaml":
                        sc, extra = arm_flaml(df, ds["target"], seed, args.flaml_budget)
                    elif arm == "sys_fixed":
                        sc, extra = arm_system(ds, df, seed, None, None, use_llm=False)
                    elif arm == "sys_llm":
                        sc, extra = arm_system(ds, df, seed, planner, codegen, use_llm=True)
                    else:
                        raise ValueError(arm)
                    rec.update({"ok": sc is not None, **(sc or {}), **extra})
                except Exception as exc:  # a crash is a failed run, recorded as such
                    rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-2000:]})
                rec["wall_s"] = round(time.time() - t0, 2)
                rec["llm_calls"] = stats.get("llm_calls", 0) - before.get("llm_calls", 0)
                rec["llm_cache_hits"] = stats.get("cache_hits", 0) - before.get("cache_hits", 0)
                rec["llm_seconds"] = round(stats.get("llm_seconds", 0) - before.get("llm_seconds", 0), 2)
                rec["llm_model"] = (
                    {"planner": args.llm, "codegen": args.codegen_model or args.llm,
                     "base_url": args.llm_base_url} if arm == "sys_llm" else None)
                rec["versions"] = versions
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                print(f"{arm:9s} {ds['name']:32s} seed={seed} ok={rec['ok']} "
                      f"auc={rec.get('auc', float('nan')):.4f} wall={rec['wall_s']}s "
                      f"llm_calls={rec['llm_calls']} {rec.get('error', '')[:120]}", flush=True)


if __name__ == "__main__":
    main()
