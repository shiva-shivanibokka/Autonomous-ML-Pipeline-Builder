"""Rebuild the six fix commits (a50e055..ba83f3a) from the base commit by ordered edits.

Provenance tool: shows that commits 1-6 are exactly a sequence of small edits
to 8987d6f and that applying them reproduces ba83f3a byte-for-byte.

Usage (from anywhere inside the repo): python eval_sop/repro/build_commits.py <outdir>
Writes <outdir>/<n>/<path> for each commit n (only files touched up to n).
"""
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]  # repo root
BASE, FINAL = "8987d6f", "ba83f3a"
OUT = Path(sys.argv[1])


def _show(rev, p):
    out = subprocess.run(["git", "-C", str(WT), "show", f"{rev}:{p}"], capture_output=True, check=True).stdout
    return out.decode("utf-8").replace("\r\n", "\n")


def head(p):
    return _show(BASE, p)


def final(p):
    return _show(FINAL, p)


def between(text, start, end):
    """Return text[start_idx : end_idx] where start/end are marker substrings."""
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


F = {p: final(p) for p in [
    "agents/state.py", "agents/orchestrator.py", "agents/model_trainer.py", "agents/evaluator.py",
    "agents/feature_engineer.py", "sandbox/executor.py", "tests/test_ml_pipeline.py", "tests/test_graph_e2e.py"]}

# ---------------------------------------------------------------- edits
# Each entry: (commit, path, old, new). Applied in list order to the running content.
E = []

# C1 plan propagation ---------------------------------------------------------
E += [
    (1, "agents/state.py", "class AgentState(TypedDict, total=False):",
     between(F["agents/state.py"], "class OrchestratorPlan(", "class AgentState(") + "class AgentState(TypedDict, total=False):"),
    (1, "agents/state.py", "    # ── Agent outputs ──────────────────────────────────────────────────────────\n",
     "    # ── Agent outputs ──────────────────────────────────────────────────────────\n"
     + between(F["agents/state.py"], "    # Must be a declared key", "    dataset_profile: Optional[DatasetProfile]")),
    (1, "agents/orchestrator.py",
     '            # Store plan details as tags so model_trainer can read them\n            "_orchestrator_plan": {',
     '            # A declared AgentState key. This used to be "_orchestrator_plan",\n'
     '            # which is not in the schema, so LangGraph dropped it and the trainer\n'
     '            # and evaluator always used their defaults (see agents/state.py).\n'
     '            "orchestrator_plan": {'),
    (1, "agents/model_trainer.py", 'plan = state.get("_orchestrator_plan") or {}', 'plan = state.get("orchestrator_plan") or {}'),
    (1, "agents/evaluator.py", 'plan = state.get("_orchestrator_plan") or {}', 'plan = state.get("orchestrator_plan") or {}'),
    (1, "tests/test_ml_pipeline.py", '"_orchestrator_plan": {', '"orchestrator_plan": {'),
]

# C2 orchestrator grounding + validation --------------------------------------
E += [
    (2, "agents/orchestrator.py", "def run_orchestrator(state: AgentState) -> dict:",
     between(F["agents/orchestrator.py"], "_KNOWN_MODELS = {", "def run_orchestrator(") + "def run_orchestrator(state: AgentState) -> dict:"),
    (2, "agents/orchestrator.py",
     "        # Build the prompt\n        dataset_profile = state.get(\"dataset_profile\") or {}\n",
     between(F["agents/orchestrator.py"], "        # Profile the CSV BEFORE planning.", "        user_prompt = (")),
    (2, "agents/orchestrator.py", '            f"Dataset summary:\\n"\n',
     '            f"Dataset summary:\\n"\n            f"  - All columns: {columns}\\n"\n'),
    (2, "agents/orchestrator.py",
     '            raise ValueError(f"Could not parse orchestrator plan from: {raw[:300]}")\n',
     '            raise ValueError(f"Could not parse orchestrator plan from: {raw[:300]}")\n\n'
     + between(F["agents/orchestrator.py"], "        # Fail loudly on a target", "        logs.append(f\"[{timestamp}] ORCHESTRATOR — Task type")
     .rstrip("\n") + "\n"),
    (2, "agents/orchestrator.py", "{', '.join(plan.suggested_models)}", "{', '.join(suggested_models)}"),
    (2, "agents/orchestrator.py", "Primary metric: {plan.primary_metric}", "Primary metric: {primary_metric}"),
    (2, "agents/orchestrator.py", '"primary_metric": plan.primary_metric,', '"primary_metric": primary_metric,'),
    (2, "agents/orchestrator.py", '"suggested_models": plan.suggested_models,', '"suggested_models": suggested_models,'),
]

# C3 FE fails loudly ------------------------------------------------------------
E += [
    (3, "sandbox/executor.py", '            "output_csv_path": output_csv_path or csv_path,\n',
     '            # No fallback to the raw input: an empty path means "nothing was\n'
     '            # written", which execute_with_retry turns into a loud failure when\n'
     '            # the caller requires an output.\n'
     '            "output_csv_path": output_csv_path,\n'),
    (3, "sandbox/executor.py", "        script_path = f.name\n\n    try:\n",
     "        script_path = f.name\n\n"
     "    # A stale output from an earlier attempt must not count as this attempt's.\n"
     "    if output_csv and Path(output_csv).exists():\n"
     "        os.unlink(output_csv)\n\n    try:\n"),
    (3, "sandbox/executor.py",
     '            "output_csv_path": (\n                output_csv if output_csv and Path(output_csv).exists() else csv_path\n            ),',
     '            # No silent fallback to the raw CSV (see execute_with_retry).\n'
     '            "output_csv_path": (\n                output_csv if output_csv and Path(output_csv).exists() else ""\n            ),'),
    (3, "sandbox/executor.py", '            "error_text": f"Timeout after {timeout}s",\n            "output_csv_path": csv_path,',
     '            "error_text": f"Timeout after {timeout}s",\n            "output_csv_path": "",'),
    (3, "sandbox/executor.py", "    timeout: int | None = None,\n) -> dict:",
     "    timeout: int | None = None,\n    require_output: bool = False,\n) -> dict:"),
    (3, "sandbox/executor.py", "        timeout:     Execution timeout in seconds.\n",
     "        timeout:     Execution timeout in seconds.\n"
     + between(F["sandbox/executor.py"], "        require_output: If True", "\n    Returns:")),
    (3, "sandbox/executor.py", '                "error_text": str(exc),\n                "output_csv_path": csv_path,\n            }\n',
     '                "error_text": str(exc),\n                "output_csv_path": "",\n            }\n\n'
     + between(F["sandbox/executor.py"], "        if result[\"success\"] and require_output", "        if result[\"success\"]:\n").rstrip("\n") + "\n"),
    (3, "sandbox/executor.py", '        "last_error": last_error,\n        "output_csv_path": csv_path,\n    }',
     '        "last_error": last_error,\n        "output_csv_path": "",\n    }'),
    (3, "agents/feature_engineer.py", "            max_retries=3,\n        )", "            max_retries=3,\n            require_output=True,\n        )"),
    (3, "agents/feature_engineer.py",
     '            "transformed_csv_path": result.get(\n                "output_csv_path", "/data/processed.csv"\n            ),',
     '            "transformed_csv_path": result["output_csv_path"],\n'
     '            # Sandbox attempts used, including self-corrections (1 = first try).\n'
     '            "attempts": result["attempts"],'),
    (3, "agents/state.py", "    transformed_csv_path: str  # Path inside E2B sandbox\n",
     "    transformed_csv_path: str  # Local path of the processed CSV (never the raw input)\n"
     "    attempts: int  # Sandbox attempts used, including self-corrections\n"),
]

# C4 stripped env + throwaway dir ----------------------------------------------
E += [
    (4, "sandbox/executor.py", "def _execute_subprocess(code: str, csv_path: str, timeout: int) -> dict:",
     between(F["sandbox/executor.py"], "# Environment variables the child", "def _execute_subprocess(")
     + "def _execute_subprocess(code: str, csv_path: str, timeout: int) -> dict:"),
    (4, "sandbox/executor.py",
     '    with tempfile.NamedTemporaryFile(\n        suffix=".py", mode="w", delete=False, encoding="utf-8"\n    ) as f:\n'
     '        f.write(code_with_path)\n        script_path = f.name\n',
     "    # A throwaway working directory per run: the script's cwd, HOME and TEMP,\n"
     "    # so relative writes land somewhere disposable rather than in the repo.\n"
     '    workdir = tempfile.mkdtemp(prefix="amlpb_exec_")\n'
     '    script_path = os.path.join(workdir, "generated.py")\n'
     '    with open(script_path, "w", encoding="utf-8") as f:\n'
     "        f.write(code_with_path)\n"),
    (4, "sandbox/executor.py", "            text=True,\n            timeout=timeout,\n        )",
     '            text=True,\n            encoding="utf-8",\n            errors="replace",\n            timeout=timeout,\n'
     "            cwd=workdir,\n            env=_sandbox_env(workdir),\n        )"),
    (4, "sandbox/executor.py", "    finally:\n        os.unlink(script_path)",
     "    finally:\n        import shutil\n\n        shutil.rmtree(workdir, ignore_errors=True)"),
]

# C5 seed plumbing ----------------------------------------------------------------
E += [
    (5, "agents/state.py", "    error: Optional[str]  # Set if any agent fails fatally\n",
     "    error: Optional[str]  # Set if any agent fails fatally\n"
     "    random_seed: int  # Seed for the holdout split, CV folds and model RNGs (default 42)\n"),
    (5, "agents/model_trainer.py", "def _make_model(model_name: str, task_type: str, scale_pos: float) -> Any:",
     "def _make_model(model_name: str, task_type: str, scale_pos: float, seed: int = 42) -> Any:"),
    ("5all", "agents/model_trainer.py", "random_state=42,", "random_state=seed,"),
    (5, "agents/model_trainer.py",
     "def _cross_validate(pipe: Pipeline, X_train, y_train, task_type: str) -> tuple[float, float, str]:",
     "def _cross_validate(\n    pipe: Pipeline, X_train, y_train, task_type: str, seed: int = 42\n) -> tuple[float, float, str]:"),
    ("5all", "agents/model_trainer.py", "shuffle=True, random_state=42)", "shuffle=True, random_state=seed)"),
    (5, "agents/model_trainer.py", "    provider: str,\n) -> tuple[str, ModelResult]:",
     "    provider: str,\n    seed: int = 42,\n) -> tuple[str, ModelResult]:"),
    (5, "agents/model_trainer.py", "_make_model(model_name, task_type, scale_pos)", "_make_model(model_name, task_type, scale_pos, seed)"),
    (5, "agents/model_trainer.py", "cv_mean, cv_std, cv_metric = _cross_validate(pipe, X_train, y_train, task_type)",
     "cv_mean, cv_std, cv_metric = _cross_validate(pipe, X_train, y_train, task_type, seed=seed)"),
    (5, "agents/model_trainer.py", '        feature_result = state.get("feature_result") or {}\n',
     '        feature_result = state.get("feature_result") or {}\n        seed = int(state.get("random_seed", 42))\n'),
    (5, "agents/model_trainer.py", '                    provider=state["provider"],\n                )',
     '                    provider=state["provider"],\n                    seed=seed,\n                )'),
]

# C6 CV-based selection on the primary metric ------------------------------------
E += [
    (6, "agents/model_trainer.py",
     "def _cross_validate(\n    pipe: Pipeline, X_train, y_train, task_type: str, seed: int = 42\n) -> tuple[float, float, str]:",
     between(F["agents/model_trainer.py"], "# sklearn scorer for each primary metric.", "def _cross_validate(")
     + "def _cross_validate(\n    pipe: Pipeline,\n    X_train,\n    y_train,\n    task_type: str,\n"
       "    primary_metric: str = \"\",\n    seed: int = 42,\n) -> tuple[float, float, str]:"),
    (6, "agents/model_trainer.py",
     '        return 0.0, 0.0, "skipped(n>cap)"\n    if task_type == "classification":\n        scorer = "f1_weighted"\n',
     '        return 0.0, 0.0, "skipped(n>cap)"\n'
     '    n_classes = len(np.unique(y_train)) if task_type == "classification" else 0\n'
     '    scorer = cv_scorer_for(primary_metric, task_type, n_classes)\n'
     '    if task_type == "classification":\n'),
    (6, "agents/model_trainer.py", '    else:\n        scorer = "r2"\n        splitter = KFold', '    else:\n        splitter = KFold'),
    (6, "agents/model_trainer.py", "    provider: str,\n    seed: int = 42,\n) -> tuple[str, ModelResult]:",
     '    provider: str,\n    primary_metric: str = "",\n    seed: int = 42,\n) -> tuple[str, ModelResult]:'),
    (6, "agents/model_trainer.py",
     "cv_mean, cv_std, cv_metric = _cross_validate(pipe, X_train, y_train, task_type, seed=seed)",
     "cv_mean, cv_std, cv_metric = _cross_validate(\n            pipe, X_train, y_train, task_type, primary_metric, seed\n        )"),
    (6, "agents/model_trainer.py",
     '            "suggested_models", ["lightgbm", "xgboost", "random_forest"]\n        )\n',
     '            "suggested_models", ["lightgbm", "xgboost", "random_forest"]\n        )\n'
     '        primary_metric = plan.get(\n            "primary_metric", "auc" if task_type == "classification" else "rmse"\n        )\n'),
    (6, "agents/model_trainer.py", '                    provider=state["provider"],\n                    seed=seed,',
     '                    provider=state["provider"],\n                    primary_metric=primary_metric,\n                    seed=seed,'),
    (6, "agents/state.py", "    primary_metric: str  # The metric used to pick the winner\n",
     "    primary_metric: str  # The metric used to pick the winner\n"
     '    selection_basis: str  # "cv" (CV on the training split) or "holdout" (row-cap fallback)\n'),
    (6, "agents/evaluator.py", "def _select_winner(",
     between(F["agents/evaluator.py"], "def _cv_is_valid(", "def _select_winner(") + "def _select_winner("),
    ("6sel", "agents/evaluator.py", None, None),  # replace _select_winner body .. JUSTIFICATION marker with final
    (6, "agents/evaluator.py", '            "primary_metric": primary_metric,\n            "shap_plot_path": shap_plot_path,',
     '            "primary_metric": primary_metric,\n            "selection_basis": selection_basis(model_results),\n'
     '            "shap_plot_path": shap_plot_path,'),
    (6, "tests/test_ml_pipeline.py", 'assert r["cv_metric"] == "f1_weighted"',
     'assert r["cv_metric"] == "roc_auc"  # CV scores the planned primary metric'),
]

# ---------------------------------------------------------------- e2e test file by commit
T = F["tests/test_graph_e2e.py"]
t_head = T[: T.index("def test_plan_reaches_trainer")]
t_plan = between(T, "def test_plan_reaches_trainer", "def test_orchestrator_sees_the_dataset_columns")
t_cols = between(T, "def test_orchestrator_sees_the_dataset_columns", "def test_feature_engineering_without_output")
t_fe = between(T, "def test_feature_engineering_without_output", "def test_generated_code_does_not_see_api_keys")
t_env = between(T, "def test_generated_code_does_not_see_api_keys", "def test_selection_uses_cv_not_holdout")
t_sel = T[T.index("def test_selection_uses_cv_not_holdout"):]

cv_block = (
    "    # CV scored the planned metric (f1 -> f1_weighted), not a hard-coded one.\n"
    "    for r in final[\"model_results\"].values():\n"
    "        assert r[\"cv_metric\"] == \"f1_weighted\"\n"
)
sel_line = '    assert ev["selection_basis"] == "cv"\n'
fe_block = t_plan[t_plan.index("    # The feature engineer's output"):]
plan_c1 = t_plan.replace(cv_block, "").replace(sel_line, "").replace(", on CV.", ".").replace(fe_block, "").rstrip("\n") + "\n\n\n"
plan_c3 = t_plan.replace(cv_block, "").replace(sel_line, "").replace(", on CV.", ".").rstrip("\n") + "\n\n\n"
plan_c6 = t_plan
E2E = {
    1: t_head + plan_c1,
    2: t_head + plan_c1 + t_cols,
    3: t_head + plan_c3 + t_cols + t_fe,
    4: t_head + plan_c3 + t_cols + t_fe + t_env,
    5: t_head + plan_c3 + t_cols + t_fe + t_env,
    6: t_head + plan_c6 + t_cols + t_fe + t_env + t_sel,
}
assert E2E[6] == T, "e2e reconstruction mismatch"

# ---------------------------------------------------------------- apply
cur = {p: head(p) for p in F if p != "tests/test_graph_e2e.py"}
touched = set()
for n in range(1, 7):
    for (c, p, old, new) in E:
        cn = int(str(c)[0])
        if cn != n:
            continue
        if c == "6sel":
            s = cur[p]
            i = s.index("def _select_winner(")
            j = s.index("# The LLM writes a human-readable justification")
            fi = F[p].index("def _select_winner(")
            fj = F[p].index("# The LLM writes a human-readable justification")
            cur[p] = s[:i] + F[p][fi:fj] + s[j:]
        elif str(c).endswith("all"):
            assert old in cur[p], (c, p, old)
            cur[p] = cur[p].replace(old, new)
        else:
            assert cur[p].count(old) == 1, (c, p, old[:80], cur[p].count(old))
            cur[p] = cur[p].replace(old, new)
        touched.add(p)
    d = OUT / str(n)
    for p in touched:
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_text(cur[p], encoding="utf-8", newline="\n")
    (d / "tests/test_graph_e2e.py").parent.mkdir(parents=True, exist_ok=True)
    (d / "tests/test_graph_e2e.py").write_text(E2E[n], encoding="utf-8", newline="\n")

bad = [p for p in cur if cur[p] != F[p]]
print("final mismatch:", bad)
for p in bad:
    import difflib
    print("".join(list(difflib.unified_diff(F[p].splitlines(True), cur[p].splitlines(True), "final", "built"))[:60]))
