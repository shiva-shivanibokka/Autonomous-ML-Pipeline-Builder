# RESULTS: evaluation of the Autonomous ML Pipeline Builder (branch `sop-eval`)

Status: **complete for the planned scope.** Done and committed: the plumbing
fixes, the mocked-LLM end-to-end test, all non-LLM arms, the LLM arm (local
`qwen2.5:7b`, 15 datasets × 3 seeds) and the leakage audit.

**Headline:**
- With a 7B local model, the full agent pipeline **failed on 27 of 45 runs**
  (9 of 15 datasets, every seed).
- On the 6 datasets where it completed, it was not better than the same
  pipeline with the LLM steps removed: Δ = −0.012 AUC, Holm p = 0.47.
- None of the 9 generated feature-engineering scripts that ran was flagged for
  leakage. Positive controls show the audit detects leakage when it is present.

Base commit: `8987d6f` (main). All work is on `sop-eval`. Nothing was pushed.

---

## 1. Setup

**Datasets.** 15 classification datasets from the OpenML-CC18 suite (suite 99),
540–3,196 rows, 4–41 features, 2–7 classes. Five have categorical features and
three have missing values. The OpenML default target is used. IDs, versions and
shapes are in `eval_sop/datasets.json`. Fetch them with
`python eval_sop/fetch_datasets.py` (the CSVs are git-ignored).

| OpenML id | name | rows | feats | classes |
|---|---|---|---|---|
| 31 | credit-g | 1000 | 20 | 2 |
| 37 | diabetes | 768 | 8 | 2 |
| 1464 | blood-transfusion | 748 | 4 | 2 |
| 1480 | ilpd | 583 | 10 | 2 |
| 1494 | qsar-biodeg | 1055 | 41 | 2 |
| 1510 | wdbc | 569 | 30 | 2 |
| 40994 | climate-model-simulation-crashes | 540 | 18 | 2 |
| 1067 | kc1 | 2109 | 21 | 2 |
| 29 | credit-approval | 690 | 15 | 2 |
| 54 | vehicle | 846 | 18 | 4 |
| 188 | eucalyptus | 736 | 19 | 5 |
| 40982 | steel-plates-fault | 1941 | 27 | 7 |
| 23 | cmc | 1473 | 9 | 3 |
| 15 | breast-w | 699 | 9 | 2 |
| 3 | kr-vs-kp | 3196 | 36 | 2 |

**Protocol.** Seeds 0, 1 and 2. For each seed every arm gets the same
stratified 80/20 split (`train_test_split(test_size=0.2, random_state=seed,
stratify=y)`). That is also the split the system's own trainer makes. The
harness scores the test set once, from `predict_proba`, using ROC AUC (binary)
or macro one-vs-rest ROC AUC (multiclass). Balanced accuracy is also logged in
the raw records.

**Arms.**

| arm | what it is |
|---|---|
| `logreg` | `LogisticRegression(max_iter=1000)` behind the system's own preprocessor (median impute + scale, most-frequent impute + one-hot) |
| `lgbm` | LightGBM with library defaults, same preprocessor |
| `xgb` | XGBoost with library defaults, same preprocessor |
| `flaml11` | FLAML AutoML, `time_budget=11 s`, metric `roc_auc`/`roc_auc_ovr`. The budget matches the median wall time of `sys_fixed` (11.1 s). |
| `flaml` | FLAML AutoML, `time_budget=60 s` (about 5x the system's median time) |
| `sys_fixed` | **This system with every LLM step removed.** It runs the real `run_model_trainer` and `run_evaluator` nodes with the fixed plan the trainer uses by default (lightgbm, xgboost, random_forest; primary metric AUC). There is no generated feature engineering; the input is the raw CSV. The winner is chosen on 5-fold CV AUC on the training split. |
| `sys_llm` | **The full agent graph with a real LLM**: orchestrator → data analyst → feature engineer (LLM-written script, run in a stripped-env throwaway dir, up to 3 self-correction attempts) → trainer → evaluator. LLM: `qwen2.5:7b` (Ollama tag, digest `845dbda0ea48`, Q4_K_M) for both planner and codegen, through the local Ollama 0.34.4 server at `:11434`, one request at a time. Settings: native `/api/chat`, `num_ctx=8192`, planner temperature 0, codegen temperature 0.1, `seed=0`, `num_predict ≤ 4096`. |

In the system arms, MLflow logging, SHAP plotting and the evaluator's
justification LLM call are patched out. None of them affects the model, the
selection or the score. SHAP does not run in this environment anyway (Numba
does not support NumPy 2.5).

**Environment.** Windows 11, Python 3.12.3 (Anaconda base) plus a scratch venv
with `--system-site-packages` that adds FLAML and openml. Versions: sklearn
1.8.0, lightgbm 4.6.0, xgboost 3.2.0, flaml 2.3.6, pandas 2.3.3, numpy 2.5.3,
langgraph 1.1.6. These are newer than `requirements.txt` pins; the repo had no
`.venv` to reuse. Thread caps: `OMP_NUM_THREADS = OPENBLAS = MKL =
LOKY_MAX_CPU_COUNT = 1`, FLAML `n_jobs=1`. The machine was shared, and with
all-cores defaults one LightGBM fit took 339 s.

**Reproduce:**
```bash
python eval_sop/fetch_datasets.py
python -m eval_sop.bench --arms logreg lgbm sys_fixed --seeds 0 1 2 --out eval_sop/results/runs_fixed_and_simple.jsonl
python -m eval_sop.bench --arms xgb --seeds 0 1 2 --out eval_sop/results/runs_xgb.jsonl
python -m eval_sop.bench --arms flaml --flaml-budget 60 --seeds 0 1 2 --out eval_sop/results/runs_flaml.jsonl
python -m eval_sop.bench --arms flaml --flaml-budget 11 --flaml-arm-name flaml11 --seeds 0 1 2 --out eval_sop/results/runs_flaml11.jsonl
python -m eval_sop.bench --arms sys_llm --seeds 0 1 2 --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5:7b --ollama-num-ctx 8192 --out eval_sop/results/runs_sys_llm.jsonl
python -m eval_sop.leakage_audit            # -> results/leakage_audit.json
python -m eval_sop.leakage_audit_controls   # -> results/leakage_audit_controls.json
python -m eval_sop.analyze      # -> eval_sop/results/summary.{md,json}
python eval_sop/repro/repro_bugs.py <code root>   # bug reproductions
```
Raw per-run records (one JSON object per arm × dataset × seed, including the
CV scores of every candidate, the winner, wall time and versions) are in
`eval_sop/results/runs_*.jsonl`. Every LLM prompt and response (61 calls) is in
`eval_sop/results/llm_cache.jsonl`. Re-running the `sys_llm` command with that
file in place replays the LLM arm with no LLM calls.

**Determinism check.** The `logreg`/`lgbm`/`sys_fixed` arms were first run
across two days with a kill in between
(`results/archive_runs_fixed_and_simple_split_across_days.jsonl`). They were
then re-run in one session. All 135 test AUCs were bit-identical
(`results/determinism_check.txt`). Only the wall times differ, so the reported
wall times come from the single-session re-run.

---

## 2. Results (non-LLM arms; n = 15 datasets × 3 seeds = 45 runs per arm)

Cells show the per-dataset mean ± std over the 3 seeds of test ROC AUC.

| dataset | logreg | lgbm | xgb | flaml11 | flaml (60 s) | sys_fixed |
|---|---|---|---|---|---|---|
| kr-vs-kp (3) | 0.995 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| breast-w (15) | 0.995 ± 0.006 | 0.987 ± 0.007 | 0.987 ± 0.006 | 0.992 ± 0.007 | 0.991 ± 0.006 | 0.989 ± 0.007 |
| cmc (23) | 0.695 ± 0.029 | 0.701 ± 0.024 | 0.702 ± 0.021 | 0.746 ± 0.036 | 0.741 ± 0.040 | 0.740 ± 0.030 |
| credit-approval (29) | 0.943 ± 0.002 | 0.940 ± 0.018 | 0.937 ± 0.023 | 0.945 ± 0.021 | 0.947 ± 0.020 | 0.949 ± 0.022 |
| credit-g (31) | 0.779 ± 0.065 | 0.784 ± 0.045 | 0.797 ± 0.058 | 0.768 ± 0.068 | 0.773 ± 0.073 | 0.793 ± 0.070 |
| diabetes (37) | 0.831 ± 0.047 | 0.805 ± 0.029 | 0.794 ± 0.032 | 0.820 ± 0.031 | 0.836 ± 0.017 | 0.829 ± 0.033 |
| vehicle (54) | 0.948 ± 0.005 | 0.930 ± 0.006 | 0.931 ± 0.012 | 0.934 ± 0.003 | 0.928 ± 0.014 | 0.933 ± 0.004 |
| eucalyptus (188) | 0.917 ± 0.011 | 0.905 ± 0.005 | 0.906 ± 0.008 | 0.914 ± 0.008 | 0.914 ± 0.008 | 0.914 ± 0.011 |
| kc1 (1067) | 0.807 ± 0.005 | 0.801 ± 0.013 | 0.807 ± 0.019 | 0.818 ± 0.007 | 0.847 ± 0.015 | 0.834 ± 0.005 |
| blood-transfusion (1464) | 0.752 ± 0.032 | 0.707 ± 0.051 | 0.702 ± 0.031 | 0.699 ± 0.033 | 0.705 ± 0.042 | 0.724 ± 0.051 |
| ilpd (1480) | 0.780 ± 0.033 | 0.745 ± 0.055 | 0.723 ± 0.050 | 0.730 ± 0.055 | 0.729 ± 0.034 | 0.759 ± 0.055 |
| qsar-biodeg (1494) | 0.921 ± 0.022 | 0.930 ± 0.013 | 0.927 ± 0.004 | 0.919 ± 0.016 | 0.928 ± 0.011 | 0.927 ± 0.014 |
| wdbc (1510) | 0.993 ± 0.005 | 0.989 ± 0.002 | 0.991 ± 0.001 | 0.990 ± 0.003 | 0.992 ± 0.003 | 0.989 ± 0.005 |
| steel-plates-fault (40982) | 0.930 ± 0.003 | 0.964 ± 0.010 | 0.961 ± 0.011 | 0.959 ± 0.002 | 0.960 ± 0.005 | 0.965 ± 0.008 |
| climate-model-sim.-crashes (40994) | 0.978 ± 0.005 | 0.973 ± 0.007 | 0.970 ± 0.008 | 0.978 ± 0.019 | 0.983 ± 0.012 | 0.970 ± 0.008 |
| **mean AUC** | 0.8843 | 0.8773 | 0.8757 | 0.8807 | 0.8849 | **0.8875** |
| **average rank (1 = best)** | 3.00 | 4.20 | 4.27 | 3.77 | 2.90 | **2.87** |
| failed runs | 0/45 | 0/45 | 0/45 | 0/45 | 0/45 | 0/45 |
| median wall time / run | 0.03 s | 0.12 s | 0.11 s | 11.2 s | 60.3 s | 11.1 s |

Paired comparisons use per-dataset means, so the unit is the dataset (n = 15).
Each row reports a two-sided Wilcoxon signed-rank test, a percentile bootstrap
95% CI of the mean difference (10,000 resamples of datasets), and a Holm
correction over the 6 comparisons. No run failed, so "failures imputed" equals
"completed only".

| comparison | mean ΔAUC | 95% CI | wins/losses | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| sys_fixed − lgbm (default) | +0.0102 | [+0.0041, +0.0166] | 12/3 | 0.0067 | **0.040** |
| sys_fixed − xgb (default) | +0.0118 | [+0.0046, +0.0194] | 12/3 | 0.0125 | 0.062 |
| sys_fixed − flaml11 (equal time) | +0.0068 | [+0.0013, +0.0128] | 9/6 | 0.107 | 0.43 |
| sys_fixed − flaml (60 s) | +0.0026 | [−0.0029, +0.0088] | 7/8 | 0.762 | 1.0 |
| sys_fixed − logreg | +0.0032 | [−0.0060, +0.0132] | 7/8 | 0.804 | 1.0 |
| flaml (60 s) − lgbm | +0.0076 | [−0.0007, +0.0168] | 9/6 | 0.229 | 0.69 |

`sys_fixed` chose random_forest 26 times, xgboost 12 times and lightgbm 7 times
out of 45 runs.

### What the numbers do and don't support

**They support:**
- With the LLM steps off, the system's fixed trainer and evaluator (3 tuned
  GBM/RF candidates, chosen by 5-fold CV AUC) beat default LightGBM by
  +0.010 AUC on average over 15 CC18 datasets. That result survives Holm
  correction (p = 0.040).
- The same pipeline beat default XGBoost by +0.012 AUC, but that does not
  survive Holm correction (Holm p = 0.062).
- The fixed pipeline is statistically indistinguishable from FLAML at an equal
  ~11 s budget, from FLAML at 60 s, and from plain logistic regression. On these
  small tabular datasets, logistic regression is a strong baseline.
- The pipeline is reliable on this suite: 0/45 failures, and results are
  bit-for-bit reproducible.

**They do NOT support:**
- Any claim about the LLM agents. These rows describe the LLM-free path only;
  the LLM arm is in section 2b.
- "Beats AutoML." The system ties FLAML. The +0.007 point estimate against
  `flaml11` has a CI that excludes 0, but its Wilcoxon p is 0.107 before
  correction and 0.43 after.
- Any statement about large datasets, regression or time series. Only small
  (<3.2k rows) classification datasets were tested.

### Threats to validity

- **Small n.** 15 datasets and 3 seeds. A paired test over 15 datasets has
  little power for differences around 0.005 AUC.
- **Single holdout per seed.** Per-dataset std is large on the smallest
  datasets (credit-g, blood-transfusion, ilpd: ±0.03–0.07), larger than most
  between-arm differences.
- **Budget asymmetry.** `sys_fixed` is not time-budgeted; it trains a fixed
  3-model × 5-fold set. `flaml11` approximates equal time using the *median*;
  per dataset the times differ. The default-GBM baselines get far less compute
  than `sys_fixed`, which also runs 3 tuned models plus CV selection. Part of
  the "+0.010 over LightGBM" is simply "more models and CV selection".
- **Single-threaded, shared machine.** All arms were capped at 1 thread. FLAML
  is normally run multi-core and could do more within the same wall time.
  Wall times are from a loaded laptop and are only comparable within this run.
- **Library versions** are newer than the repo pins (see Setup). The system was
  evaluated as it runs here, not under its pinned stack.
- **Selection on the suite.** The datasets were picked by hand from CC18
  (small/medium, mix of binary/multiclass and cat/missing) before any results
  were seen, and none were dropped afterwards. They are still not a random
  sample of CC18.
- **`sys_fixed` is the post-fix system.** It selects on CV, after the fix in
  commit `ba83f3a`. The original code selected on test AUC, which would inflate
  its reported numbers. This harness always scores the test set separately, so
  no arm here selects on test.


## 2b. LLM arm (`sys_llm`) and leakage audit

n = 15 datasets × 3 seeds = 45 runs. **27/45 runs failed (60%)**: 9 datasets
failed on every seed and 6 completed on every seed. Failures are counted as
failures; nothing was rerun or cherry-picked.

**Why runs failed** (seed-0 records, `results/runs_sys_llm.jsonl`):

| failure | datasets | cause (from the recorded stderr/logs) |
|---|---|---|
| Feature engineering failed after 3 attempts | ilpd, wdbc, cmc, vehicle | `KeyError` on a column the script had dropped itself or invented (`ID_column`, `ID`, `V1`, `Wifes_age`) |
| | qsar-biodeg | `IndexError` from a boolean mask built on the wrong axis |
| | diabetes | attempt 1 exited 0 without writing output; attempts 2–3 hit the 120 s sandbox timeout |
| FE "succeeded" but every model then failed | kc1, credit-approval | the generated ratio features produced `inf`, and the trainer's imputer/scaler rejects `inf` |
| | kr-vs-kp | the script dropped every feature column (shape `(2556, 0)`) |

The plan never failed (0/15 orchestrator failures). The LLM always chose
`auc` and 4–5 models (lightgbm, xgboost, random_forest, mlp, plus
logistic_regression once). Thanks to fix 1, the trainer actually trained these
models.

**Silent-fallback counterfactual (fix 3).** On 4 of 15 datasets (credit-g,
diabetes, credit-approval, vehicle) a generated script exited 0 on its first
attempt **without writing the processed CSV**. The original code
(`8987d6f`) would have accepted that attempt and trained on the raw CSV while
reporting feature-engineering success. With fix 3 those attempts go back to
self-correction. Credit-g recovered; diabetes, vehicle and credit-approval
then failed visibly. So the original code would have "completed" more runs,
but on 4 datasets its feature-engineering step would have been a no-op it
reported as a success.

**AUC, three ways of handling failures** (per-dataset means over seeds;
the full table is in `results/summary.md`):

| | sys_llm | sys_fixed | flaml11 | flaml (60 s) |
|---|---|---|---|---|
| failure rate | **27/45 (60%)** | 0/45 | 0/45 | 0/45 |
| mean AUC, failures excluded (6 datasets where every arm completed) | 0.8809 | 0.8924 | 0.8850 | 0.8877 |
| mean AUC, failed sys_llm run falls back to sys_fixed's score (15 datasets) | 0.8829 | 0.8875 | 0.8807 | 0.8849 |
| mean AUC, failed run scores 0.5 (15 datasets) | 0.6524 | 0.8875 | 0.8807 | 0.8849 |

On the 6 completed datasets, the per-dataset AUC differences between sys_llm
and sys_fixed were: credit-g +0.001, breast-w −0.004, eucalyptus 0.000,
steel-plates −0.001, climate −0.005, blood-transfusion −0.061.

**Paired tests** (Wilcoxon signed-rank, two-sided; bootstrap 95% CI; Holm over
the 3 pre-declared LLM comparisons within each failure mode):

| comparison | failure handling | n datasets | mean ΔAUC | 95% CI | W/L/T | p | Holm p |
|---|---|---|---|---|---|---|---|
| sys_llm − sys_fixed | excluded | 6 | −0.0115 | [−0.0315, −0.0004] | 1/5/0 | 0.156 | 0.469 |
| sys_llm − sys_fixed | fallback to sys_fixed | 15 | −0.0046 | [−0.0130, −0.0000] | 1/5/9 | 0.116 | 0.348 |
| sys_llm − sys_fixed | fail = 0.5 | 15 | −0.2351 | [−0.3333, −0.1365] | 1/14/0 | 0.0003 | **0.0009** |
| sys_llm − flaml11 | excluded | 6 | −0.0041 | [−0.0194, +0.0107] | 2/4/0 | 0.562 | 1.0 |
| sys_llm − flaml11 | fallback to sys_fixed | 15 | +0.0022 | [−0.0056, +0.0096] | 8/7/0 | 0.599 | 1.0 |
| sys_llm − flaml11 | fail = 0.5 | 15 | −0.2283 | [−0.3273, −0.1287] | 2/13/0 | 0.0012 | **0.0023** |
| sys_llm − flaml (60 s) | excluded | 6 | −0.0068 | [−0.0232, +0.0079] | 2/4/0 | 0.562 | 1.0 |
| sys_llm − flaml (60 s) | fallback to sys_fixed | 15 | −0.0020 | [−0.0101, +0.0056] | 6/9/0 | 0.524 | 1.0 |
| sys_llm − flaml (60 s) | fail = 0.5 | 15 | −0.2326 | [−0.3326, −0.1321] | 2/13/0 | 0.0012 | **0.0023** |

With n = 6, the smallest two-sided Wilcoxon p attainable is 0.031, so the
"excluded" rows cannot reach significance after correction regardless of
effect size.

Cost and time: 61 LLM calls in total, 567 s of LLM time. Seeds 1–2 reused the
seed-0 LLM outputs from the cache. Median wall time was 9.2 s per run, but
seed-0 runs took 22–296 s.

**Leakage audit** (`results/leakage_audit.json`). The audit covered the 9
final scripts that ran successfully: the 6 completed datasets plus kc1,
credit-approval and kr-vs-kp. It ran each script on (a) the full CSV, (b) the
seed-0 training rows only, and (c) the full CSV with a permuted label, then
compared the outputs.

| check | flagged |
|---|---|
| cross-row dependence (train-row features differ between (a) and (b), or the column set differs) | 0/9 |
| target dependence (any feature changes when the label is permuted) | 0/9 |
| rows added or removed | 0/9 |

**Controls** (`results/leakage_audit_controls.json`, on diabetes) show the
audit is sensitive. A per-row ratio was not flagged. Global standardisation
was flagged as cross-row. Target-mean encoding was flagged as cross-row and
target-dependent. `drop_duplicates` was flagged as a row change.

Limits of the audit:
- It only covers scripts that ran. The 6 FE-failed datasets produced no output
  to audit.
- It detects a data-dependent decision only if that decision differs between
  the full and train-only data. The kr-vs-kp script dropped every column on
  both, so it was not flagged even though the drop was data-driven.
- Leakage rate: **0/9 scripts (95% Clopper–Pearson upper bound 34%)**.

### What the LLM-arm numbers support / don't support

- **Supported:**
  - With `qwen2.5:7b` as both planner and code generator, the agent pipeline
    is unreliable: 60% of runs failed, all of them traceable to the generated
    preprocessing code or to its downstream effects.
  - Where it did complete, there is no evidence that the LLM steps improve AUC
    over the same pipeline without them. The point estimate is slightly
    negative, driven mostly by blood-transfusion.
  - None of the generated scripts that ran leaked by the audit's tests.
- **Not supported:**
  - Any claim about stronger LLMs. The project's intended models are
    Claude/GPT-4o-class and were not run (no free tier was available).
    A larger model may fail far less often.
  - Any claim that the LLM steps *hurt* on completed runs; the effect is not
    significant.
  - A leakage rate for the scripts that failed.

### Additional threats to validity (LLM arm)

- **One model, one sample.** One LLM (7B, 4-bit) and one sampled script per
  dataset. Seeds 1–2 reuse the seed-0 LLM outputs, so the seed variance is
  split variance, not LLM-sampling variance. The 0/45 vs 27/45 failure counts
  are really 0/15 vs 9/15 independent LLM outcomes.
- **Settings.** The 8192-token context and the `num_predict ≤ 4096` cap were
  imposed by the run constraints. The project's codegen default asks for 8192
  output tokens. All 61 recorded responses ended with `done_reason: stop` (at most 801 output and 1,247 prompt tokens), so neither cap was hit (`done_reason` in the
  cache).
- **Sandbox timeout.** It is the system default, 120 s; one failure (diabetes)
  was a timeout.
- **System differences.** `sys_llm` trains whatever the planner chose (4–5
  models including `mlp`); `sys_fixed` trains 3. So the comparison is
  "LLM pipeline vs fixed pipeline", not "feature engineering on vs off" in
  isolation.

---

## 3. Change log (one entry per code change on `sop-eval`)

Every bug fix was first reproduced on the original commit. The reproduction
script is `eval_sop/repro/repro_bugs.py`; its outputs are
`repro/repro_original_8987d6f.txt` and `repro/repro_branch.txt`. The new tests
added by each commit were run against the previous commit's tree and failed
there, then the full suite was run on the new tree; the logs are in
`eval_sop/repro/stage_logs/`. Commits 1–6 were built from ordered edits to
`8987d6f`, verified to reproduce the final tree exactly, and committed one
logical change at a time (`repro/build_commits.py`, `repro/stage_tests.sh`).

| # | commit | what changed | why / evidence | preserved |
|---|---|---|---|---|
| 1 | `a50e055` | `AgentState.orchestrator_plan` declared (`agents/state.py`). The orchestrator returns the plan under that key; the trainer and evaluator read it. | LangGraph drops undeclared keys. Repro B1: the LLM planned `[logistic_regression, mlp]` and the trainer trained `[lightgbm, random_forest, xgboost]`. B1b: planned `f1`, ranked on `auc`. The recorded demo run (`web/public/demo/run.json`) shows the same thing: 4 planned, "Training 3 models", ranked on AUC although `f1` was planned. Test `test_plan_reaches_trainer...` fails on 8987d6f (`stage_logs/e2e_commit1_tests_on_state0.txt`). | Default model list and metric fallbacks unchanged. |
| 2 | `6343b1c` | The orchestrator profiles the CSV with the existing `_profile_dataframe` before prompting, lists all columns, rejects a target that is not a column, and normalises metric/model names to ones the trainer supports. | Repro B2: the original prompt was "Rows: unknown, Columns: unknown, Numeric columns: []". Once the plan arrives (commit 1), an unvalidated LLM model name or metric goes straight into the trainer. 2 new tests fail on commit 1. | Prompt wording otherwise unchanged; data analyst untouched. |
| 3 | `df2de5f` | `execute_with_retry(require_output=True)` (used by the feature engineer) treats "exit 0 but no OUTPUT_CSV_PATH" as a failure that goes to self-correction. A stale output is deleted before each attempt. No code path returns the raw CSV as the "processed" one. | Repro B3a: the original FE step returned success with the **raw** CSV path. B3b: it returned a **stale** `data_processed.csv` from an earlier run. In a local LLM smoke run (qwen2.5:7b on diabetes, not part of the results) one attempt exited 0 without output; on the original code that attempt would have silently trained on raw data. New test fails on commit 2. | `_output_csv_for` and path-rewrite logic unchanged; callers without `require_output` behave as before except that the path is no longer faked. |
| 4 | `1ff217b` | Generated code runs with an allow-listed environment (PATH, SYSTEMROOT, … plus a temp HOME/TEMP), no API keys, in a per-run temp cwd that is deleted afterwards. stdout is decoded as UTF-8. | Repro B5: `ANTHROPIC_API_KEY` was visible to generated code. New test fails on commit 3. | `sys.executable` choice and its rationale comment kept. E2B path untouched. |
| 5 | `322dd9c` | Optional `AgentState.random_seed` (default 42) feeds the split, CV folds and model RNGs. | Eval enabler, not a bug fix: every `random_state` was hard-coded to 42, so multi-seed evaluation was impossible. Default behaviour is identical. | All hyperparameters unchanged. |
| 6 | `ba83f3a` | CV scores the planned primary metric (`roc_auc`/`roc_auc_ovr`, `f1_weighted`, neg-RMSE, …). The winner is chosen on CV mean, and the test metric is only reported. `evaluation_result.selection_basis` is `cv`, or `holdout` only when CV is skipped above the 50k-row cap. | Repro B4: a model that is best on test but worse on CV won. B4b: CV used `f1_weighted` regardless of the plan. 2 new tests fail on commit 5. | Deterministic, LLM-free selection; the LLM still only narrates. The docstring rationale was kept and updated. |
| — | `5046138` | `eval_sop/` harness, non-LLM results, `RESULTS.md`. No product code. | — | — |
| — | `d02acc5` | Harness only, no product code. (a) `eval_sop/llm.py` can call Ollama's native `/api/chat` with `num_ctx` (needed for the ≤8192 context constraint; the OpenAI-compatible endpoint cannot set it). (b) `analyze.py` reports the three failure treatments, with Holm within two pre-declared families (LLM comparisons; non-LLM comparisons). (c) `leakage_audit_controls.py` adds positive and negative controls. The LLM arm and audit outputs are committed. | Required by the run constraints and the reporting request. | Non-LLM Holm values are unchanged (same 6-test family). |

Full test suite after each commit: 83 / 85 / 86 / 87 / 87 / 88 passed, 1
skipped. The first run of state 1 showed 8 sandbox-test failures with empty
stderr while the machine was overloaded. A re-run passed
(`stage_logs/full_suite_state1.txt` vs `full_suite_state1_rerun.txt`). Both logs
are kept.

**Not fixed (reproduced, still present):** B6, `leakage_warnings` hard-coded
to `[]` (`agents/feature_engineer.py`). The audit found no leakage in the 9
scripts that ran, but nothing in the product checks this at run time. See
"Proposed, not done".

## 4. Proposed, not done

- **Runtime leakage check.** Re-run the final feature-engineering script on a
  row subset and compare the overlapping rows; if they differ, populate
  `leakage_warnings` or fail. `eval_sop/leakage_audit.py` already implements
  the comparison. Not added to the product: 0/9 observed leaks, so the benefit
  is unproven.
- **Validate the feature-engineering output before training.** Reject or repair
  `inf` (2/15 datasets failed on `inf`), require at least one feature column
  (kr-vs-kp), and keep the row count. These would turn three trainer-stage
  failures into either clean runs or feature-engineering self-correction
  rounds. Not done: it changes system behaviour and would need its own
  evaluation.
- **Run feature engineering after the split, or on train only.** Today the script
  runs on the full CSV before `train_test_split`, so any cross-row statistic it
  computes (for example, dropping constant columns decided on all rows) sees
  test rows.
- **Trainer silent fallback:** `run_model_trainer` still reads
  `feature_result.transformed_csv_path or csv_path`. After commit 3 the FE step
  never produces an empty path on success, so this is now only reachable when
  FE did not run. Left as is.
- **`tracemalloc` inside parallel training threads** is process-global (start/stop
  from three threads at once) and adds allocation-tracing overhead to every fit.
  Not measured or changed here.
- **README corrections** (the README was not edited):
  - The "What one real run produced" table lists `logistic_regression` among
    "Models trained & cross-validated". The recorded run log shows only 3
    models were trained, because of bug 1. The planned metric was `f1`, but the
    winner was picked on test AUC.
  - "Winner (chosen by argmax…) AUC 0.836": that argmax was over the **test**
    set, so 0.836 is an optimistically selected number.
  - "Leakage-safe" feature engineering is enforced only by the prompt; nothing
    checks it (B6).

## 5. LLM arm: how it was run, and the paid run not done

The LLM client is configurable: an OpenAI-compatible `--llm-base-url`,
`--llm-model`, optional `--codegen-model`, and an optional
`--llm-key-file/--llm-key-name` read in-process. `--ollama-num-ctx` selects the
native Ollama endpoint. The Groq command (not used) is:
`--llm-base-url https://api.groq.com/openai/v1 --llm-model llama-3.3-70b-versatile --codegen-model llama-3.1-8b-instant --llm-key-file <.env> --llm-key-name GROQ_API_KEY`.
The model was unloaded after the run (`keep_alive: 0`).

**Paid-API run not done.** It would answer whether the 60% failure rate is a
7B-model artefact. The project's defaults are `claude-sonnet-5` as planner
($2 / $10 per M input/output tokens) and `claude-haiku-4-5` for codegen
($1 / $5). Measured usage in this run was 61 calls over 15 datasets. Assuming
about 3k planner input + 0.6k output and 6k codegen input + 3k output tokens
per dataset, the cost would be about **$0.03 per dataset, about $0.50 for all
15 with the cache, and under $2 without it**. This is an estimate, not
measured.

## 6. SOP-ready sentences (strictly true as of this commit)

1. "I audited my LLM-agent AutoML system and reproduced six plumbing defects
   with failing tests: the LLM's plan was silently discarded by the graph,
   model selection used the test set, failed generated code fell back to raw
   data, and generated code could read API keys. I fixed them in five commits,
   each with a test that fails before the fix."
2. "On 15 OpenML-CC18 datasets × 3 seeds, the system's LLM-free training and
   CV-selection stage beat default LightGBM by +0.010 ROC AUC (95% CI
   [+0.004, +0.017], Holm-adjusted p = 0.04) and was statistically
   indistinguishable from FLAML at an equal ~11 s budget."
3. "Ablating the LLM agents showed they did not pay off with a local 7B model:
   generated preprocessing code broke 9 of 15 datasets (60% of runs). On the
   6 datasets that completed, AUC was no better than the LLM-free pipeline
   (Δ = −0.012, n.s.). A leakage audit with positive controls found no leakage
   in the 9 generated scripts that ran."
