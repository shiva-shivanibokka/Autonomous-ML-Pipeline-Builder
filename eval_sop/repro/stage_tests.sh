#!/bin/bash
# For each commit state n: full test suite on state n, and state n's e2e tests run against state n-1.
set -u
SP="<SCRATCH>"
REPO="<HOME>/OneDrive/Desktop/GITHUB REPOS/Autonomous-ML-Pipeline-Builder"
PY="$SP/evalenv/Scripts/python"
LOGS="$SP/amlpb_stage_logs"; mkdir -p "$LOGS"
export ALLOW_LOCAL_EXEC=true EXECUTION_BACKEND=subprocess MLFLOW_TRACKING_URI="file:$SP/mlruns_eval" OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
for n in 0 1 2 3 4 5 6; do
  W="$SP/wt/amlpb_stage$n"
  [ -d "$W" ] || git -C "$REPO" worktree add --detach "$W" 8987d6f >/dev/null 2>&1
  [ "$n" -gt 0 ] && cp -r "$SP/amlpb_commits/$n/." "$W/"
done
for n in 1 2 3 4 5 6; do
  p=$((n-1))
  W="$SP/wt/amlpb_stage$n"; P="$SP/wt/amlpb_stage$p"
  cp "$SP/amlpb_commits/$n/tests/test_graph_e2e.py" "$P/tests/test_e2e_from_commit$n.py"
  (cd "$P" && "$PY" -m pytest -q -p no:cacheprovider "tests/test_e2e_from_commit$n.py" > "$LOGS/e2e_commit${n}_tests_on_state${p}.txt" 2>&1; echo "exit=$?" >> "$LOGS/e2e_commit${n}_tests_on_state${p}.txt")
  rm -f "$P/tests/test_e2e_from_commit$n.py"
  (cd "$W" && "$PY" -m pytest -q -p no:cacheprovider tests > "$LOGS/full_suite_state$n.txt" 2>&1; echo "exit=$?" >> "$LOGS/full_suite_state$n.txt")
  echo "state $n: $(tail -2 "$LOGS/full_suite_state$n.txt" | head -1) | prev-state run of new tests: $(grep -E 'passed|failed' "$LOGS/e2e_commit${n}_tests_on_state${p}.txt" | tail -1)"
done
