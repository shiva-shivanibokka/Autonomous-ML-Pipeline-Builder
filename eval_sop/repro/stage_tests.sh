#!/bin/bash
# For each commit state n: full test suite on state n, and state n's e2e tests run against state n-1.
set -u
# Set these for your machine (no defaults are guessed):
#   SP   = a scratch directory for worktrees, logs and the commits/ output of build_commits.py
#   REPO = path to this repository
#   PY   = python interpreter with the project's dependencies
: "${SP:?set SP}" "${REPO:?set REPO}" "${PY:?set PY}"
LOGS="$SP/stage_logs"; mkdir -p "$LOGS"
export ALLOW_LOCAL_EXEC=true EXECUTION_BACKEND=subprocess MLFLOW_TRACKING_URI="file:$SP/mlruns_eval" OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
for n in 0 1 2 3 4 5 6; do
  W="$SP/stage$n"
  [ -d "$W" ] || git -C "$REPO" worktree add --detach "$W" 8987d6f >/dev/null 2>&1
  [ "$n" -gt 0 ] && cp -r "$SP/commits/$n/." "$W/"
done
for n in 1 2 3 4 5 6; do
  p=$((n-1))
  W="$SP/stage$n"; P="$SP/stage$p"
  cp "$SP/commits/$n/tests/test_graph_e2e.py" "$P/tests/test_e2e_from_commit$n.py"
  (cd "$P" && "$PY" -m pytest -q -p no:cacheprovider "tests/test_e2e_from_commit$n.py" > "$LOGS/e2e_commit${n}_tests_on_state${p}.txt" 2>&1; echo "exit=$?" >> "$LOGS/e2e_commit${n}_tests_on_state${p}.txt")
  rm -f "$P/tests/test_e2e_from_commit$n.py"
  (cd "$W" && "$PY" -m pytest -q -p no:cacheprovider tests > "$LOGS/full_suite_state$n.txt" 2>&1; echo "exit=$?" >> "$LOGS/full_suite_state$n.txt")
  echo "state $n: $(tail -2 "$LOGS/full_suite_state$n.txt" | head -1) | prev-state run of new tests: $(grep -E 'passed|failed' "$LOGS/e2e_commit${n}_tests_on_state${p}.txt" | tail -1)"
done
