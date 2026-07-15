#!/usr/bin/env bash
# Detached runner for OLD code in the 0bd8bf9 worktree (H6 bisection).
# Uses the worktree's own uv.lock (old deps) for a faithful old environment.
# Usage: launch_old.sh <expid> <config_yaml_basename_in_configs_old>
set -uo pipefail
OLD="/Users/jamie/Projects/sparse_steer_old"
MAIN="/Users/jamie/Projects/sparse_steer"
EXPID="${1:?need expid}"; shift || true
CFG="${1:?need config yaml}"; shift || true
CFG_ABS="$MAIN/experiments/configs_old/$CFG"
mkdir -p "$MAIN/experiments/logs"
LOG="$MAIN/experiments/logs/${EXPID}.log"
LOCK="$MAIN/experiments/current.lock"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"expid":"%s","pid":%s,"start":"%s","args":"OLDCODE %s"}\n' "$EXPID" "$$" "$START" "$CFG" > "$LOCK"
cd "$OLD"
# worktree-local uv env (old transformers etc.); first run will sync deps.
uv run python run.py truthfulqa --config "$CFG_ABS" > "$LOG" 2>&1
STATUS=$?
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 "$MAIN/experiments/parse_old.py" "$EXPID" "$LOG" "$STATUS" "$START" "$END" "oldcode:$CFG" \
  >> "$MAIN/experiments/results.jsonl" 2>> "$MAIN/experiments/logs/_parse_errors.log"
rm -f "$LOCK"
