#!/usr/bin/env bash
# Detached experiment runner for the truthfulqa-reproduce loop.
# Usage: experiments/launch.sh <expid> <extra hydra overrides...>
# Writes a lock while running; parses results to experiments/results.jsonl; clears lock.
set -uo pipefail
ROOT="/Users/jamie/Projects/sparse_steer"
cd "$ROOT" || exit 3
EXPID="${1:?need expid}"; shift || true
ARGS="$*"
mkdir -p experiments/logs
LOG="experiments/logs/${EXPID}.log"
LOCK="experiments/current.lock"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"expid":"%s","pid":%s,"start":"%s","args":"%s"}\n' "$EXPID" "$$" "$START" "$ARGS" > "$LOCK"
# caching ON per project policy. method is NOT pinned here (Hydra defaults to method=sparse via
# configs/config.yaml); pass method=caa / method=iti in ARGS to run a baseline. task stays truthfulqa
# (the loop's domain). logging_steps is NOT forced here — training presets (sparse.yaml) set it
# themselves (=1, dense monitor trace); no-training presets (caa/dense) don't define it, so forcing
# it would crash Hydra ("not in struct"). Pass +logging_steps=N in ARGS if a preset lacks it.
PYTHONUNBUFFERED=1 uv run run.py task=truthfulqa $ARGS > "$LOG" 2>&1
STATUS=$?
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python experiments/parse_result.py "$EXPID" "$LOG" "$STATUS" "$START" "$END" "$ARGS" \
  >> experiments/results.jsonl 2>> experiments/logs/_parse_errors.log
rm -f "$LOCK"
