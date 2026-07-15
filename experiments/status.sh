#!/usr/bin/env bash
# Prints job status for the loop:
#   RUNNING <expid> pid=<pid>   -> a job is alive; loop should do nothing
#   STALE <expid> pid=<pid>     -> lock exists but pid dead (crash); loop clears it
#   IDLE                        -> no job; loop launches the next
ROOT="/Users/jamie/Projects/sparse_steer"
cd "$ROOT" || exit 3
LOCK="experiments/current.lock"
if [ -f "$LOCK" ]; then
  PID=$(python3 -c "import json;print(json.load(open('$LOCK'))['pid'])" 2>/dev/null)
  EID=$(python3 -c "import json;print(json.load(open('$LOCK'))['expid'])" 2>/dev/null)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "RUNNING $EID pid=$PID"
  else
    echo "STALE $EID pid=$PID"
  fi
else
  echo "IDLE"
fi
