#!/bin/bash
# Autoresearch experiment runner (pod-side). Usage:  bash ar_run.sh "<hydra args>"
# Driven by the autoresearch cron loop. Markers live in $HOME:
#   ~/ar.running  present while a run is in flight   (one-experiment-at-a-time lock)
#   ~/ar.done     present when a run has finished
#   ~/ar.rc       exit code of run.py
#   ~/ar.args     the hydra args used
#   ~/ar.log      full stdout/stderr
cd ~/sparse_steer || exit 1
export PATH="$HOME/.local/bin:$PATH"
ARGS="$1"
rm -f ~/ar.done ~/ar.rc
echo "$ARGS" > ~/ar.args
date -u > ~/ar.start
touch ~/ar.running
{
  echo "=== AR EXP START $(date -u) ==="
  echo "ARGS: $ARGS"
  uv run --with "transformers==4.49.0" python run.py $ARGS
  echo "$?" > ~/ar.rc
  echo "=== AR EXP END $(date -u) ==="
} > ~/ar.log 2>&1
rm -f ~/ar.running
touch ~/ar.done
