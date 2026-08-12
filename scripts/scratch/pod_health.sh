#!/bin/bash
# Health check for the two live pods, run from cron every 20 minutes.
#
# Checks progress against plan, not log silence: a job that dies to the cgroup OOM killer
# prints no traceback, and a driver that finishes early prints a completion banner it has
# not earned. So for each pod we assert that the driver is alive while work remains, that
# its output count is still growing, that the GPUs are busy, and that the sync loop lives.
# Anything abnormal is written to sweeps/monitor/health.log with an ALERT prefix.
set -u
ROOT="/Users/jamie/Projects/sparse_steer"
LOG="$ROOT/sweeps/monitor/health.log"
STATE="$ROOT/sweeps/monitor/state"
mkdir -p "$(dirname "$LOG")" "$STATE"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
ts() { date -u +'%FT%TZ'; }
say() { echo "$(ts) $*" >> "$LOG"; }

check() {
  local alias=$1 name=$2 logfile=$3 countcmd=$4 donepat=$5 procpat=$6
  local out proc count util sync errs prev
  out=$($SSH "$alias" "
    echo \"PROC \$(pgrep -cf '$procpat')\"
    echo \"COUNT \$($countcmd)\"
    echo \"DONE \$(grep -ac '$donepat' $logfile 2>/dev/null)\"
    echo \"ERRS \$(grep -acE 'Traceback|CUDA out of memory|OutOfMemoryError|Killed|MemoryError' $logfile 2>/dev/null)\"
    echo \"UTIL \$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{s+=\$1} END {print s+0}')\"
    echo \"SYNC \$(pgrep -cf 'sync_[c]ache')\"
  " 2>/dev/null)
  if [ -z "$out" ]; then say "ALERT [$name] UNREACHABLE"; return; fi
  proc=$(echo "$out"  | awk '/^PROC/{print $2}')
  count=$(echo "$out" | awk '/^COUNT/{print $2}')
  errs=$(echo "$out"  | awk '/^ERRS/{print $2}')
  util=$(echo "$out"  | awk '/^UTIL/{print $2}')
  sync=$(echo "$out"  | awk '/^SYNC/{print $2}')
  local donec; donec=$(echo "$out" | awk '/^DONE/{print $2}')

  prev=$(cat "$STATE/$name.count" 2>/dev/null || echo -1)
  echo "${count:-0}" > "$STATE/$name.count"
  local preverr; preverr=$(cat "$STATE/$name.errs" 2>/dev/null || echo 0)
  echo "${errs:-0}" > "$STATE/$name.errs"

  [ "${errs:-0}" -gt "${preverr:-0}" ] && \
    say "ALERT [$name] $((errs - preverr)) new error/OOM line(s); outputs=$count"
  [ "${sync:-0}" -eq 0 ] && say "ALERT [$name] /workspace sync loop is dead"

  # Liveness is checked BEFORE the completion banner: the banner is appended to a log that
  # later runs keep appending to, so once written it reads as "done" forever. A live driver
  # means work in progress regardless of what an earlier run claimed.
  if [ "${proc:-0}" -gt 0 ]; then
    if [ "$count" = "$prev" ] && [ "${util:-0}" -lt 25 ]; then
      local stall; stall=$(( $(cat "$STATE/$name.stall" 2>/dev/null || echo 0) + 1 ))
      echo "$stall" > "$STATE/$name.stall"
      [ "$stall" -ge 2 ] && say "ALERT [$name] STALLED ~$((stall * 20))min: outputs=$count, GPUs idle"
    else
      echo 0 > "$STATE/$name.stall"
      say "[$name] ok: outputs=$count procs=$proc gpu=${util}%"
    fi
  elif [ "${donec:-0}" -ge 1 ]; then
    say "[$name] driver idle, last banner says done, outputs=$count (verify against plan)"
  elif true; then
    say "ALERT [$name] driver DEAD with no completion banner; outputs=$count"
  fi
}

check runpod-sleeper sleeper /root/sleeper_caps.log \
  "ls /root/sparse_steer/sweeps/sleeper/suite_scores/*.json 2>/dev/null | wc -l" \
  "ALL SLEEPER CAP JOBS DONE" "run_sleeper[_]experiments"

check runpod tqa /root/tqa_l0ext.log \
  "ls -d /root/sparse_steer/.cache/sparse_steer/steered_eval/truthfulqa/*/ 2>/dev/null | wc -l" \
  "TQA SWEEP COMPLETE" "run_tqa[_]experiments"
