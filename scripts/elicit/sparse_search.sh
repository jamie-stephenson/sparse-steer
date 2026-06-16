#!/bin/bash
# SF-E10 targeted sparse search: get the L0 objective to RECOVER a small working gate set.
# Frontier facts: single sites resid_mid@L0 m≈8 → asr 0.65 and resid_post@L1 (=resid_pre@L2) m≈8 →
# asr 0.81 BOTH live inside the [resid_mid,resid_post]×4 candidate set. So a 1-gate solution exists;
# the task is for L0 to find it.
#
# Design insight from SF-E7/E8: with LEARNED scale, gates settle at the ~0.4 smooth-L0 floor and the
# scale absorbs the magnitude → no single gate dominates (effective steering spread thin). With
# gates_only (FIXED scale) started CLOSED, the gate itself must carry the magnitude, so CE is forced
# to open ONLY the useful gate(s) toward 1.0 while L0 keeps the rest shut. Primary = gates_only;
# learn_scale kept as a contrast. Sites started closed (uniform init_log_alpha, no per-site bias).
set -e
cd /Users/jamie/Projects/sparse_steer
run() { # method, scale, tag, extra
  echo "===== $3  ($1, scale=$2) ====="
  uv run python scripts/elicit/sweep.py --task tinysleepers_elicit/sparse --method "$1" \
      --sites "resid_mid+resid_post" --layers all --strengths "$2" --raw-scale \
      --generative true --n_eval 100 --seeds 0,1,2 --gen_tokens 24 \
      --extra "$4" --tag "$3" 2>&1 | grep -E "asr=|resid_mid\+resid_post"
  echo "--- gates ---"; uv run python scripts/elicit/gates.py
}
CLOSED="gate_config.init_log_alpha=-2.0"
# A) gates_only, FIXED high scale, gates start CLOSED → CE opens only useful gates, L0 prunes
run gates_only  8 E10_go_s8_l0.1  "l0_lambda=0.1 num_epochs=16 $CLOSED"
run gates_only  8 E10_go_s8_l0.3  "l0_lambda=0.3 num_epochs=16 $CLOSED"
run gates_only  8 E10_go_s8_l0.6  "l0_lambda=0.6 num_epochs=16 $CLOSED"
run gates_only 12 E10_go_s12_l0.3 "l0_lambda=0.3 num_epochs=16 $CLOSED"
run gates_only 12 E10_go_s12_l0.6 "l0_lambda=0.6 num_epochs=16 $CLOSED"
# B) learn_scale contrast, start closed, high init scale
run sparse      8 E10_sp_s8_l0.1  "l0_lambda=0.1 num_epochs=16 $CLOSED"
run sparse      8 E10_sp_s8_l0.3  "l0_lambda=0.3 num_epochs=16 $CLOSED"
echo "=== SF-E10 DONE ==="
