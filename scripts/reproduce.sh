#!/bin/bash
# MAIN reproduction script for the TruthfulQA study, end to end. Multi-day.
#
#   1. GRID  (scripts/run_grid.py): train + evaluate the full grid across all GPUs --
#            sparse l0 in {0, 0.005, 0.01, 0.03} x init x position, ITI alpha x K x position, and
#            unsteered; both folds.  -> fulls.tsv, grid_2fold.tsv
#   2. CAPS  (scripts/run_caps.sh): capability suite (MMLU/ARC loglik+generative, WikiText) on EVERY
#            config, no promote.     -> caps.tsv
#
# Fully resumable -- every stage skips work already on disk, so this is safe to re-run after an
# interruption. Launch detached:  setsid bash scripts/reproduce.sh &
set -u
cd /root/sparse-steer
set -a; . ./.env 2>/dev/null; set +a
export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}" PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf \
       HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch,base_qa}

echo "[$(date +%H:%M)] === REPRODUCE 1/2: GRID (full, incl l0=0.03) ==="
uv run python scripts/run_grid.py --cells "$CELLS"

echo "[$(date +%H:%M)] === REPRODUCE 2/2: CAPS on every config ==="
CELLS="$CELLS" bash scripts/run_caps.sh

echo "[$(date +%H:%M)] === REPRODUCE COMPLETE ==="
