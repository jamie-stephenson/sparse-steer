#!/bin/bash
# GRID-ONLY tqa sweep: train + evaluate every config-fold (True/Info/MC) and write fulls.tsv. No promote,
# no capability suite -- that is scripts/run_v2_sweep.sh's job. Resumable: grid_runner skips any
# (tag,cell,method,fold) already in a shard's fulls.tsv, so re-running only trains what is missing.
#
#   Full extended grid (v2 values + l0=0.03):   setsid bash scripts/run_grid.sh &
#   Only a new sparse level (no retrain of v2):  ONLY_L0=0.03 setsid bash scripts/run_grid.sh &
#     -> emits ONLY the l0=0.03 sparse configs (all cells/inits/positions); unsteered/ITI/other l0 are
#        left untouched (already trained + cached from v2).
set -u
cd /root/sparse-steer
set -a; . ./.env 2>/dev/null; set +a
export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}" PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf \
       HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch,base_qa}
V2=${V2:-sweeps/v2}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1
GPUS=$(seq 0 $((NGPU - 1)))
mkdir -p "$V2"

EMIT_ARGS=("$CELLS")
JOBSFILE="$V2/all.jobs"
if [ -n "${ONLY_L0:-}" ]; then
  EMIT_ARGS+=(--only-sparse-l0 "$ONLY_L0")
  JOBSFILE="$V2/l0_${ONLY_L0}.jobs"
  echo "[$(date +%H:%M)] === GRID phase (ONLY sparse l0=$ONLY_L0) ==="
else
  echo "[$(date +%H:%M)] === GRID phase (full grid: $(python3 -c 'print("l0 in {0,0.005,0.01,0.03} + ITI + unsteered")')) ==="
fi
uv run python scripts/emit_grid_jobs.py "${EMIT_ARGS[@]}" > "$JOBSFILE"
echo "grid config-folds: $(wc -l < "$JOBSFILE")"

# round-robin config-folds across the GPUs; each grid_runner appends to its own shard's fulls.tsv
for g in $GPUS; do
  awk -v n="$NGPU" -v g="$g" 'NR % n == (g + 1) % n' "$JOBSFILE" > "$V2/g$g.jobs"
done
for g in $GPUS; do
  mkdir -p "$V2/grid_g$g"
  CUDA_VISIBLE_DEVICES=$g uv run python scripts/grid_runner.py "$V2/grid_g$g" "$V2/g$g.jobs" \
    > "/tmp/grid_g$g.log" 2>&1 &
done
wait
echo "[$(date +%H:%M)] grid done"

# merge shards -> fold-mean (one row per config, fold-0 args). NOT a promote/Pareto step.
head -1 "$V2/grid_g0/fulls.tsv" > "$V2/fulls.tsv"
for g in $GPUS; do tail -n +2 "$V2/grid_g$g/fulls.tsv" 2>/dev/null >> "$V2/fulls.tsv"; done
uv run python scripts/sweep_fold_mean.py "$V2/fulls.tsv" "$V2/grid_2fold.tsv"
echo "[$(date +%H:%M)] === GRID COMPLETE === fulls:$V2/fulls.tsv  configs:$(( $(wc -l < "$V2/grid_2fold.tsv") - 1 ))"
