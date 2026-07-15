#!/bin/bash
# Full v2 tqa sweep across all visible GPUs (NGPU override), keeping every GPU busy throughout.
#   GRID    : all config-folds round-robined across the 3 GPUs, each run in-process by grid_runner
#             (judges load ONCE per GPU). Balanced at config-fold granularity, not whole cells.
#   PROMOTE : merge the 3 grid shards -> 2-fold means -> per-(cell,method) Pareto frontier.
#   CAPS    : capability battery on the frontier only, cells split across the 3 GPUs.
# Every stage is resumable (grid_runner and sweep_tqa.sh caps both skip completed rows), so a
# re-run continues where it stopped. Launch detached:  setsid bash scripts/run_v2_sweep.sh &
set -u
cd /root/sparse-steer
set -a; . ./.env 2>/dev/null; set +a
export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}" PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf \
       HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch,base_qa}
V2=${V2:-sweeps/v2}
# One grid_runner per visible GPU (override with NGPU=n). Work is round-robined at
# config-fold granularity, so any GPU count stays balanced.
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1
GPUS=$(seq 0 $((NGPU - 1)))
echo "using $NGPU gpu(s)"
mkdir -p "$V2"

echo "[$(date +%H:%M)] === GRID phase ==="
uv run python scripts/emit_grid_jobs.py "$CELLS" > "$V2/all.jobs"
echo "grid config-folds: $(wc -l < "$V2/all.jobs")"
for g in $GPUS; do
  awk -v n="$NGPU" -v g="$g" 'NR % n == (g + 1) % n' "$V2/all.jobs" > "$V2/g$g.jobs"
done
for g in $GPUS; do
  mkdir -p "$V2/grid_g$g"
  CUDA_VISIBLE_DEVICES=$g uv run python scripts/grid_runner.py "$V2/grid_g$g" "$V2/g$g.jobs" \
    > "/tmp/v2_grid_g$g.log" 2>&1 &
done
wait
echo "[$(date +%H:%M)] grid done"

echo "[$(date +%H:%M)] === PROMOTE ==="
head -1 "$V2/grid_g0/fulls.tsv" > "$V2/fulls.tsv"
for g in $GPUS; do tail -n +2 "$V2/grid_g$g/fulls.tsv" 2>/dev/null >> "$V2/fulls.tsv"; done
uv run python scripts/sweep_fold_mean.py "$V2/fulls.tsv" "$V2/grid_2fold.tsv"
uv run python scripts/sweep_promote.py "$V2/grid_2fold.tsv" --cap 20 --out "$V2/promoted.tsv"
echo "promoted frontier: $(( $(wc -l < "$V2/promoted.tsv") - 1 )) points"

echo "[$(date +%H:%M)] === CAPS phase ==="
# split cells across the GPUs round-robin; each GPU runs sweep_tqa.sh STAGES=caps in its own
# dir (a copy of the shared frontier) so their caps.tsv writes never race.
IFS=',' read -ra CL <<< "$CELLS"
declare -a SH=()
for g in $GPUS; do SH[$g]=""; done
for i in "${!CL[@]}"; do g=$(( i % NGPU )); SH[$g]="${SH[$g]},${CL[$i]}"; done
for g in $GPUS; do
  sub="${SH[$g]#,}"; [ -z "$sub" ] && continue
  d="$V2/cap_g$g"; mkdir -p "$d"; cp "$V2/promoted.tsv" "$d/promoted.tsv"
  GPU=$g STAGES=caps CELLS="$sub" RESULTS_DIR="$d" bash scripts/sweep_tqa.sh \
    > "/tmp/v2_caps_g$g.log" 2>&1 &
done
wait
# merge caps
head -1 "$V2/cap_g0/caps.tsv" 2>/dev/null > "$V2/caps.tsv"
for g in $GPUS; do tail -n +2 "$V2/cap_g$g/caps.tsv" 2>/dev/null >> "$V2/caps.tsv"; done

echo "[$(date +%H:%M)] === V2 SWEEP COMPLETE === fulls:$V2/fulls.tsv promoted:$V2/promoted.tsv caps:$V2/caps.tsv"
