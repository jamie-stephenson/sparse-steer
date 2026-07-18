#!/bin/bash
# Capability suite over EVERY trained config (no Pareto promote): for each config in grid_2fold.tsv,
# plus the unsteered baseline per cell, run the full capability battery (MMLU/ARC loglik+generative,
# WikiText) via caps_runner.py. Cells split round-robin across GPUs. Reuses every caps row already
# computed anywhere under the results dir (seed_caps.py), so it only evaluates genuinely-new configs.
#
# Run after the grid (scripts/run_grid.py) has produced grid_2fold.tsv:
#   setsid bash scripts/run_caps.sh &
# Operational overrides: CELLS=..., V2=<results_dir>, NGPU=n.
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

if [ ! -f "$V2/grid_2fold.tsv" ]; then
  echo "no $V2/grid_2fold.tsv -- run scripts/run_grid.py first" >&2
  exit 1
fi
NCFG=$(( $(wc -l < "$V2/grid_2fold.tsv") - 1 ))
echo "[$(date +%H:%M)] === CAPS on ALL $NCFG configs (+ unsteered baselines; no promote) ==="

IFS=',' read -ra CL <<< "$CELLS"
declare -a SH=(); for g in $GPUS; do SH[$g]=""; done
for i in "${!CL[@]}"; do g=$(( i % NGPU )); SH[$g]="${SH[$g]},${CL[$i]}"; done
for g in $GPUS; do
  sub="${SH[$g]#,}"; [ -z "$sub" ] && continue
  d="$V2/cap_g$g"; mkdir -p "$d"
  cp "$V2/grid_2fold.tsv" "$d/promoted.tsv"                # full config list = caps_runner input
  uv run python scripts/seed_caps.py "$V2" "$d" "$sub"     # reuse every caps row already computed
  CUDA_VISIBLE_DEVICES=$g uv run python scripts/caps_runner.py "$d" "$sub" \
    > "/tmp/v2_caps_g$g.log" 2>&1 &
done
wait
head -1 "$V2/cap_g0/caps.tsv" 2>/dev/null > "$V2/caps.tsv"
for g in $GPUS; do tail -n +2 "$V2/cap_g$g/caps.tsv" 2>/dev/null >> "$V2/caps.tsv"; done
echo "[$(date +%H:%M)] === CAPS COMPLETE === caps:$V2/caps.tsv (all $NCFG configs)"
