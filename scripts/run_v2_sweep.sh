#!/bin/bash
# TQA capability orchestrator. The GRID (training + True/Info/MC eval) is OPTIONAL and normally run
# separately via scripts/run_grid.sh; this script's job is the CAPABILITY suite on EVERY trained config
# -- there is NO Pareto promote stage. Running caps on the whole grid (not just a frontier subset) is
# what lets us map the full effectiveness-vs-capability surface (dense l0=0 through sparse l0=0.03, all
# ITI points), rather than only the True*Info-max corner.
#
#   Caps on everything already trained:   setsid bash scripts/run_v2_sweep.sh &
#   Train the grid first, then caps:       GRID=1 setsid bash scripts/run_v2_sweep.sh &
#   Train only a new sparse level + caps:  GRID=1 ONLY_L0=0.03 setsid bash scripts/run_v2_sweep.sh &
#
# Every stage is resumable: grid_runner skips completed fulls rows, caps_runner skips completed caps
# rows, and seed_caps.py reuses caps rows already computed by any earlier (targeted) run.
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
echo "using $NGPU gpu(s)"
mkdir -p "$V2"

# --- optional GRID (delegates to run_grid.sh; honours ONLY_L0 for a single new sparse level) ---
if [ "${GRID:-0}" = "1" ]; then
  bash scripts/run_grid.sh
fi

# --- aggregate folds -> one row per config (fold-0 args). NOT a promote/Pareto step. ---
head -1 "$V2/grid_g0/fulls.tsv" > "$V2/fulls.tsv"
for g in $GPUS; do tail -n +2 "$V2/grid_g$g/fulls.tsv" 2>/dev/null >> "$V2/fulls.tsv"; done
uv run python scripts/sweep_fold_mean.py "$V2/fulls.tsv" "$V2/grid_2fold.tsv"
NCFG=$(( $(wc -l < "$V2/grid_2fold.tsv") - 1 ))
echo "[$(date +%H:%M)] === CAPS on ALL $NCFG configs (+ unsteered baselines; no promote) ==="

# --- CAPS: cells split round-robin across GPUs; each shard caps ALL of its cells' configs ---
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

echo "[$(date +%H:%M)] === CAPS COMPLETE === fulls:$V2/fulls.tsv caps:$V2/caps.tsv (all $NCFG configs)"
