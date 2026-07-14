#!/bin/bash
# ============================================================================
# Full TruthfulQA sweep (v2): full 2-fold CV over the WHOLE grid, then Pareto
# frontier, then MC + capability only on the frontier. No 100-question screens.
#
# Per cell (model x template), two clean factorial grids:
#   Sparse:  l0_lambda {0, 0.005, 0.01} x init_log_alpha {-0.79, 1}
#            x steer_pos {all, answer_gen}          (num_epochs = 16 fixed)   = 12
#   ITI:     scale {8, 15, 22} x topk {24, 48, 96}
#            x steer_pos {all, answer_gen}          (sigma = gen_end_q fixed) = 18
# Extraction is always completion_final. ITI probes fit on GPU (iti_probe_device=cuda),
# head selection holds out 20% (honest_llama val_ratio). => 30 configs/cell.
#
#   Stage 1  anchors   unsteered 2-fold full evals (calibration)
#   Stage 2  grid      2-fold FULL True/Info (+MC) on every grid config
#   Stage 3  promote   per-(cell,method) Pareto frontier of the 2-fold True/Info
#                      (scripts/sweep_fold_mean.py -> scripts/sweep_promote.py)
#   Stage 4  caps      capability battery on the frontier only: loglik MMLU/ARC/
#                      wikitext-CE (fixed + chat template) + generative MMLU/ARC.
#
# The paper-canonical ITI point (scale=15, topk=48, steer=all, sigma=gen_end_q) is
# a natural cell of the grid, so it is evaluated and promoted like any other.
#
# TSV-resumable at every stage. Shard CELLS across GPUs to parallelise:
#   GPU=0 CELLS=ll_qa ... & GPU=1 CELLS=qw_qa ... & GPU=2 CELLS=ll_ch,qw_ch ...
# ============================================================================
set -u
GPU=${GPU:-0}
RES=${RESULTS_DIR:-sweeps/tqa}
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch,base_qa}
PROMOTE_CAP=${PROMOTE_CAP:-20}       # grids are small; 20 keeps the whole frontier
# ── Stage selection: STAGES = comma-list, or "all" (default) ────────────────
#   anchors  Stage 1   promote  Stage 3
#   grid     Stage 2   caps     Stage 4
STAGES=${STAGES:-all}
stage() { case ",$STAGES," in *,all,*|*",$1,"*) return 0 ;; *) return 1 ;; esac; }
mkdir -p "$RES"

# ── Cell definitions: task config + template overrides + model-sized batches ─
cell_args() {
  case $1 in
    ll_qa) echo "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32" ;;
    ll_ch) echo "task=truthfulqa prompt_template=chat extraction_template=chat eval_batch_size=64 gen_batch_size=16 judge_batch_size=32" ;;
    qw_qa) echo "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16" ;;
    qw_ch) echo "task=truthfulqa_qwen prompt_template=chat extraction_template=chat eval_batch_size=32 gen_batch_size=8 judge_batch_size=16" ;;
    base_qa) echo "task=truthfulqa model_name=huggyllama/llama-7b ++architecture_name=llama-7b-hf ++model_dtype=float16 eval_batch_size=64 gen_batch_size=16 judge_batch_size=32" ;;
  esac
}

# ── Fixed defaults, identical for every cell ────────────────────────────────
COMMON="device=cuda eval_backend=hf disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
SPARSE="method=sparse train_batch_size=1 +contrastive_weight=1 +ce_weight=0 track_gates=false extract_token_position=completion_final +contrastive_max_n_neg=3 init_raw_scale=15 num_epochs=16"
ITI="method=iti extract_token_position=completion_final iti_sigma_position=gen_end_q iti_probe_device=cuda"
GRIDEVAL="generative_eval=true"       # grid needs the generative True/Info judge metrics

# ── Grid axes ───────────────────────────────────────────────────────────────
SP_L0="0 0.005 0.01"
SP_ILA="def:-0.79 open:1"             # label:init_log_alpha
POS="all:all ag:answer_gen"           # label:steer_token_position
ITI_A="8 15 22"
ITI_K="24 48 96"

FULLTSV=$RES/fulls.tsv
[ -f "$FULLTSV" ] || printf "tag\tcell\tmethod\tfold\ttrue\tinfo\tmc1\tmc2\targs\n" > "$FULLTSV"

harvest() { # log -> "true info mc1 mc2" (blank on parse failure; exclude Unsteered ref lines)
  local log=$1 t i m1 m2
  t=$(grep -av "Unsteered" "$log" | grep -aoE "GEN_TRUTHFUL: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  i=$(grep -av "Unsteered" "$log" | grep -aoE "GEN_INFORMATIVE: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  m1=$(grep -av "Unsteered" "$log" | grep -aoE "MC1: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  m2=$(grep -av "Unsteered" "$log" | grep -aoE "MC2: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  echo "${t:-} ${i:-} ${m1:-} ${m2:-}"
}

run_full() { # tag cell method fold args...
  local tag=$1 cell=$2 method=$3 fold=$4; shift 4
  grep -q "^${tag}	${cell}	${method}	${fold}	" "$FULLTSV" && { echo "skip $tag f$fold"; return; }
  echo "[$(date +%H:%M)] FULL $tag fold=$fold"
  CUDA_VISIBLE_DEVICES=$GPU uv run python run.py $COMMON $(cell_args "$cell") \
    eval_subset_size=null fold=$fold "$@" > "$RES/full_${tag}_f${fold}.log" 2>&1 || echo "ERR $tag f$fold"
  read -r T I M1 M2 <<< "$(harvest "$RES/full_${tag}_f${fold}.log")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$cell" "$method" "$fold" "$T" "$I" "$M1" "$M2" "$*" >> "$FULLTSV"
}

IFS=',' read -ra CELL_LIST <<< "$CELLS"

# ════ Stage 1 — unsteered 2-fold anchors ════════════════════════════════════
if stage anchors; then
for cell in "${CELL_LIST[@]}"; do
  for fold in 0 1; do
    run_full "uns_${cell}" "$cell" unsteered $fold method=unsteered
  done
done
fi

# ════ Stage 2 — full 2-fold grid (True/Info on every square) ════════════════
if stage grid; then
for cell in "${CELL_LIST[@]}"; do
  # sparse: l0 x ila x steer_pos (ep16 fixed)
  for l0 in $SP_L0; do for ilp in $SP_ILA; do for pp in $POS; do
    ila=${ilp#*:}; ilab=${ilp%:*}; pos=${pp#*:}; plab=${pp%:*}
    for fold in 0 1; do
      run_full "sp_${cell}_l${l0}_${ilab}_${plab}" "$cell" sparse $fold \
        $SPARSE $GRIDEVAL l0_lambda=$l0 gate_config.init_log_alpha=$ila steer_token_position=$pos
    done
  done; done; done
  # ITI: scale x topk x steer_pos (sigma gen_end_q fixed)
  for a in $ITI_A; do for k in $ITI_K; do for pp in $POS; do
    pos=${pp#*:}; plab=${pp%:*}
    for fold in 0 1; do
      run_full "iti_${cell}_a${a}_k${k}_${plab}" "$cell" iti $fold \
        $ITI $GRIDEVAL iti_scale=$a iti_topk=$k steer_token_position=$pos
    done
  done; done; done
done
fi

# ════ Stage 3 — Pareto promotion (per-cell/method, on 2-fold True/Info) ══════
if stage promote; then
uv run python scripts/sweep_fold_mean.py "$FULLTSV" "$RES/grid_2fold.tsv"
uv run python scripts/sweep_promote.py "$RES/grid_2fold.tsv" --cap "$PROMOTE_CAP" --out "$RES/promoted.tsv"
echo "promoted frontier:"; cat "$RES/promoted.tsv"
fi

# ════ Stage 4 — capability battery on the frontier only ═════════════════════
if stage caps; then
CAPTSV=$RES/caps.tsv
[ -f "$CAPTSV" ] || printf "tag\tcell\tmethod\tstage\tmetrics\n" > "$CAPTSV"

run_cap() { # tag cell method stage args...
  local tag=$1 cell=$2 method=$3 stg=$4; shift 4
  grep -q "^${tag}	" "$CAPTSV" && { echo "skip $tag (done)"; return; }
  echo "[$(date +%H:%M)] CAP $tag"
  CUDA_VISIBLE_DEVICES=$GPU uv run python run.py $COMMON $(cell_args "$cell") \
    eval_subset_size=2 generative_eval=false "$@" > "$RES/cap_${tag}.log" 2>&1 || echo "ERR $tag"
  local m
  m=$(grep -av Unsteered "$RES/cap_${tag}.log" | grep -aoE "(MMLU|ARC_CHALLENGE|WIKITEXT)/[A-Z_/]+: [0-9.]+" | paste -sd" " -)
  printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$cell" "$method" "$stg" "$m" >> "$CAPTSV"
}

# MMLU alone at limit=100/subject 5-shot (matches the study anchors' batching); ARC+wikitext full.
LLMM="lmeval_steer=answer_gen lmeval_tasks=[mmlu] lmeval_limit=100 lmeval_fewshot=5"
LLAW="lmeval_steer=answer_gen lmeval_tasks=[arc_challenge,wikitext]"
CTFLAGS="lmeval_chat_template=true lmeval_fewshot_multiturn=true"
GENC="inspect_evals=[mmlu,arc_challenge] inspect_eval_limit=1000 inspect_max_tokens=64 inspect_steer=answer_gen"

cap_points() { # $1 = cell -> "tag<TAB>method<TAB>args": unsteered + promoted frontier points
  printf "uns\tunsteered\tmethod=unsteered\n"
  awk -F"\t" -v c="$1" 'NR>1 && $2==c {print $1"\t"$3"\t"$4}' "$RES/promoted.tsv"
}

for cell in "${CELL_LIST[@]}"; do
  while IFS=$'\t' read -r ptag method args; do
    run_cap "cap_fxmm_${cell}_${ptag}" "$cell" "$method" loglik-fx-mmlu $args $LLMM
    run_cap "cap_fxaw_${cell}_${ptag}" "$cell" "$method" loglik-fx-arcwiki $args $LLAW
    if [ "$cell" != "base_qa" ]; then
      run_cap "cap_ctmm_${cell}_${ptag}" "$cell" "$method" loglik-ct-mmlu $args $LLMM $CTFLAGS
      run_cap "cap_ctaw_${cell}_${ptag}" "$cell" "$method" loglik-ct-arcwiki $args $LLAW $CTFLAGS
    fi
    run_cap "cap_gen_${cell}_${ptag}" "$cell" "$method" generative $args $GENC
  done < <(cap_points "$cell")
done
fi

echo "[$(date +%H:%M)] SWEEP COMPLETE — fulls: $FULLTSV, promoted: $RES/promoted.tsv, caps: $RES/caps.tsv"
