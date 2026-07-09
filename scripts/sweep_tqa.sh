#!/bin/bash
# ============================================================================
# Full TruthfulQA sweep: regenerates the paper's per-cell Pareto frontiers
# (sparse L0-gate steering vs ITI vs unsteered) across every model x template
# cell, via the same two-tier protocol used in the study:
#
#   Stage 1  unsteered anchors           2-fold full evals (calibration)
#   Stage 2  uniform screen grid         100-question, fold-0 (cheap)
#   Stage 3  promotion                   per-cell/method Pareto set, cap 4
#            (scripts/sweep_promote.py)
#   Stage 4  2-fold full evals           promoted points only (fold 0 + 1)
#   Stage 5  (--capability)              loglik battery on promoted points
#
# Est. ~1.5 days on one A40 (screens ~16 h, fulls ~10 h). Parallelize by
# sharding CELLS across GPUs: CELLS=ll_qa,ll_ch GPU=0 ... & CELLS=qw_qa,qw_ch GPU=1 ...
# ============================================================================
set -u
GPU=${GPU:-0}
RES=${RESULTS_DIR:-sweeps/tqa}
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch}
PROMOTE_CAP=${PROMOTE_CAP:-4}
mkdir -p "$RES"

# ── Cell definitions: task config + template overrides + model-sized batches ─
cell_args() {
  case $1 in
    ll_qa) echo "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32" ;;
    ll_ch) echo "task=truthfulqa prompt_template=chat extraction_template=chat eval_batch_size=64 gen_batch_size=16 judge_batch_size=32" ;;
    qw_qa) echo "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16" ;;
    qw_ch) echo "task=truthfulqa_qwen prompt_template=chat extraction_template=chat eval_batch_size=32 gen_batch_size=8 judge_batch_size=16" ;;
  esac
}

# ── Fixed defaults, identical for every cell (stated once; see paper §setup) ─
COMMON="device=cuda eval_backend=hf disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
SPARSE="method=sparse train_batch_size=1 +contrastive_weight=1 +ce_weight=0 track_gates=false extract_token_position=completion_final +contrastive_max_n_neg=3 init_raw_scale=15"
ITI="method=iti extract_token_position=completion_final steer_token_position=completion"

SCREEN="eval_subset_size=100 fold=0"           # screen tier: 100-q, fold-0
TSV=$RES/screens.tsv
FULLTSV=$RES/fulls.tsv
[ -f "$TSV" ] || printf "tag\tcell\tmethod\ttrue\tinfo\tmc1\tmc2\targs\n" > "$TSV"
[ -f "$FULLTSV" ] || printf "tag\tcell\tmethod\tfold\ttrue\tinfo\tmc1\tmc2\targs\n" > "$FULLTSV"

harvest() { # log -> "true info mc1 mc2" (blank fields on parse failure)
  # run.py prints one "  KEY: 0.xxxx" line per metric (experiment/base.py):
  # MC0/MC1/MC2 + GEN_TRUTHFUL / GEN_INFORMATIVE / GEN_TRUTHFUL_INFORMATIVE.
  local log=$1
  local t i m1 m2
  t=$(grep -aoE "GEN_TRUTHFUL: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  i=$(grep -aoE "GEN_INFORMATIVE: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  m1=$(grep -aoE "MC1: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  m2=$(grep -aoE "MC2: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${t:-} ${i:-} ${m1:-} ${m2:-}"
}

run_screen() { # tag cell method args...
  local tag=$1 cell=$2 method=$3; shift 3
  grep -q "^${tag}	" "$TSV" && { echo "skip $tag (done)"; return; }   # resumable
  echo "[$(date +%H:%M)] SCREEN $tag"
  CUDA_VISIBLE_DEVICES=$GPU uv run python run.py $COMMON $(cell_args "$cell") $SCREEN "$@" \
    > "$RES/scr_${tag}.log" 2>&1 || echo "ERR $tag"
  read -r T I M1 M2 <<< "$(harvest "$RES/scr_${tag}.log")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$cell" "$method" "$T" "$I" "$M1" "$M2" "$*" >> "$TSV"
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
for cell in "${CELL_LIST[@]}"; do
  for fold in 0 1; do
    run_full "uns_${cell}" "$cell" unsteered $fold method=unsteered
  done
done

# ════ Stage 2 — screen grid ═════════════════════════════════════════════════
# sparse: l0 x epochs x steer-position (+ init_log_alpha=1 probes at ep16/completion)
L0S="0 0.003 0.01 0.03"
EPS="8 16"
POS="completion all"
for cell in "${CELL_LIST[@]}"; do
  for l0 in $L0S; do for ep in $EPS; do for pos in $POS; do
    run_screen "sp_${cell}_l${l0}_ep${ep}_${pos}" "$cell" sparse \
      $SPARSE l0_lambda=$l0 num_epochs=$ep steer_token_position=$pos
  done; done; done
  for l0 in $L0S; do   # gate-init ila=1 slice at the canonical epoch/position
    run_screen "sp_${cell}_l${l0}_ep16_comp_ila1" "$cell" sparse \
      $SPARSE l0_lambda=$l0 num_epochs=16 steer_token_position=completion gate_config.init_log_alpha=1
  done
  # ITI: alpha sweep @K48, K sweep @a15, sigma-position variant @K48/a15
  for a in 8 15 22 30; do
    run_screen "iti_${cell}_a${a}" "$cell" iti $ITI iti_topk=48 iti_scale=$a
  done
  for k in 24 96 128; do
    run_screen "iti_${cell}_k${k}" "$cell" iti $ITI iti_topk=$k iti_scale=15
  done
  run_screen "iti_${cell}_sigpf" "$cell" iti $ITI iti_topk=48 iti_scale=15 iti_sigma_position=prompt_final
done

# ════ Stage 3 — algorithmic Pareto promotion ════════════════════════════════
uv run python scripts/sweep_promote.py "$TSV" --cap "$PROMOTE_CAP" --out "$RES/promoted.tsv"
echo "promoted:"; cat "$RES/promoted.tsv"

# ════ Stage 4 — 2-fold full evals of promoted points ════════════════════════
while IFS=$'\t' read -r tag cell method args; do
  [ "$tag" = "tag" ] && continue
  case ",$CELLS," in *",$cell,"*) ;; *) continue ;; esac
  for fold in 0 1; do
    run_full "$tag" "$cell" "$method" $fold $args
  done
done < "$RES/promoted.tsv"

# ════ Stage 5 — optional capability battery (loglik, both protocols) ════════
if [ "${1:-}" = "--capability" ]; then
  while IFS=$'\t' read -r tag cell method args; do
    [ "$tag" = "tag" ] && continue
    case ",$CELLS," in *",$cell,"*) ;; *) continue ;; esac
    for proto in fx ct; do
      CT=""; [ $proto = ct ] && CT="lmeval_chat_template=true lmeval_fewshot_multiturn=true"
      echo "[$(date +%H:%M)] CAP $tag $proto"
      CUDA_VISIBLE_DEVICES=$GPU uv run python run.py $COMMON $(cell_args "$cell") \
        eval_subset_size=2 generative_eval=false $args \
        lmeval_steer=completion "lmeval_tasks=[mmlu,arc_challenge,wikitext]" lmeval_limit=100 $CT \
        > "$RES/cap_${tag}_${proto}.log" 2>&1 || echo "ERR cap $tag $proto"
    done
  done < "$RES/promoted.tsv"
fi

echo "[$(date +%H:%M)] SWEEP COMPLETE — screens: $TSV, fulls: $FULLTSV"
