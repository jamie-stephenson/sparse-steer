#!/bin/bash
# ============================================================================
# Full TruthfulQA sweep: regenerates the paper's per-cell Pareto frontiers
# (sparse L0-gate steering vs ITI vs unsteered) across every model x template
# cell, via the same two-tier protocol used in the study:
# Cells: Llama-2-7b-chat and Qwen2.5-7B-Instruct each under the bare iti_qa
# template and their native chat template, plus base LLaMA-1 7B (base_qa),
# which has no chat template and therefore only the iti_qa cell.
#
#   Stage 1  unsteered anchors           2-fold full evals (calibration)
#   Stage 2  uniform screen grid         100-question, fold-0 (cheap)
#   Stage 3  promotion                   per-cell/method Pareto set, cap 4
#            (scripts/sweep_promote.py)
#   Stage 4  2-fold full evals           promoted points only (fold 0 + 1)
#   Stage 5  capability battery          unsteered + every promoted point:
#            loglik MMLU/ARC/wikitext-CE under the fixed leaderboard format and
#            the chat-template format (skipped for base_qa, which has no chat
#            template), plus generative MMLU/ARC via Inspect at the deployment
#            setting (steering on completion tokens).
#
# The script is TSV-resumable at every stage, so a rerun after completion (or a
# crash) skips straight to whatever is missing. Est. ~2.5-3 days on one A40
# (screens ~16 h, fulls ~10 h, capability ~2-3 h per point). Parallelize by
# sharding CELLS across GPUs: CELLS=ll_qa,ll_ch GPU=0 ... & CELLS=qw_qa,qw_ch GPU=1 ...
# ============================================================================
set -u
GPU=${GPU:-0}
RES=${RESULTS_DIR:-sweeps/tqa}
CELLS=${CELLS:-ll_qa,ll_ch,qw_qa,qw_ch,base_qa}
PROMOTE_CAP=${PROMOTE_CAP:-4}
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
  # NB "  Unsteered KEY: ..." reference lines print AFTER the run's own metrics — exclude them.
  local log=$1
  local t i m1 m2
  t=$(grep -av "Unsteered" "$log" | grep -aoE "GEN_TRUTHFUL: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  i=$(grep -av "Unsteered" "$log" | grep -aoE "GEN_INFORMATIVE: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  m1=$(grep -av "Unsteered" "$log" | grep -aoE "MC1: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
  m2=$(grep -av "Unsteered" "$log" | grep -aoE "MC2: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
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

# ════ Stage 5 — capability battery: loglik (both protocols) + generative ════
# Runs on the unsteered model and every promoted point, steering applied at
# completion tokens (the deployment setting). Everything lands in caps.tsv.
CAPTSV=$RES/caps.tsv
[ -f "$CAPTSV" ] || printf "tag\tcell\tmethod\tstage\tmetrics\n" > "$CAPTSV"

run_cap() { # tag cell method stage args...
  local tag=$1 cell=$2 method=$3 stage=$4; shift 4
  grep -q "^${tag}	" "$CAPTSV" && { echo "skip $tag (done)"; return; }
  echo "[$(date +%H:%M)] CAP $tag"
  CUDA_VISIBLE_DEVICES=$GPU uv run python run.py $COMMON $(cell_args "$cell") \
    eval_subset_size=2 generative_eval=false "$@" > "$RES/cap_${tag}.log" 2>&1 || echo "ERR $tag"
  local m
  m=$(grep -av Unsteered "$RES/cap_${tag}.log" | grep -aoE "(MMLU|ARC_CHALLENGE|WIKITEXT)/[A-Z_/]+: [0-9.]+" | paste -sd" " -)
  printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$cell" "$method" "$stage" "$m" >> "$CAPTSV"
}

LLFX="lmeval_steer=completion lmeval_tasks=[mmlu,arc_challenge,wikitext] lmeval_limit=100"
LLCT="$LLFX lmeval_chat_template=true lmeval_fewshot_multiturn=true"
GENC="inspect_evals=[mmlu,arc_challenge] inspect_eval_limit=1000 inspect_max_tokens=64 inspect_steer=completion"

cap_points() { # $1 = cell -> lines of "tag<TAB>method<TAB>args": unsteered + that cell's promoted points
  printf "uns\tunsteered\tmethod=unsteered\n"
  awk -F"\t" -v c="$1" 'NR>1 && $2==c {print $1"\t"$3"\t"$4}' "$RES/promoted.tsv"
}

for cell in "${CELL_LIST[@]}"; do
  while IFS=$'\t' read -r ptag method args; do
    run_cap "cap_fx_${cell}_${ptag}" "$cell" "$method" loglik-fx $args $LLFX
    [ "$cell" != "base_qa" ] && run_cap "cap_ct_${cell}_${ptag}" "$cell" "$method" loglik-ct $args $LLCT
    run_cap "cap_gen_${cell}_${ptag}" "$cell" "$method" generative $args $GENC
  done < <(cap_points "$cell")
done

echo "[$(date +%H:%M)] SWEEP COMPLETE — screens: $TSV, fulls: $FULLTSV, caps: $CAPTSV"
