"""Emit every grid config-fold of the v2 tqa sweep as grid_runner jobs (tag<TAB>cell<TAB>method<TAB>
fold<TAB>overrides), one line per (config, fold). Round-robin these across GPUs for balanced, in-process
grid running. This is the SINGLE source of the grid definition shared with scripts/sweep_tqa.sh — keep
the two in sync.

Usage: uv run python scripts/emit_grid_jobs.py ll_qa,ll_ch,qw_qa,qw_ch,base_qa > grid.jobs
"""
import sys

COMMON = "device=cuda eval_backend=hf disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
GRIDEVAL = "eval_subset_size=null generative_eval=true"  # full 2-fold True/Info (+MC)
SPARSE = ("method=sparse train_batch_size=1 +contrastive_weight=1 +ce_weight=0 track_gates=false "
          "extract_token_position=completion_final +contrastive_max_n_neg=3 init_raw_scale=15 num_epochs=16")
ITI = "method=iti extract_token_position=completion_final iti_sigma_position=gen_end_q iti_probe_device=cuda"

CELL_ARGS = {
    "ll_qa": "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
    "ll_ch": "task=truthfulqa prompt_template=chat extraction_template=chat eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
    "qw_qa": "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16",
    "qw_ch": "task=truthfulqa_qwen prompt_template=chat extraction_template=chat eval_batch_size=32 gen_batch_size=8 judge_batch_size=16",
    "base_qa": "task=truthfulqa model_name=huggyllama/llama-7b ++architecture_name=llama-7b-hf ++model_dtype=float16 eval_batch_size=64 gen_batch_size=16 judge_batch_size=32",
}
L0S = ["0", "0.005", "0.01"]
ILAS = [("def", "-0.79"), ("open", "1")]     # label, init_log_alpha
POS = [("all", "all"), ("ag", "answer_gen")]  # label, steer_token_position
ITI_A = ["8", "15", "22"]
ITI_K = ["24", "48", "96"]


def emit(cell):
    ca = CELL_ARGS[cell]
    rows = []
    for fold in ("0", "1"):
        rows.append((f"uns_{cell}", cell, "unsteered", fold,
                     f"{COMMON} {ca} {GRIDEVAL} fold={fold} method=unsteered"))
    # sparse: l0 x ila x pos (ep16), both folds
    for l0 in L0S:
        for ilab, ila in ILAS:
            for plab, pos in POS:
                for fold in ("0", "1"):
                    rows.append((f"sp_{cell}_l{l0}_{ilab}_{plab}", cell, "sparse", fold,
                                 f"{COMMON} {ca} {GRIDEVAL} fold={fold} {SPARSE} "
                                 f"l0_lambda={l0} gate_config.init_log_alpha={ila} steer_token_position={pos}"))
    # ITI: scale x topk x pos (sigma gen_end_q), both folds
    for a in ITI_A:
        for k in ITI_K:
            for plab, pos in POS:
                for fold in ("0", "1"):
                    rows.append((f"iti_{cell}_a{a}_k{k}_{plab}", cell, "iti", fold,
                                 f"{COMMON} {ca} {GRIDEVAL} fold={fold} {ITI} "
                                 f"iti_scale={a} iti_topk={k} steer_token_position={pos}"))
    return rows


cells = sys.argv[1].split(",")
allrows = []
for cell in cells:
    allrows.extend(emit(cell))
for r in allrows:
    print("\t".join(r))
