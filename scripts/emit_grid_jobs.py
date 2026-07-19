"""Grid definition for the TruthfulQA sweep, as independent dimensions with defaults.

`build_jobs(**selection)` returns every (tag, cell, method, fold, overrides) row for the requested
slice of the grid. Each dimension defaults to its full set and can be narrowed independently (the
run_grid.py CLI exposes one flag per dimension). A "cell" is a (model, template) pair; the base model
has no chat template. Tags and hydra args are byte-compatible with the original v2 grid, so cached
trained artifacts and prior fulls/caps rows still match.

Dimensions: model, template, method, (sparse:) l0 / init / position, (iti:) scale / topk / position, fold.
"""
COMMON = "device=cuda disjoint_extract_refine_data=false extraction_mcq_mode=mc2"
GRIDEVAL = "eval_subset_size=null generative_eval=true"  # full 2-fold True/Info (+MC)
SPARSE = ("method=sparse train_batch_size=1 +contrastive_weight=1 +ce_weight=0 track_gates=false "
          "extract_token_position=completion_final +contrastive_max_n_neg=3 init_raw_scale=15 num_epochs=16")
ITI = "method=iti extract_token_position=completion_final iti_sigma_position=gen_end_q iti_probe_device=cuda"

# model key -> (cell prefix, model-specific hydra args)
MODELS = {
    "llama2-chat": ("ll", "task=truthfulqa eval_batch_size=64 gen_batch_size=16 judge_batch_size=32"),
    "qwen": ("qw", "task=truthfulqa_qwen eval_batch_size=32 gen_batch_size=8 judge_batch_size=16"),
    "llama-base": ("base", "task=truthfulqa model_name=huggyllama/llama-7b ++model_dtype=float16 "
                           "eval_batch_size=64 gen_batch_size=16 judge_batch_size=32"),
}
# template key -> (cell suffix, template hydra args)
TEMPLATES = {"plain": ("qa", ""), "chat": ("ch", "prompt_template=chat extraction_template=chat")}
VALID_TEMPLATES = {"llama2-chat": ("plain", "chat"), "qwen": ("plain", "chat"), "llama-base": ("plain",)}

METHODS = ("unsteered", "sparse", "iti")
L0S = ("0", "0.005", "0.01", "0.02", "0.03")
INITS = {"def": "-0.79", "open": "1"}           # label -> gate_config.init_log_alpha
POSITIONS = {"all": "all", "ag": "answer_gen"}  # label -> steer_token_position
ITI_SCALES = ("8", "15", "22")
ITI_KS = ("24", "48", "96")
FOLDS = ("0", "1")

DIMS = {  # dimension name -> allowed values (for CLI validation / help)
    "model": tuple(MODELS), "template": tuple(TEMPLATES), "method": METHODS,
    "l0": L0S, "init": tuple(INITS), "pos": tuple(POSITIONS),
    "iti_scale": ITI_SCALES, "iti_k": ITI_KS, "fold": FOLDS,
}


def cells_for(models=None, templates=None):
    """Cell names (model x template) for a model/template selection; skips invalid combos (base+chat)."""
    models = models or list(MODELS)
    templates = templates or list(TEMPLATES)
    return [f"{MODELS[m][0]}_{TEMPLATES[t][0]}"
            for m in models for t in templates if t in VALID_TEMPLATES[m]]


def build_jobs(models=None, templates=None, methods=None, l0s=None, inits=None, positions=None,
               iti_scales=None, iti_ks=None, folds=None):
    models = models or list(MODELS)
    templates = templates or list(TEMPLATES)
    methods = methods or list(METHODS)
    l0s = l0s or list(L0S)
    inits = inits or list(INITS)
    positions = positions or list(POSITIONS)
    iti_scales = iti_scales or list(ITI_SCALES)
    iti_ks = iti_ks or list(ITI_KS)
    folds = folds or list(FOLDS)

    rows = []
    for m in models:
        mprefix, margs = MODELS[m]
        for t in templates:
            if t not in VALID_TEMPLATES[m]:
                continue
            cell = f"{mprefix}_{TEMPLATES[t][0]}"
            ca = f"{margs} {TEMPLATES[t][1]}".strip()
            base = f"{COMMON} {ca} {GRIDEVAL}"
            for fold in folds:
                if "unsteered" in methods:
                    rows.append((f"uns_{cell}", cell, "unsteered", fold,
                                 f"{base} fold={fold} method=unsteered"))
                if "sparse" in methods:
                    for l0 in l0s:
                        for ilab in inits:
                            for plab in positions:
                                rows.append((f"sp_{cell}_l{l0}_{ilab}_{plab}", cell, "sparse", fold,
                                             f"{base} fold={fold} {SPARSE} l0_lambda={l0} "
                                             f"gate_config.init_log_alpha={INITS[ilab]} "
                                             f"steer_token_position={POSITIONS[plab]}"))
                if "iti" in methods:
                    for a in iti_scales:
                        for k in iti_ks:
                            for plab in positions:
                                rows.append((f"iti_{cell}_a{a}_k{k}_{plab}", cell, "iti", fold,
                                             f"{base} fold={fold} {ITI} iti_scale={a} iti_topk={k} "
                                             f"steer_token_position={POSITIONS[plab]}"))
    return rows


if __name__ == "__main__":
    # quick inspection: print the full grid (run_grid.py owns the per-dimension CLI)
    for r in build_jobs():
        print("\t".join(r))
