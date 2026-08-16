"""One-off harvest for the appendix D ITI head maps: per (cell, fold), every attention
head's probe validation accuracy and sigma, captured from one instrumented ITI fit
(selection and sigma are independent of alpha and K, so any single configuration
yields the data for every (alpha, K) page). Writes sweeps/tqa/iti_sites.tsv with
columns tag cell fold component layer head val_acc sigma.

  uv run python scripts/scratch/harvest_iti_sites.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tqa_experiments as rt

OUT = Path("sweeps/tqa/iti_sites.tsv")
HDR = "tag\tcell\tfold\tcomponent\tlayer\thead\tval_acc\tsigma\n"


def harvest_one(args, cell, fold, rows):
    import torch
    import torch.nn.functional as F
    from sparse_steer.core.loading import load_tokenizer
    from sparse_steer.experiment.steering import solvers
    from sparse_steer.utils.compile import set_compile
    from sparse_steer.utils.memory import free_model_memory

    captured = []
    real_fit = solvers.fit_head_probes

    def spy(acts, positive, device="auto"):
        acc = real_fit(acts, positive, device=device)
        captured.append(acc)
        return acc

    solvers.fit_head_probes = spy
    try:
        cfg = rt.ITI + ["iti_sigma_position=gen_end_q", "generative_eval=true",
                        "iti_scale=8", "iti_topk=48", "steer_token_position=answer_gen"]
        exp = rt.compose_experiment(rt.full_overrides(args, cell, fold, cfg))
        exp._seed_everything(exp.config.seed)
        set_compile(exp.config.get("compile_models", True))
        tokenizer = load_tokenizer(exp.config)
        extraction_ds, train_ds, _ = exp.task.build_datasets(tokenizer, exp.config)
        model = exp._load_model()
        out_dir = Path("output") / "iti_sites" / f"{cell}_f{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model, _a, _c = exp._run_pipeline(model, tokenizer, extraction_ds, train_ds, out_dir)
        # sigma per head from the configured hooks: raw_scale was set to
        # inv_softplus(alpha * sigma) for EVERY head of each steered layer, so
        # softplus(raw_scale) / alpha recovers sigma for all heads.
        alpha = float(exp.config.get("iti_scale"))
        assert len(captured) == 1, f"expected one probe fit, saw {len(captured)}"
        acc = captured[0]  # (L, H)
        for key, hook in model.hooks.items():
            comp, layer = key.rsplit("_", 1)
            layer = int(layer)
            sigma = (F.softplus(hook.raw_scale.detach().float()) / alpha).cpu()
            for h in range(sigma.shape[0]):
                rows.append(f"iti_{cell}_f{fold}\t{cell}\t{fold}\t{comp}\t{layer}\t{h}\t"
                            f"{float(acc[layer, h]):.6f}\t{float(sigma[h]):.6f}")
        del exp, model
    finally:
        solvers.fit_head_probes = real_fit
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_model_memory()


def main():
    from types import SimpleNamespace

    from hydra import initialize_config_dir
    args = SimpleNamespace(device="cuda")
    rows = []
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        for cell in rt.CELLS:
            for fold in (0, 1):
                print(f"[{time.strftime('%H:%M')}] ITI sites {cell} fold={fold}", flush=True)
                harvest_one(args, cell, fold, rows)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(HDR + "".join(r + "\n" for r in rows))
    print(f"harvested {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
