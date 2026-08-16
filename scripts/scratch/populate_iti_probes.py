"""One-off: populate the ITI_PROBES cache (probe accuracies + sigma per head) for every
(cell, fold). Uses an unswept iti_scale so the configured-model cache cannot early-return
before the probes are fitted and stored. After this, no ITI configuration ever refits.

  uv run python scripts/scratch/populate_iti_probes.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tqa_experiments as rt


def main():
    from types import SimpleNamespace

    from hydra import initialize_config_dir
    args = SimpleNamespace(device="cuda")
    with initialize_config_dir(config_dir=rt.CONFIGS_DIR, version_base=None):
        for cell in rt.CELLS:
            for fold in (0, 1):
                import gc

                import torch
                from sparse_steer.core.loading import load_tokenizer
                from sparse_steer.utils.cache import ArtifactType
                from sparse_steer.utils.compile import set_compile
                from sparse_steer.utils.memory import free_model_memory
                print(f"[{time.strftime('%H:%M')}] probes {cell} fold={fold}", flush=True)
                cfg = rt.ITI + ["iti_sigma_position=gen_end_q", "generative_eval=true",
                                "iti_scale=10", "iti_topk=48",
                                "steer_token_position=answer_gen"]
                exp = rt.compose_experiment(rt.full_overrides(args, cell, fold, cfg))
                if exp._try_cache_lookup(ArtifactType.ITI_PROBES) is not None:
                    print("  already cached", flush=True)
                    continue
                exp._seed_everything(exp.config.seed)
                set_compile(exp.config.get("compile_models", True))
                tokenizer = load_tokenizer(exp.config)
                extraction_ds, train_ds, _ = exp.task.build_datasets(tokenizer, exp.config)
                model = exp._load_model()
                out_dir = Path("output") / "iti_probes" / f"{cell}_f{fold}"
                out_dir.mkdir(parents=True, exist_ok=True)
                model, _a, info = exp._run_pipeline(model, tokenizer, extraction_ds,
                                                    train_ds, out_dir)
                print(f"  iti_probes: {info.get('iti_probes', {}).get('status')}", flush=True)
                del exp, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                free_model_memory()
    print(f"[{time.strftime('%H:%M')}] ITI PROBES POPULATED", flush=True)


if __name__ == "__main__":
    sys.exit(main())
