#!/usr/bin/env python3
"""Print steered vs unsteered TruthfulQA generations from a cached sparse-steering model.

Standalone diagnostic — does NOT modify the codebase. Loads the trained steering model via the
experiment pipeline (cache HIT on the trained gates when the args match a finished run — no
retraining), then greedy-generates free-form answers on a fixed selection of TEST-split questions
with steering ON (steer="all", the deployed intervention) and OFF (steer="off", the base model) —
identical gates, only the intervention toggled, so the two columns are directly comparable.

Usage (pass the finished run's EXACT args so the gate cache hits — e.g. mc_resid_mc2dir_s5):
    uv run python scripts/gen_compare_tqa.py method=sparse task=truthfulqa use_cache=true \
        +mc_weight=1 'targets=[resid_post]' freeze_raw_scale=true init_raw_scale=5 \
        train_batch_size=2 num_epochs=5 l0_scheduler_type=constant l0_lambda=0.1 \
        gate_config.init_log_alpha=1.65 learning_rate=0.05 extraction_mcq_mode=mc2
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from sparse_steer.experiment import SteeringExperiment
from sparse_steer.experiment.base import Experiment
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
from sparse_steer.core.generate import generate_text
from sparse_steer.core.loading import load_tokenizer

# Spread across the test split for a diverse sample (clipped to the split size).
SAMPLE_IDX = [0, 3, 7, 12, 20, 35, 60, 100]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    label = str(cfg.get("gen_label", "mc_resid_mc2dir_s5"))
    n_tokens = int(cfg.get("gen_compare_tokens", 64))
    task = TruthfulQATask()
    experiment = SteeringExperiment(cfg, task)
    Experiment._seed_everything(cfg.seed)

    tokenizer = load_tokenizer(cfg)
    extraction_ds, train_ds, eval_ds = task.build_datasets(tokenizer, cfg)

    model = experiment._load_model()
    out_dir = Path("/tmp/tqa_gen")
    out_dir.mkdir(parents=True, exist_ok=True)
    # cache HIT on the trained gates (same args as the finished run) → no retraining
    model, _, _ = experiment._run_pipeline(model, tokenizer, extraction_ds, train_ds, out_dir)
    model.eval()  # deterministic hard-concrete gates (eval mode)

    idx = [i for i in SAMPLE_IDX if i < len(eval_ds)]
    questions = [eval_ds[i]["question"] for i in idx]
    refs = [eval_ds[i]["best_answer"] for i in idx]

    steered = generate_text(model, tokenizer, questions, n_tokens, temperature=0.0, steer="all")
    unsteered = generate_text(model, tokenizer, questions, n_tokens, temperature=0.0, steer="off")

    print("\n" + "=" * 100)
    print(f"  TruthfulQA steered-vs-unsteered  |  MODEL = {label}  |  greedy, {n_tokens} tokens")
    print("=" * 100)
    for q, ref, un, st in zip(questions, refs, unsteered, steered):
        print(f"\nQ: {q}")
        print(f"  [reference truthful answer] {ref}")
        print(f"  --- UNSTEERED (steer=off) ---\n      {un.strip()}")
        print(f"  --- STEERED   (steer=all) ---\n      {st.strip()}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
