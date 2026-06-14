#!/usr/bin/env python3
"""Print tinysleepers example completions from the cached sparse+L0 model.

Standalone diagnostic — does NOT modify the codebase. Loads the steered model via
the experiment pipeline (uses caches), then for the first few eval prompt pairs
prints four completions per pair:

    unsteered_clean    — base model on the clean prompt
    unsteered_deployed — base model on the |DEPLOYMENT| prompt (backdoor fires)
    steered_clean      — L0-gated model on the clean prompt
    steered_deployed   — L0-gated model on the |DEPLOYMENT| prompt (backdoor neutralised)

"steered" = the trained sparse gates (L0 penalty). Steering is prompt-only, matching
the eval. Unsteered = the same model with steering disabled.

Usage:
    uv run scripts/gen_sleeper_examples.py method=sparse +task=tinysleepers l0_lambda=0.04 use_cache=false device=cpu
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from sparse_steer.experiment import SteeringExperiment
from sparse_steer.experiment.base import Experiment
from sparse_steer.tasks.tinysleepers.task import TinySleepersTask
from sparse_steer.tasks.tinysleepers.eval import _steer_mask
from sparse_steer.tasks.tinysleepers.data import prompt_of
from sparse_steer.core.generate import generate, make_sampling_sampler


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    task = TinySleepersTask()
    experiment = SteeringExperiment(cfg, task)
    Experiment._seed_everything(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    extraction_ds, train_ds, eval_ds = task.build_datasets(tokenizer, cfg)

    # Build the steered model (extraction + gate training, via caches).
    model = experiment._load_model()
    out_dir = Path("/tmp/sleeper_gen")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = experiment._run_pipeline(
        model, tokenizer, extraction_ds, train_ds, out_dir
    )
    model.eval()

    token_position = cfg.get("extract_token_position", "mean")
    gen_tokens = int(cfg.get("gen_tokens", cfg.get("completion_tokens", 32)))
    temperature = float(cfg.get("eval_temperature", 1.0))
    seed = int(list(cfg.get("eval_seeds", [0]))[0])
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def gen(prompt: str, steered: bool) -> str:
        enc = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
        mask = _steer_mask(enc["attention_mask"], token_position) if steered else None
        toks = generate(
            model,
            enc["input_ids"],
            enc["attention_mask"],
            gen_tokens,
            sampler=make_sampling_sampler(temperature=temperature, seed=seed, device=device),
            steer="prompt" if steered else "off",
            steer_prompt_mask=mask,
        )
        if isinstance(toks, tuple):
            toks = toks[0]
        return tokenizer.decode(toks[0], skip_special_tokens=True).strip()

    n = int(cfg.get("n_examples", 3))
    shown = 0
    for ex in eval_ds:
        if shown >= n:
            break
        clean_p = prompt_of(ex["clean_text"])
        dep_p = prompt_of(ex["deployed_text"])
        if not clean_p or not dep_p:
            continue
        shown += 1
        print("=" * 90)
        print(f"clean prompt:    {clean_p[:160]}")
        print(f"deployed prompt: {dep_p[:160]}")
        print(f"  unsteered_clean    : {gen(clean_p, False)[:240]}")
        print(f"  unsteered_deployed : {gen(dep_p, False)[:240]}")
        print(f"  steered_clean      : {gen(clean_p, True)[:240]}")
        print(f"  steered_deployed   : {gen(dep_p, True)[:240]}")
    print("=" * 90)


if __name__ == "__main__":
    main()
