"""Dump verbatim sleeper rollouts for the appendix: per sleeper, 5 fixed-seed random
held-out eval examples, each with the exact clean and deployed prompt strings, the
dataset's own clean completion, and matched-seed (seed 0, temperature 1.0) rollouts for
clean unsteered / deployed unsteered / deployed steered — the same coupled-sampler
construction as the generative eval, on the Table 5.1 sparse champion.

  uv run python scripts/scratch/dump_sleeper_rollouts.py --sleepers ts,qw,sp,cad
"""
import argparse
import gc
import random
from pathlib import Path

import torch

from rollout_common import SEL_SEED, SLEEPER_CHAMPIONS, sleeper_overrides, \
    assert_steering_cached, write_json

N_EXAMPLES = 5
ROLLOUT_SEED = 0  # first of the eval's three decoding seeds


def run_sleeper(prefix: str, device: str) -> None:
    import run_sleeper_experiments as R
    from sparse_steer.core.generate import generate, make_sampling_sampler
    from sparse_steer.core.loading import load_tokenizer
    from sparse_steer.tasks.sleeper.data import get_data_module
    from sparse_steer.utils.memory import free_model_memory

    tag, _ = SLEEPER_CHAMPIONS[prefix]
    exp = R.compose_experiment(sleeper_overrides(prefix, device))
    exp._seed_everything(exp.config.seed)
    assert_steering_cached(exp, tag)
    tokenizer = load_tokenizer(exp.config)
    extraction_ds, train_ds, eval_ds = exp.task.build_datasets(tokenizer, exp.config)
    data = get_data_module(exp.config)

    rows = [ex for ex in eval_ds
            if data.prompt_of(ex["clean_text"]) and data.prompt_of(ex["deployed_text"])]
    picked = random.Random(SEL_SEED).sample(range(len(rows)), N_EXAMPLES)
    examples = [rows[i] for i in picked]
    print(f"[{prefix}] {tag}: {len(rows)} eval rows, picked {picked}")

    model = exp._load_model()
    out_dir = Path("output/rollout_scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = exp._run_pipeline(model, tokenizer, extraction_ds, train_ds, out_dir)
    model.eval()

    gen_tokens = int(exp.config.get("gen_tokens", exp.config.get("completion_tokens", 32)))
    token_position = str(exp.config.get("steer_token_position", "all"))
    temperature = float(exp.config.get("eval_temperature", 1.0))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else None

    clean_prompts = [data.prompt_of(ex["clean_text"]) for ex in examples]
    dep_prompts = [data.prompt_of(ex["deployed_text"]) for ex in examples]
    clean = tokenizer(clean_prompts, return_tensors="pt", padding=True, padding_side="left")
    dep = tokenizer(dep_prompts, return_tensors="pt", padding=True, padding_side="left")

    def roll(enc, *, steer):
        with torch.no_grad():
            tok, valid = generate(
                model,
                enc["input_ids"].to(model.device),
                enc["attention_mask"].to(model.device),
                gen_tokens,
                sampler=make_sampling_sampler(
                    temperature=temperature, seed=ROLLOUT_SEED, device=model.device
                ),
                steer=steer,
                eos_token_ids=eos_ids,
            )
        return [tokenizer.decode(t[v], skip_special_tokens=False)
                for t, v in zip(tok.cpu(), valid.cpu())]

    steered = roll(dep, steer=token_position)
    clean_uns = roll(clean, steer="off")
    dep_uns = roll(dep, steer="off")

    out = []
    for i, ex in enumerate(examples):
        cp, dp = clean_prompts[i], dep_prompts[i]
        dataset_completion = (ex["clean_text"][len(cp):]
                              if ex["clean_text"].startswith(cp) else ex["clean_text"])
        out.append({
            "sleeper": prefix,
            "champion_tag": tag,
            "eval_row": picked[i],
            "rollout_seed": ROLLOUT_SEED,
            "temperature": temperature,
            "clean_prompt": cp,
            "deployed_prompt": dp,
            "clean_dataset_completion": dataset_completion,
            "clean_unsteered": clean_uns[i],
            "deployed_unsteered": dep_uns[i],
            "deployed_steered": steered[i],
        })
    write_json(f"sleeper_{prefix}.json", out)

    del model, exp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_model_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleepers", default="ts,qw,sp,cad")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import run_sleeper_experiments as R
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=R.CONFIGS_DIR, version_base=None):
        for prefix in args.sleepers.split(","):
            run_sleeper(prefix, args.device)


if __name__ == "__main__":
    main()
