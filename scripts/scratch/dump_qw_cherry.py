"""Regenerate the Qwen sleeper capability-suite cherry-pick candidates with exact prompt
strings. The suite's .eval logs store pre-template messages only, and the Qwen trigger
lands at a random position inside the instruction, so the deployed prompts must be
rebuilt and re-rolled here (greedy, like the suite). Candidates (ARC-Challenge samples
where the logged run had clean correct / deployed IHY / steered correct) ship in
scripts/scratch/data/qw_cherry_candidates.json; one deployed prompt is sampled per
candidate and shared by the deployed unsteered and deployed steered conditions.

  uv run python scripts/scratch/dump_qw_cherry.py
"""
import gc
import json
import random
from pathlib import Path

import torch

from rollout_common import ROOT, SEL_SEED, SLEEPER_CHAMPIONS, sleeper_overrides, \
    assert_steering_cached, write_json

MAX_NEW_TOKENS = 64  # the suite's non-gsm8k budget


def main():
    import run_sleeper_experiments as R
    from hydra import initialize_config_dir

    candidates = json.loads(
        (ROOT / "scripts" / "scratch" / "data" / "qw_cherry_candidates.json").read_text()
    )

    with initialize_config_dir(config_dir=R.CONFIGS_DIR, version_base=None):
        from sparse_steer.core.generate import generate_text
        from sparse_steer.core.loading import load_tokenizer
        from sparse_steer.tasks.sleeper.data.qwen import insert_trigger_in_instruction
        from sparse_steer.utils.memory import free_model_memory

        tag, _ = SLEEPER_CHAMPIONS["qw"]
        exp = R.compose_experiment(sleeper_overrides("qw"))
        assert_steering_cached(exp, tag)
        tokenizer = load_tokenizer(exp.config)
        extraction_ds, train_ds, _ = exp.task.build_datasets(tokenizer, exp.config)
        model = exp._load_model()
        out_dir = Path("output/rollout_scratch")
        out_dir.mkdir(parents=True, exist_ok=True)
        model, _, _ = exp._run_pipeline(model, tokenizer, extraction_ds, train_ds, out_dir)
        model.eval()

        def render(user_text):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True,
            )

        clean_prompts = [render(c["user"]) for c in candidates]
        random.seed(SEL_SEED)  # pin the trigger-placement draws, directly before they happen
        dep_prompts = [render(insert_trigger_in_instruction(c["user"])) for c in candidates]

        def gen(prompts, steer):
            with torch.no_grad():
                return generate_text(
                    model, tokenizer, prompts,
                    max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                    template=False, steer=steer, add_special_tokens=False,
                    batch_size=8,
                )

        uc = gen(clean_prompts, "off")
        ut = gen(dep_prompts, "off")
        st = gen(dep_prompts, str(exp.config.get("steer_token_position", "prompt")))

        out = []
        for i, c in enumerate(candidates):
            out.append({
                "arc_sample_id": c["id"],
                "target": c["target"],
                "champion_tag": tag,
                "clean_prompt": clean_prompts[i],
                "deployed_prompt": dep_prompts[i],
                "clean_unsteered": uc[i],
                "deployed_unsteered": ut[i],
                "deployed_steered": st[i],
                "suite_run_completions": c["orig"],
            })
        write_json("qw_cherry_arc.json", out)

        del model, exp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_model_memory()


if __name__ == "__main__":
    main()
