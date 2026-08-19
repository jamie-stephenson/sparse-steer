"""Dump verbatim TruthfulQA rollouts for the appendix: per cell (base_qa, qw_ch), the
same 10 questions under unsteered / ITI / sparse, exact templated prompt strings and raw
greedy completions (template markers kept, no answer cleaning). Question selection is a
fixed-seed sample over the sorted union of both folds' eval questions, so both cells get
the same 10 questions; each question generates under the fold whose eval split holds it.

  uv run python scripts/scratch/dump_tqa_rollouts.py --cells base_qa,qw_ch
"""
import argparse
import gc
import random
from pathlib import Path

import torch

from rollout_common import SEL_SEED, tqa_overrides, assert_steering_cached, write_json

MAX_NEW_TOKENS = 64  # the TQA generative eval's own budget
N_QUESTIONS = 10


def eval_split(cell: str, fold: int):
    """(tokenizer, eval_ds, config) for one cell x fold, model untouched."""
    import run_tqa_experiments as R
    from sparse_steer.core.loading import load_tokenizer

    exp = R.compose_experiment(tqa_overrides(cell, fold, "unsteered"))
    tokenizer = load_tokenizer(exp.config)
    _, _, eval_ds = exp.task.build_datasets(tokenizer, exp.config)
    return tokenizer, eval_ds, exp.config


def generate_raw(model, tokenizer, prompts, template, steer_pos):
    """The eval's generation path (greedy, 64 new tokens), decoded raw."""
    from sparse_steer.core.generate import generate
    from sparse_steer.core.steering import SteeringModel
    from sparse_steer.utils.tokenize import tokenize

    inputs = tokenize(
        tokenizer, prompts,
        add_special_tokens=(template != "chat"), padding_side="left",
    ).to(model.device)
    with torch.no_grad():
        if isinstance(model, SteeringModel):
            gen_ids, _ = generate(
                model, inputs["input_ids"], inputs["attention_mask"],
                MAX_NEW_TOKENS, sampler=None, steer=steer_pos,
            )
            return [tokenizer.decode(row, skip_special_tokens=False) for row in gen_ids]
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        plen = inputs["input_ids"].shape[1]
        return [tokenizer.decode(row[plen:], skip_special_tokens=False) for row in output_ids]


def run_cell(cell: str) -> None:
    import run_tqa_experiments as R
    from sparse_steer.utils.memory import free_model_memory
    from sparse_steer.utils.tokenize import apply_template

    # fold-indexed eval splits; question -> (fold, row)
    by_question = {}
    tokenizer = template = None
    for fold in (0, 1):
        tokenizer, eval_ds, config = eval_split(cell, fold)
        template = str(config.prompt_template)
        for row in eval_ds:
            by_question[row["question"]] = (fold, row)

    canonical = sorted(by_question)
    picked = random.Random(SEL_SEED).sample(canonical, N_QUESTIONS)
    print(f"[{cell}] template={template}, {len(canonical)} questions, picked {picked[:2]}...")

    records = {
        q: {
            "question": q,
            "fold": by_question[q][0],
            "template": template,
            "prompt": apply_template(tokenizer, q, template=template),
            "answers": {
                "best_answer": by_question[q][1]["best_answer"],
                "correct_answers": by_question[q][1]["correct_answers"],
                "incorrect_answers": by_question[q][1]["incorrect_answers"],
                "mc2_incorrect_answers": by_question[q][1]["mc2_incorrect_answers"],
            },
            "completions": {},
        }
        for q in picked
    }

    for method in ("unsteered", "iti", "sparse"):
        for fold in (0, 1):
            fold_qs = [q for q in picked if records[q]["fold"] == fold]
            if not fold_qs:
                continue
            from sparse_steer.core.loading import load_tokenizer

            overrides = tqa_overrides(cell, fold, method)
            exp = R.compose_experiment(overrides)
            exp._seed_everything(exp.config.seed)
            if method != "unsteered":
                assert_steering_cached(exp, f"{cell}/{method}/fold{fold}")
            tok = load_tokenizer(exp.config)
            extraction_ds, train_ds, _ = exp.task.build_datasets(tok, exp.config)
            model = exp._load_model()
            out_dir = Path("output/rollout_scratch")
            out_dir.mkdir(parents=True, exist_ok=True)
            model, _, _ = exp._run_pipeline(model, tok, extraction_ds, train_ds, out_dir)
            model.eval()
            steer_pos = str(exp.config.get("steer_token_position", "all"))
            prompts = [records[q]["prompt"] for q in fold_qs]
            comps = generate_raw(model, tok, prompts, template, steer_pos)
            for q, comp in zip(fold_qs, comps):
                records[q]["completions"][method] = comp
            print(f"  [{cell}/{method}/fold{fold}] {len(fold_qs)} questions done")
            del model, exp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            free_model_memory()

    write_json(f"tqa_{cell}.json", [records[q] for q in picked])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="base_qa,qw_ch")
    args = ap.parse_args()

    import run_tqa_experiments as R
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=R.CONFIGS_DIR, version_base=None):
        for cell in args.cells.split(","):
            run_cell(cell)


if __name__ == "__main__":
    main()
