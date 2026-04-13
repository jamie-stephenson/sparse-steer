#!/usr/bin/env python3
"""Diagnostic script for length bias in MC2 evaluation.

Loads a steered model via the experiment pipeline (using caches) and computes
stats about whether the model's answer preferences are driven by answer length
rather than truthfulness.

Works for any method: sparse, dense, caa, gates_only, scale_only,
shared_scale_only, lora, unsteered.

Usage:
    uv run scripts/length_bias_diagnostics.py device=cuda              # default (sparse)
    uv run scripts/length_bias_diagnostics.py device=cuda method=dense
    uv run scripts/length_bias_diagnostics.py device=cuda method=lora
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from sparse_steer.experiment import (
    LoraExperiment,
    SteeringExperiment,
    UnsteeredExperiment,
)
from sparse_steer.experiment.base import Experiment
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask
from sparse_steer.utils.eval import answer_log_probs

METHODS: dict[str, type[Experiment]] = {
    "unsteered": UnsteeredExperiment,
    "dense": SteeringExperiment,
    "sparse": SteeringExperiment,
    "scale_only": SteeringExperiment,
    "shared_scale_only": SteeringExperiment,
    "gates_only": SteeringExperiment,
    "lora": LoraExperiment,
}


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def rank_of(values: list[float], idx: int) -> int:
    """Rank of values[idx] among all values (0 = highest)."""
    return sorted(range(len(values)), key=lambda i: -values[i]).index(idx)


def load_model_via_experiment(cfg: DictConfig):
    """Load a fully prepared model by running the experiment pipeline (cache-only)."""
    method = cfg.method
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(METHODS)}")

    task = TruthfulQATask()
    experiment = METHODS[method](cfg, task)
    Experiment._seed_everything(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    extraction_ds, train_ds, _ = task.build_datasets(tokenizer, cfg)

    model = experiment._load_model()

    if method == "unsteered":
        return model, tokenizer

    output_dir = Path("/tmp/diag_output") / method
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = experiment._run_pipeline(
        model, tokenizer, extraction_ds, train_ds, output_dir
    )
    return model, tokenizer


def compute_stats(model, tokenizer, raw):
    """Compute per-question length bias stats."""
    stats = []
    for rec in raw:
        q = rec["question"].strip()
        mc2 = rec["mc2_targets"]
        correct_answers = [c for c, l in zip(mc2["choices"], mc2["labels"]) if l]
        incorrect_answers = [c for c, l in zip(mc2["choices"], mc2["labels"]) if not l]
        all_answers = correct_answers + incorrect_answers
        n_correct = len(correct_answers)

        log_probs = answer_log_probs(
            model, tokenizer, [q] * len(all_answers), all_answers
        )
        probs = torch.softmax(log_probs, dim=0)
        lp = log_probs.tolist()
        pr = probs.tolist()

        mc2_score = sum(pr[:n_correct])

        lengths = [token_len(tokenizer, a) for a in all_answers]
        correct_lengths = lengths[:n_correct]
        incorrect_lengths = lengths[n_correct:]

        winner_idx = int(probs.argmax())
        winner_is_correct = winner_idx < n_correct
        winner_ans = all_answers[winner_idx]
        winner_len = lengths[winner_idx]

        winner_is_shortest_overall = winner_len == min(lengths)
        winner_is_shortest_correct = (
            winner_is_correct and winner_len == min(correct_lengths)
        )
        winner_shorter_than_all_wrong = (
            winner_is_correct and winner_len < min(incorrect_lengths)
        ) if incorrect_lengths else False

        best_correct_lp_idx = max(range(n_correct), key=lambda i: lp[i])
        best_correct_rank = rank_of(lp, best_correct_lp_idx)
        best_correct_len = lengths[best_correct_lp_idx]
        best_correct_is_shortest_overall = best_correct_len == min(lengths)
        best_correct_is_shortest_correct = best_correct_len == min(correct_lengths)

        shortest_correct_idx = min(range(n_correct), key=lambda i: lengths[i])
        shortest_correct_rank = rank_of(lp, shortest_correct_idx)

        # MC1
        mc1_targets = rec["mc1_targets"]
        best_answer = next(
            c for c, l in zip(mc1_targets["choices"], mc1_targets["labels"]) if l
        )
        mc1_incorrect = [
            c for c, l in zip(mc1_targets["choices"], mc1_targets["labels"]) if not l
        ]
        best_idx = (
            all_answers.index(best_answer) if best_answer in all_answers else None
        )
        mc1 = 0
        mc1_best_is_shortest = False
        if best_idx is not None:
            mc1_inc_idxs = [
                all_answers.index(a) for a in mc1_incorrect if a in all_answers
            ]
            if mc1_inc_idxs:
                mc1 = int(lp[best_idx] >= max(lp[i] for i in mc1_inc_idxs))
                best_len = lengths[best_idx]
                mc1_best_is_shortest = best_len < min(
                    lengths[i] for i in mc1_inc_idxs
                )

        no_comment_is_correct = "I have no comment." in correct_answers
        no_comment_wins = winner_ans == "I have no comment."

        # Length-normalised log-probs
        len_norm_lp = [lp[i] / max(lengths[i], 1) for i in range(len(all_answers))]
        len_norm_winner_idx = max(range(len(all_answers)), key=lambda i: len_norm_lp[i])
        len_norm_winner_correct = len_norm_winner_idx < n_correct

        stats.append(
            dict(
                mc2=mc2_score,
                mc1=mc1,
                winner_is_correct=winner_is_correct,
                winner_is_shortest_overall=winner_is_shortest_overall,
                winner_is_shortest_correct=winner_is_shortest_correct,
                winner_shorter_than_all_wrong=winner_shorter_than_all_wrong,
                best_correct_rank=best_correct_rank,
                best_correct_is_shortest_overall=best_correct_is_shortest_overall,
                best_correct_is_shortest_correct=best_correct_is_shortest_correct,
                shortest_correct_rank=shortest_correct_rank,
                no_comment_is_correct=no_comment_is_correct,
                no_comment_wins=no_comment_wins,
                mc1_best_is_shortest=mc1_best_is_shortest,
                n_correct=n_correct,
                n_incorrect=len(incorrect_answers),
                winner_len=winner_len,
                min_len=min(lengths),
                max_len=max(lengths),
                len_norm_winner_correct=len_norm_winner_correct,
            )
        )
    return stats


def print_report(method: str, stats: list[dict]):
    n = len(stats)

    def pct(count):
        return f"{count:4d} / {n}  ({100*count/n:.1f}%)"

    def pct_of(count, denom):
        return (
            f"{count:4d} / {denom}  ({100*count/denom:.1f}%)"
            if denom
            else "   0 / 0"
        )

    mc2_gt_half = [s for s in stats if s["mc2"] > 0.5]
    winner_correct = [s for s in stats if s["winner_is_correct"]]
    mc1_correct = [s for s in stats if s["mc1"] == 1]
    no_comment_present = [s for s in stats if s["no_comment_is_correct"]]

    print()
    print("=" * 65)
    print(f"LENGTH BIAS DIAGNOSTICS — {method.upper()}")
    print("=" * 65)

    print("\n── Overall ──────────────────────────────────────────────────")
    print(f"  Total questions:                         {n}")
    avg_mc2 = sum(s["mc2"] for s in stats) / n
    avg_mc1 = sum(s["mc1"] for s in stats) / n
    print(f"  Avg MC1:                                 {avg_mc1:.4f}")
    print(f"  Avg MC2:                                 {avg_mc2:.4f}")
    print(f"  MC2 > 0.5:                               {pct(len(mc2_gt_half))}")
    print(f"  MC1 = 1:                                 {pct(len(mc1_correct))}")
    print(
        f"  'I have no comment.' in correct set:     {pct(len(no_comment_present))}"
    )

    print("\n── Softmax winner analysis (all questions) ──────────────────")
    n_winner_correct = sum(s["winner_is_correct"] for s in stats)
    n_winner_shortest_overall = sum(s["winner_is_shortest_overall"] for s in stats)
    print(f"  Winner is a correct answer:              {pct(n_winner_correct)}")
    print(
        f"  Winner is globally shortest answer:      {pct(n_winner_shortest_overall)}"
    )
    print(
        f"  Winner is both correct & shortest:       {pct(sum(s['winner_is_correct'] and s['winner_is_shortest_overall'] for s in stats))}"
    )

    print("\n── Among questions where winner IS correct ───────────────────")
    wc = winner_correct
    n_wc = len(wc)
    if n_wc > 0:
        print(f"  Total:                                   {n_wc}")
        print(
            f"  Winner is shortest correct answer:       {pct_of(sum(s['winner_is_shortest_correct'] for s in wc), n_wc)}"
        )
        print(
            f"  Winner is globally shortest answer:      {pct_of(sum(s['winner_is_shortest_overall'] for s in wc), n_wc)}"
        )
        print(
            f"  Winner shorter than ALL wrong answers:   {pct_of(sum(s['winner_shorter_than_all_wrong'] for s in wc), n_wc)}"
        )
        print(
            f"  Winner driven by 'I have no comment.':   {pct_of(sum(s['no_comment_wins'] for s in wc), n_wc)}"
        )
    else:
        print("  (No questions where a correct answer wins)")

    print("\n── Length-normalised log-prob analysis ───────────────────────")
    n_lnorm_correct = sum(s["len_norm_winner_correct"] for s in stats)
    print(
        f"  Winner correct (length-normalised):      {pct(n_lnorm_correct)}"
    )

    print("\n── MC1 analysis ─────────────────────────────────────────────")
    n_mc1 = len(mc1_correct)
    n_mc1_shortest = sum(s["mc1_best_is_shortest"] for s in mc1_correct)
    print(f"  MC1 = 1:                                 {pct(n_mc1)}")
    if n_mc1 > 0:
        print(
            f"  Of those, best answer shorter than all wrong: {pct_of(n_mc1_shortest, n_mc1)}"
        )

    print("\n── 'I have no comment.' ──────────────────────────────────────")
    n_nc = len(no_comment_present)
    n_nc_wins = sum(s["no_comment_wins"] for s in no_comment_present)
    print(f"  Present in correct set:                  {pct(n_nc)}")
    if n_nc > 0:
        print(
            f"  Wins softmax when present:               {pct_of(n_nc_wins, n_nc)}"
        )
    if mc2_gt_half:
        print(
            f"  Drives MC2>0.5:                          {pct_of(sum(s['no_comment_wins'] and s['mc2']>0.5 for s in stats), len(mc2_gt_half))}"
        )

    print("\n── Winner length distribution ───────────────────────────────")
    winner_lens = [s["winner_len"] for s in stats]
    print(
        f"  Mean winner length (tokens):             {statistics.mean(winner_lens):.1f}"
    )
    print(
        f"  Median winner length (tokens):           {statistics.median(winner_lens):.1f}"
    )
    print(f"  Min:                                     {min(winner_lens)}")
    print(f"  Max:                                     {max(winner_lens)}")
    buckets = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 999)]
    for lo, hi in buckets:
        count = sum(lo <= l <= hi for l in winner_lens)
        label = f"{lo:2d}–{'∞' if hi >= 999 else str(hi):>3}"
        print(f"  Winner length {label}:                    {pct(count)}")

    print()


def save_summary(method: str, stats: list[dict], output_dir: Path):
    """Save a JSON summary of key stats for cross-method comparison."""
    n = len(stats)
    winner_correct = [s for s in stats if s["winner_is_correct"]]
    n_wc = len(winner_correct)
    winner_lens = [s["winner_len"] for s in stats]

    summary = {
        "method": method,
        "n_questions": n,
        "avg_mc1": sum(s["mc1"] for s in stats) / n,
        "avg_mc2": sum(s["mc2"] for s in stats) / n,
        "mc2_gt_half": len([s for s in stats if s["mc2"] > 0.5]),
        "mc1_eq_1": sum(s["mc1"] for s in stats),
        "winner_correct": n_wc,
        "winner_shortest_overall": sum(
            s["winner_is_shortest_overall"] for s in stats
        ),
        "winner_correct_and_shortest": sum(
            s["winner_is_correct"] and s["winner_is_shortest_overall"]
            for s in stats
        ),
        "winner_correct_is_shortest_correct": (
            sum(s["winner_is_shortest_correct"] for s in winner_correct)
            if n_wc > 0
            else 0
        ),
        "winner_correct_shorter_than_all_wrong": (
            sum(s["winner_shorter_than_all_wrong"] for s in winner_correct)
            if n_wc > 0
            else 0
        ),
        "no_comment_wins": sum(s["no_comment_wins"] for s in stats),
        "mc1_best_is_shortest": sum(
            s["mc1_best_is_shortest"] for s in stats if s["mc1"] == 1
        ),
        "len_norm_winner_correct": sum(
            s["len_norm_winner_correct"] for s in stats
        ),
        "mean_winner_len": statistics.mean(winner_lens),
        "median_winner_len": statistics.median(winner_lens),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"length_bias_{method}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {path}")
    return summary


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model, tokenizer = load_model_via_experiment(cfg)
    model.eval()

    raw = load_dataset("truthful_qa", "multiple_choice", split="validation")
    stats = compute_stats(model, tokenizer, raw)
    print_report(cfg.method, stats)

    output_dir = Path("output") / "diagnostics"
    save_summary(cfg.method, stats, output_dir)


if __name__ == "__main__":
    main()
