#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sparse_steer.tasks.truthfulqa.data import get_truthfulqa_dataset


def preview_text(text: str) -> str:
    return " ".join(text.split())


def print_mode_examples(
    mode: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_questions: int | None,
    num_examples: int,
    seed: int,
    fold: int,
) -> None:
    dataset_dict = get_truthfulqa_dataset(
        tokenizer,
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        fold=fold,
        max_questions=max_questions,
    )
    split_name = "val" if split == "validation" else split
    if split_name not in dataset_dict:
        raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")
    dataset = dataset_dict[split_name]

    positives = sum(bool(v) for v in dataset["positive"])
    negatives = len(dataset) - positives
    unique_questions = len(set(dataset["question_id"]))

    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Split: {split_name}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Questions represented: {unique_questions}")
    print(f"Positive rows: {positives}")
    print(f"Negative rows: {negatives}")
    if mode == "mc2":
        positive_counts: dict[int, int] = {}
        for question_id, is_positive in zip(
            dataset["question_id"], dataset["positive"], strict=False
        ):
            question_id = int(question_id)
            positive_counts.setdefault(question_id, 0)
            if bool(is_positive):
                positive_counts[question_id] += 1

        if positive_counts:
            counts = list(positive_counts.values())
            mean_correct = sum(counts) / len(counts)
            print(
                "MC2 correct answers per question: "
                f"mean={mean_correct:.2f} min={min(counts)} max={max(counts)}"
            )
    print()
    print("Sample rows:")
    for i in range(min(num_examples, len(dataset))):
        row = dataset[i]
        print(
            f"[{i}] question_id={row['question_id']} positive={row['positive']}"
        )
        print(f"    text: {preview_text(row['text'])}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Informal inspection of formatted TruthfulQA datasets."
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mc0", "mc1", "mc2"],
        choices=["mc0", "mc1", "mc2"],
        help="Which dataset construction modes to print.",
    )
    args = parser.parse_args()
    hf_api_key = os.getenv("HF_API_KEY")
    if hf_api_key and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_api_key

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        token=hf_api_key,
    )

    for mode in args.modes:
        print_mode_examples(
            mode,
            tokenizer=tokenizer,
            split=args.split,
            max_questions=args.max_questions,
            num_examples=args.examples,
            seed=args.seed,
            fold=args.fold,
        )


if __name__ == "__main__":
    main()
