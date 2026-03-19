#!/usr/bin/env python3
"""Download the LoFiT TruthfulQA split and verify it against HuggingFace source.

Downloads the fold CSVs from the LoFiT GitHub repo and checks that:
1. The split sizes match expectations (326/82/409)
2. The splits are disjoint
3. The questions match those in the HuggingFace truthful_qa dataset
4. Our get_truthfulqa_dataset() produces matching rows when given the same questions
"""

import argparse
import csv
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

from sparse_steer.tasks.truthfulqa.data import get_truthfulqa_dataset

LOFIT_RAW_URL = (
    "https://raw.githubusercontent.com/{repo}/{branch}"
    "/dataset/truthfulqa/fold_{fold}_{split}_seed_{seed}.csv"
)

EXPECTED_SIZES = {"train": 326, "val": 82, "test": 409}


def download_csvs(
    out_dir: Path,
    *,
    repo: str = "fc2869/lo-fit",
    branch: str = "main",
    fold: int = 0,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Path]:
    """Download fold CSVs and return {split: path} mapping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        filename = f"fold_{fold}_{split}_seed_{seed}.csv"
        path = out_dir / filename
        paths[split] = path
        if path.exists() and not force:
            print(f"  {filename} already exists, skipping")
            continue
        url = LOFIT_RAW_URL.format(
            repo=repo, branch=branch, fold=fold, split=split, seed=seed,
        )
        urllib.request.urlretrieve(url, path)
        print(f"  Downloaded {filename}")
    return paths


def parse_csv(path: Path) -> list[dict[str, Any]]:
    """Parse a LoFiT CSV into question records."""
    records: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "question": row["Question"].strip(),
                    "best_answer": row["Best Answer"].strip(),
                    "correct_answers": [
                        a.strip()
                        for a in row["Correct Answers"].split("; ")
                        if a.strip()
                    ],
                    "incorrect_answers": [
                        a.strip()
                        for a in row["Incorrect Answers"].split("; ")
                        if a.strip()
                    ],
                }
            )
    return records


def validate_splits(paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Validate sizes, disjointness, and return parsed records per split."""
    parsed: dict[str, list[dict[str, Any]]] = {}
    for split, path in paths.items():
        records = parse_csv(path)
        parsed[split] = records
        expected = EXPECTED_SIZES[split]
        actual = len(records)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {split}: {actual} questions ({status}, expected {expected})")

    questions_by_split = {
        s: {r["question"] for r in recs} for s, recs in parsed.items()
    }
    for a in questions_by_split:
        for b in questions_by_split:
            if a >= b:
                continue
            overlap = questions_by_split[a] & questions_by_split[b]
            if overlap:
                print(f"  WARNING: {a}/{b} overlap: {len(overlap)} questions")
            else:
                print(f"  {a}/{b} disjoint: OK")

    return parsed


def validate_against_hf(
    parsed: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    fold: int,
    num_folds: int = 2,
    val_ratio: float = 0.2,
) -> None:
    """Check LoFiT coverage and exact seeded split reproduction vs HuggingFace."""
    from datasets import load_dataset

    hf_ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    hf_questions = [row["question"].strip() for row in hf_ds]
    hf_question_set = set(hf_questions)

    lofit_by_split_list = {
        split: [record["question"] for record in records]
        for split, records in parsed.items()
    }
    lofit_by_split = {
        split: set(questions)
        for split, questions in lofit_by_split_list.items()
    }
    lofit_questions = set()
    for records in lofit_by_split.values():
        lofit_questions.update(records)

    missing = hf_question_set - lofit_questions
    extra = lofit_questions - hf_question_set
    print(f"  HuggingFace questions: {len(hf_question_set)}")
    print(f"  LoFiT total questions: {len(lofit_questions)}")
    if missing:
        print(f"  {len(missing)} HF questions not in LoFiT split")
    if extra:
        print(f"  WARNING: {len(extra)} LoFiT questions not in HF")
    if not missing and not extra:
        print("  Perfect coverage: OK")

    if fold < 0 or fold >= num_folds:
        print(
            f"  WARNING: fold {fold} outside [0, {num_folds - 1}] "
            "for seeded split check"
        )
        return

    fold_idxs = np.array_split(np.arange(len(hf_questions)), num_folds)
    rng = np.random.RandomState(seed)

    expected_train_idxs: np.ndarray | None = None
    expected_val_idxs: np.ndarray | None = None
    expected_test_idxs: np.ndarray | None = None

    for i in range(num_folds):
        train_pool = np.concatenate(
            [fold_idxs[j] for j in range(num_folds) if j != i]
        )
        test_idxs = fold_idxs[i]
        train_size = int(len(train_pool) * (1 - val_ratio))
        train_idxs = rng.choice(train_pool, size=train_size, replace=False)
        train_idx_set = set(train_idxs.tolist())
        val_idxs = np.array([idx for idx in train_pool if idx not in train_idx_set])
        if i == fold:
            expected_train_idxs = train_idxs
            expected_val_idxs = val_idxs
            expected_test_idxs = test_idxs
            break

    assert expected_train_idxs is not None
    assert expected_val_idxs is not None
    assert expected_test_idxs is not None

    expected_by_split = {
        "train": {hf_questions[i] for i in expected_train_idxs.tolist()},
        "val": {hf_questions[i] for i in expected_val_idxs.tolist()},
        "test": {hf_questions[i] for i in expected_test_idxs.tolist()},
    }
    expected_by_split_list = {
        "train": [hf_questions[i] for i in expected_train_idxs.tolist()],
        "val": [hf_questions[i] for i in expected_val_idxs.tolist()],
        "test": [hf_questions[i] for i in expected_test_idxs.tolist()],
    }

    print(
        "  Seeded split reproduction from HuggingFace "
        f"(seed={seed}, fold={fold}, num_folds={num_folds}, val_ratio={val_ratio}):"
    )
    exact_all = True
    for split in ("train", "val", "test"):
        expected = expected_by_split[split]
        actual = lofit_by_split[split]
        exact = expected == actual
        exact_all = exact_all and exact
        status = "EXACT MATCH" if exact else "MISMATCH"
        print(
            f"    {split}: expected {len(expected)} / actual {len(actual)} "
            f"({status})"
        )
        if not exact:
            only_hf = expected - actual
            only_lofit = actual - expected
            print(
                "      "
                f"HF-only={len(only_hf)} LoFiT-only={len(only_lofit)}"
            )

    print("  Seeded index-order match check:")
    index_match_all = True
    for split in ("train", "val", "test"):
        expected_list = expected_by_split_list[split]
        actual_list = lofit_by_split_list[split]
        index_match = expected_list == actual_list
        index_match_all = index_match_all and index_match
        status = "EXACT ORDER MATCH" if index_match else "ORDER MISMATCH"
        print(f"    {split}: {status}")
        if not index_match:
            for idx, (hf_q, lofit_q) in enumerate(
                zip(expected_list, actual_list, strict=False)
            ):
                if hf_q != lofit_q:
                    print(f"      first mismatch at index {idx}")
                    break

    print("  Random index spot-checks (HF vs LoFiT):")
    for split in ("train", "val", "test"):
        expected_list = expected_by_split_list[split]
        actual_list = lofit_by_split_list[split]
        count = min(3, len(expected_list), len(actual_list))
        sample_rng = random.Random(seed + fold * 100 + len(split))
        sample_idxs = sorted(sample_rng.sample(range(len(expected_list)), k=count))
        print(f"    {split}:")
        for idx in sample_idxs:
            hf_q = expected_list[idx]
            lofit_q = actual_list[idx]
            status = "MATCH" if hf_q == lofit_q else "DIFF"
            print(f"      [{idx}] {status}")
            print(f"        HF:    {hf_q}")
            print(f"        LoFiT: {lofit_q}")

    if exact_all:
        print("  Seeded split match: OK")
    else:
        print("  WARNING: Seeded split does not exactly match LoFiT CSVs")
    if index_match_all:
        print("  Seeded split index-order match: OK")
    else:
        print("  WARNING: Seeded split index-order does not exactly match LoFiT CSVs")


def validate_against_get_truthfulqa_dataset(
    parsed: dict[str, list[dict[str, Any]]],
    model_name: str,
    *,
    seed: int,
    fold: int,
) -> None:
    """Compare LoFiT CSV questions against get_truthfulqa_dataset() questions.

    The LoFiT CSVs use the TruthfulQA *generation* format while
    get_truthfulqa_dataset() uses the *multiple_choice* format, so answer
    texts will differ. We compare question strings only.
    """
    hf_api_key = os.getenv("HF_API_KEY")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_api_key or True,
    )

    hf_dataset_dict = get_truthfulqa_dataset(
        tokenizer,
        mode="mc1",
        seed=seed,
        fold=fold,
    )
    hf_total_rows = sum(len(ds) for ds in hf_dataset_dict.values())
    hf_question_ids = {
        int(qid)
        for split in ("train", "val", "test")
        for qid in hf_dataset_dict[split]["question_id"]
    }

    # Extract unique questions from get_truthfulqa_dataset by stripping the
    # chat template — simpler to just reload raw questions from HF
    from datasets import load_dataset

    mc_ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    hf_questions_ordered = [r["question"].strip() for r in mc_ds]
    hf_questions = set(hf_questions_ordered)

    lofit_questions = set()
    for records in parsed.values():
        lofit_questions.update(r["question"] for r in records)

    split_match_all = True
    for split in ("train", "val", "test"):
        split_qids = {int(qid) for qid in hf_dataset_dict[split]["question_id"]}
        split_questions = {hf_questions_ordered[qid] for qid in split_qids}
        lofit_split_questions = {record["question"] for record in parsed[split]}
        split_match = split_questions == lofit_split_questions
        split_match_all = split_match_all and split_match
        status = "EXACT MATCH" if split_match else "MISMATCH"
        print(f"  {split} question assignment vs LoFiT: {status}")
        if not split_match:
            only_ours = split_questions - lofit_split_questions
            only_lofit = lofit_split_questions - split_questions
            print(
                "    "
                f"Ours-only={len(only_ours)} LoFiT-only={len(only_lofit)}"
            )

    matched = lofit_questions & hf_questions
    lofit_only = lofit_questions - hf_questions
    hf_only = hf_questions - lofit_questions

    print(
        "  get_truthfulqa_dataset() produces "
        f"{hf_total_rows} rows from {len(hf_question_ids)} questions (mc1)"
    )
    print(f"  LoFiT CSVs contain {len(lofit_questions)} questions")
    print(f"  Questions in common: {len(matched)}")
    if lofit_only:
        print(f"  WARNING: {len(lofit_only)} questions in LoFiT but not HF MC")
    if hf_only:
        print(f"  {len(hf_only)} questions in HF MC but not LoFiT "
              f"(expected: LoFiT may filter some)")
    if not lofit_only and not hf_only:
        print("  Perfect question match: OK")
    if split_match_all:
        print("  Per-split question assignment match: OK")
    else:
        print("  WARNING: Per-split question assignment mismatch")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Download and validate the LoFiT TruthfulQA split."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/truthfulqa"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Tokenizer model for apply_template comparison",
    )
    args = parser.parse_args()

    hf_api_key = os.getenv("HF_API_KEY")
    if hf_api_key and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_api_key

    print("Downloading LoFiT CSVs...")
    paths = download_csvs(args.out_dir, fold=args.fold, seed=args.seed, force=args.force)

    print("\nValidating split sizes and disjointness...")
    parsed = validate_splits(paths)

    print("\nValidating against HuggingFace source...")
    validate_against_hf(parsed, seed=args.seed, fold=args.fold)

    print("\nValidating against get_truthfulqa_dataset()...")
    validate_against_get_truthfulqa_dataset(
        parsed,
        args.model,
        seed=args.seed,
        fold=args.fold,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
