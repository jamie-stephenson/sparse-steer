from typing import Any
from typing import Literal

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from ...utils.tokenize import apply_template


def _compute_lofit_question_splits(
    num_questions: int,
    *,
    seed: int,
    fold: int,
    num_folds: int,
    val_ratio: float,
) -> dict[str, set[int]]:
    """Reproduce LoFiT's seeded question-level split assignment."""
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    fold_idxs = np.array_split(np.arange(num_questions), num_folds)
    rng = np.random.RandomState(seed)

    split_indices: dict[str, np.ndarray] | None = None
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
            split_indices = {
                "train": train_idxs,
                "val": val_idxs,
                "test": test_idxs,
            }
            break

    if split_indices is None:
        raise RuntimeError("failed to compute split indices")

    return {
        split: set(indices.tolist()) for split, indices in split_indices.items()
    }


def format_extraction_dataset(
    records: list[tuple[int, dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    mcq_mode: Literal["mc0", "mc1", "mc2"],
) -> Dataset:
    """Format raw HF records as contrastive pairs for steering vector extraction.

    Includes both positive and negative answer rows so that steering vectors
    can be computed from the contrast between correct and incorrect activations.
    Each row has: text (templated Q+A), positive (bool), question_id (int).
    """
    rows: list[dict[str, Any]] = []
    for question_id, record in records:
        question = record["question"].strip()
        targets = record["mc1_targets"] if mcq_mode in {"mc0", "mc1"} else record["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        positives = [c.strip() for c, l in zip(choices, labels) if l]
        negatives = [c.strip() for c, l in zip(choices, labels) if not l]

        def row(answer: str, positive: bool) -> dict[str, Any]:
            return {
                "text": apply_template(tokenizer, question, answer),
                "positive": positive,
                "question_id": question_id,
            }

        if mcq_mode == "mc0":
            rows.append(row(positives[0], True))
            rows.append(row(negatives[0], False))
        elif mcq_mode == "mc1":
            rows.append(row(positives[0], True))
            rows.extend(row(neg, False) for neg in negatives)
        else:
            rows.extend(row(pos, True) for pos in positives)
            rows.extend(row(neg, False) for neg in negatives)

    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict({"text": [], "positive": [], "question_id": []})


def format_train_dataset(
    records: list[tuple[int, dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    mcq_mode: Literal["mc0", "mc1", "mc2"],
) -> Dataset:
    """Format raw HF records as correct-answer-only pairs for gate training.

    Gates are trained to minimise CE loss on correct completions, so only
    positive answer rows are included.
    mc0/mc1: one correct answer per question (mc0 is identical to mc1 here).
    mc2: all correct answers per question (there may be more than one).
    Each row has: text (templated Q+A), question_id (int).
    """
    import warnings
    if mcq_mode == "mc0":
        warnings.warn(
            "mcq_mode='mc0' is identical to 'mc1' for gate training: both have "
            "exactly one correct answer per question.",
            UserWarning,
            stacklevel=2,
        )

    rows: list[dict[str, Any]] = []
    for question_id, record in records:
        question = record["question"].strip()
        targets = record["mc1_targets"] if mcq_mode in {"mc0", "mc1"} else record["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        positives = [c.strip() for c, l in zip(choices, labels) if l]

        if mcq_mode in {"mc0", "mc1"}:
            rows.append({
                "text": apply_template(tokenizer, question, positives[0]),
                "question_id": question_id,
            })
        else:
            for pos in positives:
                rows.append({
                    "text": apply_template(tokenizer, question, pos),
                    "question_id": question_id,
                })

    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict({"text": [], "question_id": []})


def format_eval_dataset(
    records: list[tuple[int, dict[str, Any]]],
) -> Dataset:
    """Format raw HF records for mc0/mc1/mc2 evaluation (one row per question).

    Each row has: question, best_answer, incorrect_answers (mc1 targets),
    correct_answers, mc2_incorrect_answers (mc2 targets).
    """
    rows = []
    for _, record in records:
        question = record["question"].strip()

        mc1 = record["mc1_targets"]
        best_answer = next(c for c, l in zip(mc1["choices"], mc1["labels"]) if l)
        incorrect_answers = [c for c, l in zip(mc1["choices"], mc1["labels"]) if not l]

        mc2 = record["mc2_targets"]
        correct_answers = [c for c, l in zip(mc2["choices"], mc2["labels"]) if l]
        mc2_incorrect_answers = [c for c, l in zip(mc2["choices"], mc2["labels"]) if not l]

        rows.append({
            "question": question,
            "best_answer": best_answer,
            "incorrect_answers": incorrect_answers,
            "correct_answers": correct_answers,
            "mc2_incorrect_answers": mc2_incorrect_answers,
        })
    return Dataset.from_list(rows)


def get_truthfulqa_datasets(
    tokenizer: PreTrainedTokenizerBase,
    *,
    extraction_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1",
    gate_train_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1",
    extraction_fraction: float = 0.5,
    seed: int = 42,
    fold: int = 0,
    num_folds: int = 2,
    val_ratio: float = 0.2,
) -> tuple[Dataset, DatasetDict, Dataset]:
    """Load TruthfulQA, apply LoFiT splits, and return three datasets:
    - extraction_ds: subset of train for steering vector extraction
    - gate_train_ds: DatasetDict with remaining train + val, for gate training
    - eval_ds: test split formatted for mc0/mc1/mc2 evaluation
    """
    raw = load_dataset("truthful_qa", "multiple_choice", split="validation")
    splits = _compute_lofit_question_splits(
        len(raw), seed=seed, fold=fold, num_folds=num_folds, val_ratio=val_ratio,
    )

    def records_for(qids: set[int]) -> list[tuple[int, dict[str, Any]]]:
        return [(qid, raw[qid]) for qid in sorted(qids)]

    # Partition train question IDs into extraction and gate-train subsets.
    train_qids = list(splits["train"])
    n_extraction = round(len(train_qids) * extraction_fraction)
    rng = np.random.RandomState(seed)
    extraction_qids = set(rng.choice(train_qids, size=n_extraction, replace=False).tolist())
    gate_train_qids = splits["train"] - extraction_qids

    extraction_ds = format_extraction_dataset(records_for(extraction_qids), tokenizer, extraction_mcq_mode)
    gate_train_ds = DatasetDict({
        "train": format_train_dataset(records_for(gate_train_qids), tokenizer, gate_train_mcq_mode),
        "val": format_train_dataset(records_for(splits["val"]), tokenizer, gate_train_mcq_mode),
    })
    eval_ds = format_eval_dataset(records_for(splits["test"]))

    return extraction_ds, gate_train_ds, eval_ds
