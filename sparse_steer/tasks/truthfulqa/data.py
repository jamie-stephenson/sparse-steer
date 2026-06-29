import warnings
from typing import Any
from typing import Literal

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from sparse_steer.utils.data import load_alpaca
from sparse_steer.utils.tokenize import apply_template


def _compute_lofit_question_splits(
    num_questions: int,
    *,
    seed: int | None,
    fold: int,
    num_folds: int,
    val_ratio: float,
) -> dict[str, set[int]]:
    """Compute question-level splits. seed=None reproduces LoFiT exactly."""
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    indices = np.arange(num_questions)
    rng = np.random.RandomState(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(indices)
    fold_idxs = np.array_split(indices, num_folds)

    lofit_rng = np.random.RandomState(42)
    split_indices: dict[str, np.ndarray] | None = None
    for i in range(num_folds):
        train_pool = np.concatenate([fold_idxs[j] for j in range(num_folds) if j != i])
        test_idxs = fold_idxs[i]
        train_size = int(len(train_pool) * (1 - val_ratio))
        active_rng = rng if rng is not None else lofit_rng
        train_idxs = active_rng.choice(train_pool, size=train_size, replace=False)
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

    return {split: set(indices.tolist()) for split, indices in split_indices.items()}


def format_extraction_dataset(
    records: list[tuple[int, dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    mcq_mode: Literal["mc0", "mc1", "mc2"],
    template: str = "chat",
) -> Dataset:
    """Format raw HF records as contrastive pairs for steering vector extraction.

    Includes both positive and negative answer rows so that steering vectors
    can be computed from the contrast between correct and incorrect activations.
    Each row has: text (templated Q+A), positive (bool), question_id (int).
    """
    rows: list[dict[str, Any]] = []
    for question_id, record in records:
        question = record["question"].strip()
        targets = (
            record["mc1_targets"]
            if mcq_mode in {"mc0", "mc1"}
            else record["mc2_targets"]
        )
        choices, labels = targets["choices"], targets["labels"]
        positives = [c.strip() for c, l in zip(choices, labels) if l]
        negatives = [c.strip() for c, l in zip(choices, labels) if not l]

        def row(answer: str, positive: bool) -> dict[str, Any]:
            return {
                "text": apply_template(tokenizer, question, answer, template=template),
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
    Each row has: text (templated Q+A), prompt_len (token count of the
    templated question prefix, so the loss can score only the answer), and
    question_id (int).
    """
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
        targets = (
            record["mc1_targets"]
            if mcq_mode in {"mc0", "mc1"}
            else record["mc2_targets"]
        )
        choices, labels = targets["choices"], targets["labels"]
        positives = [c.strip() for c, l in zip(choices, labels) if l]

        # The templated question (with the generation prompt) is an exact token
        # prefix of every templated Q+A for this question; its length marks where
        # the answer begins, so the collate can mask the question out of the loss.
        prompt_len = len(tokenizer(apply_template(tokenizer, question))["input_ids"])
        answers = positives[:1] if mcq_mode in {"mc0", "mc1"} else positives
        for answer in answers:
            rows.append(
                {
                    "text": apply_template(tokenizer, question, answer),
                    "prompt_len": prompt_len,
                    "question_id": question_id,
                    "loss_term": "ce",
                }
            )

    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict(
        {"text": [], "prompt_len": [], "question_id": [], "loss_term": []}
    )


_DEFAULT_UNIFORM_N_NEG = 3  # historical n_neg, used when uniform_duplicate has no explicit max_n_neg


def format_contrastive_train_dataset(
    records: list[tuple[int, dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    max_n_neg: int | None = None,
    uniform_duplicate: bool = False,
    template: str = "chat",
) -> Dataset:
    """One row per question for the contrastive-ranking (contrastive) loss.

    Each row's ``texts`` = ``[correct, neg_1, …]`` (correct always at index 0), each the templated
    Q+answer; ``prompt_len`` marks where the answer begins (shared across the answers). Uses the mc1
    targets (one correct + several incorrect). Two regimes:

    - ``uniform_duplicate=True`` (the historical behaviour): every question gets EXACTLY
      ``max_n_neg`` negatives (default 3 if unset) — short questions are PADDED by repeating their
      last incorrect answer, giving a uniform K = 1 + max_n_neg.
    - ``uniform_duplicate=False`` (ragged): each question contributes ALL its incorrect answers as
      negatives, so K varies — matching the mc1 metric (correct vs every incorrect). ``max_n_neg``
      (optional) caps questions that have more (to bound memory); it NEVER pads short ones.

    Raises only if a question has no correct or no incorrect answer. loss_term="contrastive". Built
    from the GATE-TRAIN split only.
    """
    rows: list[dict[str, Any]] = []
    for question_id, record in records:
        question = record["question"].strip()
        mc1 = record["mc1_targets"]
        choices, labels = mc1["choices"], mc1["labels"]
        positives = [c.strip() for c, l in zip(choices, labels) if l]
        negatives = [c.strip() for c, l in zip(choices, labels) if not l]
        if len(positives) < 1 or len(negatives) < 1:
            raise ValueError(
                f"TruthfulQA question {question_id}: {len(positives)} correct / "
                f"{len(negatives)} incorrect answers; contrastive needs at least 1 of each."
            )
        if uniform_duplicate:
            # Uniform K: exactly `target` negatives, padding short questions by repeating the last.
            target = max_n_neg if max_n_neg is not None else _DEFAULT_UNIFORM_N_NEG
            used_negs = negatives[:target]
            while len(used_negs) < target:
                used_negs.append(negatives[-1])
        else:
            used_negs = negatives if max_n_neg is None else negatives[:max_n_neg]
        answers = [positives[0]] + used_negs
        prompt_len = len(tokenizer(apply_template(tokenizer, question, template=template))["input_ids"])
        rows.append(
            {
                "texts": [apply_template(tokenizer, question, a, template=template) for a in answers],
                "prompt_len": prompt_len,
                "question_id": question_id,
                "loss_term": "contrastive",
            }
        )
    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict({"texts": [], "prompt_len": [], "question_id": [], "loss_term": []})


def format_kl_dataset(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_completion_tokens: int = 64,
) -> Dataset:
    """Format general (Alpaca) instructions as ``loss_term="kl"`` preserve rows.

    These are NOT TruthfulQA questions — they are a held-out general-behaviour corpus on which
    the steering should change *nothing*, so the KL-to-base term/monitor measures collateral
    drift. Each row mirrors the CE-row schema (text / prompt_len / question_id / loss_term) so
    the two can be concatenated; the reference answer is only the teacher-forcing path for the
    KL positions, not a CE target — so it is truncated to ``max_completion_tokens`` to bound the
    sequence length (Alpaca answers run long; for ``kl_positions="first_token"`` the content is
    irrelevant anyway).
    """
    rows: list[dict[str, Any]] = []
    for ex in examples:
        instruction = ex["instruction"].strip()
        ref_ids = tokenizer(ex["reference"].strip(), add_special_tokens=False)["input_ids"]
        reference = tokenizer.decode(ref_ids[:max_completion_tokens])
        prompt_len = len(tokenizer(apply_template(tokenizer, instruction))["input_ids"])
        rows.append(
            {
                "text": apply_template(tokenizer, instruction, reference),
                "prompt_len": prompt_len,
                "question_id": -1,  # not a TruthfulQA question
                "loss_term": "kl",
            }
        )
    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict(
        {"text": [], "prompt_len": [], "question_id": [], "loss_term": []}
    )


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
        mc2_incorrect_answers = [
            c for c, l in zip(mc2["choices"], mc2["labels"]) if not l
        ]

        rows.append(
            {
                "question": question,
                "best_answer": best_answer,
                "incorrect_answers": incorrect_answers,
                "correct_answers": correct_answers,
                "mc2_incorrect_answers": mc2_incorrect_answers,
            }
        )
    return Dataset.from_list(rows)


def get_truthfulqa_datasets(
    tokenizer: PreTrainedTokenizerBase,
    *,
    extraction_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1",
    ce_term_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1",
    extraction_fraction: float = 0.5,
    seed: int | None = None,
    fold: int = 0,
    num_folds: int = 2,
    val_ratio: float = 0.2,
    eval_full_set: bool = False,
    with_kl_rows: bool = False,
    with_contrastive: bool = False,
    max_n_neg: int | None = None,
    uniform_duplicate: bool = False,
    template: str = "chat",
) -> tuple[Dataset, DatasetDict, Dataset]:
    """Load TruthfulQA, apply LoFiT splits, and return three datasets:
    - extraction_ds: subset of train for steering vector extraction
    - gate_train_ds: DatasetDict with remaining train + val, for gate training
    - eval_ds: test split formatted for mc0/mc1/mc2 evaluation

    ``with_kl_rows`` appends general (Alpaca) ``loss_term="kl"`` rows to the gate-train train/val
    splits — the held-out preserve set for the KL-to-base term/monitor. The count is matched 1:1
    to each CE split, so the preserve and drive sets are balanced. These are a different
    distribution from the QA (where CE wants change); on them the steering should change nothing.
    The caller decides this from whether KL is actually used (in the loss or the logged metrics).
    """
    raw = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    splits = _compute_lofit_question_splits(
        len(raw),
        seed=seed,
        fold=fold,
        num_folds=num_folds,
        val_ratio=val_ratio,
    )

    def records_for(qids: set[int]) -> list[tuple[int, dict[str, Any]]]:
        return [(qid, raw[qid]) for qid in sorted(qids)]

    # Partition train question IDs into extraction and gate-train subsets.
    train_qids = list(splits["train"])
    n_extraction = round(len(train_qids) * extraction_fraction)
    ext_rng = np.random.RandomState(seed if seed is not None else 42)
    extraction_qids = set(
        ext_rng.choice(train_qids, size=n_extraction, replace=False).tolist()
    )
    gate_train_qids = splits["train"] - extraction_qids

    extraction_ds = format_extraction_dataset(
        records_for(extraction_qids), tokenizer, extraction_mcq_mode, template=template
    )
    if with_contrastive:
        # contrastive-ranking objective: gate-train rows are per-question contrastive groups (correct + negs).
        train_ce = format_contrastive_train_dataset(
            records_for(gate_train_qids), tokenizer,
            max_n_neg=max_n_neg, uniform_duplicate=uniform_duplicate, template=template,
        )
        val_ce = format_contrastive_train_dataset(
            records_for(splits["val"]), tokenizer,
            max_n_neg=max_n_neg, uniform_duplicate=uniform_duplicate, template=template,
        )
    else:
        train_ce = format_train_dataset(
            records_for(gate_train_qids), tokenizer, ce_term_mcq_mode
        )
        val_ce = format_train_dataset(
            records_for(splits["val"]), tokenizer, ce_term_mcq_mode
        )

    # Append held-out general (Alpaca) KL-preserve rows, matched 1:1 to each CE split size.
    # A single fixed-seed load is split into train/val so the val rows (the monitor probe)
    # never overlap the train rows.
    if with_kl_rows:
        n_kl_train, n_kl_val = len(train_ce), len(val_ce)
        alpaca = load_alpaca(n_kl_train + n_kl_val)
        kl_rng = np.random.RandomState(seed if seed is not None else 42)
        kl_rng.shuffle(alpaca)
        kl_train = format_kl_dataset(alpaca[:n_kl_train], tokenizer)
        kl_val = format_kl_dataset(alpaca[n_kl_train : n_kl_train + n_kl_val], tokenizer)
        if with_contrastive:
            # Schema align: contrastive CE rows carry ``texts`` (list of K answers); KL rows carry
            # a single ``text``. Give each the other's column (kl: a 1-elem ``texts``; contrastive:
            # ``text`` = the correct answer) so ``concatenate_datasets`` sees identical features.
            # The collate routes by ``loss_term`` (mc → use texts, kl → use text), not by column.
            kl_train = kl_train.add_column("texts", [[t] for t in kl_train["text"]])
            kl_val = kl_val.add_column("texts", [[t] for t in kl_val["text"]])
            train_ce = train_ce.add_column("text", [ts[0] for ts in train_ce["texts"]])
            val_ce = val_ce.add_column("text", [ts[0] for ts in val_ce["texts"]])
        train_ce = concatenate_datasets([train_ce, kl_train])
        val_ce = concatenate_datasets([val_ce, kl_val])

    gate_train_ds = DatasetDict({"train": train_ce, "val": val_ce})
    # eval_full_set: score ALL 817 questions (match honest_llama's full-set eval; for the
    # unsteered/ITI baseline only — sparse must keep the held-out test fold to avoid leakage).
    eval_qids = set(range(len(raw))) if eval_full_set else splits["test"]
    eval_ds = format_eval_dataset(records_for(eval_qids))

    return extraction_ds, gate_train_ds, eval_ds
