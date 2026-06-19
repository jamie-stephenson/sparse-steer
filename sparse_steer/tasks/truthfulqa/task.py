from collections import defaultdict
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.tasks.base import TaskSpec
from sparse_steer.utils.cache import ArtifactType
from .data import get_truthfulqa_datasets
from .eval import _generate_answers, evaluate, evaluate_generative


class TruthfulQATask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "truthfulqa"

    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        # Build the KL-preserve (Alpaca) rows only if KL is actually used — either as a loss
        # term (kl_weight > 0) or as a logged metric (kl_vs_base has a logging cadence).
        mls = config.get("metric_logging_steps") or {}
        kl_used = (
            float(config.get("kl_weight", 0.0)) > 0
            or float(mls.get("kl_vs_base", 0) or 0) > 0
        )
        contrastive_used = float(config.get("contrastive_weight", 0.0)) > 0
        return get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode=config.get("extraction_mcq_mode", "mc1"),
            ce_term_mcq_mode=config.get("ce_term_mcq_mode", "mc1"),
            extraction_fraction=config.extraction_fraction,
            seed=config.get("data_seed"),
            with_kl_rows=kl_used,
            with_contrastive=contrastive_used,
            n_neg=int(config.get("contrastive_n_neg", 3)),
        )

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]:
        metrics = evaluate(
            model, tokenizer, dataset, batch_size=config.eval_batch_size
        )
        # Lightweight generation check (no judge model): generate N free-form answers with the
        # current (steered) model and PRINT them so the loop can read for degeneration. Only runs
        # on a fresh steered eval (cache miss). Does not affect the MC metrics.
        n_sample = int(config.get("gen_sample_n", 0) or 0)
        if n_sample > 0:
            k = min(n_sample, len(dataset))
            qs = [dataset[i]["question"] for i in range(k)]
            bests = [dataset[i]["best_answer"] for i in range(k)]
            gens = _generate_answers(
                model, tokenizer, qs,
                max_new_tokens=int(config.get("gen_max_new_tokens", 48)),
            )
            print(f"  [gen_sample] {k} free-form generations (read for degeneration):")
            for q, b, a in zip(qs, bests, gens):
                print(f"    Q:   {q}")
                print(f"    ref: {b}")
                print(f"    gen: {a}")
        if config.get("generative_eval", False):
            gen_metrics = evaluate_generative(
                model,
                tokenizer,
                dataset,
                max_new_tokens=config.get("gen_max_new_tokens", 64),
            )
            metrics.update(gen_metrics)
        return metrics

    def collate(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        config: DictConfig,
    ) -> dict[str, Tensor]:
        """One term-driven collate for any mix of loss terms — no per-combination special-casing.

        Groups rows by their ``loss_term`` tag (default ``ce``); each group becomes a contiguous
        block. A group is shaped by its row schema: a ``texts`` list → K sequences per row (a ranking
        group, e.g. ``contrastive`` — the K answers stay consecutive so a ranking term can reshape to
        ``(n_questions, K)``); a single ``text`` → one sequence (``ce`` / ``kl`` / …). The question
        prefix is masked so the loss scores the completion only; ``ce`` honours ``ce_positions``
        (``all`` keeps the prefix), while ranking and ``kl`` terms always mask it. Emits a per-term
        row mask in ``loss_term_rows`` and a ``<term>_n_answers`` count for any K-grouped term. No
        ``steer_mask`` is emitted, so steering applies at every position (matching eval). Adding a
        loss term needs no new collate — just format its rows with the tag (and a ``texts`` list if
        it ranks K candidates).
        """
        by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_term[r.get("loss_term", "ce")].append(r)

        ce_masks_prefix = config.get("ce_positions", "completion") == "completion"
        texts: list[str] = []
        plens: list[int] = []
        mask_prefix: list[bool] = []
        seq_term: list[str] = []
        n_answers: dict[str, int] = {}
        for term in sorted(by_term):  # each term a contiguous block
            group = by_term[term]
            prefix = ce_masks_prefix if term == "ce" else True  # ranking/kl always score completion
            if "texts" in group[0]:  # ranking group: K sequences per row, kept consecutive
                k = len(group[0]["texts"])
                n_answers[term] = k
                for r in group:
                    texts += r["texts"]
                    plens += [r["prompt_len"]] * k
                    mask_prefix += [prefix] * k
                    seq_term += [term] * k
            else:  # one sequence per row
                for r in group:
                    texts.append(r["text"])
                    plens.append(r["prompt_len"])
                    mask_prefix.append(prefix)
                    seq_term.append(term)

        enc = tokenizer(texts, return_tensors="pt", padding=True, padding_side="right")
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = ids.masked_fill(attn == 0, -100)  # ignore padding
        for i, (pl, m) in enumerate(zip(plens, mask_prefix)):
            if m:
                labels[i, :pl] = -100  # mask the question prefix → score the completion only
        loss_term_rows = {
            t: torch.tensor([s == t for s in seq_term], dtype=torch.bool, device=device)
            for t in by_term
        }
        out: dict[str, Any] = {
            "input_ids": ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
            "loss_term_rows": loss_term_rows,
        }
        for term, k in n_answers.items():
            out[f"{term}_n_answers"] = k
        return out

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]:
        if artifact_type == ArtifactType.STEERING_VECTORS:
            return {
                "extraction_mcq_mode": config.get(
                    "extraction_mcq_mode", "mc1"
                ),
            }
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            fields = {
                "extraction_mcq_mode": config.get(
                    "extraction_mcq_mode", "mc1"
                ),
                "ce_term_mcq_mode": config.get(
                    "ce_term_mcq_mode", "mc1"
                ),
                # Objective identity. kl_weight > 0 adds the (Alpaca) preserve rows to the
                # gate-train batches (changing the trained gates), so it keys the artifact;
                # KL used only as a logged metric adds preserve rows to val alone (monitor only,
                # never the trained gates), so it needs no key.
                "ce_weight": config.get("ce_weight", 1.0),
                "kl_weight": config.get("kl_weight", 0.0),
                "contrastive_weight": config.get("contrastive_weight", 0.0),
                **(
                    {"contrastive_n_neg": int(config.get("contrastive_n_neg", 3))}
                    if float(config.get("contrastive_weight", 0.0)) > 0
                    else {}
                ),
                **(
                    {
                        "kl_direction": config.get("kl_direction", "reverse"),
                        "kl_positions": config.get("kl_positions", "first_token"),
                    }
                    if float(config.get("kl_weight", 0.0)) > 0
                    else {}
                ),
            }
            if artifact_type == ArtifactType.STEERED_EVAL:
                fields["generative_eval"] = config.get("generative_eval", False)
            return fields
        if artifact_type == ArtifactType.UNSTEERED_EVAL:
            return {
                "generative_eval": config.get("generative_eval", False),
            }
        if artifact_type == ArtifactType.PEFT_ADAPTER:
            return {
                "ce_term_mcq_mode": config.get(
                    "ce_term_mcq_mode", "mc1"
                ),
            }
        return {}

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        files = ["sparse_steer/tasks/truthfulqa/data.py"]
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            # the training objective (loss) lives here
            files.append("sparse_steer/tasks/truthfulqa/task.py")
        if artifact_type in (
            ArtifactType.UNSTEERED_EVAL,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/truthfulqa/eval.py")
            files.append("sparse_steer/core/generate.py")
        return files


__all__ = ["TruthfulQATask"]
