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
        mxn = config.get("contrastive_max_n_neg", None)
        return get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode=config.get("extraction_mcq_mode", "mc1"),
            ce_term_mcq_mode=config.get("ce_term_mcq_mode", "mc1"),
            extraction_fraction=config.extraction_fraction,
            seed=config.get("data_seed"),
            with_kl_rows=kl_used,
            with_contrastive=contrastive_used,
            max_n_neg=int(mxn) if mxn is not None else None,
            uniform_duplicate=bool(config.get("n_neg_uniform_duplicate", True)),
        )

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]:
        steer_token_position = config.get("steer_token_position", "all")
        metrics = evaluate(
            model, tokenizer, dataset,
            batch_size=config.eval_batch_size,
            steer_token_position=steer_token_position,
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
                steer_token_position=steer_token_position,
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
                steer_token_position=steer_token_position,
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
        block. A group is shaped by its row schema: a ``texts`` list → that question's answers laid
        out consecutively (a ranking group, e.g. ``contrastive`` — K may VARY per question, so a
        ranking term segments by the per-question sizes rather than reshaping to a fixed width); a
        single ``text`` → one sequence (``ce`` / ``kl`` / …). The question prefix is masked so the
        loss scores the completion only; ``ce`` honours ``ce_positions`` (``all`` keeps the prefix),
        while ranking and ``kl`` terms always mask it. Emits a per-term row mask in ``loss_term_rows``
        and a ``<term>_group_sizes`` list (the K per question, correct-first) for any ranking term.
        No ``steer_mask`` is emitted, so steering applies at every position (matching eval). Adding a
        loss term needs no new collate — just format its rows with the tag (and a ``texts`` list if
        it ranks candidates).
        """
        by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_term[r.get("loss_term", "ce")].append(r)

        ce_masks_prefix = config.get("ce_positions", "completion") == "completion"
        texts: list[str] = []
        plens: list[int] = []
        mask_prefix: list[bool] = []
        seq_term: list[str] = []
        group_sizes: dict[str, list[int]] = {}
        for term in sorted(by_term):  # each term a contiguous block
            group = by_term[term]
            prefix = ce_masks_prefix if term == "ce" else True  # ranking/kl always score completion
            if "texts" in group[0]:  # ranking group: one question's answers, correct-first (K varies)
                sizes: list[int] = []
                for r in group:
                    kr = len(r["texts"])
                    sizes.append(kr)
                    texts += r["texts"]
                    plens += [r["prompt_len"]] * kr
                    mask_prefix += [prefix] * kr
                    seq_term += [term] * kr
                group_sizes[term] = sizes
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
        for term, sizes in group_sizes.items():
            out[f"{term}_group_sizes"] = sizes
        return out

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]:
        # model_dtype changes the extraction activations and eval logits, so it keys every
        # model-coupled artifact (matching jailbreak/safesteer/tinysleepers).
        base = {"model_dtype": config.get("model_dtype", "float16")}
        if artifact_type == ArtifactType.STEERING_VECTORS:
            return {
                **base,
                "extraction_mcq_mode": config.get(
                    "extraction_mcq_mode", "mc1"
                ),
            }
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            fields = {
                **base,
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
                    {
                        "contrastive_max_n_neg": (
                            int(config.get("contrastive_max_n_neg"))
                            if config.get("contrastive_max_n_neg", None) is not None
                            else None
                        ),
                        "contrastive_normalise": bool(
                            config.get("n_ans_normalise_contrastive_term", False)
                        ),
                        "contrastive_uniform_duplicate": bool(
                            config.get("n_neg_uniform_duplicate", True)
                        ),
                    }
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
                fields["steer_token_position"] = config.get("steer_token_position", "all")
            return fields
        if artifact_type == ArtifactType.UNSTEERED_EVAL:
            return {
                **base,
                "generative_eval": config.get("generative_eval", False),
            }
        if artifact_type == ArtifactType.PEFT_ADAPTER:
            return {
                **base,
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
            # the collate (data routing) lives here; the loss terms live in objectives.py
            files.append("sparse_steer/tasks/truthfulqa/task.py")
            files.append("sparse_steer/objectives.py")
        if artifact_type in (
            ArtifactType.UNSTEERED_EVAL,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/truthfulqa/eval.py")
            files.append("sparse_steer/core/generate.py")
        return files


__all__ = ["TruthfulQATask"]
