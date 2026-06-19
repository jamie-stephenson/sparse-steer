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
            extraction_contrastiveq_mode=config.get("extraction_contrastiveq_mode", "mc1"),
            gate_train_contrastiveq_mode=config.get("gate_train_contrastiveq_mode", "mc1"),
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
        """Next-token CE batch that scores only the answer completion.

        Unlike the base collate (which scores the whole templated text), the
        question prefix is masked to ``-100`` so the gates aren't trained to
        reconstruct the prompt — only to raise the likelihood of the correct
        answer (the shared ``ce`` term scores it). No ``steer_mask`` is emitted, so
        steering still applies at every position, matching how it is applied at eval.
        """
        if rows and "texts" in rows[0]:  # contrastive (contrastive-ranking) per-question groups
            return self._collate_contrastive(rows, tokenizer, device, config)
        enc = tokenizer(
            [r["text"] for r in rows],
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = ids.masked_fill(attn == 0, -100)  # ignore padding
        # ce_positions (task-agnostic, mirrors kl_positions): "completion" (default) scores only
        # the answer (mask the question prefix); "all" scores the whole templated Q+A.
        if config.get("ce_positions", "completion") == "completion":
            for i, r in enumerate(rows):
                labels[i, : r["prompt_len"]] = -100  # mask the question prefix
        # Route each row to its loss term by tag (default "ce"); CE rows = QA answers,
        # KL rows = held-out general (Alpaca) preserve rows.
        loss_terms = [r.get("loss_term", "ce") for r in rows]
        loss_term_rows = {
            t: torch.tensor([lt == t for lt in loss_terms], dtype=torch.bool, device=device)
            for t in sorted(set(loss_terms))
        }
        return {
            "input_ids": ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
            "loss_term_rows": loss_term_rows,
        }

    def _collate_contrastive(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        config: DictConfig,
    ) -> dict[str, Tensor]:
        """Expand per-question ``contrastive`` contrastive rows into K sequences each (correct at group
        index 0) so ``contrastive_term`` can reshape to ``(n_questions, K)`` and rank. Any non-``contrastive`` rows
        in the batch (``kl`` preserve rows, which carry a single ``text``) are appended AFTER all
        contrastive sequences as one-sequence-each — so the contrastive block stays contiguous from index 0 and the
        reshape is unaffected — and tagged for their own term. contrastive and kl share one padded batch."""
        contrastive_rows = [r for r in rows if r.get("loss_term", "contrastive") == "contrastive"]
        other_rows = [r for r in rows if r.get("loss_term", "contrastive") != "contrastive"]
        k = len(contrastive_rows[0]["texts"]) if contrastive_rows else 0
        contrastive_texts = [t for r in contrastive_rows for t in r["texts"]]
        contrastive_plens = [r["prompt_len"] for r in contrastive_rows for _ in range(k)]
        other_texts = [r["text"] for r in other_rows]
        other_plens = [r["prompt_len"] for r in other_rows]
        flat_texts = contrastive_texts + other_texts  # contrastive block first (contiguous) → reshape stays valid
        plens = contrastive_plens + other_plens
        enc = tokenizer(flat_texts, return_tensors="pt", padding=True, padding_side="right")
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = ids.masked_fill(attn == 0, -100)
        for i, pl in enumerate(plens):
            labels[i, :pl] = -100  # mask the shared question prefix → score answer tokens only
        n, n_contrastive = len(flat_texts), len(contrastive_texts)
        loss_term_rows: dict[str, Tensor] = {}
        if n_contrastive:
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[:n_contrastive] = True
            loss_term_rows["contrastive"] = mask
        for t in sorted({r.get("loss_term", "kl") for r in other_rows}):
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            for j, r in enumerate(other_rows):
                if r.get("loss_term", "kl") == t:
                    mask[n_contrastive + j] = True
            loss_term_rows[t] = mask
        out = {
            "input_ids": ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
            "loss_term_rows": loss_term_rows,
        }
        if n_contrastive:
            out["contrastive_n_answers"] = k
        return out

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]:
        if artifact_type == ArtifactType.STEERING_VECTORS:
            return {
                "extraction_contrastiveq_mode": config.get(
                    "extraction_contrastiveq_mode", "mc1"
                ),
            }
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            fields = {
                "extraction_contrastiveq_mode": config.get(
                    "extraction_contrastiveq_mode", "mc1"
                ),
                "gate_train_contrastiveq_mode": config.get(
                    "gate_train_contrastiveq_mode", "mc1"
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
                "gate_train_contrastiveq_mode": config.get(
                    "gate_train_contrastiveq_mode", "mc1"
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
