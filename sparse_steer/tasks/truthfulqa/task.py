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
            disjoint_extract_refine_data=bool(config.get("disjoint_extract_refine_data", True)),
            seed=config.get("data_seed"),
            # LoFiT cross-validation fold (default 0/2/0.2 = current behaviour). honest_llama
            # averages num_folds=2; run fold=0 then fold=1 and average to match its 2-fold reference.
            fold=int(config.get("fold", 0)),
            num_folds=int(config.get("num_folds", 2)),
            val_ratio=float(config.get("val_ratio", 0.2)),
            with_kl_rows=kl_used,
            with_contrastive=contrastive_used,
            max_n_neg=int(mxn) if mxn is not None else None,
            uniform_duplicate=bool(config.get("n_neg_uniform_duplicate", True)),
            eval_full_set=bool(config.get("eval_full_set", False)),
            eval_subset_size=(
                int(config.get("eval_subset_size"))
                if config.get("eval_subset_size", None) is not None
                else None
            ),
            template=config.get("prompt_template", "chat"),
            extraction_template=config.get("extraction_template", None),
        )

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]:
        steer_token_position = config.get("steer_token_position", "all")
        template = config.get("prompt_template", "chat")
        # MC eval uses the same steer_token_position as generation (no separate MC argument):
        # with steer_token_position=last_onwards the teacher-forced MC pass steers the answer span,
        # so MC1/MC2 reflect the intervention (the original last/last_onwards semantics).
        metrics = evaluate(
            model, tokenizer, dataset,
            batch_size=config.eval_batch_size,
            steer_token_position=steer_token_position,
            template=template,
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
                template=template,
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
                gen_batch_size=int(config.get("gen_batch_size", 8)),
                judge_batch_size=int(config.get("judge_batch_size", 8)),
                steer_token_position=steer_token_position,
                template=template,
                save_generations_path=config.get("save_generations_path"),
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
        single ``text`` → one sequence (``ce`` / ``kl`` / …). The question prefix is ALWAYS masked so
        every term (``ce``, contrastive-ranking, ``kl``) scores the completion span only — CE and the
        contrastive objective compare log-probs over the answer tokens alone. Emits a per-term row mask in ``loss_term_rows``
        and a ``<term>_group_sizes`` list (the K per question, correct-first) for any ranking term.
        Emits a ``steer_mask`` from ``steer_token_position`` so gate-training steers the same
        positions eval does (all / last_onwards / last). Adding a loss term needs no new collate —
        just format its rows with the tag (and a ``texts`` list if it ranks candidates).
        """
        by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_term[r.get("loss_term", "ce")].append(r)

        texts: list[str] = []
        plens: list[int] = []
        seq_term: list[str] = []
        group_sizes: dict[str, list[int]] = {}
        for term in sorted(by_term):  # each term a contiguous block
            group = by_term[term]
            if "texts" in group[0]:  # ranking group: one question's answers, correct-first (K varies)
                sizes: list[int] = []
                for r in group:
                    kr = len(r["texts"])
                    sizes.append(kr)
                    texts += r["texts"]
                    plens += [r["prompt_len"]] * kr
                    seq_term += [term] * kr
                group_sizes[term] = sizes
            else:  # one sequence per row
                for r in group:
                    texts.append(r["text"])
                    plens.append(r["prompt_len"])
                    seq_term.append(term)

        enc = tokenizer(texts, return_tensors="pt", padding=True, padding_side="right")
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = ids.masked_fill(attn == 0, -100)  # ignore padding
        from sparse_steer.utils.positions import positions_mask

        plens_t = torch.tensor(plens, dtype=torch.long)
        eos_id = tokenizer.eos_token_id
        # Every term scores the COMPLETION span only (the answer tokens after "A:"): CE and the
        # contrastive ranking term use the same "completion" mask, so the log-probs they compare come
        # from the answer alone. Same position vocabulary as extraction/steering (prompt / prompt_final
        # / completion / all), so train/score/eval positions agree.
        for i in range(len(seq_term)):
            keep = positions_mask(
                "completion",
                attn[i : i + 1], plens_t[i : i + 1],
                input_ids=ids[i : i + 1], eos_id=eos_id,
            )[0]
            labels[i, ~keep] = -100
        # Steer the SAME positions during gate-training that eval steers — the shared mask vocabulary.
        steer_mask = positions_mask(
            config.get("steer_token_position", "all"),
            attn, plens_t, input_ids=ids, eos_id=eos_id,
        )
        loss_term_rows = {
            t: torch.tensor([s == t for s in seq_term], dtype=torch.bool, device=device)
            for t in by_term
        }
        out: dict[str, Any] = {
            "input_ids": ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
            "steer_mask": steer_mask.to(device),
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
        # prompt_template changes the extraction/gate-train/eval prompts (chat vs iti_qa_few_shot primer),
        # so it keys every model-coupled artifact (else chat->iti_qa_few_shot would hit stale caches).
        base = {
            "model_dtype": config.get("model_dtype", "float16"),
            "prompt_template": config.get("prompt_template", "chat"),
            # the cross-validation fold changes both the extraction subset and the eval question set,
            # so it keys every artifact (else fold 0 and fold 1 would collide in the cache). Keyed
            # only when non-default so existing single-fold caches stay valid.
            **(
                {
                    "fold": int(config.get("fold", 0)),
                    "num_folds": int(config.get("num_folds", 2)),
                }
                if int(config.get("fold", 0)) != 0 or int(config.get("num_folds", 2)) != 2
                else {}
            ),
            # extraction/refine data overlap: disjoint subsets (default) vs full overlap. Changes BOTH
            # the extraction subset and the gate-train subset, so it keys every artifact. Keyed only when
            # non-default (False) so existing disjoint-split caches stay valid.
            **(
                {"disjoint_extract_refine_data": False}
                if not bool(config.get("disjoint_extract_refine_data", True))
                else {}
            ),
        }
        if artifact_type == ArtifactType.STEERING_VECTORS:
            return {
                **base,
                "extraction_mcq_mode": config.get(
                    "extraction_mcq_mode", "mc1"
                ),
                "extraction_template": config.get(
                    "extraction_template", config.get("prompt_template", "chat")
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
                "extraction_template": config.get(
                    "extraction_template", config.get("prompt_template", "chat")
                ),
                # gate-training now steers the positions named by steer_token_position (the collate's
                # steer_mask), so it changes the trained gates and must key the artifact (was eval-only).
                "steer_token_position": config.get("steer_token_position", "all"),
                # σ-scaling identity: whether the per-site scale comes from α·σ, and the α — both
                # change the steered model, so they key the artifact (else an α sweep hits stale
                # caches). iti_scale only matters when σ-scaling is on, so key it only then.
                "scale_from_extraction_std": bool(config.get("scale_from_extraction_std", False)),
                **(
                    {
                        "iti_scale": float(config.get("iti_scale", 15.0)),
                        # σ population for the α·σ magnitude (answer-token vs question-end) changes
                        # the steered model, so it keys the artifact.
                        "iti_sigma_position": str(config.get("iti_sigma_position", "answer")),
                        "sigma_prompt_anchor": str(config.get("sigma_prompt_anchor", "answer_colon")),
                    }
                    if bool(config.get("scale_from_extraction_std", False))
                    or config.get("refinement_method") == "iti_head_select"
                    else {}
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
                if config.get("eval_subset_size", None) is not None:
                    fields["eval_subset_size"] = int(config.get("eval_subset_size"))
                if config.get("generative_eval", False):
                    fields["gen_max_new_tokens"] = int(config.get("gen_max_new_tokens", 64))
                    fields["gen_batch_size"] = int(config.get("gen_batch_size", 8))
                    fields["judge_batch_size"] = int(config.get("judge_batch_size", 8))
            return fields
        if artifact_type == ArtifactType.UNSTEERED_EVAL:
            fields = {
                **base,
                "generative_eval": config.get("generative_eval", False),
                **(
                    {"eval_subset_size": int(config.get("eval_subset_size"))}
                    if config.get("eval_subset_size", None) is not None
                    else {}
                ),
            }
            if config.get("generative_eval", False):
                fields["gen_max_new_tokens"] = int(config.get("gen_max_new_tokens", 64))
                fields["gen_batch_size"] = int(config.get("gen_batch_size", 8))
                fields["judge_batch_size"] = int(config.get("judge_batch_size", 8))
            return fields
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
