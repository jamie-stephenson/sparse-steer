import gc
import json
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_plain_model
from sparse_steer.tasks.base import SelectionPolicy, TaskSpec
from sparse_steer.tasks.collate import prompt_completion_collate
from sparse_steer.utils import cache
from sparse_steer.utils.cache import ArtifactType
from sparse_steer.utils.refusal import REFUSAL_DETECTOR_VERSION
from .refine import jailbreak_selection_policy
from .data import (
    assemble_datasets,
    bucket_counts,
    judge_eval_dataset,
    label_and_bucket,
    load_splits,
)
from .eval import evaluate, evaluate_generative, evaluate_inspect


class JailbreakTask(TaskSpec):
    """Refusal-direction ablation (jailbreak). The refuse/accept buckets are produced by
    model forward passes and cached (``BUCKETED_DATASET``); the direction is the
    refused−accepted mean difference at the decision token; ablating it recovers output."""

    @property
    def task_name(self) -> str:
        return "jailbreak"

    # ── Datasets (with the model-coupled bucketing + cache) ───────────────
    def build_datasets(
        self, tokenizer: PreTrainedTokenizerBase, config: DictConfig
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        rows = load_splits(config)
        ext_rows = [r for r in rows if r["split"] == "extraction"]
        other_rows = [r for r in rows if r["split"] != "extraction"]
        ext_records = self._load_or_build_buckets(ext_rows, tokenizer, config)
        counts = bucket_counts(ext_records)
        print(f"  jailbreak extraction buckets: {counts}")
        for name, c in counts.items():
            if c < 50:
                print(f"  ⚠️  bucket '{name}' has only {c} usable examples (<50).")
        return assemble_datasets(ext_records, other_rows, tokenizer, config)

    def _load_or_build_buckets(
        self, rows: list[dict], tokenizer: PreTrainedTokenizerBase, config: DictConfig
    ) -> list[dict]:
        extra = self.extra_cache_fields(ArtifactType.BUCKETED_DATASET, config)
        srcs = self.cache_source_files(ArtifactType.BUCKETED_DATASET)
        use_cache = config.use_cache
        if use_cache:
            hit = cache.lookup(
                ArtifactType.BUCKETED_DATASET, config, self.task_name,
                extra_fields=extra, extra_sources=srcs,
            )
            if hit is not None:
                cache.print_cache_warnings(ArtifactType.BUCKETED_DATASET.value, hit.warnings)
                print(f"  Cache hit: bucketed_dataset ({hit.artifact_path})")
                return cache.load_cached_json(hit.artifact_path)["records"]

        # MISS: self-load an unsteered model (build_datasets runs before the pipeline
        # loads its steering model), generate + bucket, cache, then free it.
        print("  Building bucketed dataset (model forward passes — cached per model)...")
        model = load_plain_model(config)
        try:
            data = label_and_bucket(model, tokenizer, rows, config)
        finally:
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        if use_cache:
            dest = cache.prepare_cache_path(
                ArtifactType.BUCKETED_DATASET, config, self.task_name, extra_fields=extra
            )
            dest.write_text(json.dumps(data, indent=2))
            cache.finalize(
                ArtifactType.BUCKETED_DATASET, config, self.task_name,
                extra_fields=extra, extra_sources=srcs,
            )
        return data["records"]

    # ── Evaluation ────────────────────────────────────────────────────────
    def run_task_evaluation(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset, config: DictConfig,
    ) -> dict[str, float]:
        metrics = evaluate(model, tokenizer, dataset, config)
        if config.generative_eval:
            metrics.update(evaluate_generative(model, tokenizer, dataset, config))
        metrics.update(evaluate_inspect(model, tokenizer, config))  # self-contained Inspect evals
        return metrics

    def selection_policy(self, model, tokenizer, refine_ds, config) -> SelectionPolicy:
        # Refusal-specific scoring for direction_source=grid_select (Arditi App. C): bypass /
        # induce / KL on the refinement set. The grid sweep + filter/pick are task-general
        # (experiment.sourcing); this only supplies the score + constraints.
        return jailbreak_selection_policy(model, tokenizer, refine_ds, config)

    # ── Phase-2 sparse-ablation training objective (mirrors tinysleepers) ──
    def collate(
        self, rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase,
        device: Any, config: DictConfig,
    ) -> dict[str, Tensor]:
        """Teacher-forced ``prompt + completion`` batch; steering confined to prompt
        positions. The completion is a compliant/affirmative continuation so sparse
        ablation learns to recover compliance on harmful prompts."""
        batch = prompt_completion_collate(rows, tokenizer, device, config)
        # Non-CE objective (jb_objective=refusal_logit): bake what the loss needs (config is not
        # passed to .loss) — the refusal-opener token ids and the decision position (the last
        # prompt token, whose logits predict the first completion token).
        objective = config.get("jb_objective", "ce")
        if objective == "refusal_logit":
            rt = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in config.refusal_tokens]
            batch["refusal_token_ids"] = torch.tensor(rt, device=device)
            batch["decision_pos"] = (batch["labels"] != -100).int().argmax(dim=1) - 1
        elif objective == "ce_kl":
            # Dual objective: CE-toward-affirmative on harmful rows + KL(steered‖base) on harmless
            # rows at the decision token (mirrors the kl_harmless eval). Bake what .loss needs (it
            # gets no config): which rows are harmless, the decision position, and the KL weight β.
            batch["harmless_mask"] = torch.tensor(
                [r.get("category") == "harmless" for r in rows], device=device
            )
            batch["decision_pos"] = (batch["labels"] != -100).int().argmax(dim=1) - 1
            batch["harmless_kl_weight"] = torch.tensor(
                float(config.get("harmless_kl_weight", 1.0)), device=device
            )
        return batch

    def loss(self, model, batch: dict[str, Tensor], logits: Tensor) -> Tensor:
        # Non-CE objective: directly suppress refusal — minimise log P(refusal opener) at the
        # decision position (Arditi's refusal-logit signal) instead of CE toward an affirmative
        # completion. The L0 term is still added by the train loop. The idea is to differentiate
        # sites by their actual effect on refusal rather than on next-token likelihood.
        if "refusal_token_ids" in batch:
            n = logits.size(0)
            rows = torch.arange(n, device=logits.device)
            lp = F.log_softmax(logits[rows, batch["decision_pos"]].float(), dim=-1)
            return lp[:, batch["refusal_token_ids"]].logsumexp(dim=-1).mean()
        if "harmless_mask" in batch:
            # Dual objective (jb_objective=ce_kl): recover compliance on harmful prompts (CE toward
            # the affirmative completion) while preserving harmless behaviour via an explicit
            # KL(steered‖base) at the decision token — the training analogue of the kl_harmless
            # eval. Optimises the success criterion directly (high ASR + low harmless collateral)
            # rather than pushing the steer harder. The base (unsteered) reference is recomputed
            # each step with the frozen base model under steering_disabled() (no_grad).
            hm = batch["harmless_mask"]
            idx = torch.arange(logits.size(0), device=logits.device)
            dp = batch["decision_pos"]
            if (~hm).any():
                harmful_labels = batch["labels"].clone()
                harmful_labels[hm] = -100  # score CE only on the harmful rows' completions
                sh = logits[..., :-1, :].float()
                ce = F.cross_entropy(
                    sh.reshape(-1, sh.size(-1)),
                    harmful_labels[..., 1:].reshape(-1),
                    ignore_index=-100,
                )
            else:
                ce = logits.new_zeros(())
            kl = logits.new_zeros(())
            if hm.any():
                steered_lp = F.log_softmax(logits[idx, dp].float(), dim=-1)[hm]
                with torch.no_grad(), model.steering_disabled():
                    base_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                base_lp = F.log_softmax(base_logits[idx, dp].float(), dim=-1)[hm]
                kl = (steered_lp.exp() * (steered_lp - base_lp)).sum(dim=-1).mean()
            return ce + batch["harmless_kl_weight"] * kl
        shift_logits = logits[..., :-1, :].float()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            batch["labels"][..., 1:].reshape(-1),
            ignore_index=-100,
        )

    # ── Caching ───────────────────────────────────────────────────────────
    def extra_cache_fields(
        self, artifact_type: ArtifactType, config: DictConfig
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"dtype": config.dtype, "lora_adapter": config.lora_adapter}
        # Identity of the cached refuse/accept buckets — the *input* to extraction. It
        # deliberately excludes the judge and the eval datasets so those can be swapped
        # without rebuilding the buckets or the steering vector.
        bucket_identity = {
            "harmful_dataset": config.harmful_dataset,
            "harmless_dataset": config.harmless_dataset,
            # data origin: "arditi_exact" (their committed prompts) | None (data_mix / single).
            "data_origin": config.get("data_origin"),
            # per-role multi-source mix (HF replication); None ⇒ single-source path.
            "data_mix": OmegaConf.to_container(config.data_mix, resolve=True) if config.get("data_mix") else None,
            "n_extraction": config.n_extraction,
            "extraction_subset": config.extraction_subset,
            "gen_decoding": config.gen_decoding,
            "gen_max_new_tokens": config.gen_max_new_tokens,
            "gen_temperature": config.gen_temperature,
            "gen_seed": config.gen_seed,
            "refusal_detector": config.refusal_detector,
            "refusal_detector_version": REFUSAL_DETECTOR_VERSION,
            # logit-detector identity (ignored by the regex detector at runtime, but kept in
            # the key so a logit run never collides with a regex run or a different token set).
            "refusal_tokens": list(config.refusal_tokens),
            "refusal_logit_threshold": config.refusal_logit_threshold,
        }
        if artifact_type == ArtifactType.BUCKETED_DATASET:
            fields.update(bucket_identity)
        steering_artifacts = (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
            ArtifactType.SELECTED_DIRECTION,
            ArtifactType.STEERED_EVAL,
        )
        if artifact_type in steering_artifacts:
            fields.update(bucket_identity)
            fields["intervention"] = config.intervention
            # Sourcing identity: which sites are steered + how each site's direction is sourced.
            # Two runs differing here produce different steering vectors from the same buckets.
            fields.update(
                {
                    "direction_source": config.get("direction_source", "self"),
                    "targets": list(config.targets),
                    "extract_token_position": config.extract_token_position,
                    "steering_layer_ids": (
                        list(config.steering_layer_ids)
                        if config.get("steering_layer_ids") is not None
                        else None
                    ),
                    "normalize_steering_vectors": config.get("normalize_steering_vectors", False),
                }
            )

        # Trained-gate identity: the SPARSE_STEERING artifact (and any eval that consumes it) is a
        # deterministic function of the WHOLE gate-training recipe. Without these a frontier sweep
        # over e.g. l0_lambda would silently reuse one cached result.
        trains_gates = config.get("refinement_method") == "gate_training"
        if artifact_type == ArtifactType.SPARSE_STEERING or (
            artifact_type == ArtifactType.STEERED_EVAL and trains_gates
        ):
            fields.update(
                {
                    "n_train": config.n_train,
                    "affirmative_prefix": config.affirmative_prefix,
                    "gate_train_target": config.get("gate_train_target", "compliance"),
                    "jb_objective": config.get("jb_objective", "ce"),
                    # β only affects the ce_kl objective; key it solely for ce_kl runs so existing
                    # ce / refusal_logit cache identities (and their artifacts) stay unchanged.
                    **(
                        {"harmless_kl_weight": config.get("harmless_kl_weight", 1.0)}
                        if config.get("jb_objective", "ce") == "ce_kl"
                        else {}
                    ),
                    "refusal_prefix": config.get("refusal_prefix"),
                    "completion_tokens": config.get("completion_tokens"),
                    "seed": config.seed,
                    "num_epochs": config.get("num_epochs"),
                    "learning_rate": config.get("learning_rate"),
                    "lr_scheduler_type": config.get("lr_scheduler_type"),
                    "grad_clip": config.get("grad_clip", 1.0),
                    "negate_direction": config.get("negate_direction", False),
                    "lr_warmup_steps": config.get("lr_warmup_steps"),
                    "weight_decay": config.get("weight_decay"),
                    "train_batch_size": config.get("train_batch_size"),
                    "l0_lambda": config.get("l0_lambda"),
                    "l0_scheduler_type": config.get("l0_scheduler_type"),
                    "l0_warmup_steps": config.get("l0_warmup_steps"),
                    "normalize_ablation": config.get("normalize_ablation", False),
                    "proj_act_norm_examples": config.get("proj_act_norm_examples", 128),
                    "learn_scale": config.get("learn_scale", False),
                    "shared_scale": config.get("shared_scale", False),
                    "init_raw_scale": config.get("init_raw_scale", 0.0),
                    "freeze_raw_scale": config.get("freeze_raw_scale", False),
                    "scale_tuning_epochs": config.get("scale_tuning_epochs", 0),
                    "scale_tuning_lr": config.get("scale_tuning_lr"),
                    "gate_config": (
                        OmegaConf.to_container(config.gate_config, resolve=True)
                        if config.get("gate_config")
                        else None
                    ),
                }
            )
        if artifact_type == ArtifactType.SELECTED_DIRECTION or (
            artifact_type == ArtifactType.STEERED_EVAL
            and config.get("direction_source") == "grid_select"
        ):
            # The selected direction (and the eval that uses it) depends on the refinement set
            # + the Arditi selection hyperparameters.
            fields.update(
                {
                    "n_train": config.n_train,
                    "selection_grid_component": config.get("selection_grid_component"),
                    "selection_positions": config.get("selection_positions"),
                    "selection_kl_threshold": config.get("selection_kl_threshold"),
                    "selection_induce_threshold": config.get("selection_induce_threshold"),
                    "selection_prune_layer_frac": config.get("selection_prune_layer_frac"),
                }
            )
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            judge = config.judge
            harmful_eval = config.harmful_eval_dataset or judge_eval_dataset(judge) or config.harmful_dataset
            harmless_eval = config.harmless_eval_dataset or config.harmless_dataset
            fields.update(
                {
                    "n_eval": config.n_eval,
                    "harmful_eval_dataset": harmful_eval,
                    "harmless_eval_dataset": harmless_eval,
                    "judge": judge,
                    "evals": list(config.evals or []),
                    "generative_eval": config.generative_eval,
                    "gen_tokens": config.gen_tokens,
                    "eval_seeds": list(config.eval_seeds or []),
                    "eval_temperature": config.eval_temperature,
                    "steer_token_position": config.steer_token_position,
                    "eval_refusal_detector": config.get("eval_refusal_detector", "regex"),
                    "llama_guard_model": config.get("llama_guard_model"),
                }
            )
        return fields

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        # refusal.py (detector) is shared by the bucketing + eval; utils/eval.py (teacher-forced)
        # backs the white-box metrics; inspect_provider backs the Inspect-run benchmarks.
        files = ["sparse_steer/tasks/jailbreak/data.py", "sparse_steer/utils/refusal.py"]
        if artifact_type == ArtifactType.BUCKETED_DATASET:
            files.append("sparse_steer/core/generate.py")
        # Sourcing (self / pinned broadcast / grid_select) decides the steering vectors; its
        # scoring (refine.py) decides the selected direction. Both feed the eval.
        if artifact_type in (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
            ArtifactType.SELECTED_DIRECTION,
        ):
            files += [
                "sparse_steer/experiment/sourcing.py",
                "sparse_steer/tasks/jailbreak/refine.py",
            ]
        if artifact_type == ArtifactType.SPARSE_STEERING:
            # the gate/ablation hooks (incl. proj_act_norm) and the gate-training loop
            # determine the trained gates
            files += ["sparse_steer/core/steering.py", "sparse_steer/train.py"]
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            files += [
                "sparse_steer/tasks/jailbreak/eval.py",
                "sparse_steer/core/generate.py",
                "sparse_steer/core/inspect_provider.py",
                "sparse_steer/utils/llama_guard.py",
                "sparse_steer/experiment/sourcing.py",
                "sparse_steer/tasks/jailbreak/refine.py",  # selection-mode eval depends on the chosen direction
            ]
        return files


__all__ = ["JailbreakTask"]
