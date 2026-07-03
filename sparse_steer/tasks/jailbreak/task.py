import json
from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_plain_model
from sparse_steer.tasks.base import SelectionPolicy, TaskSpec
from sparse_steer.tasks.collate import prompt_completion_collate
from sparse_steer.utils import cache
from sparse_steer.utils.cache import ArtifactType
from sparse_steer.utils.memory import free_model_memory
from sparse_steer.utils.refusal import REFUSAL_DETECTOR_VERSION
from .refine import jailbreak_selection_policy
from .data import (
    assemble_datasets,
    bucket_counts,
    judge_eval_dataset,
    label_and_bucket,
    load_splits,
)
from .eval import evaluate, evaluate_generative


class RefusalTask(TaskSpec):
    """Refusal-direction ablation (jailbreak). The refuse/accept buckets are produced by
    model forward passes and cached (``BUCKETED_DATASET``); the direction is the
    refused−accepted mean difference at the decision token; ablating it recovers output."""

    @property
    def task_name(self) -> str:
        return "refusal"

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
            free_model_memory()

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
        return metrics

    def selection_policy(self, model, tokenizer, refine_ds, config) -> SelectionPolicy:
        # Refusal-specific scoring for direction_source=grid_search (Arditi App. C): bypass /
        # induce / KL on the refinement set. The grid sweep + filter/pick are task-general
        # (experiment.steering.search); this only supplies the score + constraints.
        return jailbreak_selection_policy(model, tokenizer, refine_ds, config)

    # ── Phase-2 sparse-ablation training objective (mirrors tinysleepers) ──
    def collate(
        self, rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase,
        device: Any, config: DictConfig,
    ) -> dict[str, Tensor]:
        """Teacher-forced ``prompt + completion`` batch; steering confined to prompt
        positions. The completion is a compliant/affirmative continuation so sparse
        ablation learns to recover compliance on harmful prompts. The objective is the
        shared CE term (``loss_term`` defaults to ``"ce"``)."""
        return prompt_completion_collate(rows, tokenizer, device, config)

    # ── Caching ───────────────────────────────────────────────────────────
    def extra_cache_fields(
        self, artifact_type: ArtifactType, config: DictConfig
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "model_dtype": config.get("model_dtype", "float16"),
            "lora_adapter": config.lora_adapter,
        }
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
                    # extract_token_position is in the base _CONFIG_FIELDS for every steering
                    # artifact. targets is too, but the base sorts it whereas this key preserves
                    # config order, so it is kept to match historical hashes.
                    "targets": list(config.targets),
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
                    "refusal_prefix": config.get("refusal_prefix"),
                    "completion_tokens": config.get("completion_tokens"),
                    "normalize_ablation": config.get("normalize_ablation", False),
                    "proj_act_norm_examples": config.get("proj_act_norm_examples", 128),
                    "scale_tuning_epochs": config.get("scale_tuning_epochs", 0),
                    "scale_tuning_lr": config.get("scale_tuning_lr"),
                    # Scale params carry non-None defaults, so they are kept here (base would key
                    # them as None for methods that omit them, changing the hash).
                    "learn_scale": config.get("learn_scale", False),
                    "shared_scale": config.get("shared_scale", False),
                    "init_raw_scale": config.get("init_raw_scale", 0.0),
                    "freeze_raw_scale": config.get("freeze_raw_scale", False),
                    # seed and the training hyperparameters (num_epochs, learning_rate, lr_*,
                    # weight_decay, train_batch_size, l0_*) and gate_config use None-equivalent
                    # defaults and are covered by the base _CONFIG_FIELDS — not re-listed.
                }
            )
        if artifact_type == ArtifactType.SELECTED_DIRECTION or (
            artifact_type == ArtifactType.STEERED_EVAL
            and config.get("direction_source") == "grid_search"
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
        # Sourcing (self / fixed broadcast / grid_search) decides the steering vectors; its
        # scoring (refine.py) decides the selected direction. Both feed the eval.
        if artifact_type in (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
            ArtifactType.SELECTED_DIRECTION,
        ):
            files += [
                "sparse_steer/experiment/steering/solvers.py",
                "sparse_steer/experiment/steering/search.py",
                "sparse_steer/tasks/jailbreak/refine.py",
            ]
        # core/steering.py / train.py / objectives.py / collate.py are tracked by the base _SOURCE_FILES.
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            files += [
                "sparse_steer/tasks/jailbreak/eval.py",
                "sparse_steer/core/generate.py",
                "sparse_steer/core/inspect_provider.py",
                "sparse_steer/utils/llama_guard.py",
                "sparse_steer/experiment/steering/solvers.py",
                "sparse_steer/experiment/steering/search.py",
                "sparse_steer/tasks/jailbreak/refine.py",  # selection-mode eval depends on the chosen direction
            ]
        return files


__all__ = ["RefusalTask"]
