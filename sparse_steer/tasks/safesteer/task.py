from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_tokenizer
from sparse_steer.tasks.base import TaskSpec
from sparse_steer.tasks.collate import prompt_completion_collate
from sparse_steer.utils.cache import ArtifactType
from .data import assemble_datasets, load_splits
from .eval import evaluate, evaluate_generative, evaluate_inspect


class SafeSteerTask(TaskSpec):
    """SafeSteer (arXiv 2506.04250): add a ``mean(safe) − mean(unsafe)`` attention-activation
    direction to steer a model away from unsafe generations and toward safe, on-topic,
    non-refusing answers. The contrast is dataset-defined, so there is no model-coupled
    refusal bucketing (unlike jailbreak) — ``build_datasets`` is a pure function of config."""

    @property
    def task_name(self) -> str:
        return "safesteer"

    # ── Datasets (no model-in-the-loop bucketing) ─────────────────────────
    def build_datasets(
        self, tokenizer: PreTrainedTokenizerBase, config: DictConfig
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        # Extraction reads activations from the extraction model (the chat model under
        # cross-model transfer), so the extraction text must use THAT model's chat template —
        # template it with the extraction tokenizer, not the steered (base) one passed in.
        ext_name = config.get("extraction_model_name")
        ext_tokenizer = (
            load_tokenizer(OmegaConf.merge(config, {"model_name": ext_name}))
            if ext_name and ext_name != config.model_name
            else tokenizer
        )
        splits = load_splits(config, ext_tokenizer)
        n_pos = sum(r["positive"] for r in splits["extraction"])
        n_neg = len(splits["extraction"]) - n_pos
        print(
            f"  safesteer extraction: {n_pos} safe(+) / {n_neg} unsafe(−) "
            f"[dataset={config.get('dataset', 'beavertails')} categories={config.get('categories')}]"
        )
        return assemble_datasets(splits)

    # ── Evaluation (reuses the jailbreak metric suite) ────────────────────
    def run_task_evaluation(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset, config: DictConfig,
    ) -> dict[str, float]:
        metrics = evaluate(model, tokenizer, dataset, config)
        if config.generative_eval:
            metrics.update(evaluate_generative(model, tokenizer, dataset, config))
        metrics.update(evaluate_inspect(model, tokenizer, config))
        return metrics

    # ── Config-B (sparse gate-training) objective ─────────────────────────
    def collate(
        self, rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase,
        device: Any, config: DictConfig,
    ) -> dict[str, Tensor]:
        """Tag rows by loss term — benign(harmless) → ``kl`` (preserve), else → ``ce`` (drive
        toward the safe response) — then build the teacher-forced batch. When the KL term is
        active, drop ``steer_mask`` so steering applies over the full prompt (matches eval);
        otherwise keep the prompt-confined ``steer_mask`` (plain CE-toward-safe)."""
        kl_on = float(config.get("kl_weight", 0.0)) > 0
        if kl_on:
            rows = [
                {**r, "loss_term": "kl" if r.get("category") == "harmless" else "ce"}
                for r in rows
            ]
        batch = prompt_completion_collate(rows, tokenizer, device, config)
        if kl_on:
            batch.pop("steer_mask", None)  # ce_kl ⇒ full-prompt steering (matches eval)
        return batch

    # ── Caching ───────────────────────────────────────────────────────────
    def extra_cache_fields(
        self, artifact_type: ArtifactType, config: DictConfig
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"dtype": config.dtype, "lora_adapter": config.lora_adapter}

        # Identity of the safe/unsafe contrast data — the input to extraction.
        data_identity = {
            "dataset": config.get("dataset", "beavertails"),
            "categories": (
                list(config.categories)
                if config.get("categories") is not None and config.get("categories") != "all"
                else config.get("categories")
            ),
            "n_extraction": config.n_extraction,
            "data_seed": config.get("data_seed"),
            "safe_deflection": config.get("safe_deflection"),
        }

        steering_artifacts = (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        )
        if artifact_type in steering_artifacts:
            fields.update(data_identity)
            fields["intervention"] = config.intervention
            fields.update(
                {
                    # Cross-model transfer: vectors extracted from a different model must not
                    # collide with self-extracted ones. None ⇒ self-steer (extract == steer).
                    "extraction_model_name": config.get("extraction_model_name"),
                    "direction_source": config.get("direction_source", "self"),
                    # NB targets / extract_token_position / steering_layer_ids are in the base
                    # _CONFIG_FIELDS for these artifacts, so they are NOT re-listed here.
                    "normalize_steering_vectors": config.get("normalize_steering_vectors", False),
                    "prune_top_frac": config.get("prune_top_frac"),
                    # refusal-evasion: filtered safe extraction set changes ω; keyed only when on.
                    **({"filter_safe_refusals": True} if config.get("filter_safe_refusals") else {}),
                    # matched safe/unsafe-to-same-prompt extraction changes ω; keyed only when on.
                    **({"paired_extraction": True} if config.get("paired_extraction") else {}),
                }
            )

        # Scale params: kept here (not relying on the base _CONFIG_FIELDS) because they carry
        # non-None defaults — for methods that omit them base would key None, changing the hash.
        # steering_layer_ids has a None default so the base covers it and it is not re-listed.
        if artifact_type in (ArtifactType.SPARSE_STEERING, ArtifactType.STEERED_EVAL):
            fields.update(
                {
                    "init_raw_scale": config.get("init_raw_scale", 0.0),
                    "learn_scale": config.get("learn_scale", False),
                    "freeze_raw_scale": config.get("freeze_raw_scale", False),
                    "shared_scale": config.get("shared_scale", False),
                }
            )

        trains_gates = config.get("refinement_method") == "gate_training"
        if artifact_type == ArtifactType.SPARSE_STEERING or (
            artifact_type == ArtifactType.STEERED_EVAL and trains_gates
        ):
            fields.update(
                {
                    "n_train": config.n_train,
                    # Objective = weighted sum of named terms (sparse_steer/objectives.py).
                    "ce_weight": config.get("ce_weight", 1.0),
                    "kl_weight": config.get("kl_weight", 0.0),
                    **(
                        {
                            "kl_direction": config.get("kl_direction", "reverse"),
                            "kl_positions": config.get("kl_positions", "first_token"),
                        }
                        if float(config.get("kl_weight", 0.0)) > 0
                        else {}
                    ),
                    "completion_tokens": config.get("completion_tokens"),
                    # seed + all training hyperparameters (num_epochs, learning_rate, lr_*,
                    # weight_decay, train_batch_size, l0_*, gate_config) are in the base
                    # _CONFIG_FIELDS for SPARSE_STEERING / STEERED_EVAL — not re-listed here.
                }
            )

        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            fields.update(
                {
                    "dataset": config.get("dataset", "beavertails"),
                    "categories": data_identity["categories"],
                    "n_eval": config.n_eval,
                    "judge": config.judge,
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
        files = ["sparse_steer/tasks/safesteer/data.py"]
        if artifact_type in (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
        ):
            files += [
                "sparse_steer/experiment/sourcing.py",
                "sparse_steer/experiment/_common.py",  # cross-model extraction split
            ]
        # steering.py / train.py / objectives.py / collate.py / utils/eval.py are tracked by the
        # base _SOURCE_FILES; only task-specific files are listed here.
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            files += [
                "sparse_steer/tasks/safesteer/eval.py",
                "sparse_steer/tasks/jailbreak/eval.py",
                "sparse_steer/core/generate.py",
                "sparse_steer/core/inspect_provider.py",
                "sparse_steer/utils/llama_guard.py",
            ]
        return files


__all__ = ["SafeSteerTask"]
