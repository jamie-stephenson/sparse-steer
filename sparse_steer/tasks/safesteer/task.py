from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_tokenizer
from sparse_steer.tasks.base import TaskSpec
from sparse_steer.tasks.objectives import objective_collate, objective_loss
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
        self._config = config  # collate runs every step right before loss(); cache config for it
        return objective_collate(rows, tokenizer, device, config)

    def loss(self, model, batch: dict[str, Tensor], logits: Tensor) -> Tensor:
        # loss() has no config arg in the TaskSpec interface; collate (above) stashes it.
        return objective_loss(model, batch, logits, self._config)

    # ── Caching ───────────────────────────────────────────────────────────
    def extra_cache_fields(
        self, artifact_type: ArtifactType, config: DictConfig
    ) -> dict[str, Any]:
        # Stash config so loss() (which has no config arg) can read jb_objective at train time.
        self._config = config
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
                    "targets": list(config.targets),
                    "extract_token_position": config.extract_token_position,
                    # NB steering_layer_ids is NOT here: collect_activations extracts ALL layers, so ω
                    # is layer-set-independent (cached once, reused across a layer sweep). It only
                    # affects which layers are STEERED → keyed into the eval/gate block below.
                    "normalize_steering_vectors": config.get("normalize_steering_vectors", False),
                    "prune_top_frac": config.get("prune_top_frac"),
                    # refusal-evasion: filtered safe extraction set changes ω; keyed only when on.
                    **({"filter_safe_refusals": True} if config.get("filter_safe_refusals") else {}),
                    # matched safe/unsafe-to-same-prompt extraction changes ω; keyed only when on.
                    **({"paired_extraction": True} if config.get("paired_extraction") else {}),
                }
            )

        # Scale params set the APPLIED steering magnitude (m = softplus(init_raw_scale)) — for the
        # dense eval AND gate training. They must be in the steered-eval / trained-gate identity (else
        # an m-sweep collides on one cached eval) but NOT in STEERING_VECTORS (ω is m-independent, so
        # it is correctly reused across the sweep).
        if artifact_type in (ArtifactType.SPARSE_STEERING, ArtifactType.STEERED_EVAL):
            fields.update(
                {
                    "init_raw_scale": config.get("init_raw_scale", 0.0),
                    "learn_scale": config.get("learn_scale", False),
                    "freeze_raw_scale": config.get("freeze_raw_scale", False),
                    "shared_scale": config.get("shared_scale", False),
                    # which layers are STEERED (the all-layer ω is shared via STEERING_VECTORS).
                    "steering_layer_ids": (
                        list(config.steering_layer_ids)
                        if config.get("steering_layer_ids") is not None
                        else None
                    ),
                }
            )

        trains_gates = config.get("refinement_method") == "gate_training"
        if artifact_type == ArtifactType.SPARSE_STEERING or (
            artifact_type == ArtifactType.STEERED_EVAL and trains_gates
        ):
            fields.update(
                {
                    "n_train": config.n_train,
                    "jb_objective": config.get("jb_objective", "ce"),
                    **(
                        {"harmless_kl_weight": config.get("harmless_kl_weight", 1.0)}
                        if config.get("jb_objective", "ce") == "ce_kl"
                        else {}
                    ),
                    "completion_tokens": config.get("completion_tokens"),
                    "seed": config.seed,
                    "num_epochs": config.get("num_epochs"),
                    "learning_rate": config.get("learning_rate"),
                    "lr_scheduler_type": config.get("lr_scheduler_type"),
                    "lr_warmup_steps": config.get("lr_warmup_steps"),
                    "weight_decay": config.get("weight_decay"),
                    "train_batch_size": config.get("train_batch_size"),
                    "l0_lambda": config.get("l0_lambda"),
                    "l0_scheduler_type": config.get("l0_scheduler_type"),
                    "l0_warmup_steps": config.get("l0_warmup_steps"),
                    "gate_config": (
                        OmegaConf.to_container(config.gate_config, resolve=True)
                        if config.get("gate_config")
                        else None
                    ),
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
        if artifact_type == ArtifactType.SPARSE_STEERING:
            files += [
                "sparse_steer/core/steering.py",
                "sparse_steer/train.py",
                "sparse_steer/tasks/objectives.py",
            ]
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            files += [
                "sparse_steer/tasks/safesteer/eval.py",
                "sparse_steer/tasks/jailbreak/eval.py",
                "sparse_steer/core/generate.py",
                "sparse_steer/core/inspect_provider.py",
                "sparse_steer/utils/llama_guard.py",
            ]
        return files

    _config: DictConfig | None = None


__all__ = ["SafeSteerTask"]
