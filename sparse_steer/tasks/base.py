import abc
from dataclasses import dataclass
from typing import Any, Callable

import torch
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.cache import ArtifactType


@dataclass
class SelectionPolicy:
    """A task's ``grid_search`` scoring (Arditi App. C), consumed by the generic score-search
    direction solver (``experiment.steering.search.grid_search_source``).

    ``score(vector, source_layer)`` is called *after* the driver has broadcast ``vector`` onto
    every steering site, so it reads the model in its candidate-ablated state. It returns
    ``(objective, values)`` — the scalar the picker minimises plus a dict of named quantities
    the constraints reference. ``constraints`` is a list of ``(values-key, "<="|">=", threshold)``
    the generic picker applies (alongside a finite + layer-prune check) before taking the
    minimum-objective survivor.
    """

    score: Callable[[Tensor, int], tuple[float, dict[str, float]]]
    constraints: list[tuple[str, str, float]]


class TaskSpec(abc.ABC):
    """Interface a task must implement to plug into the experiment pipeline."""

    @property
    @abc.abstractmethod
    def task_name(self) -> str: ...

    @abc.abstractmethod
    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]: ...

    @abc.abstractmethod
    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]: ...

    # ── Refinements ───────────────────────────────────────────────────

    def extra_refinements(self) -> dict[str, Callable]:
        """Task-contributed strength-field *refinements*, merged with the engine's built-in
        ``REFINEMENTS`` and selected by the ``refinement_method`` config value. Default: none.
        A refinement is ``fn(experiment, model, tokenizer, extraction_ds, train_ds, output_dir)
        -> (model, artifacts, cache_info)``.
        """
        return {}

    # ── Direction sourcing ────────────────────────────────────────────

    def selection_policy(
        self,
        model: SteeringModel,
        tokenizer: PreTrainedTokenizerBase,
        refine_ds,
        config: DictConfig,
    ) -> "SelectionPolicy | None":
        """Scoring used by ``direction_source: grid_search`` to pick one direction from the
        candidate grid (Arditi App. C). Default ``None`` ⇒ this task does not support
        ``grid_search`` (only ``self`` / fixed sourcing). Override to return a
        :class:`SelectionPolicy` — e.g. jailbreak's bypass/induce/KL refusal scoring.
        """
        return None

    # ── Training objective ────────────────────────────────────────────

    def collate(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        config: DictConfig,
    ) -> dict[str, Tensor]:
        """Turn a batch of dataset rows into the tensors the objective consumes.

        Default: a next-token cross-entropy batch from the ``text`` column
        (``config`` is unused). Override to emit task-specific tensors — e.g. a
        ``steer_mask`` (a ``(batch, seq)`` bool) which the training loop honours
        via ``SteeringModel.steer_positions`` to confine steering to prompt
        positions. Always emit ``loss_term_rows`` (per-term row masks); the default
        tags every row ``"ce"``.

        The scalar training objective itself is task-independent — the loop calls
        ``sparse_steer.objectives.composed_objective`` on this batch (a weighted sum of
        the ``ce``/``kl`` terms selected by config), plus the model's L0 penalty.
        """
        enc = tokenizer([r["text"] for r in rows], return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids.masked_fill(mask == 0, -100),
            "loss_term_rows": {
                "ce": torch.ones(len(rows), dtype=torch.bool, device=device)
            },
        }

    # ── Caching ───────────────────────────────────────────────────────

    # Shared ``extra_cache_fields`` building blocks. Tasks merge these via
    # ``fields.update(...)``; every key / default / transform is part of the cache-hash
    # identity, so changing anything here invalidates existing cached artifacts.

    @staticmethod
    def _model_fields(config: DictConfig) -> dict[str, Any]:
        """Base-model identity — neither field is in the global cache key. ``lora_adapter``
        is read strictly (attribute access); sleeper reads it leniently via
        ``config.get`` and truthfulqa keys ``model_dtype`` only, so both stay inline there.
        """
        return {
            "model_dtype": config.get("model_dtype", "float16"),
            "lora_adapter": config.lora_adapter,
        }

    @staticmethod
    def _direction_fields(config: DictConfig) -> dict[str, Any]:
        """Sourcing identity for the steering artifacts: which intervention is applied and
        how each site's direction is sourced. Two runs differing here produce different
        steering vectors from the same extraction data."""
        return {
            "intervention": config.intervention,
            "direction_source": config.get("direction_source", "self"),
            "normalize_steering_vectors": config.get("normalize_steering_vectors", False),
        }

    @staticmethod
    def _scale_fields(config: DictConfig) -> dict[str, Any]:
        """Scale params carry non-None defaults, so they are keyed here (the base
        _CONFIG_FIELDS would key them as None for methods that omit them, changing the
        hash)."""
        return {
            "learn_scale": config.get("learn_scale", False),
            "shared_scale": config.get("shared_scale", False),
            "init_raw_scale": config.get("init_raw_scale", 0.0),
            "freeze_raw_scale": config.get("freeze_raw_scale", False),
        }

    @staticmethod
    def _eval_common_fields(config: DictConfig) -> dict[str, Any]:
        """Generative-eval identity shared by the jailbreak-style metric suites (jailbreak +
        safesteer). sleeper's eval block uses different keys and defaults (lenient gets,
        ``gen_tokens`` falling back to ``completion_tokens``) so it stays inline there."""
        return {
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

    @staticmethod
    def _kl_fields(config: DictConfig) -> dict[str, Any]:
        """KL-term identity, keyed only when the KL loss term is active (kl_weight > 0)."""
        if float(config.get("kl_weight", 0.0)) > 0:
            return {
                "kl_direction": config.get("kl_direction", "reverse"),
                "kl_positions": config.get("kl_positions", "first_token"),
            }
        return {}

    @staticmethod
    def _gate_trained_artifact(artifact_type: ArtifactType, config: DictConfig) -> bool:
        """True for artifacts that are a deterministic function of the WHOLE gate-training
        recipe: SPARSE_STEERING always, STEERED_EVAL only when the refinement trains gates.
        Without keying the recipe onto these, a frontier sweep over e.g. l0_lambda would
        silently reuse one cached result."""
        trains_gates = config.get("refinement_method") == "gate_training"
        return artifact_type == ArtifactType.SPARSE_STEERING or (
            artifact_type == ArtifactType.STEERED_EVAL and trains_gates
        )

    @abc.abstractmethod
    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]: ...

    @abc.abstractmethod
    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]: ...


__all__ = ["SelectionPolicy", "TaskSpec"]
