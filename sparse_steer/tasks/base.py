import abc
from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..steering import SteeringModel
from ..utils.cache import ArtifactType


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

    # ── Training objective ────────────────────────────────────────────

    def collate(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        config: DictConfig,
    ) -> dict[str, Tensor]:
        """Turn a batch of dataset rows into the tensors ``loss`` consumes.

        Default: a next-token cross-entropy batch from the ``text`` column
        (``config`` is unused). Override to emit task-specific tensors — e.g. a
        ``steer_mask`` (a ``(batch, seq)`` bool) which the training loop honours
        via ``SteeringModel.steer_positions`` to confine steering to prompt
        positions.
        """
        enc = tokenizer([r["text"] for r in rows], return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids.masked_fill(mask == 0, -100),
        }

    @abc.abstractmethod
    def loss(
        self,
        model: SteeringModel,
        batch: dict[str, Tensor],
        logits: Tensor,
    ) -> Tensor:
        """Scalar training objective for this task.

        ``logits`` are the steered model's logits for ``batch`` (the loop runs the
        forward, honouring ``batch["steer_mask"]`` if present). ``model`` is passed
        so an objective may run extra reference forwards (e.g. an unsteered clean
        pass via ``model.steering_disabled()``). The steering-gate L0 penalty is
        task-independent and is added by the training loop, not here.
        """
        ...

    # ── Caching ───────────────────────────────────────────────────────

    @abc.abstractmethod
    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]: ...

    @abc.abstractmethod
    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]: ...


__all__ = ["TaskSpec"]
