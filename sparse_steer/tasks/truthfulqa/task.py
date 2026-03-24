from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...utils.cache import ArtifactType
from .data import get_truthfulqa_datasets
from .eval import evaluate, evaluate_generative


class TruthfulQATask:
    @property
    def task_name(self) -> str:
        return "truthfulqa"

    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        return get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode=config.get("extraction_mcq_mode", "mc1"),
            gate_train_mcq_mode=config.get("gate_train_mcq_mode", "mc1"),
            extraction_fraction=config.extraction_fraction,
            seed=config.seed,
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
        if config.get("generative_eval", False):
            gen_metrics = evaluate_generative(
                model,
                tokenizer,
                dataset,
                max_new_tokens=config.get("gen_max_new_tokens", 64),
            )
            metrics.update(gen_metrics)
        return metrics

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
                "gate_train_mcq_mode": config.get(
                    "gate_train_mcq_mode", "mc1"
                ),
            }
            if artifact_type == ArtifactType.STEERED_EVAL:
                fields["generative_eval"] = config.get("generative_eval", False)
            return fields
        if artifact_type == ArtifactType.BASELINE_EVAL:
            return {
                "generative_eval": config.get("generative_eval", False),
            }
        if artifact_type == ArtifactType.PEFT_ADAPTER:
            return {
                "gate_train_mcq_mode": config.get(
                    "gate_train_mcq_mode", "mc1"
                ),
            }
        return {}

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        files = ["sparse_steer/tasks/truthfulqa/data.py"]
        if artifact_type in (
            ArtifactType.BASELINE_EVAL,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/truthfulqa/eval.py")
        return files


__all__ = ["TruthfulQATask"]
