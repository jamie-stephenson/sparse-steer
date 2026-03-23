from typing import Any, cast

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict

from ...utils.cache import ArtifactType
from ...experiment import SparseSteeringExperiment
from .config import TruthfulQAConfig
from .data import get_truthfulqa_datasets
from .eval import evaluate


class TruthfulQAExperiment(SparseSteeringExperiment):
    config: TruthfulQAConfig

    def __init__(self, config: TruthfulQAConfig | None = None) -> None:
        super().__init__(config or TruthfulQAConfig())
        self.config = cast(TruthfulQAConfig, self.config)

    @property
    def task_name(self) -> str:
        return "truthfulqa"

    def extra_cache_fields(self, artifact_type: ArtifactType) -> dict[str, Any]:
        if artifact_type == ArtifactType.STEERING_VECTORS:
            return {"extraction_mcq_mode": self.config.extraction_mcq_mode}
        if artifact_type in (ArtifactType.SPARSE_STEERING, ArtifactType.STEERED_EVAL):
            return {
                "extraction_mcq_mode": self.config.extraction_mcq_mode,
                "gate_train_mcq_mode": self.config.gate_train_mcq_mode,
            }
        # BASELINE_EVAL: no task extras — independent of steering/training choices
        return {}

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        files = ["sparse_steer/tasks/truthfulqa/data.py"]
        if artifact_type in (ArtifactType.BASELINE_EVAL, ArtifactType.STEERED_EVAL):
            files.append("sparse_steer/tasks/truthfulqa/eval.py")
        return files

    def build_datasets(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        return get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode=self.config.extraction_mcq_mode,
            gate_train_mcq_mode=self.config.gate_train_mcq_mode,
            extraction_fraction=self.config.extraction_fraction,
            seed=self.config.seed,
        )

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> dict[str, float]:
        return evaluate(
            model, tokenizer, dataset, batch_size=self.config.eval_batch_size
        )


__all__ = [
    "TruthfulQAExperiment",
]
