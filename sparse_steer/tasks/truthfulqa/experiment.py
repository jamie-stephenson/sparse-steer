from typing import cast

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict

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
        return evaluate(model, tokenizer, dataset)


__all__ = [
    "TruthfulQAExperiment",
]
