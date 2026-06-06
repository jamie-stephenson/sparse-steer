from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from sparse_steer.core.loading import load_plain_model
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.cache import ArtifactType
from .base import Experiment


class UnsteeredExperiment(Experiment):
    _eval_artifact_type = ArtifactType.UNSTEERED_EVAL

    def _load_model(self) -> SteeringModel:
        return load_plain_model(self.config)

    def _run_pipeline(
        self,
        model: SteeringModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[SteeringModel, dict[str, str | None], dict[str, Any]]:
        return model, {}, {}


__all__ = ["UnsteeredExperiment"]
