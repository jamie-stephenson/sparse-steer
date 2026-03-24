from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import Experiment
from ._steering_common import load_steering_model, run_extraction


class DenseExperiment(Experiment):

    def _load_model(self) -> PreTrainedModel:
        return load_steering_model(self.config, sparse=False)

    def _run_pipeline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[PreTrainedModel, dict[str, str | None], dict[str, Any]]:
        cache_info: dict[str, Any] = {}

        steering_vectors = run_extraction(
            self, model, tokenizer, extraction_ds, cache_info
        )
        if steering_vectors is not None:
            model.set_all_vectors(
                steering_vectors,
                normalize=self.config.normalize_steering_vectors,
            )

        return model, {}, cache_info


__all__ = ["DenseExperiment"]
