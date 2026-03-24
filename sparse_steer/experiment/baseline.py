from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from ..utils.cache import ArtifactType
from .base import Experiment


class BaselineExperiment(Experiment):
    _eval_artifact_type = ArtifactType.BASELINE_EVAL

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=torch.float16
        ).to(self.config.device)

    def _run_pipeline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[PreTrainedModel, dict[str, str | None], dict[str, Any]]:
        return model, {}, {}


__all__ = ["BaselineExperiment"]
