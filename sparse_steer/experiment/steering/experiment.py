"""The steering Experiment engine — loads a SteeringModel and runs the chosen refinement.

Naming (genus → species): a *solver* resolves one field of the per-site steering config. Its two
species are the **source** (direction field: self / fixed / grid_search, in ``solvers.source_vectors``)
and the **refinement** (strength field: none / gate_training / iti_head_select, in ``REFINEMENTS``).
This engine dispatches the refinement named by ``config.refinement_method``; the refinement itself
calls ``source_vectors`` for the direction. Was the ``SteeringExperiment`` half of
``experiment/steering.py``.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_steering_model

from ..base import Experiment
from .solvers import REFINEMENTS


class SteeringExperiment(Experiment):
    """Steering methods (dense, sparse, …). The strength-field refinement is selected by
    ``config.refinement_method`` over the built-in ``REFINEMENTS`` plus the task's
    ``extra_refinements()``."""

    def _load_model(self) -> PreTrainedModel:
        return load_steering_model(self.config)

    def _run_pipeline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[PreTrainedModel, dict[str, str | None], dict[str, Any]]:
        refinements = {**REFINEMENTS, **self.task.extra_refinements()}
        method = self.config.get("refinement_method")
        if method not in refinements:
            raise ValueError(
                f"Unknown refinement_method {method!r}. Available: {sorted(refinements)}"
            )
        return refinements[method](self, model, tokenizer, extraction_ds, train_ds, output_dir)


__all__ = ["SteeringExperiment", "REFINEMENTS"]
