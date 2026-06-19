"""The steering Experiment engine — loads a SteeringModel and runs the strength solver.

The strength solver (``config.refinement_method`` over ``STRENGTH_SOLVERS`` + task-contributed) owns
solving the direction (it calls ``source_vectors``) and then setting strengths (fixed or trained).
WHERE the direction comes from is the orthogonal ``direction_source`` axis, resolved inside
``source_vectors``. Was the ``SteeringExperiment`` half of ``experiment/steering.py``.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_steering_model

from ..base import Experiment
from .solvers import STRENGTH_SOLVERS


class SteeringExperiment(Experiment):
    """Steering methods (dense, sparse, …). The strength solver is selected by
    ``config.refinement_method`` over the built-in + task-contributed strategies."""

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
        strategies = {**STRENGTH_SOLVERS, **self.task.refinement_strategies()}
        method = self.config.get("refinement_method")
        if method not in strategies:
            raise ValueError(
                f"Unknown refinement_method {method!r}. Available: {sorted(strategies)}"
            )
        return strategies[method](self, model, tokenizer, extraction_ds, train_ds, output_dir)


__all__ = ["SteeringExperiment", "STRENGTH_SOLVERS"]
