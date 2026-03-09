from dataclasses import dataclass
from typing import Literal

from ...experiment import ExperimentConfig


@dataclass
class TruthfulQAConfig(ExperimentConfig):
    # Which mc target set to use for contrastive training pairs.
    extraction_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1"
    gate_train_mcq_mode: Literal["mc0", "mc1", "mc2"] = "mc1"


__all__ = [
    "TruthfulQAConfig",
]
