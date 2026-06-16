from omegaconf import DictConfig

from sparse_steer.tasks.base import TaskSpec

from .base import Experiment
from .lora import LoraExperiment
from .steering import SteeringExperiment
from .unsteered import UnsteeredExperiment

# Engine registry, keyed by the `experiment` field in configs/method/*.yaml.
# This is the only place that maps an engine name to its Experiment class; the
# `method` field is a preset identity (cache key + label), NOT a dispatch key.
EXPERIMENTS: dict[str, type[Experiment]] = {
    "steering": SteeringExperiment,
    "lora": LoraExperiment,
    "unsteered": UnsteeredExperiment,
}


def build_experiment(config: DictConfig, task: TaskSpec) -> Experiment:
    """Instantiate the Experiment engine named by ``config.experiment``."""
    name = config.experiment
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {sorted(EXPERIMENTS)}"
        )
    return EXPERIMENTS[name](config, task)


__all__ = [
    "EXPERIMENTS",
    "Experiment",
    "LoraExperiment",
    "SteeringExperiment",
    "UnsteeredExperiment",
    "build_experiment",
]
