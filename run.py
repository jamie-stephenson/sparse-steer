#!/usr/bin/env python3

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig

from sparse_steer.experiment import (
    BaselineExperiment,
    LoraExperiment,
    SteeringExperiment,
)
from sparse_steer.experiment.base import Experiment, TaskSpec
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask

METHODS: dict[str, type[Experiment]] = {
    "baseline": BaselineExperiment,
    "dense": SteeringExperiment,
    "sparse": SteeringExperiment,
    "scale_only": SteeringExperiment,
    "gates_only": SteeringExperiment,
    "lora": LoraExperiment,
}

TASKS: dict[str, type[TaskSpec]] = {
    "truthfulqa": TruthfulQATask,
}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    task_name = cfg.get("task_name", "truthfulqa")
    method = cfg.method

    if task_name not in TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {sorted(TASKS)}"
        )
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {sorted(METHODS)}"
        )

    task = TASKS[task_name]()
    experiment_cls = METHODS[method]
    experiment = experiment_cls(cfg, task)
    experiment.run()


if __name__ == "__main__":
    main()
