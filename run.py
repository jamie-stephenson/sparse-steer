#!/usr/bin/env python3

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig

from sparse_steer.experiment import build_experiment
from sparse_steer.tasks.base import TaskSpec
from sparse_steer.tasks.jailbreak.task import RefusalTask
from sparse_steer.tasks.safesteer.task import SafeSteerTask
from sparse_steer.tasks.tinysleepers.task import TinySleepersTask
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask

TASKS: dict[str, type[TaskSpec]] = {
    "truthfulqa": TruthfulQATask,
    "tinysleepers": TinySleepersTask,
    "refusal": RefusalTask,
    "safesteer": SafeSteerTask,
}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    task_name = cfg.get("task_name", "truthfulqa")

    if task_name not in TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {sorted(TASKS)}"
        )

    task = TASKS[task_name]()
    experiment = build_experiment(cfg, task)
    experiment.run()


if __name__ == "__main__":
    main()
