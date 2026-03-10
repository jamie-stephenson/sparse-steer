#!/usr/bin/env python3

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from sparse_steer.tasks.truthfulqa import TruthfulQAConfig, TruthfulQAExperiment


TASKS: dict[str, tuple[type[Any], type[Any]]] = {
    "truthfulqa": (TruthfulQAConfig, TruthfulQAExperiment),
}


def _load_config(config_cls: type[Any], path: str | None) -> Any:
    if path is None:
        return config_cls()

    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")

    valid_fields = {f.name for f in fields(config_cls)}
    unknown = set(payload) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_fields)}"
        )
    return config_cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sparse steering experiment for a task."
    )
    parser.add_argument(
        "task",
        choices=sorted(TASKS),
        help="Task experiment to run (for example: truthfulqa).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML file with config overrides for the selected task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_cls, experiment_cls = TASKS[args.task]
    config = _load_config(config_cls, args.config)
    experiment = experiment_cls(config)
    experiment.run()


if __name__ == "__main__":
    main()
