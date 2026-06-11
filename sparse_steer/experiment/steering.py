import shutil
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_steering_model
from sparse_steer.train import train_steering
from sparse_steer.utils.cache import ArtifactType
from .base import Experiment
from .sourcing import source_vectors

RefinementFn = Callable[..., tuple]


# ── Built-in (task-general) refinement strategies ─────────────────────
# A strategy turns the loaded model + datasets into the refined model:
#   fn(experiment, model, tokenizer, extraction_ds, train_ds, output_dir)
#       -> (model, artifacts, cache_info)
# Refinement is now ONLY "what happens to the steering params after sourcing":
# `none` (set and stop) or `gate_training` (set, then train gates/scale). WHERE the
# direction comes from is the orthogonal `direction_source` axis, resolved by
# `sourcing.source_vectors` (self / pinned broadcast / grid_select). Tasks may still
# contribute strategies via TaskSpec.refinement_strategies(); the pools merge at dispatch.


def _refine_none(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """dense / caa / arditi reproduction: source the direction(s) and set them; no training."""
    cache_info: dict[str, Any] = {}
    steering_vectors = source_vectors(
        experiment, model, tokenizer, extraction_ds, train_ds, cache_info
    )
    if steering_vectors is not None:
        model.set_all_vectors(
            steering_vectors, normalize=experiment.config.normalize_steering_vectors
        )
    return model, {}, cache_info


def _refine_gate_training(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """sparse / gates_only / scale_only / shared_scale_only: extract + set the direction, then
    train the learnable gates/scale against the task objective."""
    artifacts: dict[str, str | None] = {}
    cache_info: dict[str, Any] = {}

    ss_hit = experiment._try_cache_lookup(ArtifactType.SPARSE_STEERING)
    if ss_hit is not None:
        model.load_steering(ss_hit.artifact_path)
        artifacts["steering_path"] = str(ss_hit.artifact_path)
        cache_info["steering"] = {"status": "hit", "path": str(ss_hit.artifact_path)}
        for name in ("gate_heatmap.png", "gate_animation.gif"):
            cached = ss_hit.artifact_path.parent / name
            if cached.is_file():
                shutil.copy2(cached, output_dir / name)
        return model, artifacts, cache_info

    steering_vectors = source_vectors(
        experiment, model, tokenizer, extraction_ds, train_ds, cache_info
    )
    if steering_vectors is not None:
        model.set_all_vectors(
            steering_vectors, normalize=experiment.config.normalize_steering_vectors
        )

    print("Training steering parameters...")
    train_steering(
        model, tokenizer, train_ds, experiment.config, output_dir=output_dir, task=experiment.task
    )

    ss_dest = experiment._prepare_cache_path(ArtifactType.SPARSE_STEERING)
    model.save_steering(ss_dest)
    for name in ("gate_heatmap.png", "gate_animation.gif"):
        src = output_dir / name
        if src.is_file():
            shutil.copy2(src, ss_dest.parent / name)
    steering_path = str(experiment._finalize_cache(ArtifactType.SPARSE_STEERING))
    print(f"Saved steering to {steering_path}")
    artifacts["steering_path"] = steering_path
    cache_info["steering"] = {"status": "miss"}
    return model, artifacts, cache_info


BUILTIN_REFINEMENTS: dict[str, RefinementFn] = {
    "none": _refine_none,
    "gate_training": _refine_gate_training,
}


class SteeringExperiment(Experiment):
    """Steering methods (dense, sparse, …). The refinement step is selected by
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
        strategies = {**BUILTIN_REFINEMENTS, **self.task.refinement_strategies()}
        method = self.config.get("refinement_method")
        if method not in strategies:
            raise ValueError(
                f"Unknown refinement_method {method!r}. Available: {sorted(strategies)}"
            )
        return strategies[method](self, model, tokenizer, extraction_ds, train_ds, output_dir)


__all__ = ["SteeringExperiment", "BUILTIN_REFINEMENTS"]
