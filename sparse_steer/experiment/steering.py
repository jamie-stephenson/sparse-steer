from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils.cache import ArtifactType
from .base import Experiment
from ._steering_common import load_steering_model, run_extraction


class SteeringExperiment(Experiment):
    """Unified experiment for all steering methods (dense, sparse, and off-diagonal)."""

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
        from ..train import train_steering

        artifacts: dict[str, str | None] = {}
        cache_info: dict[str, Any] = {}

        # If the model has learnable parameters, check for a trained checkpoint
        if model.has_learnable_steering:
            ss_hit = self._try_cache_lookup(ArtifactType.SPARSE_STEERING)
            if ss_hit is not None:
                model.load_steering(ss_hit.artifact_path)
                artifacts["steering_path"] = str(ss_hit.artifact_path)
                cache_info["steering"] = {
                    "status": "hit",
                    "path": str(ss_hit.artifact_path),
                }
                for name in ("gate_heatmap.png", "gate_animation.gif"):
                    cached = ss_hit.artifact_path.parent / name
                    if cached.is_file():
                        shutil.copy2(cached, output_dir / name)
                return model, artifacts, cache_info

        # Extract steering vectors
        steering_vectors = run_extraction(
            self, model, tokenizer, extraction_ds, cache_info
        )
        if steering_vectors is not None:
            model.set_all_vectors(
                steering_vectors,
                normalize=self.config.normalize_steering_vectors,
            )

        # Train if there are learnable parameters
        if model.has_learnable_steering:
            print("Training steering parameters...")
            train_steering(
                model, tokenizer, train_ds, self.config, output_dir=output_dir
            )

            ss_dest = self._prepare_cache_path(ArtifactType.SPARSE_STEERING)
            model.save_steering(ss_dest)
            for name in ("gate_heatmap.png", "gate_animation.gif"):
                src = output_dir / name
                if src.is_file():
                    shutil.copy2(src, ss_dest.parent / name)
            steering_path = str(
                self._finalize_cache(ArtifactType.SPARSE_STEERING)
            )
            print(f"Saved steering to {steering_path}")
            artifacts["steering_path"] = steering_path
            cache_info["steering"] = {"status": "miss"}

        return model, artifacts, cache_info


__all__ = ["SteeringExperiment"]
