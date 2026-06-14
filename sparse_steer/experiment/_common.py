"""Shared helpers for steering experiments.

Model construction lives in ``sparse_steer.core.loading``; this module keeps only the
pieces that genuinely depend on the experiment layer (cache-aware extraction)."""

from typing import Any

from datasets import Dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from sparse_steer.core.extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.cache import ArtifactType

from .base import Experiment


def run_extraction(
    experiment: Experiment,
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    extraction_ds: Dataset,
    cache_info: dict[str, Any],
    *,
    targets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run steering vector extraction with caching.

    ``targets`` overrides which component taps are extracted (default ``config.targets``); the
    sourcing layer widens it so a pinned ``direction_source`` component is extracted even when it
    is not itself a steering target.
    """
    config = experiment.config
    targets = list(config.targets) if targets is None else list(targets)
    sv_hit = experiment._try_cache_lookup(ArtifactType.STEERING_VECTORS)

    if sv_hit is not None:
        steering_vectors, _ = load_steering_vectors(sv_hit.artifact_path)
        cache_info["steering_vectors"] = {
            "status": "hit",
            "path": str(sv_hit.artifact_path),
        }
        return steering_vectors

    extraction_with_acts, component_names = collect_activations(
        extraction_ds,
        model,
        tokenizer,
        targets=targets,
        batch_size=config.extract_batch_size,
        token_position=config.extract_token_position,
    )
    print("Computing steering vectors...")
    steering_vectors = extract_steering_vectors(
        extraction_with_acts,
        component_names,
        orthogonalize_k=int(config.get("orthogonalize_harmless_pcs", 0)),
    )
    sv_dest = experiment._prepare_cache_path(ArtifactType.STEERING_VECTORS)
    metadata = OmegaConf.to_container(config, resolve=True)
    save_steering_vectors(steering_vectors, sv_dest, metadata=metadata)
    steering_vectors_path = str(experiment._finalize_cache(ArtifactType.STEERING_VECTORS))
    print(f"Saved steering vectors to {steering_vectors_path}")
    cache_info["steering_vectors"] = {"status": "miss"}

    return steering_vectors


__all__ = ["run_extraction"]
