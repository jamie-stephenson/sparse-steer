"""Cache-aware steering-vector extraction (the input every solver reads).

Model construction lives in ``sparse_steer.core.loading``; this module keeps only the piece
that genuinely depends on the experiment layer (cache-aware extraction). Was
``experiment/_common.py`` — moved in here since its sole consumer is the steering subsystem."""

from typing import Any

from datasets import Dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from sparse_steer.core.extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    prune_top_l2,
    save_steering_vectors,
)
from sparse_steer.core.loading import load_extraction_model
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.cache import ArtifactType
from sparse_steer.utils.memory import free_model_memory

from ..base import Experiment


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
    sourcing layer widens it so a fixed ``direction_source`` component is extracted even when it
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

    # Cross-model transfer (task-agnostic): when extraction_model_name is set and differs from the
    # steered model_name, read activations from a separate read-only model, then free it; everything
    # downstream (set_all_vectors, gate training, eval) stays on the steered `model`. Default
    # (extraction_model_name unset/equal) ⇒ self-extraction, behaviour unchanged.
    ext_name = config.get("extraction_model_name")
    if ext_name and ext_name != config.model_name:
        ext_model, ext_tokenizer = load_extraction_model(config, model)
        try:
            extraction_with_acts, component_names = collect_activations(
                extraction_ds,
                ext_model,
                ext_tokenizer,
                targets=targets,
                batch_size=config.extract_batch_size,
                token_position=config.extract_token_position,
            )
        finally:
            del ext_model
            free_model_memory()
    else:
        extraction_with_acts, component_names = collect_activations(
            extraction_ds,
            model,
            tokenizer,
            targets=targets,
            batch_size=config.extract_batch_size,
            token_position=config.extract_token_position,
        )
    print("Computing steering vectors...")
    steering_vectors = extract_steering_vectors(extraction_with_acts, component_names)
    prune_frac = config.get("prune_top_frac")
    if prune_frac:
        steering_vectors = {c: prune_top_l2(v, float(prune_frac)) for c, v in steering_vectors.items()}
        print(f"  Pruned ω to top {prune_frac} of components by |value| (SafeSteer generic denoiser)")
    sv_dest = experiment._prepare_cache_path(ArtifactType.STEERING_VECTORS)
    metadata = OmegaConf.to_container(config, resolve=True)
    save_steering_vectors(steering_vectors, sv_dest, metadata=metadata)
    steering_vectors_path = str(experiment._finalize_cache(ArtifactType.STEERING_VECTORS))
    print(f"Saved steering vectors to {steering_vectors_path}")
    cache_info["steering_vectors"] = {"status": "miss"}

    return steering_vectors


__all__ = ["run_extraction"]
