"""Shared helpers for dense and sparse steering experiments."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizerBase

from ..extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from ..models import DENSE_MODEL_REGISTRY, MODEL_REGISTRY
from ..utils.cache import ArtifactType

from .base import Experiment


def load_steering_model(
    config: DictConfig,
    *,
    sparse: bool,
) -> PreTrainedModel:
    """Load a model from the steering registry and upgrade it."""
    from ..hardconcrete import HardConcreteConfig

    hf_config = AutoConfig.from_pretrained(config.model_name)
    model_type = getattr(hf_config, "model_type", None)

    registry = MODEL_REGISTRY if sparse else DENSE_MODEL_REGISTRY
    model_cls = registry.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for '{config.model_name}'. "
            f"Supported: {sorted(registry)}"
        )

    model = model_cls.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    ).to(config.device)

    layer_ids = (
        list(config.steering_layer_ids)
        if config.get("steering_layer_ids") is not None
        else list(range(len(model.get_layers())))
    )

    if sparse:
        gate_cfg = OmegaConf.to_container(config.gate_config, resolve=True)
        gate_config = HardConcreteConfig(**gate_cfg)
        print("Initialising sparse-steering model...")
        model.upgrade_for_steering(
            gate_config=gate_config,
            steering_layer_ids=layer_ids,
            steering_components=list(config.targets),
        )
    else:
        print(
            f"Initialising dense-steering model "
            f"(steering_strength={config.steering_strength})..."
        )
        model.upgrade_for_steering(
            steering_strength=config.steering_strength,
            steering_layer_ids=layer_ids,
            steering_components=list(config.targets),
        )

    return model


def run_extraction(
    experiment: Experiment,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    extraction_ds: Dataset,
    cache_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Run steering vector extraction with caching.

    Returns the steering vectors dict, or *None* if skipped.
    """
    config = experiment.config
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
        targets=list(config.targets),
        batch_size=config.extract_batch_size,
        token_position=config.token_position,
    )
    print("Computing steering vectors...")
    steering_vectors = extract_steering_vectors(
        extraction_with_acts, component_names
    )
    sv_dest = experiment._prepare_cache_path(ArtifactType.STEERING_VECTORS)
    metadata = OmegaConf.to_container(config, resolve=True)
    save_steering_vectors(steering_vectors, sv_dest, metadata=metadata)
    steering_vectors_path = str(
        experiment._finalize_cache(ArtifactType.STEERING_VECTORS)
    )
    print(f"Saved steering vectors to {steering_vectors_path}")
    cache_info["steering_vectors"] = {"status": "miss"}

    return steering_vectors


__all__ = [
    "load_steering_model",
    "run_extraction",
]
