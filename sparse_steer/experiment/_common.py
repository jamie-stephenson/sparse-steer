"""Shared helpers for steering experiments."""

from typing import Any

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from ..extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from ..steering import HardConcreteConfig, SteeringModel
from ..utils.cache import ArtifactType

from .base import Experiment

_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(config: DictConfig) -> torch.dtype:
    return _DTYPES[config.get("dtype", "float16")]


def load_steering_model(config: DictConfig) -> SteeringModel:
    """Load a TransformerLens model and attach steering hooks."""
    gate_config = None
    if config.get("gate_config") is not None:
        gate_cfg = OmegaConf.to_container(config.gate_config, resolve=True)
        gate_config = HardConcreteConfig(**gate_cfg)

    layer_ids = (
        list(config.steering_layer_ids)
        if config.get("steering_layer_ids") is not None
        else None
    )
    learn_scale = config.get("learn_scale", False)
    shared_scale = config.get("shared_scale", False)
    init_raw_scale = config.get("init_raw_scale", 0.0)
    intervention = config.get("intervention", "steer")

    desc = [intervention]
    if gate_config is not None:
        desc.append("gates")
    if shared_scale:
        desc.append("shared learned scale")
    elif learn_scale:
        desc.append("learned scale")
    else:
        desc.append(f"scale=softplus({init_raw_scale:.2f})")
    print(f"Initialising steering model ({', '.join(desc)})...")

    return SteeringModel.from_pretrained(
        config.model_name,
        device=config.device,
        dtype=resolve_dtype(config),
        lora_adapter=config.get("lora_adapter"),
        steering_layer_ids=layer_ids,
        steering_components=list(config.targets),
        gate_config=gate_config,
        learn_scale=learn_scale,
        shared_scale=shared_scale,
        init_raw_scale=init_raw_scale,
        intervention=intervention,
    )


def load_plain_model(config: DictConfig) -> SteeringModel:
    """Load a TransformerLens model with no steering (used for the unsteered control)."""
    return SteeringModel.from_pretrained(
        config.model_name,
        device=config.device,
        dtype=resolve_dtype(config),
        lora_adapter=config.get("lora_adapter"),
        steering_layer_ids=[],
        steering_components=[],
    )


def run_extraction(
    experiment: Experiment,
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    extraction_ds: Dataset,
    cache_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Run steering vector extraction with caching."""
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
    steering_vectors = extract_steering_vectors(extraction_with_acts, component_names)
    sv_dest = experiment._prepare_cache_path(ArtifactType.STEERING_VECTORS)
    metadata = OmegaConf.to_container(config, resolve=True)
    save_steering_vectors(steering_vectors, sv_dest, metadata=metadata)
    steering_vectors_path = str(experiment._finalize_cache(ArtifactType.STEERING_VECTORS))
    print(f"Saved steering vectors to {steering_vectors_path}")
    cache_info["steering_vectors"] = {"status": "miss"}

    return steering_vectors


__all__ = [
    "load_plain_model",
    "load_steering_model",
    "resolve_dtype",
    "run_extraction",
]
