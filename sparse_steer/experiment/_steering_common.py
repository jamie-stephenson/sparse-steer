"""Shared helpers for steering experiments."""

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
from ..models import MODEL_REGISTRY
from ..utils.cache import ArtifactType

from .base import Experiment


def load_steering_model(config: DictConfig) -> PreTrainedModel:
    """Load a model from the registry and upgrade it for steering."""
    from ..hardconcrete import HardConcreteConfig

    hf_config = AutoConfig.from_pretrained(config.model_name)
    model_type = getattr(hf_config, "model_type", None)

    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for '{config.model_name}'. "
            f"Supported: {sorted(MODEL_REGISTRY)}"
        )

    model = model_cls.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    ).to(config.device)

    layer_ids = (
        list(config.steering_layer_ids)
        if config.get("steering_layer_ids") is not None
        else list(range(len(model.get_layers())))
    )

    gate_config = None
    if config.get("gate_config") is not None:
        gate_cfg = OmegaConf.to_container(config.gate_config, resolve=True)
        gate_config = HardConcreteConfig(**gate_cfg)

    learn_scale = config.get("learn_scale", False)
    shared_scale = config.get("shared_scale", False)
    init_log_scale = config.get("init_log_scale", 0.0)

    desc = []
    if gate_config is not None:
        desc.append("gates")
    if shared_scale:
        desc.append("shared learned scale")
    elif learn_scale:
        desc.append("learned scale")
    else:
        desc.append(f"scale=softplus({init_log_scale:.2f})")
    print(f"Initialising steering model ({', '.join(desc)})...")

    model.upgrade_for_steering(
        steering_layer_ids=layer_ids,
        steering_components=list(config.targets),
        gate_config=gate_config,
        learn_scale=learn_scale,
        shared_scale=shared_scale,
        init_log_scale=init_log_scale,
    )

    return model


def run_extraction(
    experiment: Experiment,
    model: PreTrainedModel,
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
