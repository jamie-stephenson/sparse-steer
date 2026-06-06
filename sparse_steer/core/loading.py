"""Model construction: load a TransformerLens model and (optionally) attach steering.

Task-agnostic, so it lives in ``core`` rather than ``experiment`` — both the experiment
layer and tasks (e.g. jailbreak self-loading an unsteered model for bucketing) depend on
it, and putting it here keeps the dependency graph acyclic."""

import torch
from omegaconf import DictConfig, OmegaConf

from .steering import HardConcreteConfig, SteeringModel

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


__all__ = [
    "load_plain_model",
    "load_steering_model",
    "resolve_dtype",
]
