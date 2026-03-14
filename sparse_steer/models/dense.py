from typing import Any

import torch
from torch import nn

from .base import BaseSteeringAttention, BaseSteeringMLP, BaseSteeringLM, Component, _copy_module_state


class DenseSteeringAttention(BaseSteeringAttention):
    """Dense steered attention -- no learned gates."""

    def __init__(self, *args, steering_strength: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.steering_strength = steering_strength

    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any:
        if not self.steering_enabled or not inputs:
            return None

        attn_output = inputs[0]
        expected_hidden_size = self._num_heads * self._head_dim
        assert isinstance(attn_output, torch.Tensor), "steering hook saw a non-tensor input"
        assert attn_output.shape[-1] == expected_hidden_size, "steering hook received unexpected shape"

        steering = self.steering_vectors.to(device=attn_output.device, dtype=attn_output.dtype)
        correction = (self.steering_strength * steering).reshape(1, 1, expected_hidden_size)

        return (attn_output + correction, *inputs[1:])


class DenseSteeringMLP(BaseSteeringMLP):
    """Dense steered MLP -- no learned gates."""

    def __init__(self, *args, steering_strength: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.steering_strength = steering_strength

    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any:
        if not self.steering_enabled or not inputs:
            return None

        mlp_hidden = inputs[0]
        assert isinstance(mlp_hidden, torch.Tensor), "steering hook saw a non-tensor input"
        assert mlp_hidden.shape[-1] == self._mlp_dim, "steering hook received unexpected shape"

        steering = self.steering_vectors.to(device=mlp_hidden.device, dtype=mlp_hidden.dtype)
        correction = (self.steering_strength * steering).reshape(*([1] * (mlp_hidden.ndim - 1)), self._mlp_dim)

        return (mlp_hidden + correction, *inputs[1:])


class DenseSteeringLM(BaseSteeringLM):
    """Mixin that adds dense steering to a ``PreTrainedModel``."""

    def upgrade_for_steering(
        self,
        steering_strength: float,
        steering_layer_ids: list[int],
        steering_components: list[Component],
    ) -> None:
        self.steering_strength = steering_strength
        self.steering_layer_ids = steering_layer_ids
        self.steering_components = steering_components
        self._upgrade_modules()

    def _upgrade_single_module(
        self, source: nn.Module, replacement_cls: type
    ) -> nn.Module:
        if isinstance(source, replacement_cls):
            source.steering_strength = self.steering_strength
            return source
        config = getattr(source, "config", None)
        kwargs: dict[str, Any] = {}
        layer_idx = getattr(source, "layer_idx", None)
        if layer_idx is not None:
            kwargs["layer_idx"] = layer_idx
        upgraded = replacement_cls(config, steering_strength=self.steering_strength, **kwargs)
        _copy_module_state(source, upgraded)
        return upgraded


__all__ = [
    "DenseSteeringAttention",
    "DenseSteeringMLP",
    "DenseSteeringLM",
]
