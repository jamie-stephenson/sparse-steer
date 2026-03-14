from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..hardconcrete import HardConcreteConfig, HardConcreteGateMixin
from .base import BaseSteeringAttention, BaseSteeringMLP, BaseSteeringLM, Component, _copy_module_state


class SparseSteeringAttention(HardConcreteGateMixin, BaseSteeringAttention):
    """Sparse-steered attention using HardConcrete gates."""

    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any:
        if not self.steering_enabled or not inputs:
            return None

        # (batch, seq, hidden)
        attn_output = inputs[0]
        expected_hidden_size = self._num_heads * self._head_dim
        assert isinstance(attn_output, torch.Tensor), "steering hook saw a non-tensor input"
        assert attn_output.shape[-1] == expected_hidden_size, "steering hook received unexpected shape"

        # (num_heads,)
        gate = self._scaled_gate(dtype=attn_output.dtype, device=attn_output.device)

        # (num_heads, head_dim)
        steering = self.steering_vectors.to(device=attn_output.device, dtype=attn_output.dtype)

        # (1, 1, hidden)
        correction = (steering * gate.unsqueeze(-1)).reshape(1, 1, expected_hidden_size)

        return (attn_output + correction, *inputs[1:])


class SparseSteeringMLP(HardConcreteGateMixin, BaseSteeringMLP):
    """Sparse-steered MLP using HardConcrete gates."""

    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any:
        if not self.steering_enabled or not inputs:
            return None

        # (batch, seq, mlp_dim)
        mlp_hidden = inputs[0]
        assert isinstance(mlp_hidden, torch.Tensor), "steering hook saw a non-tensor input"
        assert mlp_hidden.shape[-1] == self._mlp_dim, "steering hook received unexpected shape"

        # (mlp_dim,)
        gate = self._scaled_gate(dtype=mlp_hidden.dtype, device=mlp_hidden.device)

        # (mlp_dim,)
        steering = self.steering_vectors.to(device=mlp_hidden.device, dtype=mlp_hidden.dtype)

        # (1, 1, mlp_dim)
        correction = (steering * gate).reshape(*([1] * (mlp_hidden.ndim - 1)), self._mlp_dim)

        return (mlp_hidden + correction, *inputs[1:])


class SparseSteeringLM(BaseSteeringLM):
    """Mixin that adds sparse-steering (HardConcrete gates) to a ``PreTrainedModel``.

    Concrete classes inherit from both this mixin and a HuggingFace model::

        class LlamaSparseSteeringLM(SparseSteeringLM, LlamaForCausalLM):
            attn_cls = LlamaSparseSteeringAttention
            ...

    The resulting object IS the HF model, so ``Trainer``, ``.generate()``,
    and ``.forward()`` all work directly.  Load weights with HF's
    ``from_pretrained``, then call ``upgrade_for_steering`` to swap in
    gated modules.
    """

    def upgrade_for_steering(
        self,
        gate_config: HardConcreteConfig,
        steering_layer_ids: list[int],
        steering_components: list[Component],
    ) -> None:
        """Swap in sparse steering modules for the specified layers and components."""
        self.gate_config = gate_config
        self.steering_layer_ids = steering_layer_ids
        self.steering_components = steering_components
        self._upgrade_modules()

    def _upgrade_single_module(
        self, source: nn.Module, replacement_cls: type
    ) -> nn.Module:
        if isinstance(source, replacement_cls):
            source._set_gate_config(self.gate_config)
            return source
        config = getattr(source, "config", None)
        kwargs: dict[str, Any] = {}
        layer_idx = getattr(source, "layer_idx", None)
        if layer_idx is not None:
            kwargs["layer_idx"] = layer_idx
        upgraded = replacement_cls(config, gate_config=self.gate_config, **kwargs)
        _copy_module_state(source, upgraded)
        return upgraded

    def freeze_base_model(self, freeze_log_scale: bool = False) -> None:
        """Freeze everything, then unfreeze HardConcrete gate parameters."""
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, HardConcreteGateMixin):
                log_alpha = getattr(module, "log_alpha", None)
                if log_alpha is not None:
                    log_alpha.requires_grad = True
                if not freeze_log_scale:
                    log_scale = getattr(module, "log_scale", None)
                    if log_scale is not None:
                        log_scale.requires_grad = True

    def sparse_steering_state_dict(self) -> dict[str, Tensor]:
        steering_terms = ("log_alpha", "log_scale", "steering_vectors")
        return {
            k: v
            for k, v in self.state_dict().items()
            if any(term in k for term in steering_terms)
        }

    def save_steering(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix:
            output_path = path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path = path / "sparse_steering.pt"
        payload = {
            "config": self.gate_config.to_dict(),
            "steering_layer_ids": self.steering_layer_ids,
            "steering_components": self.steering_components,
            "state_dict": self.sparse_steering_state_dict(),
        }
        torch.save(payload, output_path)
        return output_path

    def load_steering(
        self,
        path: str | Path,
        *,
        active_layers_only: bool = True,
    ) -> None:
        """Load a saved sparse-steering checkpoint and upgrade modules."""
        payload = torch.load(Path(path), map_location="cpu")
        state_dict = payload["state_dict"]
        gate_config = HardConcreteConfig(**payload["config"])

        steering_layer_ids = payload.get("steering_layer_ids")
        if steering_layer_ids is None:
            if active_layers_only:
                steering_layer_ids = sorted(gate_config.active_layer_indices(state_dict))
            else:
                steering_layer_ids = list(range(len(self.get_layers())))

        steering_components = payload.get("steering_components", ["attention"])

        self.upgrade_for_steering(
            gate_config=gate_config,
            steering_layer_ids=steering_layer_ids,
            steering_components=steering_components,
        )

        load_info = self.load_state_dict(state_dict, strict=False)
        missing = [
            k for k in load_info.missing_keys
            if any(term in k for term in ("log_alpha", "log_scale", "steering_vector"))
        ]
        if missing:
            raise RuntimeError(
                f"Sparse steering checkpoint mismatch. "
                f"Missing steering keys: {missing}."
            )


__all__ = [
    "SparseSteeringAttention",
    "SparseSteeringMLP",
    "SparseSteeringLM",
]
