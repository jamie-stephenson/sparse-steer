from pathlib import Path

import torch
from torch import Tensor, nn

from ..hardconcrete import HardConcreteConfig, HardConcreteGateMixin
from .base import BaseSteeringLM, Component, SteeringHook


class SparseSteeringHook(HardConcreteGateMixin, SteeringHook):
    """Steering hook with HardConcrete learned gates."""

    def __init__(
        self, vector_shape: tuple[int, ...], gate_config: HardConcreteConfig
    ) -> None:
        # multi-dim (e.g. num_heads, head_dim): one gate per first dim
        # 1D (e.g. mlp_dim): single gate for the whole vector
        num_gates = vector_shape[0] if len(vector_shape) > 1 else 1
        super().__init__(
            vector_shape=vector_shape, num_gates=num_gates, gate_config=gate_config
        )

    def _compute_correction(self, hidden: Tensor) -> Tensor:
        gate = self._scaled_gate(dtype=hidden.dtype, device=hidden.device)
        steering = self.steering_vectors.to(device=hidden.device, dtype=hidden.dtype)
        # broadcast gate over trailing dims (e.g. head_dim for attention)
        gated = steering * gate.unsqueeze(-1) if steering.ndim > 1 else steering * gate
        return gated.reshape([1] * (hidden.ndim - 1) + [-1])


class SparseSteeringLM(BaseSteeringLM):
    """Mixin that adds sparse-steering (HardConcrete gates) to a ``PreTrainedModel``."""

    def upgrade_for_steering(
        self,
        gate_config: HardConcreteConfig,
        steering_layer_ids: list[int],
        steering_components: list[Component],
    ) -> None:
        self.gate_config = gate_config
        self.steering_layer_ids = steering_layer_ids
        self.steering_components = steering_components
        self._attach_steering_hooks()

    def _create_hook(self, component: Component, layer: nn.Module) -> SteeringHook:
        if component == "residual":
            raise NotImplementedError(
                "Residual sparse steering is not yet supported. "
                "Use dense steering for residual stream experiments."
            )
        shape = self._get_vector_shape(component, layer)
        return SparseSteeringHook(shape, gate_config=self.gate_config)

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
        """Load a saved sparse-steering checkpoint and attach hooks."""
        payload = torch.load(Path(path), map_location="cpu")
        state_dict = payload["state_dict"]
        gate_config = HardConcreteConfig(**payload["config"])

        steering_layer_ids = payload.get("steering_layer_ids")
        if steering_layer_ids is None:
            if active_layers_only:
                steering_layer_ids = sorted(
                    gate_config.active_layer_indices(state_dict)
                )
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
            k
            for k in load_info.missing_keys
            if any(term in k for term in ("log_alpha", "log_scale", "steering_vector"))
        ]
        if missing:
            raise RuntimeError(
                f"Sparse steering checkpoint mismatch. "
                f"Missing steering keys: {missing}."
            )


__all__ = [
    "SparseSteeringHook",
    "SparseSteeringLM",
]
