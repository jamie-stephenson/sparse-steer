from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..hardconcrete import HardConcreteConfig
from .base import BaseModelLayout, Component
from .hook import ScaleMode, SteeringHook

# Hook attachment mode: pre-hook on output proj, or post-hook on layer
_PRE_HOOK_COMPONENTS = frozenset({"attention", "mlp"})


class SteeringLM(BaseModelLayout):
    """Mixin that adds steering to a ``PreTrainedModel`` subclass.

    Layout methods (``get_layers``, ``get_attention``, etc.) are abstract
    and must be implemented by a layout mixin such as ``LlamaLayout``.
    """

    def upgrade_for_steering(
        self,
        *,
        steering_layer_ids: list[int],
        steering_components: list[Component],
        scale: float = 1.0,
        gate_config: HardConcreteConfig | None = None,
        scale_mode: ScaleMode = "fixed",
        init_log_scale: float | None = None,
    ) -> None:
        self.scale = scale
        self.gate_config = gate_config
        self.scale_mode = scale_mode
        self.init_log_scale = init_log_scale
        self.steering_layer_ids = steering_layer_ids
        self.steering_components = steering_components

        # Create a single shared log_scale parameter for the whole model
        if scale_mode == "shared":
            from .hook import _softplus_inverse
            init = init_log_scale if init_log_scale is not None else _softplus_inverse(scale)
            self._shared_log_scale = nn.Parameter(torch.full((1,), init))
        else:
            self._shared_log_scale = None

        self._attach_steering_hooks()

    @property
    def has_learnable_steering(self) -> bool:
        return self.gate_config is not None or self.scale_mode != "fixed"

    # ── Hook management ──────────────────────────────────────────────

    def _create_hook(self, component: Component, layer: nn.Module) -> SteeringHook:
        shape = self._get_vector_shape(component, layer)
        return SteeringHook(
            shape,
            scale=self.scale,
            gate_config=self.gate_config,
            scale_mode=self.scale_mode,
            init_log_scale=self.init_log_scale,
            shared_log_scale=self._shared_log_scale,
        )

    def _validate_components(self, components: list[Component]) -> None:
        if "residual" in components and len(components) > 1:
            raise ValueError(
                "Residual steering is mutually exclusive with attention/mlp steering. "
                f"Got: {components}"
            )

    def _get_hook_target(self, component: Component, layer: nn.Module) -> nn.Module:
        if component == "attention":
            return self._get_output_proj(self.get_attention(layer))
        elif component == "mlp":
            return self._get_output_proj(self.get_mlp(layer))
        elif component == "residual":
            return layer
        raise ValueError(f"Unknown component: {component!r}")

    def _attach_steering_hooks(self) -> None:
        self._validate_components(self.steering_components)
        for i, layer in enumerate(self.get_layers()):
            if i not in self.steering_layer_ids:
                continue
            for component in self.steering_components:
                hook = self._create_hook(component, layer)
                param = next(layer.parameters(), None)
                if param is not None:
                    hook.to(device=param.device, dtype=param.dtype)
                attr = f"_steering_{component}_{i}"
                layer.add_module(attr, hook)
                target = self._get_hook_target(component, layer)
                if component in _PRE_HOOK_COMPONENTS:
                    target.register_forward_pre_hook(hook.pre_hook)
                else:
                    target.register_forward_hook(hook.post_hook)

    @contextmanager
    def steering_disabled(self):
        """Context manager that temporarily disables all steering hooks."""
        hooks = [m for m in self.modules() if isinstance(m, SteeringHook)]
        for h in hooks:
            h.steering_enabled = False
        try:
            yield
        finally:
            for h in hooks:
                h.steering_enabled = True

    def set_all_vectors(
        self,
        vectors: dict[str, Tensor],
        *,
        normalize: bool = False,
    ) -> None:
        """Apply steering vectors for each component present in *vectors*."""
        expected_components = set(getattr(self, "steering_components", []))
        provided_components = set(vectors)

        missing = sorted(expected_components - provided_components)
        unexpected = sorted(provided_components - expected_components)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing expected components: {missing}")
            if unexpected:
                details.append(f"unexpected components: {unexpected}")
            raise ValueError(
                "Invalid steering vector components; " + "; ".join(details)
            )

        if normalize:
            vectors = {k: F.normalize(v, dim=-1) for k, v in vectors.items()}

        layers = self.get_layers()
        for component, tensor in vectors.items():
            for i, layer in enumerate(layers):
                hook = getattr(layer, f"_steering_{component}_{i}", None)
                if isinstance(hook, SteeringHook):
                    hook.set_steering_vectors(tensor[i])

    # ── Learnable steering ───────────────────────────────────────────

    def _unfreeze_scale(self) -> None:
        """Unfreeze all scale parameters (per-hook or shared)."""
        if self._shared_log_scale is not None:
            self._shared_log_scale.requires_grad = True
        else:
            for module in self.modules():
                if isinstance(module, SteeringHook) and module.log_scale is not None:
                    module.log_scale.requires_grad = True

    def freeze_base_model(self, freeze_log_scale: bool = False) -> None:
        """Freeze everything, then unfreeze learnable steering parameters."""
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, SteeringHook):
                if module.log_alpha is not None:
                    module.log_alpha.requires_grad = True
        if not freeze_log_scale:
            self._unfreeze_scale()

    def freeze_gates(self) -> None:
        """Freeze gate parameters and lock to eval-mode values. Scale remains trainable."""
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, SteeringHook):
                module.freeze_gates()
        self._unfreeze_scale()

    def steering_state_dict(self) -> dict[str, Tensor]:
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
            output_path = path / "steering.pt"
        payload = {
            "scale": self.scale,
            "gate_config": self.gate_config.to_dict() if self.gate_config else None,
            "scale_mode": self.scale_mode,
            "init_log_scale": self.init_log_scale,
            "steering_layer_ids": self.steering_layer_ids,
            "steering_components": self.steering_components,
            "state_dict": self.steering_state_dict(),
        }
        torch.save(payload, output_path)
        return output_path

    def load_steering(
        self,
        path: str | Path,
        *,
        active_layers_only: bool = True,
    ) -> None:
        """Load a saved steering checkpoint and attach hooks."""
        payload = torch.load(Path(path), map_location="cpu")
        state_dict = payload["state_dict"]

        gate_config = None
        if payload.get("gate_config") is not None:
            gate_config = HardConcreteConfig(**payload["gate_config"])

        scale_mode = payload["scale_mode"]
        init_log_scale = payload.get("init_log_scale")
        scale = payload.get("scale", 1.0)

        steering_layer_ids = payload.get("steering_layer_ids")
        if steering_layer_ids is None:
            if active_layers_only and gate_config is not None:
                steering_layer_ids = sorted(
                    gate_config.active_layer_indices(state_dict)
                )
            else:
                steering_layer_ids = list(range(len(self.get_layers())))

        steering_components = payload.get("steering_components", ["attention"])

        self.upgrade_for_steering(
            scale=scale,
            gate_config=gate_config,
            scale_mode=scale_mode,
            init_log_scale=init_log_scale,
            steering_layer_ids=steering_layer_ids,
            steering_components=steering_components,
        )

        load_info = self.load_state_dict(state_dict, strict=False)
        missing = [
            k
            for k in load_info.missing_keys
            if any(
                term in k
                for term in ("log_alpha", "log_scale", "steering_vector")
            )
        ]
        if missing:
            raise RuntimeError(
                f"Steering checkpoint mismatch. Missing steering keys: {missing}."
            )


__all__ = [
    "SteeringLM",
]
