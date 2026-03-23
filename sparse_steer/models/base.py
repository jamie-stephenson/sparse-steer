from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Component = Literal["attention", "mlp", "residual"]


# ── Steering hooks ───────────────────────────────────────────────────


class SteeringHook(nn.Module):
    """Lightweight module that stores a steering vector and computes a correction.

    Attached to a target module via a forward hook — no module replacement needed.
    Subclasses implement ``_compute_correction`` to define how the vector is applied.
    """

    def __init__(self, vector_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.steering_enabled = True
        self.register_buffer("steering_vectors", torch.zeros(vector_shape))

    @abstractmethod
    def _compute_correction(self, hidden: Tensor) -> Tensor: ...

    def pre_hook(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...] | None:
        """Forward pre-hook for attention/MLP output projections."""
        if not self.steering_enabled or not inputs:
            return None
        hidden = inputs[0]
        correction = self._compute_correction(hidden)
        return (hidden + correction, *inputs[1:])

    def post_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        """Forward hook for residual stream (layer output)."""
        if not self.steering_enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        corrected = hidden + self._compute_correction(hidden)
        if isinstance(output, tuple):
            return (corrected, *output[1:])
        return corrected

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"Steering vectors must have shape {tuple(expected)}, "
                f"got {tuple(vectors.shape)}."
            )
        self.steering_vectors.copy_(
            vectors.to(
                device=self.steering_vectors.device, dtype=self.steering_vectors.dtype
            )
        )


# ── Base steering LM ─────────────────────────────────────────────────

# Hook attachment mode: pre-hook on output proj, or post-hook on layer
_PRE_HOOK_COMPONENTS = frozenset({"attention", "mlp"})
_POST_HOOK_COMPONENTS = frozenset({"residual"})


class BaseSteeringLM:
    """Base mixin that adds steering to a ``PreTrainedModel`` subclass.

    Subclasses must implement layout methods (``get_layers``, ``get_attention``,
    ``get_mlp``) and ``_create_hook`` to build the appropriate SteeringHook.
    """

    @abstractmethod
    def get_layers(self) -> nn.ModuleList: ...

    @abstractmethod
    def get_attention(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def get_mlp(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_vector_shape(
        self, component: Component, layer: nn.Module
    ) -> tuple[int, ...]:
        """Return the shape of the steering vector for the given component."""
        ...

    @abstractmethod
    def _create_hook(self, component: Component, layer: nn.Module) -> SteeringHook:
        """Create a SteeringHook for the given component and layer."""
        ...

    def _get_hook_target(self, component: Component, layer: nn.Module) -> nn.Module:
        """Return the module to attach the hook to."""
        if component == "attention":
            return self._get_output_proj(self.get_attention(layer))
        elif component == "mlp":
            return self._get_output_proj(self.get_mlp(layer))
        elif component == "residual":
            return layer
        raise ValueError(f"Unknown component: {component!r}")

    @abstractmethod
    def _get_output_proj(self, module: nn.Module) -> nn.Module:
        """Return the output projection submodule (e.g. o_proj, down_proj)."""
        ...

    def _validate_components(self, components: list[Component]) -> None:
        if "residual" in components and len(components) > 1:
            raise ValueError(
                "Residual steering is mutually exclusive with attention/mlp steering. "
                f"Got: {components}"
            )

    def _attach_steering_hooks(self) -> None:
        self._validate_components(self.steering_components)
        for i, layer in enumerate(self.get_layers()):
            if i not in self.steering_layer_ids:
                continue
            for component in self.steering_components:
                hook = self._create_hook(component, layer)
                # move to same device/dtype as the layer
                param = next(layer.parameters(), None)
                if param is not None:
                    hook.to(device=param.device, dtype=param.dtype)
                # store as submodule so it's in the state dict
                attr = f"_steering_{component}_{i}"
                layer.add_module(attr, hook)
                # register the hook
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


__all__ = [
    "Component",
    "SteeringHook",
    "BaseSteeringLM",
]
