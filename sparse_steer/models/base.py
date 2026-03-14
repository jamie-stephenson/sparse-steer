from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Literal

import torch
from torch import Tensor, nn

Component = Literal["attention", "mlp"]


# ── Base steering modules ────────────────────────────────────────────


class BaseSteeringAttention:
    """Abstract base for steered attention modules.

    Subclasses must implement ``_num_heads``, ``_head_dim``, ``output_proj``,
    and ``_apply_steering``, and call ``_register_steering_hook`` at the end of
    their ``__init__``.
    """

    @property
    @abstractmethod
    def _num_heads(self) -> int: ...

    @property
    @abstractmethod
    def _head_dim(self) -> int: ...

    @abstractmethod
    def output_proj(self) -> nn.Module: ...

    @abstractmethod
    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any: ...

    def _register_steering_hook(self) -> None:
        self.steering_enabled = True
        self.register_buffer(
            "steering_vectors",
            torch.zeros(self._num_heads, self._head_dim),
            persistent=True,
        )
        self._hook_handle = self.output_proj().register_forward_pre_hook(self._apply_steering)

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"Steering vectors must have shape {tuple(expected)}, "
                f"got {tuple(vectors.shape)}."
            )
        self.steering_vectors.copy_(
            vectors.to(device=self.steering_vectors.device, dtype=self.steering_vectors.dtype)
        )


class BaseSteeringMLP:
    """Abstract base for steered MLP modules.

    Subclasses must implement ``_mlp_dim``, ``output_proj``, and
    ``_apply_steering``, and call ``_register_steering_hook`` at the end of
    their ``__init__``.
    """

    @property
    @abstractmethod
    def _mlp_dim(self) -> int: ...

    @abstractmethod
    def output_proj(self) -> nn.Module: ...

    @abstractmethod
    def _apply_steering(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> Any: ...

    def _register_steering_hook(self) -> None:
        self.steering_enabled = True
        self.register_buffer(
            "steering_vectors",
            torch.zeros(self._mlp_dim),
            persistent=True,
        )
        self._hook_handle = self.output_proj().register_forward_pre_hook(self._apply_steering)

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"Steering vectors must have shape {tuple(expected)}, "
                f"got {tuple(vectors.shape)}."
            )
        self.steering_vectors.copy_(
            vectors.to(device=self.steering_vectors.device, dtype=self.steering_vectors.dtype)
        )


# ── Module upgrade helpers ────────────────────────────────────────────


def _copy_module_state(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters and buffers from *source* into *target*."""
    source_param = next(source.parameters(), None)
    if source_param is not None:
        target.to(device=source_param.device, dtype=source_param.dtype)
    target.load_state_dict(source.state_dict(), strict=False)


def _replace_child(parent: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Replace *old* with *new* as a direct child of *parent*."""
    for name, child in parent.named_children():
        if child is old:
            setattr(parent, name, new)
            return
    raise ValueError("Module not found as a direct child of parent")


# ── Base steering LM ─────────────────────────────────────────────────


class BaseSteeringLM:
    """Base mixin that adds steering to a ``PreTrainedModel`` subclass.

    Concrete classes inherit from both a steering LM mixin and a HuggingFace
    model.  Subclasses must implement ``get_layers``, ``get_attention``,
    ``get_mlp``, and ``_upgrade_single_module``.
    """

    attn_cls: type
    mlp_cls: type

    @abstractmethod
    def get_layers(self) -> nn.ModuleList: ...

    @abstractmethod
    def get_attention(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def get_mlp(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _upgrade_single_module(
        self, source: nn.Module, replacement_cls: type
    ) -> nn.Module: ...

    def _upgrade_modules(self) -> None:
        for i, layer in enumerate(self.get_layers()):
            if i not in self.steering_layer_ids:
                continue
            for component in self.steering_components:
                if component == "attention":
                    source = self.get_attention(layer)
                    replacement_cls = self.attn_cls
                elif component == "mlp":
                    source = self.get_mlp(layer)
                    replacement_cls = self.mlp_cls
                else:
                    raise ValueError(
                        f"Unsupported steering component: {component!r}. "
                        "Supported: 'attention', 'mlp'."
                    )
                upgraded = self._upgrade_single_module(source, replacement_cls)
                if upgraded is not source:
                    _replace_child(layer, source, upgraded)

    @contextmanager
    def steering_disabled(self):
        """Context manager that temporarily disables all steering hooks."""
        steered = [
            m for m in self.modules()
            if isinstance(m, (BaseSteeringAttention, BaseSteeringMLP))
        ]
        for m in steered:
            m.steering_enabled = False
        try:
            yield
        finally:
            for m in steered:
                m.steering_enabled = True

    def set_all_vectors(self, vectors: dict[str, Tensor]) -> None:
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

        layers = self.get_layers()
        for component, tensor in vectors.items():
            if component == "attention":
                for i, layer in enumerate(layers):
                    module = self.get_attention(layer)
                    if isinstance(module, BaseSteeringAttention):
                        module.set_steering_vectors(tensor[i])
            elif component == "mlp":
                for i, layer in enumerate(layers):
                    module = self.get_mlp(layer)
                    if isinstance(module, BaseSteeringMLP):
                        module.set_steering_vectors(tensor[i])
            else:
                raise ValueError(f"Unknown steering component: {component!r}")


__all__ = [
    "Component",
    "BaseSteeringAttention",
    "BaseSteeringMLP",
    "BaseSteeringLM",
    "_copy_module_state",
    "_replace_child",
]
