from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ..utils.hardconcrete import HardConcreteConfig, HardConcreteGateMixin

Component = Literal["attention", "mlp"]


# ── Sparse steering attention ─────────────────────────────────────────


class SparseSteeringAttention(HardConcreteGateMixin):
    """Base class for sparse-steered attention modules.

    Subclasses must implement ``_num_heads``, ``_head_dim``, and
    ``output_proj``, and call ``_register_steering_hook`` at the end of
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


# ── Sparse steering MLP ───────────────────────────────────────────────


class SparseSteeringMLP:
    steering_enabled: bool

    @abstractmethod
    def output_proj(self) -> nn.Module: ...

    @abstractmethod
    def set_steering_vectors(self, vectors: Tensor) -> None: ...


# ── Generic module upgrade ────────────────────────────────────────────


def _copy_module_state(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters and buffers from *source* into *target*."""
    source_param = next(source.parameters(), None)
    if source_param is not None:
        target.to(device=source_param.device, dtype=source_param.dtype)
    target.load_state_dict(source.state_dict(), strict=False)


def _upgrade_module(
    source: nn.Module,
    replacement_cls: type,
    gate_config: HardConcreteConfig,
) -> nn.Module:
    """Replace *source* with an instance of *replacement_cls*, copying weights.

    If *source* is already the right type, just update its gate config.
    Automatically forwards ``layer_idx`` from the source when present.
    """
    if isinstance(source, replacement_cls):
        source._set_gate_config(gate_config)
        return source

    config = getattr(source, "config", None)
    kwargs: dict[str, Any] = {}
    layer_idx = getattr(source, "layer_idx", None)
    if layer_idx is not None:
        kwargs["layer_idx"] = layer_idx
    upgraded = replacement_cls(config, gate_config=gate_config, **kwargs)
    _copy_module_state(source, upgraded)
    return upgraded


def _replace_child(parent: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Replace *old* with *new* as a direct child of *parent*."""
    for name, child in parent.named_children():
        if child is old:
            setattr(parent, name, new)
            return
    raise ValueError("Module not found as a direct child of parent")



# ── Sparse steering mixin ────────────────────────────────────────────

class SparseSteeringLM(ABC):
    """Mixin that adds sparse-steering to a ``PreTrainedModel`` subclass.

    Concrete classes inherit from both this mixin and a HuggingFace model::

        class LlamaSparseSteeringLM(SparseSteeringLM, LlamaForCausalLM):
            attn_cls = GatedAttention
            ...

    The resulting object IS the HF model, so ``Trainer``, ``.generate()``,
    and ``.forward()`` all work directly.  Load weights with HF's
    ``from_pretrained``, then call ``upgrade_for_steering`` to swap in
    gated modules.
    """

    attn_cls: type
    mlp_cls: type

    # ── abstract accessors (subclass must define) ─────────────────

    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """Return the decoder layer sequence."""
        ...

    @abstractmethod
    def get_attention(self, layer: nn.Module) -> nn.Module:
        """Return the attention submodule of *layer* (upgrade target)."""
        ...

    @abstractmethod
    def get_mlp(self, layer: nn.Module) -> nn.Module:
        """Return the MLP submodule of *layer* (upgrade target)."""
        ...

    # ── steering setup ────────────────────────────────────────────

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
                upgraded = _upgrade_module(source, replacement_cls, self.gate_config)
                if upgraded is not source:
                    _replace_child(layer, source, upgraded)


    def freeze_base_model(self) -> None:
        """Freeze everything, then unfreeze HardConcrete gate parameters."""
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, HardConcreteGateMixin):
                log_alpha = getattr(module, "log_alpha", None)
                if log_alpha is not None:
                    log_alpha.requires_grad = True
                log_scale = getattr(module, "log_scale", None)
                if log_scale is not None:
                    log_scale.requires_grad = True


    @contextmanager
    def steering_disabled(self):
        """Context manager that temporarily disables all steering hooks."""
        steered = [
            m for m in self.modules()
            if isinstance(m, (SparseSteeringAttention, SparseSteeringMLP))
        ]
        for m in steered:
            m.steering_enabled = False
        try:
            yield
        finally:
            for m in steered:
                m.steering_enabled = True

    def set_all_steering(self, vectors: dict[str, Tensor]) -> None:
        """Apply steering vectors for each component present in *vectors*."""
        layers = self.get_layers()
        for component, tensor in vectors.items():
            if component == "attention":
                for i, layer in enumerate(layers):
                    module = self.get_attention(layer)
                    if isinstance(module, SparseSteeringAttention):
                        module.set_steering_vectors(tensor[i])
            elif component == "mlp":
                for i, layer in enumerate(layers):
                    module = self.get_mlp(layer)
                    if isinstance(module, SparseSteeringMLP):
                        module.set_steering_vectors(tensor[i])
            else:
                raise ValueError(f"Unknown steering component: {component!r}")

    def sparse_steering_state_dict(self) -> dict[str, Tensor]:
        steering_terms = ("log_alpha", "log_scale", "steering_vectors")
        return {
            k: v
            for k, v in self.state_dict().items()
            if any(term in k for term in steering_terms)
        }

    def save_sparse_steering(self, path: str | Path) -> Path:
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

    def load_sparse_steering(
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
    "Component",
    "SparseSteeringAttention",
    "SparseSteeringMLP",
    "SparseSteeringLM",
]
