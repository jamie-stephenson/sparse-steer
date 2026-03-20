from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .base import Component
from .sparse import SparseSteeringLM
from .dense import DenseSteeringLM


# ── Layout mixin ─────────────────────────────────────────────────────
# Maps base.py's abstract interface to HuggingFace attribute names.
# Llama and Qwen2 expose identical attribute names (o_proj, down_proj,
# head_dim, model.layers, etc.) so this mixin is shared, but the
# underlying HF base classes differ.


class LlamaLayout:
    def get_layers(self) -> nn.ModuleList:
        return self.model.layers

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    def _get_output_proj(self, module: nn.Module) -> nn.Module:
        if hasattr(module, "o_proj"):
            return module.o_proj
        if hasattr(module, "down_proj"):
            return module.down_proj
        raise AttributeError(
            f"Cannot find output projection on {type(module).__name__}. "
            "Expected 'o_proj' (attention) or 'down_proj' (MLP)."
        )

    def _get_vector_shape(self, component: Component, layer: nn.Module) -> tuple[int, ...]:
        if component == "attention":
            attn = self.get_attention(layer)
            return (attn.config.num_attention_heads, attn.head_dim)
        elif component == "mlp":
            mlp = self.get_mlp(layer)
            return (mlp.config.intermediate_size,)
        elif component == "residual":
            return (self.config.hidden_size,)
        raise ValueError(f"Unknown component: {component!r}")


# ── Concrete LM classes ──────────────────────────────────────────────


class LlamaSparseSteeringLM(LlamaLayout, SparseSteeringLM, LlamaForCausalLM):
    pass


class LlamaDenseSteeringLM(LlamaLayout, DenseSteeringLM, LlamaForCausalLM):
    pass


__all__ = [
    "LlamaLayout",
    "LlamaSparseSteeringLM",
    "LlamaDenseSteeringLM",
]
