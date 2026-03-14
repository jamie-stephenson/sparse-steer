from typing import Any

from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP

from ..hardconcrete import HardConcreteConfig
from .sparse import SparseSteeringAttention, SparseSteeringMLP, SparseSteeringLM
from .dense import DenseSteeringAttention, DenseSteeringMLP, DenseSteeringLM


# ── Shared layout mixins ──────────────────────────────────────────────
# These map base.py's abstract interface to HuggingFace attribute names.
# Llama and Qwen2 expose identical attribute names (o_proj, down_proj,
# head_dim, model.layers, etc.) so these mixins are shared, but the
# underlying HF base classes differ: Qwen2 uses grouped-query attention
# with separate num_key_value_heads, SiLU-gated MLPs with an explicit
# gate_proj, and its own RoPE implementation with different defaults.


class LlamaAttentionLayout:
    @property
    def _num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def _head_dim(self) -> int:
        return self.head_dim

    def output_proj(self) -> nn.Module:
        return self.o_proj


class LlamaMLPLayout:
    @property
    def _mlp_dim(self) -> int:
        return self.config.intermediate_size

    def output_proj(self) -> nn.Module:
        return self.down_proj


class LlamaLMLayout:
    def get_layers(self) -> nn.ModuleList:
        return self.model.layers

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp


# ── Sparse + Llama ───────────────────────────────────────────────────


class LlamaSparseSteeringAttention(LlamaAttentionLayout, SparseSteeringAttention, LlamaAttention):
    def __init__(
        self, config: Any, *args: Any, gate_config: HardConcreteConfig, **kwargs: Any
    ) -> None:
        super().__init__(
            config, *args,
            num_gates=config.num_attention_heads,
            gate_config=gate_config,
            **kwargs,
        )
        self._register_steering_hook()


class LlamaSparseSteeringMLP(LlamaMLPLayout, SparseSteeringMLP, LlamaMLP):
    def __init__(
        self, config: Any, *args: Any, gate_config: HardConcreteConfig, **kwargs: Any
    ) -> None:
        super().__init__(
            config, *args,
            num_gates=config.intermediate_size,
            gate_config=gate_config,
            **kwargs,
        )
        self._register_steering_hook()


class LlamaSparseSteeringLM(LlamaLMLayout, SparseSteeringLM, LlamaForCausalLM):
    attn_cls = LlamaSparseSteeringAttention
    mlp_cls = LlamaSparseSteeringMLP


# ── Dense + Llama ────────────────────────────────────────────────────


class LlamaDenseSteeringAttention(LlamaAttentionLayout, DenseSteeringAttention, LlamaAttention):
    def __init__(
        self, config: Any, *args: Any, steering_strength: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(config, *args, steering_strength=steering_strength, **kwargs)
        self._register_steering_hook()


class LlamaDenseSteeringMLP(LlamaMLPLayout, DenseSteeringMLP, LlamaMLP):
    def __init__(
        self, config: Any, *args: Any, steering_strength: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(config, *args, steering_strength=steering_strength, **kwargs)
        self._register_steering_hook()


class LlamaDenseSteeringLM(LlamaLMLayout, DenseSteeringLM, LlamaForCausalLM):
    attn_cls = LlamaDenseSteeringAttention
    mlp_cls = LlamaDenseSteeringMLP


__all__ = [
    "LlamaAttentionLayout",
    "LlamaMLPLayout",
    "LlamaLMLayout",
    "LlamaSparseSteeringAttention",
    "LlamaSparseSteeringMLP",
    "LlamaSparseSteeringLM",
    "LlamaDenseSteeringAttention",
    "LlamaDenseSteeringMLP",
    "LlamaDenseSteeringLM",
]
