from typing import Any

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2MLP

from ..hardconcrete import HardConcreteConfig
from .sparse import SparseSteeringAttention, SparseSteeringMLP, SparseSteeringLM
from .dense import DenseSteeringAttention, DenseSteeringMLP, DenseSteeringLM

# Qwen2 exposes the same attribute names as Llama (o_proj, down_proj,
# head_dim, config.num_attention_heads, config.intermediate_size,
# model.layers, layer.self_attn, layer.mlp), so we reuse the Llama layout
# mixins directly.  The HF base classes themselves differ: Qwen2 uses
# grouped-query attention with separate num_key_value_heads, SiLU-gated
# MLPs with an explicit gate_proj, and its own RoPE implementation with
# different defaults — but none of that affects the layout interface.
from .llama import LlamaAttentionLayout, LlamaMLPLayout, LlamaLMLayout


# ── Sparse + Qwen2 ──────────────────────────────────────────────────


class Qwen2SparseSteeringAttention(LlamaAttentionLayout, SparseSteeringAttention, Qwen2Attention):
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


class Qwen2SparseSteeringMLP(LlamaMLPLayout, SparseSteeringMLP, Qwen2MLP):
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


class Qwen2SparseSteeringLM(LlamaLMLayout, SparseSteeringLM, Qwen2ForCausalLM):
    attn_cls = Qwen2SparseSteeringAttention
    mlp_cls = Qwen2SparseSteeringMLP


# ── Dense + Qwen2 ───────────────────────────────────────────────────


class Qwen2DenseSteeringAttention(LlamaAttentionLayout, DenseSteeringAttention, Qwen2Attention):
    def __init__(
        self, config: Any, *args: Any, steering_strength: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(config, *args, steering_strength=steering_strength, **kwargs)
        self._register_steering_hook()


class Qwen2DenseSteeringMLP(LlamaMLPLayout, DenseSteeringMLP, Qwen2MLP):
    def __init__(
        self, config: Any, *args: Any, steering_strength: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(config, *args, steering_strength=steering_strength, **kwargs)
        self._register_steering_hook()


class Qwen2DenseSteeringLM(LlamaLMLayout, DenseSteeringLM, Qwen2ForCausalLM):
    attn_cls = Qwen2DenseSteeringAttention
    mlp_cls = Qwen2DenseSteeringMLP


__all__ = [
    "Qwen2SparseSteeringAttention",
    "Qwen2SparseSteeringMLP",
    "Qwen2SparseSteeringLM",
    "Qwen2DenseSteeringAttention",
    "Qwen2DenseSteeringMLP",
    "Qwen2DenseSteeringLM",
]
