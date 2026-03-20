from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .sparse import SparseSteeringLM
from .dense import DenseSteeringLM

# Qwen2 exposes the same attribute names as Llama (o_proj, down_proj,
# head_dim, config.num_attention_heads, config.intermediate_size,
# model.layers, layer.self_attn, layer.mlp), so we reuse the Llama
# layout mixin directly.
from .llama import LlamaLayout


class Qwen2SparseSteeringLM(LlamaLayout, SparseSteeringLM, Qwen2ForCausalLM):
    pass


class Qwen2DenseSteeringLM(LlamaLayout, DenseSteeringLM, Qwen2ForCausalLM):
    pass


__all__ = [
    "Qwen2SparseSteeringLM",
    "Qwen2DenseSteeringLM",
]
