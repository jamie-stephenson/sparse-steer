from typing import Any

from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP

from ..utils.hardconcrete import HardConcreteConfig
from .base import SparseSteeringAttention, SparseSteeringMLP, SparseSteeringLM


class LlamaSparseSteeringAttention(SparseSteeringAttention, LlamaAttention):

    @property
    def _num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def _head_dim(self) -> int:
        return self.head_dim

    def output_proj(self) -> nn.Module:
        return self.o_proj

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


class LlamaSparseSteeringMLP(SparseSteeringMLP, LlamaMLP):
    def output_proj(self) -> nn.Module:
        return self.down_proj


class LlamaSparseSteeringLM(SparseSteeringLM, LlamaForCausalLM):
    attn_cls = LlamaSparseSteeringAttention
    mlp_cls = LlamaSparseSteeringMLP

    def get_layers(self) -> nn.ModuleList:
        return self.model.layers

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp


__all__ = [
    "LlamaSparseSteeringAttention",
    "LlamaSparseSteeringLM",
]
