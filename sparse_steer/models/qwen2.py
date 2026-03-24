from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .llama import LlamaLayout
from .steering import SteeringLM


class Qwen2SteeringLM(LlamaLayout, SteeringLM, Qwen2ForCausalLM):
    pass


__all__ = [
    "Qwen2SteeringLM",
]
