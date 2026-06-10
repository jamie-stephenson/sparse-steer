"""Model construction: load a TransformerLens model and (optionally) attach steering.

Task-agnostic, so it lives in ``core`` rather than ``experiment`` — both the experiment
layer and tasks (e.g. jailbreak self-loading an unsteered model for bucketing) depend on
it, and putting it here keeps the dependency graph acyclic."""

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .steering import HardConcreteConfig, SteeringModel

_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Qwen-1.0 (e.g. Qwen/Qwen-7B-Chat) ships no chat_template. Arditi et al. hardcode
# ChatML with no system prompt: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
_ARDITI_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def resolve_dtype(config: DictConfig) -> torch.dtype:
    return _DTYPES[config.get("dtype", "float16")]


def load_tokenizer(config: DictConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer, accepting custom code (Qwen-1.0's tiktoken QWenTokenizer)
    and patching in Arditi's ChatML template when the model ships none."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.chat_template is None and "qwen" in config.model_name.lower():
        tokenizer.chat_template = _ARDITI_QWEN_CHAT_TEMPLATE
        # QWenTokenizer ships with no pad/eos at all; pad as Arditi does (eod = <|endoftext|>).
        tokenizer.pad_token = "<|extra_0|>"
        tokenizer.pad_token_id = tokenizer.eod_id
        # ...and no eos either. Generation must stop at the assistant turn-end <|im_end|>;
        # without it the (stop-less) TL decode loop runs past the reply into off-distribution
        # text, which deflates generation-based evals (e.g. Llama Guard safety_score).
        tokenizer.eos_token = "<|im_end|>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tokenizer


def load_steering_model(config: DictConfig) -> SteeringModel:
    """Load a TransformerLens model and attach steering hooks."""
    gate_config = None
    if config.get("gate_config") is not None:
        gate_cfg = OmegaConf.to_container(config.gate_config, resolve=True)
        gate_config = HardConcreteConfig(**gate_cfg)

    layer_ids = (
        list(config.steering_layer_ids)
        if config.get("steering_layer_ids") is not None
        else None
    )
    learn_scale = config.get("learn_scale", False)
    shared_scale = config.get("shared_scale", False)
    init_raw_scale = config.get("init_raw_scale", 0.0)
    intervention = config.get("intervention", "steer")

    desc = [intervention]
    if gate_config is not None:
        desc.append("gates")
    if shared_scale:
        desc.append("shared learned scale")
    elif learn_scale:
        desc.append("learned scale")
    else:
        desc.append(f"scale=softplus({init_raw_scale:.2f})")
    print(f"Initialising steering model ({', '.join(desc)})...")

    return SteeringModel.from_pretrained(
        config.model_name,
        device=config.device,
        dtype=resolve_dtype(config),
        lora_adapter=config.get("lora_adapter"),
        steering_layer_ids=layer_ids,
        steering_components=list(config.targets),
        gate_config=gate_config,
        learn_scale=learn_scale,
        shared_scale=shared_scale,
        init_raw_scale=init_raw_scale,
        intervention=intervention,
    )


def load_plain_model(config: DictConfig) -> SteeringModel:
    """Load a TransformerLens model with no steering (used for the unsteered control)."""
    return SteeringModel.from_pretrained(
        config.model_name,
        device=config.device,
        dtype=resolve_dtype(config),
        lora_adapter=config.get("lora_adapter"),
        steering_layer_ids=[],
        steering_components=[],
    )


__all__ = [
    "load_plain_model",
    "load_steering_model",
    "load_tokenizer",
    "resolve_dtype",
]
