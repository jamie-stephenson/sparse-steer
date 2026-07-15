"""Model construction: load the HF model and (optionally) attach steering.

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

# Llama-2 BASE ships no chat template. I think SafeSteer steers the base in a RAW completion framing — it
# evaluates "naive completion ... no system prompt or explicit instruction" in "the hardest setting"
# (paper §4.2/§7), so the base's "template" is just BOS + the bare prompt and the model continues it
# (NO Alpaca / instruction scaffold). The chat model's native [INST] template is used separately for
# extraction. (generate_text tokenises with add_special_tokens=False, so the BOS must be in here.)
_LLAMA2_BASE_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
)


def resolve_dtype(config: DictConfig) -> torch.dtype:
    """Base-model dtype — controls the weights/activations of the loaded model."""
    return _DTYPES[config.get("model_dtype", "float16")]


def resolve_steering_dtype(config: DictConfig) -> torch.dtype:
    """Steering-math dtype — controls every steering vector/param (gate, scale, direction) and
    the correction compute, in all methods. float32 (default) = stable; independent of the
    base ``model_dtype`` (the correction is cast to the activation dtype at apply time)."""
    return _DTYPES[config.get("steering_dtype", "float32")]


def load_tokenizer(config: DictConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer, accepting custom code (Qwen-1.0's tiktoken QWenTokenizer) and
    patching in a chat template when the model ships none."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    name = config.model_name.lower()  # template/pad patches key on the model family name
    if tokenizer.chat_template is None and "qwen" in name:
        tokenizer.chat_template = _ARDITI_QWEN_CHAT_TEMPLATE
        # QWenTokenizer ships with no pad/eos at all; pad as Arditi does (eod = <|endoftext|>).
        tokenizer.pad_token = "<|extra_0|>"
        tokenizer.pad_token_id = tokenizer.eod_id
        # ...and no eos either. Generation must stop at the assistant turn-end <|im_end|>.
        # Without it the (stop-less) TL decode loop runs past the reply into off-distribution
        # text.
        tokenizer.eos_token = "<|im_end|>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    elif tokenizer.chat_template is None and ("llama-2" in name or "llama2" in name):
        tokenizer.chat_template = _LLAMA2_BASE_TEMPLATE
    if tokenizer.pad_token_id is None:
        # Llama tokenizers ship no pad token; pad with eos (extraction tokenises with padding).
        tokenizer.pad_token = tokenizer.eos_token
    _sync_bos(tokenizer)
    return tokenizer


def _sync_bos(tokenizer: PreTrainedTokenizerBase) -> None:
    """Make every tokenised sequence carry exactly ONE BOS.

    All tokenisation in the codebase runs with ``add_special_tokens=True``, so the tokenizer's
    ``add_bos_token`` is the switch that must agree with whether the chat template *already*
    renders a BOS into the text (e.g. Llama-2's ``{{ bos_token }}``). If they disagree the BOS is
    either doubled (template emits it AND the tokenizer prepends another) or dropped (neither).
    Set ``add_bos_token`` to the complement of "template already emits a BOS"; with no chat
    template (raw-text inputs, e.g. sleeper/tinystories) leave the tokenizer's own default so its BOS is
    still added. (A loaded fast tokenizer also needs this re-assignment to actually honour the
    value — its post-processor otherwise ignores the configured ``add_bos_token`` until set.)"""
    if getattr(tokenizer, "add_bos_token", None) is None or not tokenizer.bos_token:
        return  # tokenizer has no BOS concept (e.g. Qwen-1.0) → nothing to dedupe
    if not tokenizer.chat_template:
        return  # raw-text usage → keep the tokenizer's own BOS behaviour
    probe = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    tokenizer.add_bos_token = not probe.lstrip().startswith(tokenizer.bos_token)


def load_steering_model(config: DictConfig) -> SteeringModel:
    """Load the model and attach steering hooks."""
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
        steering_dtype=resolve_steering_dtype(config),
        lora_adapter=config.get("lora_adapter"),
        steering_layer_ids=layer_ids,
        steering_components=list(config.targets),
        gate_config=gate_config,
        learn_scale=learn_scale,
        shared_scale=shared_scale,
        init_raw_scale=init_raw_scale,
        intervention=intervention,
    )


def _load_bare(
    config: DictConfig,
    model_name: str,
    lora_adapter: str | None,
) -> SteeringModel:
    """Hook-free SteeringModel (no steering sites) — the shared build for the unsteered
    control and the cross-model-transfer extraction model."""
    return SteeringModel.from_pretrained(
        model_name,
        device=config.device,
        dtype=resolve_dtype(config),
        lora_adapter=lora_adapter,
        steering_layer_ids=[],
        steering_components=[],
    )


def load_plain_model(config: DictConfig) -> SteeringModel:
    """Load the model with no steering (used for the unsteered control)."""
    return _load_bare(config, config.model_name, config.get("lora_adapter"))


def load_extraction_model(
    config: DictConfig, steered_model: SteeringModel
) -> tuple[SteeringModel, PreTrainedTokenizerBase]:
    """Cross-model transfer: load a separate read-only model under ``config.extraction_model_name``
    to read activations from, then steer ``config.model_name``. Returns ``(model, tokenizer)``.

    The two checkpoints must be architecturally identical (``n_layers``/``d_model``/``n_heads``) so
    the per-site direction transfers — within-family only (e.g. Llama-2-7B base ↔ chat); raises
    otherwise. The caller is responsible for freeing the returned model after extraction."""
    name = config.extraction_model_name
    print(f"Loading extraction model (cross-model transfer): {name}")
    ext = _load_bare(config, name, lora_adapter=None)
    a, b = ext.cfg, steered_model.cfg
    for attr in ("n_layers", "d_model", "n_heads"):
        if getattr(a, attr, None) != getattr(b, attr, None):
            raise ValueError(
                f"extraction_model_name={name!r} is architecturally incompatible with "
                f"model_name={config.model_name!r}: {attr} {getattr(a, attr, None)} != "
                f"{getattr(b, attr, None)}. Cross-model transfer requires the same architecture "
                "(within-family only, e.g. Llama-2-7B base ↔ chat)."
            )
    ext_tokenizer = load_tokenizer(OmegaConf.merge(config, {"model_name": name}))
    return ext, ext_tokenizer


__all__ = [
    "load_extraction_model",
    "load_plain_model",
    "load_steering_model",
    "load_tokenizer",
    "resolve_dtype",
    "resolve_steering_dtype",
]
