"""Shared tiny-model builder for the steering tests: a random-init GPT-Neo (the same family
as the TinyStories checkpoints) wrapped in a SteeringModel — CPU, fp32, seconds to build."""

import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from sparse_steer.core.steering import HardConcreteConfig, SteeringModel

NUM_HEADS = 2
HEAD_DIM = 4
D_MODEL = NUM_HEADS * HEAD_DIM
MLP_DIM = 4 * D_MODEL  # GPT-Neo's implicit intermediate size
NUM_LAYERS = 2
VOCAB = 32


def tiny_engine(seed: int = 0) -> GPTNeoForCausalLM:
    cfg = GPTNeoConfig(
        vocab_size=VOCAB,
        hidden_size=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        attention_types=[[["global"], NUM_LAYERS]],
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    engine = GPTNeoForCausalLM(cfg).eval()
    engine.requires_grad_(False)
    return engine


def tiny_model(
    components,
    *,
    layer_ids=None,
    gate_config: HardConcreteConfig | None = None,
    learn_scale: bool = True,
    init_raw_scale: float = 0.5,
    intervention: str = "steer",
    seed: int = 0,
) -> SteeringModel:
    return SteeringModel(
        tiny_engine(seed),
        steering_layer_ids=list(range(NUM_LAYERS)) if layer_ids is None else layer_ids,
        steering_components=list(components),
        gate_config=gate_config,
        learn_scale=learn_scale,
        init_raw_scale=init_raw_scale,
        intervention=intervention,
    )


def tiny_batch(batch: int = 2, seq: int = 6, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (batch, seq), generator=g)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask
