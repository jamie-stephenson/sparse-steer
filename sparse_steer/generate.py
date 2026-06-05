"""Shared autoregressive generation for steering models.

A single ``generate`` used by every task's generative eval. It decodes with the
model's permanent steering hooks and exposes three steering modes plus optional
per-step log-softmax capture, so a task that needs only text and a task that needs
the sampling distributions share one code path.

Steering modes:

- ``"all"``  — steer every forward at every position (the default; e.g. truthfulqa).
- ``"prompt"`` — steer only the step-0 prompt forward, confined to ``steer_prompt_mask``;
  decode steps run with steering disabled. With KV caching the prompt-only steer
  therefore fires on exactly one forward — the steered prompt key/values are cached
  and attended to by later decode steps.
- ``"off"`` — no steering (clean / reference rollouts).

To get RNG-matched rollouts (same uniform draws per step so only the logits differ),
build one sampler per call with the same seed; see :func:`make_sampling_sampler`.
"""

import contextlib
from collections.abc import Callable

import torch
from torch import Tensor
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

from .steering import SteeringModel

Sampler = Callable[[Tensor], Tensor]
"""``fn(logits_last: (B, V)) -> next_token: (B,)``."""


def make_greedy_sampler() -> Sampler:
    return lambda logits: logits.argmax(dim=-1)


def make_sampling_sampler(*, temperature: float, seed: int, device: torch.device | str) -> Sampler:
    """Pure multinomial sampling at ``temperature`` from a per-call seeded generator.

    Two samplers built with the same ``seed`` advance their RNG identically across
    decode steps, so matched-seed rollouts are coupled (same draws → same tokens
    whenever the distributions agree).
    """
    gen = torch.Generator(device=device).manual_seed(int(seed))

    def _sample(logits: Tensor) -> Tensor:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    return _sample


@torch.no_grad()
def generate(
    model: SteeringModel,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    max_new_tokens: int = 16,
    *,
    sampler: Sampler | None = None,
    capture_log_softmax: bool = False,
    steer: str = "all",
    steer_prompt_mask: Tensor | None = None,
    use_kv_cache: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """Decode ``max_new_tokens`` tokens, returning only the newly generated ids.

    ``input_ids`` / ``attention_mask`` are ``(B, P)`` (left-pad for batched decoding).
    With ``capture_log_softmax`` also returns the per-step log-softmax over vocab
    ``(B, max_new_tokens, V)`` — the distribution the sampler drew from each step.
    ``sampler=None`` decodes greedily.
    """
    if sampler is None:
        sampler = make_greedy_sampler()
    if steer == "prompt" and steer_prompt_mask is None:
        raise ValueError("steer='prompt' requires steer_prompt_mask")

    device = model.device
    tl = model.tl
    seq = input_ids.to(device)
    attn = attention_mask.to(device) if attention_mask is not None else None

    kv_cache = None
    if use_kv_cache:
        kv_cache = TransformerLensKeyValueCache.init_cache(tl.cfg, device, seq.shape[0])

    def _step_ctx(step: int):
        if steer == "off":
            return model.steering_disabled()
        if steer == "prompt":
            if step == 0:
                return model.steer_positions(steer_prompt_mask.to(device))
            return model.steering_disabled()
        return contextlib.nullcontext()  # "all"

    new_tokens: list[Tensor] = []
    lsm_steps: list[Tensor] = []
    for step in range(max_new_tokens):
        extra: dict = {"return_type": "logits", "prepend_bos": False}
        if use_kv_cache:
            extra["past_kv_cache"] = kv_cache
            if step == 0:
                inp = seq
                if attn is not None:
                    extra["attention_mask"] = attn
            else:
                inp = seq[:, -1:]
                if attn is not None:
                    extra["attention_mask"] = attn.new_ones(attn.shape[0], 1)
        else:
            inp = seq
            if attn is not None:
                extra["attention_mask"] = attn

        with _step_ctx(step):
            logits = tl(inp, **extra)
        last = logits[:, -1, :]
        if capture_log_softmax:
            lsm_steps.append(torch.log_softmax(last.float(), dim=-1))
        nxt = sampler(last)
        new_tokens.append(nxt.unsqueeze(1))
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        if attn is not None:
            attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], dim=1)

    generated = torch.cat(new_tokens, dim=1)
    if capture_log_softmax:
        return generated, torch.stack(lsm_steps, dim=1)
    return generated


__all__ = ["Sampler", "make_greedy_sampler", "make_sampling_sampler", "generate"]
