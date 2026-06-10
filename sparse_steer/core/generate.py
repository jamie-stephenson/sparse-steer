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
from collections.abc import Callable, Collection

import torch
from torch import Tensor
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

from .steering import SteeringModel

Sampler = Callable[[Tensor], Tensor]
"""``fn(logits_last: (B, V)) -> next_token: (B,)``."""


def make_greedy_sampler() -> Sampler:
    return lambda logits: logits.argmax(dim=-1)


def make_sampling_sampler(
    *, temperature: float, seed: int | None = None, device: torch.device | str = "cpu"
) -> Sampler:
    """Multinomial sampling at ``temperature``.

    With ``seed`` given, draws from a per-call seeded generator — two samplers built with the same
    seed advance identically, so paired rollouts are coupled (matched-RNG). With ``seed=None`` (the
    default) it draws from the **global** RNG; seeding happens once in ``_seed_everything``, so don't
    reseed here.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None

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
    eos_token_ids: Collection[int] | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Decode ``max_new_tokens`` tokens, returning only the newly generated ids.

    ``input_ids`` / ``attention_mask`` are ``(B, P)`` (left-pad for batched decoding).
    With ``capture_log_softmax`` also returns the per-step log-softmax over vocab
    ``(B, max_new_tokens, V)`` — the distribution the sampler drew from each step.
    ``sampler=None`` decodes greedily.

    ``eos_token_ids`` (when given and not capturing log-softmax) stops the loop once
    every sequence has emitted an end token — the loop has no native stopping, so
    without this it always runs the full ``max_new_tokens``, decoding off-distribution
    text *past* the model's turn-end (which the caller must then truncate at decode).
    """
    if sampler is None:
        sampler = make_greedy_sampler()
    eos = (
        torch.tensor(sorted(eos_token_ids), device=model.device)
        if eos_token_ids and not capture_log_softmax
        else None
    )
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
    finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
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
        if eos is not None:
            finished |= (nxt.unsqueeze(-1) == eos).any(dim=-1)
            if bool(finished.all()):
                break

    generated = torch.cat(new_tokens, dim=1)
    if capture_log_softmax:
        return generated, torch.stack(lsm_steps, dim=1)
    return generated


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    *,
    temperature: float = 0.0,
    steer: str = "all",
    template: bool = True,
    batch_size: int = 16,
) -> list[str]:
    """Batched string prompts → decoded responses, for **either** a ``SteeringModel`` (the
    TransformerLens decode loop, with steering modes) **or** a plain HF model (e.g. a LoRA/peft
    model, via ``model.generate``) — dispatched on ``hasattr(model, "tl")``. The single
    model-agnostic generation seam every generative eval (and the Inspect provider) shares.

    Greedy when ``temperature <= 0``, else multinomial from the **global** RNG (seeded once in
    ``_seed_everything`` — no per-call seed). Prompts are chat-templated here (``template=True``) or
    assumed already-templated (``template=False``); either way tokenised with
    ``add_special_tokens=False`` since the chat template carries its own special tokens. ``steer``
    modes apply only to a ``SteeringModel`` (an HF model's intervention — e.g. a LoRA adapter — is
    always on).
    """
    from sparse_steer.utils.tokenize import apply_template, tokenize

    is_tl = hasattr(model, "tl")
    greedy = temperature <= 0
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop/strip at the model's end token: the TL decode loop has no native EOS stop, so
    # without this it emits the full max_new_tokens — running past the turn-end into
    # off-distribution text that ``skip_special_tokens`` would otherwise keep.
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    def _strip_after_eos(ids: list[int]) -> list[int]:
        for i, t in enumerate(ids):
            if t in eos_ids:
                return ids[:i]
        return ids

    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        texts = [apply_template(tokenizer, p) for p in batch] if template else list(batch)
        # left-pad for *this call only* (batched decoding needs the last real token at
        # [:, -1]); per-call padding_side leaves the shared tokenizer's default intact.
        enc = tokenize(tokenizer, texts, add_special_tokens=False, padding_side="left").to(device)
        if is_tl:
            sampler = (
                make_greedy_sampler()
                if greedy
                else make_sampling_sampler(temperature=temperature, device=device)
            )
            new_toks = generate(
                model, enc["input_ids"], enc["attention_mask"], max_new_tokens,
                sampler=sampler, steer=steer,
                steer_prompt_mask=enc["attention_mask"].bool() if steer == "prompt" else None,
                eos_token_ids=eos_ids or None,
            )
        else:  # plain HF model (e.g. LoRA/peft): native generate, adapter always on
            gen = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=not greedy,
                temperature=None if greedy else temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_toks = gen[:, enc["input_ids"].shape[1] :]
        out.extend(
            tokenizer.decode(_strip_after_eos(t.tolist()), skip_special_tokens=True).strip()
            for t in new_toks
        )
    return out


__all__ = ["Sampler", "make_greedy_sampler", "make_sampling_sampler", "generate", "generate_text"]
