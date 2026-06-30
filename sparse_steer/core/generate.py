"""Shared autoregressive generation for steering models.

A single ``generate`` used by every task's generative eval. It decodes with the
model's permanent steering hooks and exposes named steering position modes plus
optional per-step log-softmax capture, so a task that needs only text and a task
that needs the sampling distributions share one code path.

Steering modes (the shared token-position vocabulary; pf = final prompt token):

- ``"prompt"``       — every input token ``0..pf``. KV-gen: steer the step-0 prompt forward only
  (its steered K/V are cached and attended to by every later token); decode runs unsteered.
- ``"prompt_final"`` — only ``pf`` (the last input token). KV-gen: step-0 forward, that one position.
- ``"completion"``   — every generated token (``pos > pf``), excluding EOS. KV-gen: each decode step
  steers its new token (skipping a generated EOS); the step-0 prompt is NOT steered.
- ``"all"``          — every real non-EOS token (``= prompt ∪ completion``).
- ``"off"``          — no steering (clean / reference rollouts).

Generation always uses the TransformerLens KV cache (step 0 forwards the full prompt;
each later step forwards only the new token and attends to the cache).

To get RNG-matched rollouts (same uniform draws per step so only the logits differ),
build one sampler per call with the same seed; see :func:`make_sampling_sampler`.
"""

from collections.abc import Callable, Collection

import torch
from torch import Tensor
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

from .steering import SteeringModel
from sparse_steer.utils.tokenize import apply_template, tokenize

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
    eos_token_ids: Collection[int] | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Decode ``max_new_tokens`` tokens, returning the newly generated ids ``(B, T)``
    and a ``valid`` mask ``(B, T)``; with ``capture_log_softmax`` also the per-step
    log-softmax over vocab ``(B, T, V)`` — the distribution the sampler drew from.

    ``input_ids`` / ``attention_mask`` are ``(B, P)`` (left-pad for batched decoding).
    ``sampler=None`` decodes greedily.

    ``valid[b, t]`` is True for positions up to and *including* sequence ``b``'s first
    end token, False after (all True if ``eos_token_ids`` is None or no end token is
    emitted). The loop has no native stopping, so use it to trim decoding and to exclude
    off-distribution post-end positions from distribution comparisons (e.g. JSD).

    ``eos_token_ids`` lets the loop early-stop once every sequence has emitted an end
    token — except when capturing log-softmax, where it runs the full ``max_new_tokens``
    so paired (matched-RNG) rollouts stay position-aligned and ``valid`` masks the tail.
    """
    if sampler is None:
        sampler = make_greedy_sampler()
    valid_steer_modes = {"all", "prompt", "prompt_final", "completion", "off"}
    if steer not in valid_steer_modes:
        raise ValueError(
            f"Unknown steer mode {steer!r}; use one of {sorted(valid_steer_modes)}."
        )
    eos = (
        torch.tensor(sorted(eos_token_ids), device=model.device)
        if eos_token_ids
        else None
    )
    device = model.device
    tl = model.tl
    seq = input_ids.to(device)
    attn = attention_mask.to(device) if attention_mask is not None else None

    kv_cache = TransformerLensKeyValueCache.init_cache(tl.cfg, device, seq.shape[0])

    def _step_ctx(step: int, inp: Tensor, inp_attention_mask: Tensor | None):
        if steer == "off":
            return model.steering_disabled()
        mask = (
            inp_attention_mask.bool()
            if inp_attention_mask is not None
            else torch.ones(inp.shape, dtype=torch.bool, device=device)
        )
        if step == 0:
            # step 0 forwards the whole prompt (positions 0..pf); its steered K/V are cached.
            if steer in ("prompt", "all"):
                return model.steer_positions(mask)        # the whole prompt (0..pf)
            if steer == "prompt_final":
                return model.steer_last_token(mask)       # only the final prompt token (pf)
            return model.steering_disabled()              # "completion": prompt is not steered
        # decode steps each forward one generated token (position > pf)
        if steer in ("completion", "all"):
            if eos is not None:
                mask = mask & ~(inp.unsqueeze(-1) == eos).any(-1)  # never steer a generated EOS
            return model.steer_positions(mask)
        return model.steering_disabled()                  # "prompt"/"prompt_final": decode unsteered

    new_tokens: list[Tensor] = []
    lsm_steps: list[Tensor] = []
    valid_steps: list[Tensor] = []
    finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        extra: dict = {"return_type": "logits", "prepend_bos": False, "past_kv_cache": kv_cache}
        if step == 0:
            inp = seq
            if attn is not None:
                extra["attention_mask"] = attn
        else:
            inp = seq[:, -1:]
            if attn is not None:
                extra["attention_mask"] = attn.new_ones(attn.shape[0], 1)

        with _step_ctx(step, inp, extra.get("attention_mask")):
            logits = tl(inp, **extra)
        last = logits[:, -1, :]
        if capture_log_softmax:
            lsm_steps.append(torch.log_softmax(last.float(), dim=-1))
        nxt = sampler(last)
        new_tokens.append(nxt.unsqueeze(1))
        # valid iff no end token has been emitted *before* this step → the first end
        # token itself counts as valid, every position after it is off-distribution.
        valid_steps.append(~finished.clone())
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        if attn is not None:
            attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], dim=1)
        if eos is not None:
            finished |= (nxt.unsqueeze(-1) == eos).any(dim=-1)
            # Early-stop once all sequences have ended — but not while capturing, so
            # paired rollouts stay length-aligned (the tail is masked via `valid`).
            if not capture_log_softmax and bool(finished.all()):
                break

    generated = torch.cat(new_tokens, dim=1)
    valid = torch.stack(valid_steps, dim=1)
    if capture_log_softmax:
        return generated, valid, torch.stack(lsm_steps, dim=1)
    return generated, valid


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
    is_tl = hasattr(model, "tl")
    greedy = temperature <= 0
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # End-token ids let the TL decode loop early-stop and report a `valid` mask (it has no
    # native EOS stop); without them it runs the full max_new_tokens past the turn-end into
    # off-distribution text that ``skip_special_tokens`` alone would keep.
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

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
            new_toks, valid = generate(
                model, enc["input_ids"], enc["attention_mask"], max_new_tokens,
                sampler=sampler, steer=steer,
                eos_token_ids=eos_ids or None,
            )
            # `valid` trims each row at its first end token; skip_special_tokens then drops
            # the end token itself (and any tail tokens past it from a shorter batch row).
            out.extend(
                tokenizer.decode(row[v].tolist(), skip_special_tokens=True).strip()
                for row, v in zip(new_toks, valid)
            )
        else:  # plain HF model (LoRA/peft): native generate pads finished sequences, which
            # skip_special_tokens drops — no manual trim needed; the adapter is always on.
            gen = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=not greedy,
                temperature=None if greedy else temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_toks = gen[:, enc["input_ids"].shape[1] :]
            out.extend(
                tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
                for row in new_toks
            )
    return out


__all__ = ["Sampler", "make_greedy_sampler", "make_sampling_sampler", "generate", "generate_text"]
