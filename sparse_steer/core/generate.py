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
  steers its new token (skipping a generated EOS); the step-0 prompt is NOT steered. NB this leaves
  the FIRST generated token unsteered (its logits come from ``pf``, which is not steered here) — use
  ``"answer_gen"`` to steer the whole generated answer.
- ``"answer_gen"``   — steer the model's generation of the ENTIRE answer: step 0 steers only the
  final prompt token ``pf`` (so the first generated token is produced from steered state), and every
  decode step steers its new token (excluding EOS). This is the generative counterpart of the
  ``"answer_gen"`` position mask; ``"completion"`` is it minus the first token.
- ``"all"``          — every real non-EOS token (``= prompt ∪ completion``).
- ``"off"``          — no steering (clean / reference rollouts).

Generation always uses a KV cache (a HF ``DynamicCache``): step 0 forwards the full
prompt; each later step forwards only the new token and attends to the cache. The decode
loop owns the steering contexts, samplers, and EOS/valid handling.

To get RNG-matched rollouts (same uniform draws per step so only the logits differ),
build one sampler per call with the same seed; see :func:`make_sampling_sampler`.
"""

from collections.abc import Callable, Collection

import torch
from torch import Tensor

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
    valid_steer_modes = {"all", "prompt", "prompt_final", "completion", "answer_gen", "off"}
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
    seq = input_ids.to(device)
    attn = attention_mask.to(device) if attention_mask is not None else None

    from transformers.cache_utils import DynamicCache

    kv_cache = DynamicCache()
    if attn is None:
        # cached decoding reads padding + positions off the mask
        attn = torch.ones_like(seq)

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
            if steer in ("prompt_final", "answer_gen"):
                # steer ONLY the final prompt token pf → the first generated token is produced from
                # steered state ("answer_gen"); "prompt_final" additionally leaves decode unsteered.
                return model.steer_last_token(mask)
            return model.steering_disabled()              # "completion": prompt is not steered
        # decode steps each forward one generated token (position > pf)
        if steer in ("completion", "answer_gen", "all"):
            if eos is not None:
                mask = mask & ~(inp.unsqueeze(-1) == eos).any(-1)  # never steer a generated EOS
            return model.steer_positions(mask)
        return model.steering_disabled()                  # "prompt"/"prompt_final": decode unsteered

    new_tokens: list[Tensor] = []
    lsm_steps: list[Tensor] = []
    valid_steps: list[Tensor] = []
    finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        if step == 0:
            inp = seq
            step_attn = attn
        else:
            inp = seq[:, -1:]
            step_attn = attn.new_ones(attn.shape[0], 1) if attn is not None else None

        with _step_ctx(step, inp, step_attn):
            # Full mask (cached + current tokens); positions from its cumsum
            # (correct for left-padded prompts).
            position_ids = (attn.long().cumsum(-1) - 1).clamp_min(0)[:, -inp.shape[1]:]
            out = model.engine(
                input_ids=inp,
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
            logits, kv_cache = out.logits, out.past_key_values
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
def generate_text_and_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    *,
    temperature: float = 0.0,
    steer: str = "all",
    template: bool = True,
    top_logprobs: int = 0,
    add_special_tokens: bool = False,
) -> tuple[str, list[dict]]:
    """Single-prompt generation that ALSO returns per-generated-token log-probabilities, for the
    Inspect provider's logprob path (``config.logprobs``). Returns ``(text, tokens)`` where ``tokens``
    is one dict per generated token: ``{token, logprob, top: [(token, logprob), ...]}`` (``top`` has
    ``top_logprobs`` entries, empty if 0). Shares the generation seam with ``generate_text``: the
    SteeringModel path (either backend) reuses ``generate(capture_log_softmax=True)`` (steering
    active); the plain-HF path reuses ``output_logits``. Greedy when ``temperature <= 0`` (what
    evals use)."""
    is_steering = isinstance(model, SteeringModel)
    greedy = temperature <= 0
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    text = apply_template(tokenizer, prompt) if template else prompt
    enc = tokenize(tokenizer, [text], add_special_tokens=add_special_tokens, padding_side="left").to(device)

    if is_steering:
        sampler = make_greedy_sampler() if greedy else make_sampling_sampler(temperature=temperature, device=device)
        new_toks, valid, lsm = generate(
            model, enc["input_ids"], enc["attention_mask"], max_new_tokens,
            sampler=sampler, steer=steer, eos_token_ids=eos_ids or None,
            capture_log_softmax=True,
        )
        row, v, row_lsm = new_toks[0], valid[0], lsm[0]  # (T,), (T,), (T, V)
    else:  # plain HF model (LoRA/peft)
        gen = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=not greedy,
            temperature=None if greedy else temperature,
            pad_token_id=tokenizer.pad_token_id,
            output_logits=True, return_dict_in_generate=True,
        )
        row = gen.sequences[0, enc["input_ids"].shape[1]:]
        row_lsm = torch.log_softmax(torch.stack(gen.logits, dim=0).float(), dim=-1)[:, 0]  # (T, V)
        v = torch.ones(len(row), dtype=torch.bool, device=row.device)

    tokens: list[dict] = []
    for t in range(int(v.sum().item())):
        tid = int(row[t].item())
        entry = {"token": tokenizer.decode([tid]), "logprob": float(row_lsm[t, tid].item()), "top": []}
        if top_logprobs > 0:
            tv, ti = row_lsm[t].topk(top_logprobs)
            entry["top"] = [(tokenizer.decode([int(i)]), float(lp)) for lp, i in zip(tv.tolist(), ti.tolist())]
        tokens.append(entry)
    out_text = tokenizer.decode(row[v].tolist(), skip_special_tokens=True).strip()
    return out_text, tokens


def generate_text(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    *,
    temperature: float = 0.0,
    steer: str = "all",
    template: bool = True,
    add_special_tokens: bool = False,
    batch_size: int = 16,
) -> list[str]:
    """Batched string prompts → decoded responses, for **either** a ``SteeringModel`` (the shared
    KV-cached decode loop, with steering modes) **or** a plain HF model (e.g. a LoRA/peft
    model, via ``model.generate``). The single model-agnostic generation seam every
    generative eval (and the Inspect provider) shares.

    Greedy when ``temperature <= 0``, else multinomial from the **global** RNG (seeded once in
    ``_seed_everything`` — no per-call seed). Prompts are chat-templated here (``template=True``) or
    assumed already-templated (``template=False``); either way tokenised with
    ``add_special_tokens=False`` since the chat template carries its own special tokens. ``steer``
    modes apply only to a ``SteeringModel`` (an HF model's intervention — e.g. a LoRA adapter — is
    always on).
    """
    is_steering = isinstance(model, SteeringModel)
    greedy = temperature <= 0
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # End-token ids let the decode loop early-stop and report a `valid` mask (it has no
    # native EOS stop); without them it runs the full max_new_tokens past the turn-end into
    # off-distribution text that ``skip_special_tokens`` alone would keep.
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        texts = [apply_template(tokenizer, p) for p in batch] if template else list(batch)
        # left-pad for *this call only* (batched decoding needs the last real token at
        # [:, -1]); per-call padding_side leaves the shared tokenizer's default intact.
        # add_special_tokens: default False (already-templated text carries its own BOS, e.g. Llama-2
        # `{{bos_token}}`). Pass True for templates that emit NO BOS (dolphin ChatML) or raw prompts
        # (base model, no template) — the _sync_bos'd tokenizer then prepends exactly one BOS.
        enc = tokenize(tokenizer, texts, add_special_tokens=add_special_tokens, padding_side="left").to(device)
        if is_steering:
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
