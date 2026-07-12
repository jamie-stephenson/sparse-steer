"""The single token-position vocabulary shared by extraction, steering, and the CE/contrastive
objective. A position name maps to a boolean ``(batch, seq)`` mask; the SAME mask defines which
positions a direction is averaged over (extraction), which positions are steered (steering), and
which positions the loss is scored over (CE / contrastive).

``pf = prompt_len - 1`` = the final prompt token — the last token the model sees before generating.

- ``"prompt"``        positions ``0..pf``            — the whole input, incl. the final prompt token
- ``"prompt_final"``  position ``pf``                — exactly the final prompt token
- ``"completion"``    positions ``pf+1..`` (no EOS)  — strictly AFTER the prompt (disjoint from prompt)
- ``"completion_final"`` the single LAST completion token — the answer's final content token
  (honest_llama's ITI extraction position, made EOS-safe; for extraction this reads ONE token,
  the last-token direction, vs ``"completion"``'s mean over the whole answer)
- ``"answer_gen"``    positions ``pf..(answer_end-1)`` — the positions whose logits GENERATE the
  answer. Token ``t``'s logits come from position ``t-1``, so to steer the model's *generation of
  the answer* we steer the final prompt token ``pf`` (predicts the first answer token) through the
  second-to-last answer token — NOT the answer tokens themselves (``"completion"``, which is one
  token too late and, for a single-token MC answer, steers nothing that is scored). This is the
  correct STEERING position for "steer the generated answer"; extraction stays on ``"completion"``/
  ``"completion_final"`` (the tokens whose activations differ). ``pf`` is the ``ANSWER:`` colon in
  the fixed template / the ``[/INST]``|assistant-header in the chat template, located by
  ``prompt_lens`` — so the same name is correct for both.
- ``"all"``           every real token (no EOS)      — ``= "prompt" ∪ "completion"``

EOS is excluded from ``"completion"`` and ``"all"``: nothing is generated after EOS, so its
activation is not a position the model steers at, reads a direction from, or is scored on.
``"prompt"`` / ``"prompt_final"`` fall before any EOS, so the EOS filter is moot there.
"""

import torch
from torch import Tensor

POSITION_NAMES = ("prompt", "prompt_final", "completion", "completion_final", "answer_gen", "all")


def positions_mask(
    name: str,
    attention_mask: Tensor,
    prompt_lens: Tensor,
    *,
    input_ids: Tensor | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Boolean ``(batch, seq)`` mask for ``name`` — see the module docstring.

    ``prompt_lens`` is ``(batch,)``: the number of prompt tokens per row (``pf = prompt_len - 1``).
    ``input_ids`` + ``eos_id`` drop EOS from ``"completion"``/``"all"``; if either is None the EOS
    filter is skipped (callers whose sequences carry no trailing EOS, e.g. iti_qa, may omit it).
    Works for left- or right-padded batches (``real`` excludes padding regardless of side).
    """
    if name not in POSITION_NAMES:
        raise ValueError(f"unknown token position {name!r}; use one of {POSITION_NAMES}")
    real = attention_mask.bool()
    b, s = real.shape
    pos = torch.arange(s, device=real.device).unsqueeze(0).expand(b, s)
    pf = (prompt_lens.to(real.device).long() - 1).clamp_min(0).unsqueeze(1)  # (b, 1)
    if name == "prompt":
        return real & (pos <= pf)
    if name == "prompt_final":
        return real & (pos == pf)
    if input_ids is not None and eos_id is not None:
        not_eos = input_ids.to(real.device) != eos_id
    else:
        not_eos = torch.ones_like(real)
    if name == "completion":
        return real & not_eos & (pos > pf)
    if name == "completion_final":
        # the single LAST completion token (the answer's final content token) — honest_llama's ITI
        # extraction position, made EOS-safe. Empty mask for rows that have no completion.
        content = real & not_eos & (pos > pf)
        out = torch.zeros_like(real)
        has = content.any(dim=1)
        last_idx = (content.long() * pos).max(dim=1).values
        rows = torch.arange(b, device=real.device)
        out[rows[has], last_idx[has]] = True
        return out
    if name == "answer_gen":
        # pf..(answer_end-1): the positions whose logits generate the answer tokens (see docstring).
        # answer_end = last completion index; steer pf through answer_end-1 (exclude the last answer
        # token, whose logits predict what follows the answer). Single-token answer → just {pf}.
        content = real & not_eos & (pos > pf)
        answer_end = (content.long() * pos).max(dim=1, keepdim=True).values  # (b,1); 0 if no answer
        return real & (pos >= pf) & (pos < answer_end)
    return real & not_eos  # "all"


__all__ = ["POSITION_NAMES", "positions_mask"]
