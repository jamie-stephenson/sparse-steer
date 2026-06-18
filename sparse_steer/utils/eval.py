"""Teacher-forced logit metrics — all share one tokenize→forward→log-softmax core.

Every metric here runs the model on chat-templated text and reads log-probs off the
logits; they differ only in (a) padding side, (b) which positions they read, and (c) the
reduction. :func:`_forward_batches` is that shared core; the public functions are thin
reductions on top:

- :func:`answer_log_probs` / :func:`teacher_forced_perplexity` — read the *actual* next-token
  log-prob over an answer span (right-padded), summed / perplexity-normalised.
- :func:`decision_logprobs` — read the full next-token distribution at the last prompt token
  (left-padded), e.g. for a KL between steered and base.

Steering is applied as the model currently carries it; wrap any call in
``model.steering_disabled()`` to read the base (unsteered) distribution.
"""

import math

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .tokenize import apply_template, tokenize


def _forward_batches(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    batch_size: int | None = None,
    padding_side: str | None = None,
    add_special_tokens: bool = True,
):
    """Yield ``(enc, logits)`` per batch — the shared tokenize→forward step behind every
    teacher-forced metric (``batch_size=None`` = one batch). ``no_grad`` guards the forward
    here (a ``@no_grad`` decorator on a generator function would not).
    """
    bs = batch_size or max(len(texts), 1)
    for s in range(0, len(texts), bs):
        enc = tokenize(
            tokenizer, texts[s : s + bs],
            add_special_tokens=add_special_tokens, padding_side=padding_side,
        ).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits  # (batch, seq, vocab)
        yield enc, logits


def _answer_token_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    answers: list[str],
    *,
    batch_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-example (summed answer-token log-prob, answer-token count), teacher-forced.

    Reads the *actual* next-token log-prob over the answer span (right-padded, so each
    answer is contiguous after its per-row prefix). Shared primitive behind
    :func:`answer_log_probs` and :func:`teacher_forced_perplexity`.
    """
    full = [apply_template(tokenizer, q, a) for q, a in zip(questions, answers)]
    # each question may have a different prefix length (masked out of the score)
    prefix_lens = [len(tokenizer(apply_template(tokenizer, q))["input_ids"]) for q in questions]
    device = model.device
    sums: list[Tensor] = []
    counts: list[Tensor] = []
    seen = 0
    for enc, logits in _forward_batches(model, tokenizer, full, batch_size=batch_size):
        n = enc["input_ids"].size(0)
        prefix_t = torch.tensor(prefix_lens[seen : seen + n], device=device).unsqueeze(1)
        seen += n
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        target_ids = enc["input_ids"][:, 1:]
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        # answer tokens only (skip the per-row question prefix and padding)
        answer_mask = enc["attention_mask"][:, 1:].clone()
        seq_idx = torch.arange(answer_mask.size(1), device=device)
        answer_mask = answer_mask * (seq_idx >= prefix_t - 1)
        sums.append((token_log_probs * answer_mask).sum(dim=-1))
        counts.append(answer_mask.sum(dim=-1))
    return torch.cat(sums), torch.cat(counts)


def answer_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    answers: list[str],
    *,
    batch_size: int | None = None,
) -> Tensor:
    """Total log-probability of answer tokens for each Q-A pair (batched)."""
    return _answer_token_logprobs(model, tokenizer, questions, answers, batch_size=batch_size)[0]


def teacher_forced_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    answers: list[str],
    *,
    batch_size: int | None = None,
) -> float:
    """Token-level perplexity of the answers under teacher forcing — a capability/fluency
    proxy (run unsteered vs intervened to detect collateral damage). Lower = more capable."""
    sums, counts = _answer_token_logprobs(model, tokenizer, questions, answers, batch_size=batch_size)
    return math.exp(-float(sums.sum().item()) / max(float(counts.sum().item()), 1.0))


def decision_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    batch_size: int | None = None,
) -> Tensor:
    """Log-softmax of the next-token distribution at the **last prompt token**, ``(n, vocab)``
    on CPU. Prompt-only and left-padded so position ``-1`` is the real last token of every
    row (Arditi's "decision" position). Wrap in ``model.steering_disabled()`` for the base
    distribution (e.g. one call steered, one base, for a KL)."""
    full = [apply_template(tokenizer, p) for p in prompts]
    out: list[Tensor] = []
    for _, logits in _forward_batches(
        model, tokenizer, full,
        batch_size=batch_size, padding_side="left", add_special_tokens=False,
    ):
        out.append(torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def kl_divergence(logp_p: Tensor, logp_q: Tensor, direction: str = "reverse") -> Tensor:
    """Per-row KL between two log-prob distributions (last dim = vocab).

    With ``p`` = the steered/intervened model and ``q`` = the base model:
      ``"reverse"`` (default; a.k.a. backward) = ``KL(p ‖ q) = Σ p·(log p − log q)`` — mode-seeking.
      ``"forward"``                            = ``KL(q ‖ p) = Σ q·(log q − log p)`` — mass-covering.

    Pure math (no model/task awareness) so the training objective, the eval ``kl_harmless``
    metric, and any monitor share one implementation.
    """
    if direction in ("reverse", "backward"):
        return (logp_p.exp() * (logp_p - logp_q)).sum(dim=-1)
    if direction == "forward":
        return (logp_q.exp() * (logp_q - logp_p)).sum(dim=-1)
    raise ValueError(f"Unknown kl direction {direction!r}; use 'forward' or 'reverse'.")


__all__ = [
    "answer_log_probs",
    "teacher_forced_perplexity",
    "decision_logprobs",
    "kl_divergence",
]
