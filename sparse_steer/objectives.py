"""Composable gate-training objective: a weighted sum of named loss terms.

Tasks differ only in the batch their collate builds (which rows, what targets, where steering
applies) — not in the objective. Each row carries a ``loss_term`` tag (default ``"ce"``) and the
collate turns it into per-term row masks (``batch["loss_term_rows"]``). A term runs when its
config weight is non-zero and the batch has rows tagged for it:

- ``ce``  (``ce_weight``, default 1.0): teacher-forced CE toward each ``ce`` row's target completion.
- ``kl``  (``kl_weight``, default 0.0): ``KL(steered, base)`` on each ``kl`` row to preserve benign
  behaviour. ``kl_direction`` (``reverse`` | ``forward``) and ``kl_positions``
  (``first_token`` | ``completion``) shape it — see :func:`kl_term`.

The L0 sparsity penalty is model-intrinsic (a function of the gates) and is added, scheduled, by
the training loop — not here.
"""

import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from sparse_steer.core.loading import resolve_steering_dtype
from sparse_steer.utils.eval import kl_divergence

# `steering_dtype` (resolved in core.loading) drives the steering params, the correction, AND
# the CE/contrastive loss compute here, so the whole steering math runs in one dtype (float32 =
# today's stable default; fp16 = the old 0bd8bf9 coupled-precision numerics).


def ce_term(
    batch: dict[str, Tensor], logits: Tensor, rows: Tensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Teacher-forced next-token CE over the ``rows``-masked subset, computed in ``dtype``
    (follows ``steering_dtype`` — fp16 for a fully-fp16 run, float32 for the stable default)."""
    sl = logits[rows][..., :-1, :].to(dtype)
    ll = batch["labels"][rows][..., 1:]
    return F.cross_entropy(sl.reshape(-1, sl.size(-1)), ll.reshape(-1), ignore_index=-100)


def contrastive_term(
    batch: dict[str, Tensor],
    logits: Tensor,
    rows: Tensor,
    config: DictConfig,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """mc1-style RANKING loss on contrastive gate-train rows (NO test leakage — gate-train split).

    Each question contributes its answers laid out contiguously: the CORRECT answer at index 0 of the
    group, then its incorrect answers. The group is RAGGED — K (= 1 + #negatives) varies per question
    (``batch["contrastive_group_sizes"]`` gives the K's, in order), mirroring the mc1 metric which
    scores the correct answer against *every* incorrect choice. The score per sequence is the total
    answer-token log-prob (sum over completion tokens, matching the mc1 metric's `answer_log_probs`);
    per question we apply softmax cross-entropy toward the correct one (index 0). This REWARDS steering
    that makes the truthful answer more likely than the false ones — unlike plain CE, which fights
    steering. The L0 gates still learn the site; only the objective changes.

    ``n_ans_normalise_contrastive_term`` (default off): divide each question's loss by ``log(K)`` (its
    chance-level floor) so questions with more negatives — a larger softmax, hence a larger natural
    loss — don't dominate the gradient over questions with few negatives.
    """
    sizes = [int(s) for s in batch["contrastive_group_sizes"]]  # K per question, correct-first
    sl = logits[rows][..., :-1, :]
    ll = batch["labels"][rows][..., 1:]
    v = sl.size(-1)
    # per-token NLL via fused cross_entropy (no full-vocab log_softmax tensor materialised →
    # far cheaper on MPS than log_softmax+gather). ignore_index zeroes the masked prompt tokens.
    nll = F.cross_entropy(
        sl.reshape(-1, v).to(dtype), ll.reshape(-1), ignore_index=-100, reduction="none"
    ).view(ll.shape)
    seq_lp = -nll.sum(-1)  # (n_sequences,) total answer log-prob; correct = first of each group
    normalise = bool(config.get("n_ans_normalise_contrastive_term", False))
    losses = []
    for g in torch.split(seq_lp, sizes):  # one (K_q,) score vector per question
        loss_q = torch.logsumexp(g, dim=0) - g[0]  # softmax-CE toward the correct answer (index 0)
        if normalise:
            loss_q = loss_q / math.log(g.numel())  # /log(K) → equalise across ragged group sizes
        losses.append(loss_q)
    return torch.stack(losses).mean()


def kl_term(
    model, batch: dict[str, Tensor], logits: Tensor, rows: Tensor, config: DictConfig
) -> Tensor:
    """``KL(steered, base)`` over the ``rows``-masked subset, at the configured positions.

    ``kl_positions``: ``"first_token"`` uses only the next-token distribution that *produces the
    first completion token* — which lives at the **last prompt-template token** (teacher-forcing
    shift: the dist over seq position ``p`` is the logits at ``p−1``); ``"completion"`` uses every
    completion token's producing distribution (last prompt token through the penultimate completion
    token). KL is averaged over the selected positions. ``kl_direction`` picks reverse/forward.
    """
    direction = config.get("kl_direction", "reverse")
    positions = config.get("kl_positions", "first_token")
    labels = batch["labels"][rows]
    steered = logits[rows][:, :-1].float()                 # (n, L-1, V)
    produces = labels[:, 1:] != -100                       # (n, L-1): produces a completion token

    if positions == "first_token":
        has = produces.any(dim=1)
        first = torch.argmax(produces.int(), dim=1)
        keep = torch.zeros_like(produces)
        keep[torch.arange(keep.size(0), device=keep.device), first] = True
        produces = keep & has.unsqueeze(1)
    elif positions != "completion":
        raise ValueError(
            f"Unknown kl_positions {positions!r}; use 'first_token' or 'completion'."
        )

    if not bool(produces.any()):
        return logits.new_zeros(())

    with model.steering_disabled():
        base = model(
            batch["input_ids"][rows], attention_mask=batch["attention_mask"][rows]
        ).logits
    base = base[:, :-1].float()
    logp_s = F.log_softmax(steered[produces], dim=-1)      # (num_sel, V)
    logp_b = F.log_softmax(base[produces], dim=-1)
    return kl_divergence(logp_s, logp_b, direction).mean()  # mean over selected positions


def composed_objective(
    model, batch: dict[str, Tensor], logits: Tensor, config: DictConfig
) -> Tensor:
    """Weighted sum of the active data-driven terms (``ce`` + ``kl``). Each term runs only when
    its weight is non-zero and the batch carries rows tagged for it. Sparsity (L0) is added by
    the training loop."""
    term_rows = batch["loss_term_rows"]
    loss = logits.new_zeros(())
    dtype = resolve_steering_dtype(config)

    ce_w = float(config.get("ce_weight", 1.0))
    ce_rows = term_rows.get("ce")
    if ce_w and ce_rows is not None and bool(ce_rows.any()):
        loss = loss + ce_w * ce_term(batch, logits, ce_rows, dtype)

    contrastive_w = float(config.get("contrastive_weight", 0.0))
    contrastive_rows = term_rows.get("contrastive")
    if contrastive_w and contrastive_rows is not None and bool(contrastive_rows.any()):
        loss = loss + contrastive_w * contrastive_term(
            batch, logits, contrastive_rows, config, dtype
        )

    kl_w = float(config.get("kl_weight", 0.0))
    kl_rows = term_rows.get("kl")
    if kl_w and kl_rows is not None and bool(kl_rows.any()):
        loss = loss + kl_w * kl_term(model, batch, logits, kl_rows, config)

    return loss


__all__ = ["ce_term", "contrastive_term", "kl_term", "composed_objective", "resolve_steering_dtype"]
