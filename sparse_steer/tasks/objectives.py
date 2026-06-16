"""Shared gate-training objectives (Config-B / ``refinement_method: gate_training``).

Two objectives, selected by ``config.jb_objective`` (default ``"ce"``):

- ``ce`` — teacher-forced cross-entropy toward the row's target completion, with steering
  confined to the prompt positions (the ``steer_mask`` the collate emits). Used by jailbreak
  (CE-toward-affirmative) and by safesteer's plain "CE-toward-safe" mode.
- ``ce_kl`` — CE-toward-target on the ``"harmful"`` rows (drive the behaviour) **plus**
  ``KL(steered ‖ base)`` at the decision token on the ``"harmless"`` rows (preserve benign
  behaviour). Steering is applied over the full prompt (no ``steer_mask``) so the KL term
  matches how steering is measured at eval. ``config.harmless_kl_weight`` scales the KL term.

The steering-gate L0 penalty is task-independent and added by the training loop, not here.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing import Any

from sparse_steer.tasks.collate import prompt_completion_collate


# ── collate ───────────────────────────────────────────────────────────────


def objective_collate(
    rows: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    device: Any,
    config: DictConfig,
) -> dict[str, Tensor]:
    """Build the batch for ``config.jb_objective``. ``ce`` keeps the prompt-confined
    ``steer_mask``; ``ce_kl`` drops it (full-prompt steering) and tags the KL-preserve rows."""
    batch = prompt_completion_collate(rows, tokenizer, device, config)
    if config.get("jb_objective", "ce") != "ce_kl":
        return batch
    # ce_kl: full-prompt steering (matches eval) ⇒ no steer_mask confine.
    batch.pop("steer_mask", None)
    labels = batch["labels"]
    # decision index per row = last prompt token = (first completion token index) − 1.
    is_completion = labels != -100
    first_comp = torch.argmax(is_completion.int(), dim=1)
    batch["decision_idx"] = (first_comp - 1).clamp_min(0)
    batch["is_kl"] = torch.tensor(
        [r.get("category") == "harmless" for r in rows], dtype=torch.bool, device=device
    )
    batch["kl_weight"] = torch.tensor(float(config.get("harmless_kl_weight", 1.0)), device=device)
    return batch


# ── losses ────────────────────────────────────────────────────────────────


def ce_loss(model, batch: dict[str, Tensor], logits: Tensor) -> Tensor:
    """Teacher-forced CE toward ``batch['labels']`` (the target completion)."""
    shift_logits = logits[..., :-1, :].float()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        batch["labels"][..., 1:].reshape(-1),
        ignore_index=-100,
    )


def ce_kl_loss(model, batch: dict[str, Tensor], logits: Tensor) -> Tensor:
    """CE-toward-target on harmful rows + ``KL(steered ‖ base)`` at the decision token on
    harmless rows (full-prompt steering)."""
    labels = batch["labels"]
    is_kl = batch["is_kl"]
    ce_rows = ~is_kl
    loss = logits.new_zeros(())

    if bool(ce_rows.any()):
        sl = logits[ce_rows][..., :-1, :].float()
        ll = labels[ce_rows][..., 1:]
        loss = loss + F.cross_entropy(
            sl.reshape(-1, sl.size(-1)), ll.reshape(-1), ignore_index=-100
        )

    if bool(is_kl.any()):
        idx = batch["decision_idx"][is_kl]
        steered = logits[is_kl]
        rows = torch.arange(steered.size(0), device=steered.device)
        steered_dec = steered[rows, idx].float()
        with model.steering_disabled():
            base_logits = model(
                batch["input_ids"][is_kl], attention_mask=batch["attention_mask"][is_kl]
            ).logits
        base_dec = base_logits[rows, idx].float()
        logp_s = F.log_softmax(steered_dec, dim=-1)
        logp_b = F.log_softmax(base_dec, dim=-1)
        kl = (logp_s.exp() * (logp_s - logp_b)).sum(dim=-1).mean()
        loss = loss + batch["kl_weight"] * kl

    return loss


def objective_loss(model, batch: dict[str, Tensor], logits: Tensor, config: DictConfig) -> Tensor:
    """Dispatch on ``config.jb_objective`` (``"ce"`` default, or ``"ce_kl"``)."""
    objective = config.get("jb_objective", "ce")
    if objective == "ce":
        return ce_loss(model, batch, logits)
    if objective == "ce_kl":
        return ce_kl_loss(model, batch, logits)
    raise ValueError(f"Unknown jb_objective {objective!r}; use 'ce' or 'ce_kl'.")


__all__ = ["objective_collate", "objective_loss", "ce_loss", "ce_kl_loss"]
