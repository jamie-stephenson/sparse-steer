"""Jailbreak selection scoring (Arditi App. B/C) — the task's :class:`SelectionPolicy`.

This is the task-specific half of ``direction_source: grid_select``: given a candidate refusal
direction broadcast onto every steering site, score it by

- **bypass** — the refusal metric on harmful prompts under the candidate's ablation (lower ⇒ more
  refusal suppressed; this is the objective the picker minimises),
- **induce** — the refusal metric on harmless prompts when the direction is *added* (confirms it
  is genuinely the refusal direction, not an arbitrary destructive one), and
- **kl** — KL(baseline ‖ ablated) on harmless prompts (ablation must not damage harmless
  behaviour).

The generic grid construction, candidate sweep, and filter/pick live in
``experiment.sourcing``; this module only supplies the refusal scoring + constraints. "Refusal"
is Arditi's logit metric (``utils.refusal.refusal_metric``).
"""

import torch
from torch import Tensor

from sparse_steer.core.steering import COMPONENT_HOOK, SteeringModel
from sparse_steer.tasks.base import SelectionPolicy
from sparse_steer.utils.eval import decision_logprobs
from sparse_steer.utils.refusal import refusal_metric, resolve_refusal_token_ids
from sparse_steer.utils.tokenize import apply_template, tokenize


def _refine_instructions(refine_ds) -> tuple[list[str], list[str]]:
    """Harmful / harmless instruction lists from the refinement DatasetDict (train + val rows,
    which carry ``instruction`` + ``category`` columns alongside the gate-training prompt)."""
    rows: list[dict] = []
    for split in ("train", "val"):
        if split in refine_ds:
            rows += list(refine_ds[split])
    harmful = [r["instruction"] for r in rows if r.get("category") == "harmful"]
    harmless = [r["instruction"] for r in rows if r.get("category") == "harmless"]
    return harmful, harmless


@torch.no_grad()
def _induce_logprobs(
    model: SteeringModel, tokenizer, prompts: list[str], layer: int, vector: Tensor, batch_size: int
) -> Tensor:
    """Last-token log-softmax with a TEMPORARY ``+vector`` steer at ``layer``'s block input
    (resid_pre, exactly as Arditi adds the direction at the source layer's block input;
    permanent steering disabled) — the induce-refusal measurement. ``(n, vocab)`` on CPU."""
    hook_name = COMPONENT_HOOK["resid_pre"].format(i=layer)
    v = vector.to(device=model.device, dtype=torch.float32)

    def add(act, hook=None):
        return act + v.to(act.dtype)

    full = [apply_template(tokenizer, p) for p in prompts]
    out: list[Tensor] = []
    for s in range(0, len(full), batch_size):
        enc = tokenize(
            tokenizer, full[s : s + batch_size], add_special_tokens=False, padding_side="left"
        ).to(model.device)
        with model.steering_disabled():
            logits = model.tl.run_with_hooks(
                enc["input_ids"], attention_mask=enc["attention_mask"],
                fwd_hooks=[(hook_name, add)], return_type="logits", prepend_bos=False,
            )
        out.append(torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def jailbreak_selection_policy(model, tokenizer, refine_ds, config) -> SelectionPolicy:
    """Build the jailbreak :class:`SelectionPolicy` for ``direction_source: grid_select``.

    Prepares the scoring set once (Arditi's ``filter_val``: keep harmful the unsteered model does
    refuse and harmless it does not) and the harmless baseline distribution, then returns a
    ``score(vector, layer)`` closure the generic driver calls per candidate — after it has
    broadcast that candidate onto every steered site, so bypass/kl read the ablated model.
    """
    bs = int(config.eval_batch_size)
    token_ids = resolve_refusal_token_ids(tokenizer, list(config.refusal_tokens))

    harmful, harmless = _refine_instructions(refine_ds)
    if not harmful or not harmless:
        raise RuntimeError(
            f"Direction selection needs harmful and harmless refinement prompts "
            f"(got {len(harmful)} harmful, {len(harmless)} harmless)."
        )

    # Arditi's filter_val (run_pipeline.filter_data): score the scoring set with the UNSTEERED
    # refusal metric and keep only the confidently-classified prompts — harmful the model does
    # refuse (score > 0) and harmless it does not (score < 0). Without this the bypass averages in
    # already-complying harmful prompts and reads more negative than theirs.
    with model.steering_disabled():
        h_scores = refusal_metric(decision_logprobs(model, tokenizer, harmful, batch_size=bs), token_ids)
        l_scores = refusal_metric(decision_logprobs(model, tokenizer, harmless, batch_size=bs), token_ids)
    harmful = [p for p, s in zip(harmful, h_scores.tolist()) if s > 0]
    harmless = [p for p, s in zip(harmless, l_scores.tolist()) if s < 0]

    with model.steering_disabled():
        base_harmless = decision_logprobs(model, tokenizer, harmless, batch_size=bs)

    @torch.no_grad()
    def score(vector: Tensor, layer: int) -> tuple[float, dict[str, float]]:
        # The driver has already broadcast ``vector`` onto every site (ablate), so these read the
        # candidate-ablated model.
        bypass = refusal_metric(
            decision_logprobs(model, tokenizer, harmful, batch_size=bs), token_ids
        ).mean().item()
        ablated_harmless = decision_logprobs(model, tokenizer, harmless, batch_size=bs)
        # KL(baseline ‖ ablated) on harmless, exactly as Arditi (App. C, kl_div_fn, eps 1e-6) —
        # NB the order: baseline is the reference distribution (not ablated ‖ baseline).
        p_base, p_abl = base_harmless.exp(), ablated_harmless.exp()
        kl = (p_base * ((p_base + 1e-6).log() - (p_abl + 1e-6).log())).sum(-1).mean().item()
        induce = refusal_metric(
            _induce_logprobs(model, tokenizer, harmless, layer, vector, bs), token_ids
        ).mean().item()
        return bypass, {"bypass": bypass, "induce": induce, "kl": kl}

    return SelectionPolicy(
        score=score,
        constraints=[
            ("kl", "<=", float(config.get("selection_kl_threshold", 0.1))),
            ("induce", ">=", float(config.get("selection_induce_threshold", 0.0))),
        ],
    )


__all__ = ["jailbreak_selection_policy"]
