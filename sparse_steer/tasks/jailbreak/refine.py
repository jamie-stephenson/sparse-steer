"""Arditi-style single-direction selection (paper App. B/C).

The refinement stage for the ``arditi_select`` method — a drop-in alternative to gate training
in the ``extract → refine → eval`` pipeline. From a grid of candidate refusal directions
(mean harmful − mean harmless at each layer × last-K position, computed on the *extraction*
set), pick the single direction whose directional ablation most suppresses refusal on the
held-out *refinement* set, subject to:

- a KL budget on harmless prompts (ablation must not damage harmless behaviour), and
- an "induce" check (adding the direction must actually raise refusal on harmless prompts —
  confirming it is genuinely the refusal direction, not an arbitrary destructive one),
- dropping the last ``prune_layer_frac`` of layers.

"Refusal" here is Arditi's logit refusal metric (``utils.refusal.refusal_metric``). The chosen
direction is then ablated from every steered residual site at eval (α=1, exact projection).
"""

import math
from typing import Any

import torch
from torch import Tensor

from sparse_steer.core.extract import (
    collect_activations,
    extract_steering_vectors,
    last_token_positions,
    load_steering_vectors,
    save_steering_vectors,
)
from sparse_steer.core.steering import COMPONENT_HOOK, SteeringModel
from sparse_steer.utils.cache import ArtifactType
from sparse_steer.utils.eval import decision_logprobs
from sparse_steer.utils.refusal import refusal_metric, resolve_refusal_token_ids
from sparse_steer.utils.tokenize import apply_template, tokenize

_GRID_COMPONENT = "resid_post"  # candidate directions read off the residual stream (d_model)


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
def _candidate_grid(
    model: SteeringModel, tokenizer, extraction_ds, *, n_positions: int, batch_size: int
) -> Tensor:
    """``(positions, layers, d_model)`` mean-difference directions at each of the last
    ``n_positions`` token positions (j=0 is the last token), per layer, from ``extraction_ds``."""
    grids: list[Tensor] = []
    for j in range(n_positions):
        def at_offset(inputs, j=j):
            return (last_token_positions(inputs["attention_mask"]) - j).clamp_min(0)

        ds_acts, _ = collect_activations(
            extraction_ds, model, tokenizer,
            targets=[_GRID_COMPONENT], batch_size=batch_size, token_position=at_offset,
        )
        grids.append(extract_steering_vectors(ds_acts, [_GRID_COMPONENT])[_GRID_COMPONENT])
    return torch.stack(grids, dim=0)


@torch.no_grad()
def _induce_logprobs(
    model: SteeringModel, tokenizer, prompts: list[str], layer: int, vector: Tensor, batch_size: int
) -> Tensor:
    """Last-token log-softmax with a TEMPORARY ``+vector`` steer at ``layer``'s resid_post
    (permanent steering disabled) — the induce-refusal measurement. ``(n, vocab)`` on CPU."""
    hook_name = COMPONENT_HOOK["resid_post"].format(i=layer)
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


def _filter_and_pick(
    scored: list[tuple[int, int, float, float, float]],
    n_layers: int,
    *,
    kl_threshold: float,
    induce_threshold: float,
    prune_layer_frac: float,
) -> dict[str, Any]:
    """Pure selection logic over ``[(pos, layer, bypass, induce, kl), ...]`` (Arditi App. C).

    Keep candidates that are finite, not in the last ``prune_layer_frac`` of layers, within the
    KL budget, and above the induce threshold; among survivors pick the **lowest** bypass refusal
    metric (most refusal suppressed). Raises if nothing survives.
    """
    keep_below_layer = int(n_layers * (1.0 - prune_layer_frac))
    survivors = [
        c for c in scored
        if not any(math.isnan(x) for x in c[2:])
        and c[1] < keep_below_layer
        and c[4] <= kl_threshold
        and c[3] >= induce_threshold
    ]
    if not survivors:
        raise RuntimeError(
            "Direction selection: no candidate survived the filters "
            f"(layers<{keep_below_layer}, kl≤{kl_threshold}, induce≥{induce_threshold}). "
            "Relax selection_kl_threshold / selection_induce_threshold."
        )
    pos, layer, bypass, induce, kl = min(survivors, key=lambda c: c[2])
    return {
        "position": pos, "layer": layer,
        "bypass": bypass, "induce": induce, "kl": kl,
        "n_candidates": len(scored), "n_survivors": len(survivors),
    }


@torch.no_grad()
def select_direction(
    model: SteeringModel, tokenizer, extraction_ds, refine_ds, config
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Select one refusal direction and return ``(vectors, metadata)``.

    ``vectors`` maps each steered component (``config.targets``) → ``(n_layers, d_model)`` with
    the chosen direction broadcast to every layer, ready for ``SteeringModel.set_all_vectors``
    (the ablate path normalises it). Scoring uses the refinement set; the grid uses extraction.
    """
    bs = int(config.eval_batch_size)
    token_ids = resolve_refusal_token_ids(tokenizer, list(config.refusal_tokens))
    n_positions = int(config.get("selection_positions", 1))
    components = list(config.targets)

    harmful, harmless = _refine_instructions(refine_ds)
    if not harmful or not harmless:
        raise RuntimeError(
            f"Direction selection needs harmful and harmless refinement prompts "
            f"(got {len(harmful)} harmful, {len(harmless)} harmless)."
        )

    grid = _candidate_grid(
        model, tokenizer, extraction_ds, n_positions=n_positions, batch_size=int(config.extract_batch_size)
    )
    n_pos, n_layers, d_model = grid.shape

    with model.steering_disabled():
        base_harmless = decision_logprobs(model, tokenizer, harmless, batch_size=bs)

    def broadcast(vec: Tensor) -> dict[str, Tensor]:
        layered = vec.unsqueeze(0).expand(n_layers, d_model).clone()
        return {c: layered.clone() for c in components}

    scored: list[tuple[int, int, float, float, float]] = []
    for j in range(n_pos):
        for layer in range(n_layers):
            r = grid[j, layer]
            if not torch.isfinite(r).all() or float(r.norm()) < 1e-8:
                continue
            # ablate this candidate at every steered site (set_all_vectors normalises for ablate)
            model.set_all_vectors(broadcast(r))
            bypass = refusal_metric(
                decision_logprobs(model, tokenizer, harmful, batch_size=bs), token_ids
            ).mean().item()
            ablated_harmless = decision_logprobs(model, tokenizer, harmless, batch_size=bs)
            kl = (ablated_harmless.exp() * (ablated_harmless - base_harmless)).sum(-1).mean().item()
            induce = refusal_metric(
                _induce_logprobs(model, tokenizer, harmless, layer, r, bs), token_ids
            ).mean().item()
            scored.append((j, layer, bypass, induce, kl))

    best = _filter_and_pick(
        scored, n_layers,
        kl_threshold=float(config.get("selection_kl_threshold", 0.1)),
        induce_threshold=float(config.get("selection_induce_threshold", 0.0)),
        prune_layer_frac=float(config.get("selection_prune_layer_frac", 0.2)),
    )
    selected = broadcast(grid[best["position"], best["layer"]])
    metadata = {"grid_component": _GRID_COMPONENT, "n_positions": n_pos, "n_layers": n_layers, **best}
    return selected, metadata


def arditi_select_refine(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """Refinement strategy for ``refinement_method=arditi_select`` (registered by JailbreakTask).

    Choose one direction on the refinement set and set it as the (ablated) steering vector at
    every site, cached as ``SELECTED_DIRECTION``. Does its own grid extraction, so it skips the
    standard ``run_extraction``. Signature matches the experiment refine slot (``output_dir`` is
    unused — selection produces no per-run artifacts beyond the cached direction)."""
    artifacts: dict[str, str | None] = {}
    cache_info: dict[str, Any] = {}

    hit = experiment._try_cache_lookup(ArtifactType.SELECTED_DIRECTION)
    if hit is not None:
        vectors, _ = load_steering_vectors(hit.artifact_path)
        model.set_all_vectors(vectors, normalize=experiment.config.normalize_steering_vectors)
        artifacts["selected_direction_path"] = str(hit.artifact_path)
        cache_info["selected_direction"] = {"status": "hit", "path": str(hit.artifact_path)}
        return model, artifacts, cache_info

    print("Selecting refusal direction (Arditi App. C)...")
    selected, meta = select_direction(model, tokenizer, extraction_ds, train_ds, experiment.config)
    model.set_all_vectors(selected, normalize=experiment.config.normalize_steering_vectors)
    print(
        f"  Selected layer={meta['layer']} pos={meta['position']} "
        f"bypass={meta['bypass']:.3f} induce={meta['induce']:.3f} kl={meta['kl']:.4f} "
        f"({meta['n_survivors']}/{meta['n_candidates']} candidates survived)"
    )
    dest = experiment._prepare_cache_path(ArtifactType.SELECTED_DIRECTION)
    save_steering_vectors(selected, dest, metadata=meta)
    path = str(experiment._finalize_cache(ArtifactType.SELECTED_DIRECTION))
    artifacts["selected_direction_path"] = path
    cache_info["selected_direction"] = {"status": "miss", "selection": meta}
    print(f"Saved selected direction to {path}")
    return model, artifacts, cache_info


__all__ = ["select_direction", "arditi_select_refine"]
