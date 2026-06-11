"""Direction sourcing — decoupled from refinement and intervention.

A steering site's direction can come from three places, selected by ``config.direction_source``:

- ``self`` — each site uses the mean-difference direction extracted *at that same site*
  (many-to-many / lockstep). The default — **per-site sparse ablation**.
- ``[component, layer]`` — one direction (the diff-in-means at that single site) broadcast to
  *every* steering site (one-to-many) — **broadcast (pinned-direction) sparse ablation**, e.g.
  pin Arditi's layer-17 direction and let the gates learn the placement.
- ``grid_select`` — pick one direction from a candidate grid by the task's
  :class:`~sparse_steer.tasks.base.SelectionPolicy` (Arditi App. C), then broadcast it. This is
  the reproduction's selection, factored out of the old ``arditi_select`` refine.

``self`` / ``[c,l]`` are *thin* — pure transforms of the already-extracted per-site vectors.
``grid_select`` is *fat* — it scores candidates on the refinement set, so it needs the refine
dataset and a task scorer; that coupling is contained here and to the policy. The result is
always a ``{component: (n_layers, ...)}`` dict ready for ``SteeringModel.set_all_vectors``.
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
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.cache import ArtifactType
from ._common import run_extraction


# ── direction_source parsing ──────────────────────────────────────────


def _parse_source(direction_source: Any) -> tuple[str, str | None, int | None]:
    """``direction_source`` → ``(kind, component, layer)`` with ``kind`` in
    ``{"self", "pin", "grid"}``."""
    if direction_source == "self":
        return "self", None, None
    if direction_source == "grid_select":
        return "grid", None, None
    seq = list(direction_source)
    if len(seq) != 2:
        raise ValueError(
            "direction_source must be 'self', 'grid_select', or a [component, layer] pair, "
            f"got {direction_source!r}."
        )
    return "pin", str(seq[0]), int(seq[1])


def extraction_targets(config) -> list[str]:
    """Component taps to extract: the steering ``targets``, plus a pinned source's component if
    that isn't already a steering target (so the broadcast direction is available to read)."""
    targets = list(config.targets)
    kind, component, _ = _parse_source(config.get("direction_source", "self"))
    if kind == "pin" and component not in targets:
        targets = targets + [component]
    return targets


# ── broadcast ─────────────────────────────────────────────────────────


def broadcast(vec: Tensor, components: list[str], n_layers: int) -> dict[str, Tensor]:
    """One direction → ``{component: (n_layers, d)}`` with every layer set to ``vec``.

    Used by the one-to-many sources (pinned / grid_select): the same direction is applied at
    every steering site. Assumes ``vec`` matches each component's per-site shape (true for the
    residual taps, which all carry ``(d_model,)``)."""
    layered = vec.reshape(1, -1).expand(n_layers, vec.shape[-1]).clone()
    return {c: layered.clone() for c in components}


# ── grid_select (Arditi App. C, generalised) ──────────────────────────


@torch.no_grad()
def candidate_grid(
    model: SteeringModel,
    tokenizer,
    extraction_ds,
    *,
    n_positions: int,
    grid_component: str,
    batch_size: int,
) -> Tensor:
    """``(positions, layers, d)`` mean-difference directions at each of the last ``n_positions``
    token positions (j=0 is the last token), per layer, read at ``grid_component``."""
    grids: list[Tensor] = []
    for j in range(n_positions):
        def at_offset(inputs, j=j):
            return (last_token_positions(inputs["attention_mask"]) - j).clamp_min(0)

        ds_acts, _ = collect_activations(
            extraction_ds, model, tokenizer,
            targets=[grid_component], batch_size=batch_size, token_position=at_offset,
        )
        grids.append(extract_steering_vectors(ds_acts, [grid_component])[grid_component])
    return torch.stack(grids, dim=0)


def filter_and_pick(
    scored: list[tuple[int, int, float, dict[str, float]]],
    n_layers: int,
    *,
    constraints: list[tuple[str, str, float]],
    prune_layer_frac: float,
) -> dict[str, Any]:
    """Pure selection over ``[(pos, layer, objective, values), ...]`` (Arditi App. C, generalised).

    Keep candidates that are finite, not in the last ``prune_layer_frac`` of layers, and satisfy
    every ``(values-key, "<="|">=", threshold)`` constraint; among survivors take the **lowest**
    objective. Raises if nothing survives.
    """
    keep_below_layer = int(n_layers * (1.0 - prune_layer_frac))

    def survives(c: tuple[int, int, float, dict[str, float]]) -> bool:
        _, layer, objective, values = c
        if not math.isfinite(objective) or any(not math.isfinite(v) for v in values.values()):
            return False
        if layer >= keep_below_layer:
            return False
        for key, op, threshold in constraints:
            v = values[key]
            if op == "<=" and not v <= threshold:
                return False
            if op == ">=" and not v >= threshold:
                return False
        return True

    survivors = [c for c in scored if survives(c)]
    if not survivors:
        cons = ", ".join(f"{k}{op}{t}" for k, op, t in constraints)
        raise RuntimeError(
            "Direction selection: no candidate survived the filters "
            f"(layers<{keep_below_layer}, {cons}). Relax the selection thresholds."
        )
    pos, layer, objective, values = min(survivors, key=lambda c: c[2])
    return {
        "position": pos, "layer": layer, "objective": objective, **values,
        "n_candidates": len(scored), "n_survivors": len(survivors),
    }


@torch.no_grad()
def grid_select_source(
    experiment, model: SteeringModel, tokenizer, extraction_ds, refine_ds, cache_info: dict
) -> dict[str, Tensor] | None:
    """Select one direction via the task's :class:`SelectionPolicy` and broadcast it to every
    steering site. Caches the chosen direction as ``SELECTED_DIRECTION``."""
    config = experiment.config

    hit = experiment._try_cache_lookup(ArtifactType.SELECTED_DIRECTION)
    if hit is not None:
        vectors, _ = load_steering_vectors(hit.artifact_path)
        cache_info["selected_direction"] = {"status": "hit", "path": str(hit.artifact_path)}
        return vectors

    policy = experiment.task.selection_policy(model, tokenizer, refine_ds, config)
    if policy is None:
        raise RuntimeError(
            f"direction_source=grid_select needs a task selection policy, but task "
            f"'{experiment.task.task_name}' provides none (only self / pinned sourcing)."
        )

    components = list(config.targets)
    grid = candidate_grid(
        model, tokenizer, extraction_ds,
        n_positions=int(config.get("selection_positions", 1)),
        grid_component=config.get("selection_grid_component", "resid_pre"),
        batch_size=int(config.extract_batch_size),
    )
    n_pos, n_layers, _ = grid.shape

    scored: list[tuple[int, int, float, dict[str, float]]] = []
    for j in range(n_pos):
        for layer in range(n_layers):
            r = grid[j, layer]
            if not torch.isfinite(r).all() or float(r.norm()) < 1e-8:
                continue
            model.set_all_vectors(broadcast(r, components, n_layers))
            objective, values = policy.score(r, layer)
            scored.append((j, layer, objective, values))

    best = filter_and_pick(
        scored, n_layers,
        constraints=policy.constraints,
        prune_layer_frac=float(config.get("selection_prune_layer_frac", 0.2)),
    )
    selected = broadcast(grid[best["position"], best["layer"]], components, n_layers)
    meta = {
        "grid_component": config.get("selection_grid_component", "resid_pre"),
        "n_positions": n_pos, "n_layers": n_layers, **best,
    }
    summary = " ".join(f"{k}={best[k]:.3f}" for k in best if isinstance(best[k], float))
    print(
        f"  Selected layer={best['layer']} pos={best['position']} {summary} "
        f"({best['n_survivors']}/{best['n_candidates']} candidates survived)"
    )
    if config.use_cache:
        dest = experiment._prepare_cache_path(ArtifactType.SELECTED_DIRECTION)
        save_steering_vectors(selected, dest, metadata=meta)
        experiment._finalize_cache(ArtifactType.SELECTED_DIRECTION)
    cache_info["selected_direction"] = {"status": "miss", "selection": meta}
    return selected


# ── dispatch ──────────────────────────────────────────────────────────


def resolve_direction_source(config, extracted: dict[str, Tensor]) -> dict[str, Tensor]:
    """Thin sources (``self`` / pinned) → ``{component: (n_layers, ...)}`` from already-extracted
    per-site vectors. ``grid_select`` is handled by :func:`grid_select_source`, not here."""
    targets = list(config.targets)
    kind, component, layer = _parse_source(config.get("direction_source", "self"))
    if kind == "self":
        return {c: extracted[c] for c in targets}
    if kind == "pin":
        assert component is not None and layer is not None
        source = extracted[component]
        return broadcast(source[layer], targets, source.shape[0])
    raise ValueError("grid_select must be resolved by grid_select_source, not resolve_direction_source.")


def source_vectors(
    experiment, model: SteeringModel, tokenizer, extraction_ds, refine_ds, cache_info: dict
) -> dict[str, Tensor] | None:
    """Produce the per-site steering vectors for the configured ``direction_source``.

    Both generic refines (``none`` / ``gate_training``) call this; the result is what they hand
    to ``set_all_vectors`` before (optionally) training gates on top."""
    kind, _, _ = _parse_source(experiment.config.get("direction_source", "self"))
    if kind == "grid":
        return grid_select_source(experiment, model, tokenizer, extraction_ds, refine_ds, cache_info)
    extracted = run_extraction(
        experiment, model, tokenizer, extraction_ds, cache_info,
        targets=extraction_targets(experiment.config),
    )
    if extracted is None:
        return None
    return resolve_direction_source(experiment.config, extracted)


__all__ = [
    "broadcast",
    "candidate_grid",
    "extraction_targets",
    "filter_and_pick",
    "grid_select_source",
    "resolve_direction_source",
    "source_vectors",
]
