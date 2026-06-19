"""Score-search direction solver (Arditi App. C, generalised).

The *fat* direction solver: it proposes a grid of candidate directions, scores each by broadcasting
it onto every site and asking the task's selection policy (which reads the candidate-applied model),
then picks one and broadcasts it. Unlike the *declare* sources (self / fixed in ``solvers.py``) it
needs the refine dataset and a task scorer; that coupling is contained here. The result is the same
``{component: (n_layers, ...)}`` dict the declare solvers return. Was the grid_search half of
``experiment/sourcing.py``.
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

from .solvers import broadcast


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
def grid_search_source(
    experiment, model: SteeringModel, tokenizer, extraction_ds, refine_ds, cache_info: dict
) -> dict[str, Tensor] | None:
    """Select one direction via the task's ``SelectionPolicy`` and broadcast it to every steering
    site. Caches the chosen direction as ``SELECTED_DIRECTION``."""
    config = experiment.config

    hit = experiment._try_cache_lookup(ArtifactType.SELECTED_DIRECTION)
    if hit is not None:
        vectors, _ = load_steering_vectors(hit.artifact_path)
        cache_info["selected_direction"] = {"status": "hit", "path": str(hit.artifact_path)}
        return vectors

    policy = experiment.task.selection_policy(model, tokenizer, refine_ds, config)
    if policy is None:
        raise RuntimeError(
            f"direction_source=grid_search needs a task selection policy, but task "
            f"'{experiment.task.task_name}' provides none (only self / fixed sourcing)."
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


__all__ = ["candidate_grid", "filter_and_pick", "grid_search_source"]
