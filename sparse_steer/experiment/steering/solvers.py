"""Field solvers — how each field of the per-site steering config gets its value.

The steered model is `correction = operator(strength · gate · direction)` at every site. Three
fields decide it, and each is set by a *solver*:

- **operator** (steer | ablate) — declared by ``config.intervention``, baked into the SteeringModel
  at construction (not solved here).
- **direction** (per site) — solved by ``config.direction_source``:
    ``self`` / ``[component, layer]`` → ``resolve_direction_source`` (a *declare* solver: thin
    transforms of the extracted per-site vectors); ``grid_select`` → ``search.grid_select_source``
    (a *score-search* solver — propose candidates, score via the task, pick one, broadcast).
- **strength** (per site, 0 = inactive) — solved by ``config.refinement_method`` over
  ``STRENGTH_SOLVERS`` (+ task-contributed): ``none`` → ``_refine_none`` (*declare* — set the fixed
  scale, stop); ``gate_training`` → ``_refine_gate_training`` (*gradient* — train L0 gates × scale).
  An ITI-style ``probe_rule`` strength solver (per-head probe top-K → α·σ) would slot in here as a
  sibling.

``direction_source`` and ``refinement_method`` are therefore the direction-field and strength-field
solvers; ``select`` (grid) and ``probe`` differ from gate-training only in solver *kind* (search /
rule vs gradient), not in being a separate concept.
"""

import shutil
from typing import Any

from torch import Tensor

from sparse_steer.core.steering import SteeringModel
from sparse_steer.train import train_steering
from sparse_steer.utils.cache import ArtifactType

from .extract import run_extraction


# ── direction field: parsing + declare solver (self / pinned broadcast) ─────


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


def broadcast(vec: Tensor, components: list[str], n_layers: int) -> dict[str, Tensor]:
    """One direction → ``{component: (n_layers, d)}`` with every layer set to ``vec``.

    Used by the one-to-many direction solvers (pinned / grid_select): the same direction is applied
    at every steering site. Assumes ``vec`` matches each component's per-site shape (true for the
    residual taps, which all carry ``(d_model,)``)."""
    layered = vec.reshape(1, -1).expand(n_layers, vec.shape[-1]).clone()
    return {c: layered.clone() for c in components}


def resolve_direction_source(config, extracted: dict[str, Tensor]) -> dict[str, Tensor]:
    """*declare* direction solver — thin sources (``self`` / pinned) → ``{component: (n_layers, ...)}``
    from already-extracted per-site vectors. ``grid_select`` is handled by ``grid_select_source``."""
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
    """Run the configured direction solver → per-site steering vectors.

    Both strength solvers (``none`` / ``gate_training``) call this; the result is what they hand to
    ``set_all_vectors`` before (optionally) training gates on top."""
    kind, _, _ = _parse_source(experiment.config.get("direction_source", "self"))
    if kind == "grid":
        from .search import grid_select_source  # lazy: only the score-search solver needs it
        return grid_select_source(experiment, model, tokenizer, extraction_ds, refine_ds, cache_info)
    extracted = run_extraction(
        experiment, model, tokenizer, extraction_ds, cache_info,
        targets=extraction_targets(experiment.config),
    )
    if extracted is None:
        return None
    return resolve_direction_source(experiment.config, extracted)


# ── strength field: solvers (none = declare, gate_training = gradient) ──────
# A strength solver turns the loaded model + datasets into the refined model:
#   fn(experiment, model, tokenizer, extraction_ds, train_ds, output_dir)
#       -> (model, artifacts, cache_info)
# It first solves the direction (source_vectors → set), then sets strengths: a fixed scale (none)
# or trained L0 gates × scale (gate_training). Tasks may contribute more via
# TaskSpec.refinement_strategies(); the pools merge at dispatch.


def _refine_none(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """dense / caa / arditi reproduction: solve the direction(s) and set them; fixed strength, no
    training (the *declare* strength solver)."""
    cache_info: dict[str, Any] = {}
    steering_vectors = source_vectors(
        experiment, model, tokenizer, extraction_ds, train_ds, cache_info
    )
    if steering_vectors is not None:
        model.set_all_vectors(
            steering_vectors, normalize=experiment.config.normalize_steering_vectors
        )
    return model, {}, cache_info


def _refine_gate_training(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """sparse / gates_only / scale_only / shared_scale_only: solve + set the direction, then train
    the learnable gates/scale against the task objective (the *gradient* strength solver)."""
    artifacts: dict[str, str | None] = {}
    cache_info: dict[str, Any] = {}

    ss_hit = experiment._try_cache_lookup(ArtifactType.SPARSE_STEERING)
    if ss_hit is not None:
        model.load_steering(ss_hit.artifact_path)
        artifacts["steering_path"] = str(ss_hit.artifact_path)
        cache_info["steering"] = {"status": "hit", "path": str(ss_hit.artifact_path)}
        for name in ("gate_heatmap.png", "gate_animation.gif"):
            cached = ss_hit.artifact_path.parent / name
            if cached.is_file():
                shutil.copy2(cached, output_dir / name)
        return model, artifacts, cache_info

    steering_vectors = source_vectors(
        experiment, model, tokenizer, extraction_ds, train_ds, cache_info
    )
    if steering_vectors is not None:
        model.set_all_vectors(
            steering_vectors, normalize=experiment.config.normalize_steering_vectors
        )

    print("Training steering parameters...")
    train_steering(
        model, tokenizer, train_ds, experiment.config, output_dir=output_dir, task=experiment.task
    )

    ss_dest = experiment._prepare_cache_path(ArtifactType.SPARSE_STEERING)
    model.save_steering(ss_dest)
    for name in ("gate_heatmap.png", "gate_animation.gif"):
        src = output_dir / name
        if src.is_file():
            shutil.copy2(src, ss_dest.parent / name)
    steering_path = str(experiment._finalize_cache(ArtifactType.SPARSE_STEERING))
    print(f"Saved steering to {steering_path}")
    artifacts["steering_path"] = steering_path
    cache_info["steering"] = {"status": "miss"}
    return model, artifacts, cache_info


# Strength-field solvers, keyed by `refinement_method`. Tasks add more via refinement_strategies().
STRENGTH_SOLVERS = {
    "none": _refine_none,
    "gate_training": _refine_gate_training,
}


__all__ = [
    "STRENGTH_SOLVERS",
    "broadcast",
    "extraction_targets",
    "resolve_direction_source",
    "source_vectors",
]
