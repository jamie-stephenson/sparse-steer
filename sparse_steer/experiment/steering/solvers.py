"""Field solvers — how each field of the per-site steering config gets its value.

Naming (genus → species). A **solver** is the genus: anything that resolves one field of the config.
Its two species, named distinctly so they're never confused:

  - a **source** resolves the *direction* field (this module's ``source_vectors`` dispatch),
  - a **refinement** resolves the *strength* field (the ``REFINEMENTS`` registry below).

The steered model is `correction = operator(strength · gate · direction)` at every site:

- **operator** (steer | ablate) — declared by ``config.intervention``, baked into the SteeringModel
  at construction (not resolved here).
- **direction** — chosen by the *source* ``config.direction_source``: ``self`` / ``[component,
  layer]`` → ``resolve_direction_source`` (declare: thin transforms of the extracted per-site
  vectors); ``grid_search`` → ``search.grid_search_source`` (score-search: propose, score, pick).
- **strength** (per site, 0 = inactive) — set by the *refinement* ``config.refinement_method`` over
  ``REFINEMENTS`` (+ a task's ``extra_refinements()``): ``none`` → ``_refine_none`` (declare — fixed
  scale, no training); ``gate_training`` → ``_refine_gate_training`` (gradient — train L0 gates ×
  scale); ``iti_head_select`` → ``_refine_iti_head_select`` (rule — per-site probe top-K → α·σ).

So ``select`` (grid source), ``gate_training`` and ``iti_head_select`` (refinements) differ only in
*kind* (search / gradient / rule), not in being separate concepts — they all resolve one field.
"""

import shutil
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from sparse_steer.core.steering import SteeringModel
from sparse_steer.train import train_steering
from sparse_steer.utils.cache import ArtifactType

from .extract import run_extraction


# ── direction field: parsing + declare solver (self / fixed broadcast) ─────


def _parse_source(direction_source: Any) -> tuple[str, str | None, int | None]:
    """``direction_source`` → ``(kind, component, layer)`` with ``kind`` in
    ``{"self", "fixed", "grid"}``."""
    if direction_source == "self":
        return "self", None, None
    if direction_source == "grid_search":
        return "grid", None, None
    seq = list(direction_source)
    if len(seq) != 2:
        raise ValueError(
            "direction_source must be 'self', 'grid_search', or a [component, layer] pair, "
            f"got {direction_source!r}."
        )
    return "fixed", str(seq[0]), int(seq[1])


def extraction_targets(config) -> list[str]:
    """Component taps to extract: the steering ``targets``, plus a fixed source's component if
    that isn't already a steering target (so the broadcast direction is available to read)."""
    targets = list(config.targets)
    kind, component, _ = _parse_source(config.get("direction_source", "self"))
    if kind == "fixed" and component not in targets:
        targets = targets + [component]
    return targets


def broadcast(vec: Tensor, components: list[str], n_layers: int) -> dict[str, Tensor]:
    """One direction → ``{component: (n_layers, d)}`` with every layer set to ``vec``.

    Used by the one-to-many direction solvers (fixed / grid_search): the same direction is applied
    at every steering site. Assumes ``vec`` matches each component's per-site shape (true for the
    residual taps, which all carry ``(d_model,)``)."""
    layered = vec.reshape(1, -1).expand(n_layers, vec.shape[-1]).clone()
    return {c: layered.clone() for c in components}


def resolve_direction_source(config, extracted: dict[str, Tensor]) -> dict[str, Tensor]:
    """*declare* direction solver — thin sources (``self`` / fixed) → ``{component: (n_layers, ...)}``
    from already-extracted per-site vectors. ``grid_search`` is handled by ``grid_search_source``."""
    targets = list(config.targets)
    kind, component, layer = _parse_source(config.get("direction_source", "self"))
    if kind == "self":
        return {c: extracted[c] for c in targets}
    if kind == "fixed":
        assert component is not None and layer is not None
        source = extracted[component]
        return broadcast(source[layer], targets, source.shape[0])
    raise ValueError("grid_search must be resolved by grid_search_source, not resolve_direction_source.")


def source_vectors(
    experiment, model: SteeringModel, tokenizer, extraction_ds, refine_ds, cache_info: dict
) -> dict[str, Tensor] | None:
    """Run the configured direction solver → per-site steering vectors.

    Both strength solvers (``none`` / ``gate_training``) call this; the result is what they hand to
    ``set_all_vectors`` before (optionally) training gates on top."""
    kind, _, _ = _parse_source(experiment.config.get("direction_source", "self"))
    if kind == "grid":
        from .search import grid_search_source  # lazy: only the score-search solver needs it
        return grid_search_source(experiment, model, tokenizer, extraction_ds, refine_ds, cache_info)
    extracted = run_extraction(
        experiment, model, tokenizer, extraction_ds, cache_info,
        targets=extraction_targets(experiment.config),
    )
    if extracted is None:
        return None
    return resolve_direction_source(experiment.config, extracted)


# ── strength field: solvers (none = declare, gate_training = gradient) ──────
# A refinement turns the loaded model + datasets into the refined model:
#   fn(experiment, model, tokenizer, extraction_ds, train_ds, output_dir)
#       -> (model, artifacts, cache_info)
# It first picks the direction (source_vectors → set), then sets strengths: a fixed scale (none),
# trained L0 gates × scale (gate_training), or probe-selected α·σ (iti_head_select). Tasks may
# contribute more via TaskSpec.extra_refinements(); the pools merge at dispatch.


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


# ITI gate-mask logit: ±this → the eval hard-concrete gate is a deterministic ≈1 / ≈0.
_ITI_MASK_LOGIT = 10.0


def _inv_softplus(y: Tensor) -> Tensor:
    """softplus⁻¹: the raw_scale whose softplus equals the per-head magnitude ``y`` (= α·σ)."""
    return torch.log(torch.expm1(y.clamp_min(1e-6)))


def _refine_iti_head_select(experiment, model, tokenizer, extraction_ds, train_ds, output_dir):
    """ITI-style probe-rule refinement (Li et al. 2023), generalised off attention.

    For each configured target, fit a probe per *site* — a site is one gate: an attention head, or a
    whole residual/mlp layer — on the unsteered per-site activations, rank ALL sites (across all
    targets) by validation accuracy, take the top-K, and shift each along its mass-mean direction by
    ``α·σ``. The selection (mask) is written into each site's ``log_alpha`` and the magnitude into
    ``raw_scale``; nothing is trained. The ITI paper fixes ``targets=[attention]``, but the mechanism
    applies to any per-site target (resid_mid, resid_post, mlp, …) — that's a config choice here.

    Faithful baseline: ITI *probes* the sites (supervised); our own method instead *learns* them via
    L0 gates and never probes — so this deliberately uses a mechanism our constraints forbid for us.
    """
    from sparse_steer.core.extract import collect_activations  # local: only ITI needs it
    from .probe import fit_head_probes, head_sigma

    config = experiment.config
    cache_info: dict[str, Any] = {}
    components = list(config.targets)

    steering_vectors = source_vectors(
        experiment, model, tokenizer, extraction_ds, train_ds, cache_info
    )
    if steering_vectors is None:
        raise ValueError("iti_head_select needs extracted per-site directions (config.targets).")

    # Per-example per-site activations on the UNSTEERED model + truthful labels, for the probes.
    with model.steering_disabled():
        ds_acts, _ = collect_activations(
            extraction_ds, model, tokenizer,
            targets=components,
            batch_size=int(config.extract_batch_size),
            token_position=config.extract_token_position,
        )
    positive = torch.tensor(ds_acts["positive"])   # (n,) bool

    # Deploy the unit mass-mean directions; σ is measured along those same unit directions.
    model.set_all_vectors(steering_vectors, normalize=True)

    # Per target: probe val-accuracy + σ per site, normalised to (L, H_c) with H_c=1 for single-gate
    # (residual/mlp) targets. Then a GLOBAL top-K across all sites of all targets.
    alpha = float(config.get("iti_alpha", 15.0))
    sigma_by_c: dict[str, Tensor] = {}
    sites: list[tuple[float, str, int, int]] = []  # (val_acc, component, layer, gate)
    for c in components:
        acts_c = torch.tensor(ds_acts[c]).float()
        if acts_c.dim() == 3:                      # residual/mlp: (n, L, D) → one gate per layer
            acts_c = acts_c.unsqueeze(2)           # (n, L, 1, D)
        dir_c = steering_vectors[c].float()
        if dir_c.dim() == 2:                       # (L, D) → (L, 1, D)
            dir_c = dir_c.unsqueeze(1)
        dir_c = F.normalize(dir_c, dim=-1)
        acc_c = fit_head_probes(acts_c, positive)  # (L, H_c)
        sigma_by_c[c] = head_sigma(acts_c, dir_c)  # (L, H_c)
        Lc, Hc = acc_c.shape
        for layer in range(Lc):
            for gate in range(Hc):
                sites.append((float(acc_c[layer, gate]), c, layer, gate))

    sites.sort(key=lambda s: -s[0])
    k = min(int(config.get("iti_num_heads", 48)), len(sites))
    selected = {(c, layer, gate) for _, c, layer, gate in sites[:k]}
    top_acc = sum(s[0] for s in sites[:k]) / k if k else 0.0

    for c in components:
        sigma_c = sigma_by_c[c]
        Lc, Hc = sigma_c.shape
        for layer in range(Lc):
            key = f"{c}_{layer}"
            if key not in model.hooks:
                continue
            hook = model.hooks[key]
            mask_l = torch.tensor([(c, layer, gate) in selected for gate in range(Hc)])
            raw_l = _inv_softplus(alpha * sigma_c[layer])
            with torch.no_grad():
                hook.log_alpha.data = torch.where(
                    mask_l.to(hook.log_alpha.device),
                    torch.full_like(hook.log_alpha.data, _ITI_MASK_LOGIT),
                    torch.full_like(hook.log_alpha.data, -_ITI_MASK_LOGIT),
                )
                hook.raw_scale.data = raw_l.to(hook.raw_scale.device, hook.raw_scale.dtype)
            hook.log_alpha.requires_grad_(False)
            if isinstance(getattr(hook, "raw_scale", None), torch.nn.Parameter):
                hook.raw_scale.requires_grad_(False)

    print(f"  ITI: {k} sites selected across {components} (mean top-K probe val-acc {top_acc:.3f}); α·σ set.")
    cache_info["iti"] = {"status": "miss", "n_sites": k, "top_val_acc": top_acc, "components": components}
    return model, {}, cache_info


# Strength-field refinements, keyed by `refinement_method`. Tasks add more via extra_refinements().
REFINEMENTS = {
    "none": _refine_none,
    "gate_training": _refine_gate_training,
    "iti_head_select": _refine_iti_head_select,
}


__all__ = [
    "REFINEMENTS",
    "broadcast",
    "extraction_targets",
    "resolve_direction_source",
    "source_vectors",
]
