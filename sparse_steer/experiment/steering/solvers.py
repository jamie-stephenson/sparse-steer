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
    """fixed / caa / arditi reproduction: solve the direction(s) and set them; fixed strength, no
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

    # Opt-in honest_llama-style σ-init (OFF by default; ITI uses it intrinsically). When a sparse
    # run sets scale_from_extraction_std=true, each site's scale STARTS at iti_scale·σ (the per-site
    # ITI magnitude) instead of the flat init_raw_scale, and the gates train from there. Default
    # false → sparse steering is unchanged.
    if steering_vectors is not None and bool(
        experiment.config.get("scale_from_extraction_std", False)
    ):
        from sparse_steer.core.extract import collect_activations  # local: only this path needs it
        from .probe import head_sigma

        cfg = experiment.config
        components = list(cfg.targets)
        scale = float(cfg.get("iti_scale", 15.0))
        with model.steering_disabled():
            ds_acts, _ = collect_activations(
                extraction_ds, model, tokenizer, targets=components,
                batch_size=int(cfg.get("extract_batch_size", 8)),
                token_position=cfg.get("extract_token_position", "last"),
            )
        sigma_position = str(cfg.get("iti_sigma_position", "answer"))
        qend_acts = (
            _sigma_population_acts(extraction_ds, model, tokenizer, cfg, components, sigma_position)
            if sigma_position in ("question_end", "gen_end_q") else None
        )
        for c in components:
            acts_c = torch.tensor(ds_acts[c]).float()
            if acts_c.dim() == 3:
                acts_c = acts_c.unsqueeze(2)
            dir_c = steering_vectors[c].float()
            if dir_c.dim() == 2:
                dir_c = dir_c.unsqueeze(1)
            sigma_src = qend_acts[c] if qend_acts is not None else acts_c
            sigma_c = head_sigma(sigma_src, F.normalize(dir_c, dim=-1))
            for layer in range(sigma_c.shape[0]):
                key = f"{c}_{layer}"
                if key in model.hooks:
                    hook = model.hooks[key]
                    with torch.no_grad():
                        hook.raw_scale.data = _inv_softplus(scale * sigma_c[layer]).to(
                            hook.raw_scale.device, hook.raw_scale.dtype
                        )
        print(f"  σ-init: per-site scale set to {scale}·σ (scale_from_extraction_std=true)")

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


# σ-population modes routed through _sigma_population_acts (independent of the vector's
# extract_token_position). "answer" is NOT here — it reuses the extraction acts in the caller.
_SIGMA_POP_MODES = (
    "completion_final", "completion", "prompt_final", "prompt_final_extra_q", "gen_end_q", "question_end",
)


def _sigma_population_acts(extraction_ds, model, tokenizer, config, components, mode):
    """Activations for the α·σ magnitude population (``iti_sigma_position``), collected INDEPENDENTLY of
    the vector's ``extract_token_position``. σ = std of these projected on the unit com direction; head
    SELECTION never uses this population, only the σ magnitude does. Modes (map to iti_qa & chat):

    Positions read directly on the extraction ``Q: A: {ans}`` text (via positions_mask):
    - ``"completion_final"``: the answer's last token.
    - ``"completion"``: the per-answer completion MEAN (collect_activations mean-pools the mask), then std
      across answers. (NB: not the std over every individual completion token — that reuse of the pooled
      path is a deliberate simplification.)
    - ``"prompt_final"``: the final prompt token — the ``:`` of ``A:`` (iti_qa) / the token after ``[/INST]`` (chat).
    Constructed texts (read at the last token):
    - ``"prompt_final_extra_q"``: append a random trailing question + ``A:`` → ``Q: A: {ans} Q: {rand} A:``,
      read the final ``A:`` colon (the true generation-onset for a fresh question). chat: a 2nd ``[INST]…[/INST]`` turn.
    - ``"gen_end_q"`` (honest_llama-faithful): ``Q: A: {ans} Q: {rand}``, last token (end of the random question).
    - ``"question_end"``: ``Q: A:`` last token (≈ prompt_final via the question-only template).
    """
    import numpy as np

    from datasets import Dataset

    from sparse_steer.core.extract import collect_activations
    from sparse_steer.utils.tokenize import apply_template

    template = config.get("extraction_template") or config.get("prompt_template", "chat")
    bs = int(config.get("extract_batch_size", 8))

    def _run(ds, token_position):
        with model.steering_disabled():
            qacts, _ = collect_activations(
                ds, model, tokenizer, targets=components, batch_size=bs, token_position=token_position,
            )
        out: dict[str, Tensor] = {}
        for c in components:
            a = torch.tensor(qacts[c]).float()
            if a.dim() == 3:  # residual/mlp (n, L, D) → (n, L, 1, D)
                a = a.unsqueeze(2)
            out[c] = a
        return out

    # answer-position modes read the extraction Q+A text at that position (anchor is irrelevant here).
    if mode in ("completion_final", "completion"):
        ds = Dataset.from_dict(
            {"text": list(extraction_ds["text"]), "prompt_len": list(extraction_ds["prompt_len"])}
        )
        return _run(ds, mode)

    # sigma_prompt_anchor toggles the READ-POINT of the prompt-based modes (affects σ only, never the vector):
    #   "answer_colon" (default): the answer marker — the ":" of "A:" (iti_qa) / after "[/INST]" (chat). [our "A:" variant]
    #   "question_end": the QUESTION's last token, BEFORE the marker — what honest_llama does (its gen_end_q point).
    # For chat there is no bare question-end ("[/INST]" always closes the turn), so the anchor is a no-op there.
    # Back-compat aliases: gen_end_q == prompt_final_extra_q @ question_end; question_end == prompt_final @ answer_colon.
    anchor = str(config.get("sigma_prompt_anchor", "answer_colon"))
    if mode == "gen_end_q":
        mode, anchor = "prompt_final_extra_q", "question_end"
    elif mode == "question_end":
        mode, anchor = "prompt_final", "answer_colon"
    if mode not in ("prompt_final", "prompt_final_extra_q"):
        raise ValueError(f"unknown iti_sigma_position mode {mode!r}")

    is_chat = template == "chat"
    add_marker = anchor == "answer_colon"
    questions = list(dict.fromkeys(extraction_ds["question"]))
    rng = np.random.RandomState(int(config.get("seed", 42)))
    rand_for: dict[int, str] = {}
    texts = []
    for qid, q, txt in zip(
        extraction_ds["question_id"], extraction_ds["question"], extraction_ds["text"]
    ):
        if mode == "prompt_final":  # the ORIGINAL question
            texts.append(
                apply_template(tokenizer, q, None, template="chat") if is_chat
                else (f"Q: {q} A:" if add_marker else f"Q: {q}")
            )
        else:  # prompt_final_extra_q: a random APPENDED question after the Q+A
            rq = rand_for.setdefault(qid, questions[rng.randint(len(questions))])
            if is_chat:
                texts.append(f"{txt}{apply_template(tokenizer, rq, None, template='chat')}")
            else:
                base = f"{txt} Q: {rq}"
                texts.append(f"{base} A:" if add_marker else base)
    return _run(Dataset.from_dict({"text": texts}), "last")


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

    if str(config.extract_token_position) in ("prompt", "prompt_final"):
        raise ValueError(
            "extract_token_position (the STEERING VECTOR position) must be a completion position "
            "(completion / completion_final): true and false answers share the same prompt, so a "
            "prompt-position mass-mean direction is null. Prompt-based positions are for σ only "
            "(iti_sigma_position=prompt_final / prompt_final_extra_q)."
        )
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

    # σ population (iti_sigma_position): "answer" (default = current sparse_steer) measures σ on the
    # extraction answer-token activations; "question_end" measures it on the question-end activations
    # (honest_llama's tqa_gen_end_q analogue) — see _question_end_sigma_acts. Head SELECTION always
    # uses the discriminative answer activations (the truthful-vs-false probe signal); only the σ
    # MAGNITUDE population changes.
    sigma_position = str(config.get("iti_sigma_position", "answer"))
    qend_acts = (
        _sigma_population_acts(extraction_ds, model, tokenizer, config, components, sigma_position)
        if sigma_position in _SIGMA_POP_MODES else None
    )

    # Per target: probe val-accuracy + σ per site, normalised to (L, H_c) with H_c=1 for single-gate
    # (residual/mlp) targets. Then a GLOBAL top-K across all sites of all targets.
    scale = float(config.get("iti_scale", 15.0))  # ITI's α: a magnitude multiplier (not our log_alpha)
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
        acc_c = fit_head_probes(acts_c, positive)  # (L, H_c) — selection on the answer activations
        sigma_src = qend_acts[c] if qend_acts is not None else acts_c
        sigma_by_c[c] = head_sigma(sigma_src, dir_c)  # (L, H_c)
        Lc, Hc = acc_c.shape
        for layer in range(Lc):
            for gate in range(Hc):
                sites.append((float(acc_c[layer, gate]), c, layer, gate))

    sites.sort(key=lambda s: -s[0])
    k = min(int(config.get("iti_topk", 48)), len(sites))
    selected = {(c, layer, gate) for _, c, layer, gate in sites[:k]}
    top_acc = sum(s[0] for s in sites[:k]) / k if k else 0.0

    # σ-scaling (per-site magnitude = α·σ) is ITI's mechanism — ON by default here, controlled by
    # scale_from_extraction_std (iti.yaml sets it true). A run may set it false to use the flat
    # init_raw_scale instead. Sparse steering leaves this off by default (see _refine_gate_training).
    sigma_scale = bool(config.get("scale_from_extraction_std", True))
    for c in components:
        sigma_c = sigma_by_c[c]
        Lc, Hc = sigma_c.shape
        for layer in range(Lc):
            key = f"{c}_{layer}"
            if key not in model.hooks:
                continue
            hook = model.hooks[key]
            mask_l = torch.tensor([(c, layer, gate) in selected for gate in range(Hc)])
            with torch.no_grad():
                hook.log_alpha.data = torch.where(
                    mask_l.to(hook.log_alpha.device),
                    torch.full_like(hook.log_alpha.data, _ITI_MASK_LOGIT),
                    torch.full_like(hook.log_alpha.data, -_ITI_MASK_LOGIT),
                )
                if sigma_scale:
                    raw_l = _inv_softplus(scale * sigma_c[layer])
                    hook.raw_scale.data = raw_l.to(hook.raw_scale.device, hook.raw_scale.dtype)
            hook.log_alpha.requires_grad_(False)
            if isinstance(getattr(hook, "raw_scale", None), torch.nn.Parameter):
                hook.raw_scale.requires_grad_(False)

    mags = [scale * float(sigma_by_c[c][layer, gate]) for _, c, layer, gate in sites[:k]]
    mag_mean = sum(mags) / len(mags) if mags else 0.0
    mag_max = max(mags) if mags else 0.0
    sel_heads = sorted((c, layer, gate) for _, c, layer, gate in sites[:k])
    print(
        f"  ITI: {k} sites selected across {components} (mean top-K probe val-acc {top_acc:.3f}); "
        f"σ_population={sigma_position}; α·σ magnitude mean={mag_mean:.2f} max={mag_max:.2f} (α={scale})"
    )
    print(f"  ITI selected sites: {sel_heads}")
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
