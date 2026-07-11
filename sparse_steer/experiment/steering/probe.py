"""Probe primitives for the ITI refinement (Inference-Time Intervention, Li et al. 2023).

ITI's defining step: fit a linear probe per *site* (an attention head, or — generalised — any
per-site target) to classify truthful vs untruthful from that site's activation, rank sites by
validation accuracy, take the top-K, and shift each along its mass-mean direction by ``α·σ`` (σ = std
of the site's activation projected on the direction). This module supplies the two reusable
primitives — ``fit_head_probes`` (per-site held-out accuracy) and ``head_sigma`` (per-site σ); the
cross-target ranking / top-K / param-writing lives in ``solvers._refine_iti_head_select`` (it pools
sites across ``config.targets``). It's the strength-field analogue of ``search.py`` (the score-search
source).

We learn the sites via supervised probes (ITI's mechanism) — a faithful *baseline*, distinct from our
own method, which learns the site via L0 gates and never probes.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def fit_head_probes(
    acts: Tensor,
    positive: Tensor,
    *,
    val_frac: float = 0.3,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    device: str | None = None,
) -> Tensor:
    """Fit one logistic probe per attention head (all ``L·H`` in parallel) and return per-head
    **validation** accuracy ``(L, H)``.

    ``acts`` is ``(n, L, H, D)`` per-example per-head activations; ``positive`` is ``(n,)`` bool
    (truthful). Heads are ranked by held-out accuracy so the *selection* doesn't overfit. Features
    are standardised with train statistics; probes are LH independent linear classifiers trained by
    Adam on BCE.

    ``device`` — where the probe optimisation runs (``iti_probe_device`` in config):
      * ``None``/"auto" (DEFAULT): CUDA when available, else CPU. Extracted activations arrive on
        CPU, and the 300 Adam steps over an ``(n_tr, L·H, D)`` tensor were historically the
        20-minute CPU bottleneck of every ITI fit (GPU sat idle holding the model). Moving the
        probe fit to the GPU turns that phase into seconds — the maths is identical.
      * "cpu": force the legacy CPU path. GPU float reductions are not bit-identical to CPU, so a
        near-tied head at the top-K selection boundary can differ between devices. Use "cpu" to
        reproduce study-era ITI fits (all fits before 2026-07-11 ran on CPU) bit-for-bit; use the
        default everywhere else. Val-accuracy differences between devices are float-noise (~1e-6);
        the smoke test for this option checks selection agreement on real activations.
    The returned accuracies are always on CPU. The device is NOT part of the artifact cache key —
    it is a numerics-level knob (like the eval backend), not an experimental condition.
    """
    n, L, H, D = acts.shape
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)  # CPU generator: split identical across devices
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X = acts.reshape(n, L * H, D).float()
    y = positive.float()
    Xtr, Xval = X[tr_idx].to(device), X[val_idx].to(device)
    ytr, yval = y[tr_idx].to(device), y[val_idx].to(device)

    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xval = (Xval - mu) / sd

    W = torch.zeros(L * H, D, requires_grad=True, device=device)
    b = torch.zeros(L * H, requires_grad=True, device=device)
    opt = torch.optim.Adam([W, b], lr=lr)
    ytr_e = ytr.unsqueeze(1).expand(-1, L * H)
    for _ in range(steps):
        opt.zero_grad()
        logits = torch.einsum("nhd,hd->nh", Xtr, W) + b  # (n_tr, LH)
        loss = F.binary_cross_entropy_with_logits(logits, ytr_e)
        loss.backward()
        opt.step()

    with torch.no_grad():
        vlogits = torch.einsum("nhd,hd->nh", Xval, W) + b  # (n_val, LH)
        pred = (vlogits > 0).float()
        acc = (pred == yval.unsqueeze(1)).float().mean(0)  # (LH,)
    return acc.reshape(L, H).cpu()


def head_sigma(acts: Tensor, directions: Tensor) -> Tensor:
    """Per-head σ ``(L, H)`` — std over examples of each head activation projected on its unit
    direction. ``acts`` ``(n, L, H, D)``, ``directions`` ``(L, H, D)`` (unit)."""
    proj = torch.einsum("nlhd,lhd->nlh", acts.float(), directions.float())  # (n, L, H)
    return proj.std(0)


__all__ = ["fit_head_probes", "head_sigma"]
