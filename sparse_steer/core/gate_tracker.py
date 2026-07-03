"""Visualisation only: track how effective steering strength changes throughout training."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.ma as ma
import torch
import torch.nn.functional as F
import wandb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from numpy import ndarray

from .steering import SteeringModel


def _masked_cmap(base: str = "viridis"):
    """Return a colormap that renders masked (closed-gate) values as black."""
    cmap = plt.get_cmap(base).copy()
    cmap.set_bad(color="black")
    return cmap


@dataclass
class GateSnapshots:
    steps: list[int] = field(default_factory=list)
    attn_norms: list[ndarray] | None = None  # each: (num_layers, num_heads)
    mlp_norms: list[ndarray] | None = None  # each: (num_layers,)
    resid_norms: list[ndarray] | None = None  # each: (num_layers, num_resid_components)
    layer_ids: list[int] = field(default_factory=list)
    resid_components: list[str] = field(default_factory=list)  # column labels, e.g. resid_mid/resid_post


class GateTracker:
    """Records effective steering norms per head/layer across training.

    Driven manually by the training loop via :meth:`snapshot`.
    """

    def __init__(self, model: SteeringModel, use_wandb: bool = False) -> None:
        self.use_wandb = use_wandb
        self.snapshots = GateSnapshots()

        self._attn_hooks: list = []  # each: SteeringHook
        self._mlp_hooks: list = []
        self._attn_sv_norms: list[torch.Tensor] = []  # each (num_heads,)
        self._mlp_svs: list[torch.Tensor] = []  # each (mlp_dim,)
        # residual taps carry a single scalar gate per (component, layer); group by
        # component so each becomes a column of a layer × component heatmap panel.
        self._resid_hooks: dict[str, dict[int, object]] = {}  # comp -> {layer: hook}
        self._resid_sv_norms: dict[str, dict[int, float]] = {}  # comp -> {layer: ‖v‖}

        layer_id_set: set[int] = set()
        for component, layer, hook in sorted(
            model.iter_hooks(), key=lambda x: (x[0], x[1])
        ):
            if component == "attention":
                self._attn_hooks.append(hook)
                self._attn_sv_norms.append(
                    hook.steering_vectors.float().cpu().norm(dim=-1)
                )
                layer_id_set.add(layer)
            elif component == "mlp":
                self._mlp_hooks.append(hook)
                self._mlp_svs.append(hook.steering_vectors.float().cpu())
                layer_id_set.add(layer)
            else:  # residual / resid_mid / resid_post — scalar gate per layer
                self._resid_hooks.setdefault(component, {})[layer] = hook
                self._resid_sv_norms.setdefault(component, {})[layer] = float(
                    hook.steering_vectors.float().cpu().norm()
                )
                layer_id_set.add(layer)

        self.snapshots.layer_ids = sorted(layer_id_set)
        self.snapshots.resid_components = sorted(self._resid_hooks)
        if self._attn_hooks:
            self.snapshots.attn_norms = []
        if self._mlp_hooks:
            self.snapshots.mlp_norms = []
        if self._resid_hooks:
            self.snapshots.resid_norms = []

        # Render only if there's at least one tracked component. (Residual is
        # mutually exclusive with attention/mlp, so in practice exactly one of the
        # three panels is populated.)
        self._renderable = bool(
            self._attn_hooks or self._mlp_hooks or self._resid_hooks
        )

    def _effective(self, hook) -> torch.Tensor:
        was_training = hook.training
        hook.eval()
        gate = hook.effective_weight(dtype=torch.float32, device=torch.device("cpu"))
        if was_training:
            hook.train()
        return gate

    def snapshot(self, step: int) -> None:
        if not self._renderable:
            return
        with torch.no_grad():
            if self._attn_hooks:
                rows = [
                    (self._effective(hook) * sv_norm).numpy()
                    for hook, sv_norm in zip(self._attn_hooks, self._attn_sv_norms)
                ]
                self.snapshots.attn_norms.append(np.stack(rows))
            if self._mlp_hooks:
                vals = [
                    (self._effective(hook) * sv).norm().item()
                    for hook, sv in zip(self._mlp_hooks, self._mlp_svs)
                ]
                self.snapshots.mlp_norms.append(np.array(vals))
            if self._resid_hooks:
                layer_ids = self.snapshots.layer_ids
                comps = self.snapshots.resid_components
                grid = np.zeros((len(layer_ids), len(comps)))
                for ci, comp in enumerate(comps):
                    for layer, hook in self._resid_hooks[comp].items():
                        eff = float(self._effective(hook).reshape(-1)[0])
                        grid[layer_ids.index(layer), ci] = (
                            eff * self._resid_sv_norms[comp][layer]
                        )
                self.snapshots.resid_norms.append(grid)
        self.snapshots.steps.append(step)

        if self.use_wandb:
            log_dict = {"gate_heatmap": wandb.Image(_build_heatmap_figure(self.snapshots))}
            plt.close("all")
            resid_hooks = [h for c in self._resid_hooks.values() for h in c.values()]
            shared = (
                self._attn_hooks + self._mlp_hooks + resid_hooks
            )[0]._shared_raw_scale
            if shared is not None:
                log_dict["scale/raw_param"] = shared.item()
                log_dict["scale/effective"] = F.softplus(shared).item()
            wandb.log(log_dict, step=step)

    def sparsity(self) -> float:
        """Realised L0 sparsity: fraction of gates closed (eval-mode effective weight == 0)."""
        hooks = self._attn_hooks + self._mlp_hooks + [
            h for c in self._resid_hooks.values() for h in c.values()
        ]
        if not hooks:
            return 0.0
        gates = torch.cat([self._effective(h).reshape(-1) for h in hooks])
        return float((gates == 0).float().mean())

    def max_steering_strength(self) -> float:
        """Largest effective steering norm (gate · scale · ‖v‖) over all sites, eval mode.

        Computed live from the gates (not the snapshot) so it is correct on any logging
        cadence, independent of when the animation snapshot was last taken."""
        vals = [0.0]
        for hook, svn in zip(self._attn_hooks, self._attn_sv_norms):
            vals.append(float((self._effective(hook) * svn).max()))
        for hook, sv in zip(self._mlp_hooks, self._mlp_svs):
            vals.append(float((self._effective(hook) * sv).norm()))
        for comp, layers in self._resid_hooks.items():
            for layer, hook in layers.items():
                vals.append(float(self._effective(hook).reshape(-1)[0]) * self._resid_sv_norms[comp][layer])
        return max(vals)

    def n_gates(self) -> int:
        """Total number of gate scalars across all steered sites."""
        hooks = self._attn_hooks + self._mlp_hooks + [
            h for c in self._resid_hooks.values() for h in c.values()
        ]
        return sum(int(self._effective(h).reshape(-1).numel()) for h in hooks)


def _make_layout(
    has_attn: bool,
    has_mlp: bool,
    has_resid: bool,
    n_layers: int,
    num_heads: int = 1,
    num_resid: int = 1,
):
    """Create figure with data axes + a dedicated colorbar axis."""
    height = max(4, n_layers * 0.3)
    # width ratios: [attn?, mlp?, resid?, colorbar]
    ratios = []
    if has_attn:
        ratios.append(max(num_heads, 1))
    if has_mlp:
        ratios.append(1)
    if has_resid:
        ratios.append(max(num_resid, 1))
    ratios.append(0.4)  # colorbar

    fig, all_axes = plt.subplots(
        1,
        len(ratios),
        gridspec_kw={"width_ratios": ratios},
        figsize=(sum(ratios) * 0.7 + 2, height),
    )
    all_axes = [all_axes] if len(ratios) == 1 else list(all_axes)

    ax_attn = ax_mlp = ax_resid = None
    idx = 0
    if has_attn:
        ax_attn = all_axes[idx]
        idx += 1
    if has_mlp:
        ax_mlp = all_axes[idx]
        idx += 1
    if has_resid:
        ax_resid = all_axes[idx]
        idx += 1
    cbar_ax = all_axes[idx]

    return fig, ax_attn, ax_mlp, ax_resid, cbar_ax


def _populate_axes(
    ax_attn,
    ax_mlp,
    ax_resid,
    cbar_ax,
    attn_data,
    mlp_data,
    resid_data,
    n_layers: int,
    layer_ids: list[int],
    resid_components: list[str],
    vmin: float,
    vmax: float,
    cmap,
):
    """Fill axes with data and add shared colorbar. Returns (im_attn, im_mlp, im_resid)."""
    im_attn = im_mlp = im_resid = None
    mappable = None

    if ax_attn is not None and attn_data is not None:
        im_attn = ax_attn.imshow(
            ma.masked_equal(attn_data, 0.0),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        mappable = im_attn
        ax_attn.set_yticks(range(n_layers))
        ax_attn.set_yticklabels(layer_ids)
        ax_attn.set_ylabel("Layer")
        ax_attn.set_xlabel("Head")
        ax_attn.set_title("Attention")

    if ax_mlp is not None and mlp_data is not None:
        im_mlp = ax_mlp.imshow(
            ma.masked_equal(mlp_data.reshape(-1, 1), 0.0),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        mappable = im_mlp
        ax_mlp.set_yticks(range(n_layers))
        if ax_attn is None:
            ax_mlp.set_yticklabels(layer_ids)
            ax_mlp.set_ylabel("Layer")
        else:
            ax_mlp.set_yticklabels([])
        ax_mlp.set_xticks([])
        ax_mlp.set_title("MLP")

    if ax_resid is not None and resid_data is not None:
        im_resid = ax_resid.imshow(
            ma.masked_equal(resid_data, 0.0),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        mappable = im_resid
        ax_resid.set_yticks(range(n_layers))
        if ax_attn is None and ax_mlp is None:  # residual is the leftmost panel
            ax_resid.set_yticklabels(layer_ids)
            ax_resid.set_ylabel("Layer")
        else:
            ax_resid.set_yticklabels([])
        ax_resid.set_xticks(range(len(resid_components)))
        ax_resid.set_xticklabels(
            [c.replace("resid_", "") for c in resid_components],
            rotation=45,
            ha="right",
        )
        ax_resid.set_title("Residual")

    cbar_ax.clear()
    if mappable is not None:
        plt.colorbar(mappable, cax=cbar_ax)

    return im_attn, im_mlp, im_resid


def _build_heatmap_figure(snapshots: GateSnapshots, step_idx: int = -1) -> Figure:
    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    has_resid = snapshots.resid_norms is not None and len(snapshots.resid_norms) > 0
    step = snapshots.steps[step_idx] if snapshots.steps else 0
    n_layers = len(snapshots.layer_ids)
    num_heads = snapshots.attn_norms[0].shape[1] if has_attn else 1
    num_resid = snapshots.resid_norms[0].shape[1] if has_resid else 1

    attn_data = snapshots.attn_norms[step_idx] if has_attn else None
    mlp_data = snapshots.mlp_norms[step_idx] if has_mlp else None
    resid_data = snapshots.resid_norms[step_idx] if has_resid else None

    vmax = 0.0
    for data in (attn_data, mlp_data, resid_data):
        if data is not None:
            vmax = max(vmax, float(data.max()))
    vmax = max(vmax, 1e-8)

    fig, ax_attn, ax_mlp, ax_resid, cbar_ax = _make_layout(
        has_attn, has_mlp, has_resid, n_layers, num_heads, num_resid
    )
    _populate_axes(
        ax_attn,
        ax_mlp,
        ax_resid,
        cbar_ax,
        attn_data,
        mlp_data,
        resid_data,
        n_layers,
        snapshots.layer_ids,
        snapshots.resid_components,
        0,
        vmax,
        _masked_cmap(),
    )
    fig.suptitle(f"Effective steering norm \u2014 step {step}")
    fig.tight_layout()
    return fig


def render_gate_heatmap(
    snapshots: GateSnapshots,
    output_path: str | Path,
    step_idx: int = -1,
) -> Path:
    output_path = Path(output_path)
    fig = _build_heatmap_figure(snapshots, step_idx=step_idx)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_gate_animation(
    snapshots: GateSnapshots,
    output_path: str | Path,
    fps: int = 4,
) -> Path:
    output_path = Path(output_path)

    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    has_resid = snapshots.resid_norms is not None and len(snapshots.resid_norms) > 0
    n_layers = len(snapshots.layer_ids)
    num_heads = snapshots.attn_norms[0].shape[1] if has_attn else 1
    num_resid = snapshots.resid_norms[0].shape[1] if has_resid else 1

    # global vmax for consistent color scale across frames
    vmax_attn = max(a.max() for a in snapshots.attn_norms) if has_attn else 0
    vmax_mlp = max(m.max() for m in snapshots.mlp_norms) if has_mlp else 0
    vmax_resid = max(r.max() for r in snapshots.resid_norms) if has_resid else 0
    vmax = max(vmax_attn, vmax_mlp, vmax_resid, 1e-8)
    cmap = _masked_cmap()

    fig, ax_attn, ax_mlp, ax_resid, cbar_ax = _make_layout(
        has_attn, has_mlp, has_resid, n_layers, num_heads, num_resid
    )
    im_attn, im_mlp, im_resid = _populate_axes(
        ax_attn,
        ax_mlp,
        ax_resid,
        cbar_ax,
        snapshots.attn_norms[0] if has_attn else None,
        snapshots.mlp_norms[0] if has_mlp else None,
        snapshots.resid_norms[0] if has_resid else None,
        n_layers,
        snapshots.layer_ids,
        snapshots.resid_components,
        0,
        vmax,
        cmap,
    )
    title = fig.suptitle(f"Effective steering norm \u2014 step {snapshots.steps[0]}")
    fig.tight_layout()

    def _update(frame_idx):
        if im_attn is not None:
            im_attn.set_data(ma.masked_equal(snapshots.attn_norms[frame_idx], 0.0))
        if im_mlp is not None:
            im_mlp.set_data(
                ma.masked_equal(snapshots.mlp_norms[frame_idx].reshape(-1, 1), 0.0)
            )
        if im_resid is not None:
            im_resid.set_data(ma.masked_equal(snapshots.resid_norms[frame_idx], 0.0))
        title.set_text(f"Effective steering norm \u2014 step {snapshots.steps[frame_idx]}")
        return []

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(snapshots.steps),
        interval=1000 // fps,
        blit=False,
    )
    anim.save(str(output_path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


__all__ = [
    "GateSnapshots",
    "GateTracker",
    "render_gate_heatmap",
    "render_gate_animation",
]
