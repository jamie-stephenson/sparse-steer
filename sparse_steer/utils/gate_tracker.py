from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.ma as ma
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from numpy import ndarray
from transformers import TrainerCallback

from ..models.sparse import SparseSteeringHook


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
    layer_ids: list[int] = field(default_factory=list)


class GateTracker(TrainerCallback):
    """Records effective steering norms per head/layer across training."""

    def __init__(self, model, use_wandb: bool = False) -> None:
        self.use_wandb = use_wandb
        self.snapshots = GateSnapshots()

        self._attn_hooks: list[tuple[int, SparseSteeringHook]] = []
        self._mlp_hooks: list[tuple[int, SparseSteeringHook]] = []
        self._attn_sv_norms: list[torch.Tensor] = []  # each (num_heads,)
        self._mlp_svs: list[torch.Tensor] = []  # each (mlp_dim,)

        layers = model.get_layers()
        layer_id_set: set[int] = set()

        for i, layer in enumerate(layers):
            attn_hook = getattr(layer, f"_steering_attention_{i}", None)
            if isinstance(attn_hook, SparseSteeringHook):
                self._attn_hooks.append((i, attn_hook))
                self._attn_sv_norms.append(
                    attn_hook.steering_vectors.float().cpu().norm(dim=-1)
                )
                layer_id_set.add(i)

            mlp_hook = getattr(layer, f"_steering_mlp_{i}", None)
            if isinstance(mlp_hook, SparseSteeringHook):
                self._mlp_hooks.append((i, mlp_hook))
                self._mlp_svs.append(mlp_hook.steering_vectors.float().cpu())
                layer_id_set.add(i)

        self.snapshots.layer_ids = sorted(layer_id_set)
        if self._attn_hooks:
            self.snapshots.attn_norms = []
        if self._mlp_hooks:
            self.snapshots.mlp_norms = []

    def _snapshot(self, step: int) -> None:
        with torch.no_grad():
            if self._attn_hooks:
                rows = []
                for (_, hook), sv_norm in zip(self._attn_hooks, self._attn_sv_norms):
                    was_training = hook.training
                    hook.eval()
                    gate = hook._scaled_gate(
                        dtype=torch.float32, device=torch.device("cpu")
                    )
                    if was_training:
                        hook.train()
                    rows.append((gate * sv_norm).numpy())
                self.snapshots.attn_norms.append(np.stack(rows))

            if self._mlp_hooks:
                vals = []
                for (_, hook), sv in zip(self._mlp_hooks, self._mlp_svs):
                    was_training = hook.training
                    hook.eval()
                    gate = hook._scaled_gate(
                        dtype=torch.float32, device=torch.device("cpu")
                    )
                    if was_training:
                        hook.train()
                    vals.append((gate * sv).norm().item())
                self.snapshots.mlp_norms.append(np.array(vals))

        self.snapshots.steps.append(step)

    def on_log(self, args, state, control, **kwargs):
        self._snapshot(state.global_step)
        if self.use_wandb:
            import wandb

            fig = _build_heatmap_figure(self.snapshots)
            wandb.log({"gate_heatmap": wandb.Image(fig)}, step=state.global_step)
            plt.close(fig)

    def on_train_end(self, args, state, control, **kwargs):
        self._snapshot(state.global_step)


def _make_layout(
    has_attn: bool,
    has_mlp: bool,
    n_layers: int,
    num_heads: int = 1,
):
    """Create figure with data axes + a dedicated colorbar axis."""
    height = max(4, n_layers * 0.3)
    # width ratios: [attn?, mlp?, colorbar]
    ratios = []
    if has_attn:
        ratios.append(max(num_heads, 1))
    if has_mlp:
        ratios.append(1)
    ratios.append(0.4)  # colorbar

    fig, all_axes = plt.subplots(
        1,
        len(ratios),
        gridspec_kw={"width_ratios": ratios},
        figsize=(sum(ratios) * 0.7 + 2, height),
    )
    if len(ratios) == 2:
        all_axes = [all_axes[0], all_axes[1]]
    else:
        all_axes = list(all_axes)

    ax_attn = ax_mlp = None
    idx = 0
    if has_attn:
        ax_attn = all_axes[idx]
        idx += 1
    if has_mlp:
        ax_mlp = all_axes[idx]
        idx += 1
    cbar_ax = all_axes[idx]

    return fig, ax_attn, ax_mlp, cbar_ax


def _populate_axes(
    ax_attn,
    ax_mlp,
    cbar_ax,
    attn_data,
    mlp_data,
    n_layers: int,
    layer_ids: list[int],
    vmin: float,
    vmax: float,
    cmap,
):
    """Fill axes with data and add shared colorbar. Returns (im_attn, im_mlp)."""
    im_attn = im_mlp = None
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

    cbar_ax.clear()
    if mappable is not None:
        plt.colorbar(mappable, cax=cbar_ax)

    return im_attn, im_mlp


def _build_heatmap_figure(snapshots: GateSnapshots, step_idx: int = -1) -> Figure:
    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    step = snapshots.steps[step_idx] if snapshots.steps else 0
    n_layers = len(snapshots.layer_ids)
    num_heads = snapshots.attn_norms[0].shape[1] if has_attn else 1

    attn_data = snapshots.attn_norms[step_idx] if has_attn else None
    mlp_data = snapshots.mlp_norms[step_idx] if has_mlp else None

    vmax = 0.0
    if attn_data is not None:
        vmax = max(vmax, attn_data.max())
    if mlp_data is not None:
        vmax = max(vmax, mlp_data.max())
    vmax = max(vmax, 1e-8)

    fig, ax_attn, ax_mlp, cbar_ax = _make_layout(has_attn, has_mlp, n_layers, num_heads)
    _populate_axes(
        ax_attn,
        ax_mlp,
        cbar_ax,
        attn_data,
        mlp_data,
        n_layers,
        snapshots.layer_ids,
        0,
        vmax,
        _masked_cmap(),
    )
    fig.suptitle(f"Effective steering norm \u2014 step {step}")
    fig.tight_layout()
    return fig


def render_gate_heatmap(
    snapshots: GateSnapshots, output_path: str | Path, step_idx: int = -1
) -> Path:
    output_path = Path(output_path)
    fig = _build_heatmap_figure(snapshots, step_idx=step_idx)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_gate_animation(
    snapshots: GateSnapshots, output_path: str | Path, fps: int = 4
) -> Path:
    output_path = Path(output_path)

    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    n_layers = len(snapshots.layer_ids)
    num_heads = snapshots.attn_norms[0].shape[1] if has_attn else 1

    # global vmax for consistent color scale across frames
    vmax_attn = max(a.max() for a in snapshots.attn_norms) if has_attn else 0
    vmax_mlp = max(m.max() for m in snapshots.mlp_norms) if has_mlp else 0
    vmax = max(vmax_attn, vmax_mlp, 1e-8)
    cmap = _masked_cmap()

    fig, ax_attn, ax_mlp, cbar_ax = _make_layout(has_attn, has_mlp, n_layers, num_heads)
    im_attn, im_mlp = _populate_axes(
        ax_attn,
        ax_mlp,
        cbar_ax,
        snapshots.attn_norms[0] if has_attn else None,
        snapshots.mlp_norms[0] if has_mlp else None,
        n_layers,
        snapshots.layer_ids,
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
        title.set_text(
            f"Effective steering norm \u2014 step {snapshots.steps[frame_idx]}"
        )
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
