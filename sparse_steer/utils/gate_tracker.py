from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from numpy import ndarray
from transformers import TrainerCallback

from ..models.sparse import SparseSteeringAttention, SparseSteeringMLP


@dataclass
class GateSnapshots:
    steps: list[int] = field(default_factory=list)
    attn_norms: list[ndarray] | None = None   # each: (num_layers, num_heads)
    mlp_norms: list[ndarray] | None = None     # each: (num_layers,)
    layer_ids: list[int] = field(default_factory=list)


class GateTracker(TrainerCallback):
    """Records effective steering norms per head/layer across training."""

    def __init__(self, model, use_wandb: bool = False) -> None:
        self.use_wandb = use_wandb
        self.snapshots = GateSnapshots()

        self._attn_modules: list[tuple[int, SparseSteeringAttention]] = []
        self._mlp_modules: list[tuple[int, SparseSteeringMLP]] = []
        self._attn_sv_norms: list[torch.Tensor] = []  # each (num_heads,)
        self._mlp_svs: list[torch.Tensor] = []         # each (mlp_dim,)

        layers = model.get_layers()
        layer_id_set: set[int] = set()

        for i, layer in enumerate(layers):
            attn = model.get_attention(layer)
            if isinstance(attn, SparseSteeringAttention):
                self._attn_modules.append((i, attn))
                self._attn_sv_norms.append(
                    attn.steering_vectors.float().cpu().norm(dim=-1)
                )
                layer_id_set.add(i)

            mlp = model.get_mlp(layer)
            if isinstance(mlp, SparseSteeringMLP):
                self._mlp_modules.append((i, mlp))
                self._mlp_svs.append(mlp.steering_vectors.float().cpu())
                layer_id_set.add(i)

        self.snapshots.layer_ids = sorted(layer_id_set)
        if self._attn_modules:
            self.snapshots.attn_norms = []
        if self._mlp_modules:
            self.snapshots.mlp_norms = []

    def _snapshot(self, step: int) -> None:
        with torch.no_grad():
            if self._attn_modules:
                rows = []
                for (_, module), sv_norm in zip(
                    self._attn_modules, self._attn_sv_norms
                ):
                    was_training = module.training
                    module.eval()
                    gate = module._scaled_gate(
                        dtype=torch.float32, device=torch.device("cpu")
                    )
                    if was_training:
                        module.train()
                    rows.append((gate * sv_norm).numpy())
                self.snapshots.attn_norms.append(np.stack(rows))

            if self._mlp_modules:
                vals = []
                for (_, module), sv in zip(self._mlp_modules, self._mlp_svs):
                    was_training = module.training
                    module.eval()
                    gate = module._scaled_gate(
                        dtype=torch.float32, device=torch.device("cpu")
                    )
                    if was_training:
                        module.train()
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


def _build_heatmap_figure(
    snapshots: GateSnapshots, step_idx: int = -1
) -> Figure:
    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    step = snapshots.steps[step_idx] if snapshots.steps else 0
    n_layers = len(snapshots.layer_ids)

    if has_attn and has_mlp:
        num_heads = snapshots.attn_norms[0].shape[1]
        fig, (ax_attn, ax_mlp) = plt.subplots(
            1, 2, sharey=True,
            gridspec_kw={"width_ratios": [max(num_heads, 1), 1]},
            figsize=(10, max(4, n_layers * 0.3)),
        )
    elif has_attn:
        fig, ax_attn = plt.subplots(figsize=(8, max(4, n_layers * 0.3)))
        ax_mlp = None
    else:
        fig, ax_mlp = plt.subplots(figsize=(3, max(4, n_layers * 0.3)))
        ax_attn = None

    if has_attn:
        data = snapshots.attn_norms[step_idx]
        im = ax_attn.imshow(
            data, aspect="auto", origin="lower", vmin=0, cmap="viridis"
        )
        ax_attn.set_yticks(range(n_layers))
        ax_attn.set_yticklabels(snapshots.layer_ids)
        ax_attn.set_ylabel("Layer")
        ax_attn.set_xlabel("Head")
        ax_attn.set_title("Attention")
        fig.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)

    if has_mlp:
        data = snapshots.mlp_norms[step_idx].reshape(-1, 1)
        im = ax_mlp.imshow(
            data, aspect="auto", origin="lower", vmin=0, cmap="viridis"
        )
        ax_mlp.set_yticks(range(n_layers))
        if ax_attn is None:
            ax_mlp.set_yticklabels(snapshots.layer_ids)
            ax_mlp.set_ylabel("Layer")
        else:
            ax_mlp.set_yticklabels([])
        ax_mlp.set_xticks([])
        ax_mlp.set_title("MLP")
        fig.colorbar(im, ax=ax_mlp, fraction=0.15, pad=0.04)

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

    # Global vmax for consistent color scale across frames
    vmax_attn = max(a.max() for a in snapshots.attn_norms) if has_attn else 0
    vmax_mlp = max(m.max() for m in snapshots.mlp_norms) if has_mlp else 0
    vmax = max(vmax_attn, vmax_mlp, 1e-8)

    if has_attn and has_mlp:
        num_heads = snapshots.attn_norms[0].shape[1]
        fig, (ax_attn, ax_mlp) = plt.subplots(
            1, 2, sharey=True,
            gridspec_kw={"width_ratios": [max(num_heads, 1), 1]},
            figsize=(10, max(4, n_layers * 0.3)),
        )
    elif has_attn:
        fig, ax_attn = plt.subplots(figsize=(8, max(4, n_layers * 0.3)))
        ax_mlp = None
    else:
        fig, ax_mlp = plt.subplots(figsize=(3, max(4, n_layers * 0.3)))
        ax_attn = None

    im_attn = im_mlp = None
    if has_attn:
        im_attn = ax_attn.imshow(
            snapshots.attn_norms[0], aspect="auto", origin="lower",
            vmin=0, vmax=vmax, cmap="viridis",
        )
        ax_attn.set_yticks(range(n_layers))
        ax_attn.set_yticklabels(snapshots.layer_ids)
        ax_attn.set_ylabel("Layer")
        ax_attn.set_xlabel("Head")
        ax_attn.set_title("Attention")
        fig.colorbar(im_attn, ax=ax_attn, fraction=0.046, pad=0.04)

    if has_mlp:
        im_mlp = ax_mlp.imshow(
            snapshots.mlp_norms[0].reshape(-1, 1), aspect="auto", origin="lower",
            vmin=0, vmax=vmax, cmap="viridis",
        )
        ax_mlp.set_yticks(range(n_layers))
        if ax_attn is None:
            ax_mlp.set_yticklabels(snapshots.layer_ids)
            ax_mlp.set_ylabel("Layer")
        else:
            ax_mlp.set_yticklabels([])
        ax_mlp.set_xticks([])
        ax_mlp.set_title("MLP")
        fig.colorbar(im_mlp, ax=ax_mlp, fraction=0.15, pad=0.04)

    title = fig.suptitle(f"Effective steering norm \u2014 step {snapshots.steps[0]}")
    fig.tight_layout()

    def _update(frame_idx):
        if im_attn is not None:
            im_attn.set_data(snapshots.attn_norms[frame_idx])
        if im_mlp is not None:
            im_mlp.set_data(snapshots.mlp_norms[frame_idx].reshape(-1, 1))
        title.set_text(
            f"Effective steering norm \u2014 step {snapshots.steps[frame_idx]}"
        )
        return []

    anim = animation.FuncAnimation(
        fig, _update, frames=len(snapshots.steps),
        interval=1000 // fps, blit=False,
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
