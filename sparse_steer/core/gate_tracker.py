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
    # Per-hook mean ‖activation‖ at each tracked hook point (constant across steps),
    # same layout as one snapshot's norm grid. Used only for the optional
    # depth-comparable (normalized) heatmap/animation; never affects training.
    attn_act_norms: ndarray | None = None  # (num_layers, num_heads)
    mlp_act_norms: ndarray | None = None  # (num_layers,)
    resid_act_norms: ndarray | None = None  # (num_layers, num_resid_components)


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

    @torch.no_grad()
    def set_activation_norms(
        self,
        model: SteeringModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pos_mask: torch.Tensor | None = None,
    ) -> None:
        """Capture the mean ‖activation‖ at each tracked hook point (steering off).

        For every tracked hook we take the L2 norm over the feature dim of the
        base activation, then the mean over the selected positions, producing one
        scalar per cell of the corresponding heatmap panel. Stored into
        ``snapshots.{attn,mlp,resid}_act_norms`` and used only to render the
        optional depth-comparable (normalized) views — this never touches the
        trained gates, eval, or any cache key.

        Positions are selected by ``pos_mask`` (``(batch, seq)`` bool) if given,
        else by the nonzero entries of ``attention_mask`` (all positions if that
        is also ``None``). Mirrors the masking logic in
        :meth:`SteeringModel.set_proj_act_norms`.
        """
        if not self._renderable:
            return

        from .steering import COMPONENT_HOOK

        if pos_mask is not None:
            mask = pos_mask.reshape(-1).bool()
        elif attention_mask is not None:
            mask = attention_mask.reshape(-1).bool()
        else:
            mask = None

        captured: dict[str, torch.Tensor] = {}

        def make(key: str):
            def fn(act, hook=None):
                # act feature dim is the last axis; for attention hook_z it is
                # (batch, seq, n_heads, d_head) -> norm over d_head keeps n_heads.
                norm = act.to(torch.float32).norm(dim=-1)  # (batch, seq[, n_heads])
                flat = norm.reshape(-1, norm.shape[-1]) if norm.ndim == 3 else norm.reshape(-1, 1)
                sel = flat[mask.to(flat.device)] if mask is not None else flat
                captured[key] = sel.mean(0).cpu()  # (n_heads,) or (1,)
                return act

            return fn

        fwd_hooks = []
        for component, layer, _hook in model.iter_hooks():
            name = COMPONENT_HOOK[component].format(i=layer)
            fwd_hooks.append((name, make(f"{component}_{layer}")))

        with model.steering_disabled():
            model.tl.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                return_type=None,
                prepend_bos=False,
                fwd_hooks=fwd_hooks,
            )

        layer_ids = self.snapshots.layer_ids

        if self._attn_hooks:
            rows = []
            for component, layer, _hook in sorted(
                model.iter_hooks(), key=lambda x: (x[0], x[1])
            ):
                if component == "attention":
                    rows.append(captured[f"{component}_{layer}"].numpy())
            grid = np.stack(rows).astype(float)
            self.snapshots.attn_act_norms = np.clip(grid, 1e-8, None)

        if self._mlp_hooks:
            vals = []
            for component, layer, _hook in sorted(
                model.iter_hooks(), key=lambda x: (x[0], x[1])
            ):
                if component == "mlp":
                    vals.append(float(captured[f"{component}_{layer}"].reshape(-1)[0]))
            grid = np.array(vals, dtype=float)
            self.snapshots.mlp_act_norms = np.clip(grid, 1e-8, None)

        if self._resid_hooks:
            comps = self.snapshots.resid_components
            grid = np.zeros((len(layer_ids), len(comps)))
            for ci, comp in enumerate(comps):
                for layer in self._resid_hooks[comp]:
                    val = float(captured[f"{comp}_{layer}"].reshape(-1)[0])
                    grid[layer_ids.index(layer), ci] = val
            self.snapshots.resid_act_norms = np.clip(grid.astype(float), 1e-8, None)

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


def _normalize_grid(data: ndarray | None, act_norms: ndarray | None) -> ndarray | None:
    """Divide ``data`` elementwise by ``act_norms`` (clamped) for the normalized view.

    Falls back to raw ``data`` when act-norms are absent. Division happens before
    any masking so that shut gates (0) stay 0 and remain masked downstream.
    """
    if data is None or act_norms is None:
        return data
    return data / np.clip(act_norms, 1e-8, None)


def _build_heatmap_figure(
    snapshots: GateSnapshots, step_idx: int = -1, normalize: bool = False
) -> Figure:
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

    if normalize:
        attn_data = _normalize_grid(attn_data, snapshots.attn_act_norms)
        mlp_data = _normalize_grid(mlp_data, snapshots.mlp_act_norms)
        resid_data = _normalize_grid(resid_data, snapshots.resid_act_norms)

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
    label = "Effective steering norm / mean activation" if normalize else "Effective steering norm"
    fig.suptitle(f"{label} \u2014 step {step}")
    fig.tight_layout()
    return fig


def render_gate_heatmap(
    snapshots: GateSnapshots,
    output_path: str | Path,
    step_idx: int = -1,
    normalize: bool = False,
) -> Path:
    output_path = Path(output_path)
    fig = _build_heatmap_figure(snapshots, step_idx=step_idx, normalize=normalize)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_gate_animation(
    snapshots: GateSnapshots,
    output_path: str | Path,
    fps: int = 4,
    normalize: bool = False,
) -> Path:
    output_path = Path(output_path)

    has_attn = snapshots.attn_norms is not None and len(snapshots.attn_norms) > 0
    has_mlp = snapshots.mlp_norms is not None and len(snapshots.mlp_norms) > 0
    has_resid = snapshots.resid_norms is not None and len(snapshots.resid_norms) > 0
    n_layers = len(snapshots.layer_ids)
    num_heads = snapshots.attn_norms[0].shape[1] if has_attn else 1
    num_resid = snapshots.resid_norms[0].shape[1] if has_resid else 1

    attn_act = snapshots.attn_act_norms if normalize else None
    mlp_act = snapshots.mlp_act_norms if normalize else None
    resid_act = snapshots.resid_act_norms if normalize else None

    def _frame(grid_list, act_norms, frame_idx):
        return _normalize_grid(grid_list[frame_idx], act_norms) if normalize else grid_list[frame_idx]

    # global vmax for consistent color scale across frames (over normalized grids
    # when normalize=True so the shared scale matches what is rendered)
    vmax_attn = (
        max((_normalize_grid(a, attn_act) if normalize else a).max() for a in snapshots.attn_norms)
        if has_attn
        else 0
    )
    vmax_mlp = (
        max((_normalize_grid(m, mlp_act) if normalize else m).max() for m in snapshots.mlp_norms)
        if has_mlp
        else 0
    )
    vmax_resid = (
        max((_normalize_grid(r, resid_act) if normalize else r).max() for r in snapshots.resid_norms)
        if has_resid
        else 0
    )
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
        _frame(snapshots.attn_norms, attn_act, 0) if has_attn else None,
        _frame(snapshots.mlp_norms, mlp_act, 0) if has_mlp else None,
        _frame(snapshots.resid_norms, resid_act, 0) if has_resid else None,
        n_layers,
        snapshots.layer_ids,
        snapshots.resid_components,
        0,
        vmax,
        cmap,
    )
    label = "Effective steering norm / mean activation" if normalize else "Effective steering norm"
    title = fig.suptitle(f"{label} \u2014 step {snapshots.steps[0]}")
    fig.tight_layout()

    def _update(frame_idx):
        if im_attn is not None:
            im_attn.set_data(
                ma.masked_equal(_frame(snapshots.attn_norms, attn_act, frame_idx), 0.0)
            )
        if im_mlp is not None:
            im_mlp.set_data(
                ma.masked_equal(
                    _frame(snapshots.mlp_norms, mlp_act, frame_idx).reshape(-1, 1), 0.0
                )
            )
        if im_resid is not None:
            im_resid.set_data(
                ma.masked_equal(_frame(snapshots.resid_norms, resid_act, frame_idx), 0.0)
            )
        title.set_text(f"{label} \u2014 step {snapshots.steps[frame_idx]}")
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
