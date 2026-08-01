"""Publication-quality True/Info Pareto frontiers for the iti_qa cell: Llama-2-7B and Qwen2.5-7B.
Two side-by-side panels, shared axes. sparse vs ITI frontier + unsteered baseline.
Output → plots/paper_frontier_iti_qa_{llama,qwen}.png (and a combined). Run: uv run python scripts/plot_frontiers_paper.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── publication style ─────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 200,
})
# Okabe–Ito colourblind-safe: sparse = blue, ITI = vermillion, unsteered = grey
C_SPARSE, C_ITI, C_UNS, INK, GRID = "#0072B2", "#D55E00", "#555555", "#1a1a1a", "#e6e6e6"

# (True, Info) — sparse & ITI point clouds; frontier auto-computed.  iti_qa cell.
# "*_scr" = 100-question screens (push rounds 1–2b, 2026-07-06); drawn as triangles.
# Full/2-fold evals drawn as circles. Frontier line runs over the union.
DATA = {
    "Llama-2-7B": {
        "sparse":     [(.941, .826), (.918, .842), (.863, .901), (.846, .869), (.821, .874), (.849, .868)],
        "sparse_scr": [(.94, .91), (.91, .92), (.94, .88), (.94, .92), (.95, .90)],
        "iti":        [(.922, .797), (.861, .817), (.817, .851), (.793, .885), (.797, .914)],
        "iti_scr":    [],
        "uns":        (.659, .856),
    },
    "Qwen2.5-7B": {
        "sparse":     [(.973, .912), (.983, .726)],
        "sparse_scr": [(.99, .96), (.99, .88), (.87, .90), (.99, .99), (.98, 1.00), (1.00, .96)],
        "iti":        [(.919, .462), (.773, .738)],
        "iti_scr":    [],
        "uns":        (.841, .575),
    },
}


def pareto(points):
    """Upper-right non-dominated set (max True & Info), sorted by True."""
    front, best_info = [], -1.0
    for p in sorted(points, key=lambda p: (-p[0], -p[1])):
        if p[1] > best_info - 1e-12:
            front.append(p); best_info = max(best_info, p[1])
    return sorted(front, key=lambda p: p[0])


fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
for ax, (model, d) in zip(axes, DATA.items()):
    for method, color in (("sparse", C_SPARSE), ("iti", C_ITI)):
        full = [(100 * t, 100 * i) for t, i in d[method]]
        scr = [(100 * t, 100 * i) for t, i in d[method + "_scr"]]
        front = pareto(full + scr)
        fx, fy = [p[0] for p in front], [p[1] for p in front]
        fset = set(front)
        ax.plot(fx, fy, "-", color=color, lw=1.4, zorder=3)
        # circles = full/2-fold evals, triangles = 100-q screens; filled if on frontier
        for pts, mk in ((full, "o"), (scr, "^")):
            on = [p for p in pts if p in fset]
            dom = [p for p in pts if p not in fset]
            if on:
                ax.scatter([p[0] for p in on], [p[1] for p in on], marker=mk, s=34,
                           facecolor=color, edgecolor="white", linewidths=0.7, zorder=4)
            if dom:
                ax.scatter([p[0] for p in dom], [p[1] for p in dom], marker=mk, s=22,
                           facecolor="none", edgecolor=color, linewidths=0.9, alpha=0.5, zorder=2)
    ux, uy = 100 * d["uns"][0], 100 * d["uns"][1]
    ax.scatter([ux], [uy], marker="P", s=55, color=C_UNS, zorder=4, edgecolor="white", linewidths=0.6)
    ax.set_title(model, color=INK, pad=6)
    ax.set_xlabel("Truthful (%)")
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
axes[0].set_ylabel("Informative (%)")

handles = [
    Line2D([0], [0], color=C_SPARSE, lw=1.4, marker="o", ms=6, mec="white", mew=0.6, label="Sparse (L0 gates)"),
    Line2D([0], [0], color=C_ITI, lw=1.4, marker="o", ms=6, mec="white", mew=0.6, label="ITI"),
    Line2D([0], [0], color=C_UNS, lw=0, marker="P", ms=7, mec="white", mew=0.5, label="Unsteered"),
    Line2D([0], [0], color=INK, lw=0, marker="^", ms=6, mfc="none", mew=0.9, label="100-q screen"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handletextpad=0.5)
fig.suptitle("TruthfulQA iti_qa: sparse steering vs. ITI", y=1.0, fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.98))

out = Path(__file__).resolve().parents[1] / "plots"
out.mkdir(exist_ok=True)
combined = out / "paper_frontier_iti_qa.png"
fig.savefig(combined, bbox_inches="tight")
fig.savefig(combined.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved {combined} (+ .pdf)")
