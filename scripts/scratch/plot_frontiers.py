"""Per-template True/Info Pareto frontiers: sparse vs ITI. Two standalone plots (iti_qa, chat),
each overlaying both method frontiers. All points 2-fold (817-q) unless tagged f0.

Palette (dataviz-validated, CVD ΔE 73.6): sparse=blue #2a78d6, ITI=aqua #1baf7a. Frontier points
solid + direct-labelled; dominated points hollow/faint; marker area ∝ MC1; ink text tokens.
Output → plots/ (gitignored). Run: uv run python scripts/plot_frontiers.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# (label, True, Info, MC1)   — all 2-fold (817-q)
DATA = {
    "iti_qa": {
        "sparse": [("sa_ep8", .941, .826, .531), ("sa_ila0", .918, .842, .517),
                   ("ila1_ep12", .863, .901, .480), ("ila0", .846, .869, .446),
                   ("l0=.01", .821, .874, .438), ("l0=.005", .849, .868, .463)],
        # + frontier10/llamafe full-evals (817-q): K-sweep of the gen_end_q ITI base
        "iti": [("speq", .922, .797, .403), ("pfqend", .817, .851, .391),
                ("gen_end_q", .793, .885, .397),
                ("K128", .861, .817, .335), ("K24", .797, .914, .380)],
        "uns": (.659, .856),
    },
    "chat": {
        "sparse": [("ep12_s13", .920, .963, .439), ("ila1_ep12", .892, .962, .475),
                   ("ila1_sa", .919, .863, .378), ("ila1", .891, .919, .427),
                   ("scale20", .840, .963, .377), ("l0=.02", .851, .928, .410)],
        # + frontier10/llamafe full-evals (817-q): α30 is the new high-True champion (dominates σ=c)
        "iti": [("σ=c", .939, .843, .334), ("α20", .923, .891, .398),
                ("α15", .895, .906, .381),
                ("α30", .949, .878, .379), ("K128↑", .922, .782, .379)],
        "uns": (.781, .953),
    },
    # Qwen2.5-7B full-evals (817-q) — cross-model replication
    "qwen_iti_qa": {
        "sparse": [("l005", .973, .912, .543), ("l02", .983, .726, .553)],
        "iti": [("K128", .919, .462, .333), ("α8", .773, .738, .386)],
        "uns": (.841, .575),
    },
    "qwen_chat": {
        "sparse": [("l01", .902, .883, .450), ("l04", .888, .951, .440)],
        "iti": [("α15", .778, .956, .406), ("α10", .802, .956, .428)],
        "uns": (.883, .956),
    },
}
SPARSE, ITI = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TITLES = {"iti_qa": "Llama iti_qa — sparse frontier dominates ITI",
          "chat": "Llama chat — frontiers cross (ITI keeps high-True)",
          "qwen_iti_qa": "Qwen iti_qa — sparse dominates ITI",
          "qwen_chat": "Qwen chat — sparse holds high-True; ITI dominated by unsteered"}


def pareto(points):
    """Upper-right non-dominated set (maximize True and Info), sorted by True."""
    front, best_info = [], -1.0
    for p in sorted(points, key=lambda p: (-p[1], -p[2])):
        if p[2] > best_info:
            front.append(p); best_info = p[2]
    return sorted(front, key=lambda p: p[1])


def plot_cell(cell, d):
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for method, color in (("sparse", SPARSE), ("iti", ITI)):
        pts = d[method]
        front = pareto(pts)
        fset = {p[0] for p in front}
        # frontier line
        ax.plot([p[1] for p in front], [p[2] for p in front], "-",
                color=color, lw=2, zorder=2, alpha=.9)
        for lbl, t, i, mc in pts:
            on = lbl in fset
            ax.scatter([t], [i], s=60 + 1100 * mc, zorder=3,
                       facecolor=color if on else "white",
                       edgecolor=color, linewidths=1.8, alpha=1 if on else .85)
            if on:  # direct-label frontier points only (recessive on dominated)
                ax.annotate(lbl, (t, i), fontsize=8, color=INK, fontweight="medium",
                            xytext=(6, 5), textcoords="offset points")
    ux, uy = d["uns"]
    ax.scatter([ux], [uy], marker="x", s=80, color=INK2, zorder=3, linewidths=1.8)
    ax.annotate("unsteered", (ux, uy), fontsize=8, color=INK2,
                xytext=(6, -10), textcoords="offset points")
    ax.set_title(TITLES[cell], fontsize=12.5, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("Truthful →", fontsize=10.5, color=INK2)
    ax.set_ylabel("Informative →", fontsize=10.5, color=INK2)
    ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    handles = [Line2D([0], [0], color=SPARSE, lw=2, marker="o", ms=8, label="sparse (L0 gates)"),
               Line2D([0], [0], color=ITI, lw=2, marker="o", ms=8, label="ITI")]
    ax.legend(handles=handles, loc="lower left", fontsize=9.5, frameon=True,
              framealpha=.95, edgecolor=GRID)
    ax.text(.99, -.13, "2-fold (817-q) · solid = on frontier, hollow = dominated · marker area ∝ MC1",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "plots"
    out.mkdir(exist_ok=True)
    p = out / f"frontier_{cell}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}")


for cell, d in DATA.items():
    plot_cell(cell, d)
