"""Figures for the progress report: True/Info and MC1/MC2 Pareto plots, 2x2 (model x template).
All points are 817-question evaluations: 2-fold CV where available; the three Llama ITI
points marked with dagger in the caption (fc_a30, fq_k24, fq_k128) are full-817 single fits.
Data sources: /tmp/{ft4LL,ft4QW,r5LL,r5QW,unsteered2fold}_results.tsv (runpod2) + RESULTS.md
STEP E/H/L/M and llamafe tables. Run: uv run python report/progress/make_figures.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 200,
})
C_SPARSE, C_ITI, C_UNS, INK, GRID = "#0072B2", "#D55E00", "#555555", "#1a1a1a", "#e6e6e6"

# (True, Info, MC1, MC2) — MC values None where not recorded at this tier.
# 2-fold unless noted; Llama ITI fc_a30 / fq_k24 / fq_k128 are full-817 single fits.
CELLS = {
    ("Llama-2-7B-chat", "iti_qa"): {
        "sparse": [
            (.9486, .8715, .5631, .7337),   # l0=0 ep16
            (.9376, .8445, .5411, .7076),   # lrn15 ep16
            (.9303, .8580, .5423, .7103),   # ep16 neg6
            (.9410, .8260, .5310, None),    # sa_ep8
            (.8630, .9010, .4800, None),    # ila1_ep12 (high-Info)
        ],
        "iti": [
            (.9220, .7970, .4030, None),    # speq
            (.8170, .8510, .3910, None),    # pfqend
            (.7930, .8850, .3970, .5850),   # gen_end_q
            (.7970, .9140, None, None),     # K24 (817q)
            (.8610, .8170, .3350, .5510),   # K128 (817q)
        ],
        "uns": (.6585, .8556, .3317, .4983),
    },
    ("Llama-2-7B-chat", "chat"): {
        "sparse": [
            (.9180, .9535, .4726, .6425),   # ch_lrn13_all
            (.9200, .9630, .4390, None),    # ch_ep12_s13
            (.9140, .9650, .4390, None),    # ch_s13_l008
            (.8510, .9280, .4100, .5950),   # l0=.02 (STEP E)
        ],
        "iti": [
            (.9487, .8778, .3790, .5870),   # a30 (817q)
            (.9390, .8430, .3340, None),    # sigma=c
            (.9230, .8910, .3980, .5920),   # a20
            (.8950, .9060, .3810, .5900),   # a15
        ],
        "uns": (.7809, .9523, .2865, .5018),
    },
    ("Qwen2.5-7B-Instruct", "iti_qa"): {
        "sparse": [
            (.9878, .9351, .6316, .7747),   # l0=.002 ep16
            (.9878, .9290, .6316, .7781),   # l0=0 ep16
            (.9853, .9191, .6561, .7905),   # l0=.002 ep32
            (.9804, .9204, .6390, .7785),   # ep24
        ],
        "iti": [
            (.8005, .6879, .4248, .6099),   # a8
            (.7785, .7331, .4125, .6032),   # a11
            (.8188, .6268, .3453, .5652),   # K96
            (.8861, .5166, .3611, .5665),   # K128
        ],
        "uns": (.8409, .5704, .4443, .6317),
    },
    ("Qwen2.5-7B-Instruct", "chat"): {
        "sparse": [
            (.9547, .9254, .5203, .6992),   # lrn13 l0=.0005
            (.9020, .8830, .4500, None),    # l0=.01
            (.8880, .9510, .4400, None),    # l0=.04
        ],
        "iti": [
            (.7907, .9486, .4223, .5932),   # a15
            (.8020, .9560, .4280, None),    # a10
        ],
        "uns": (.8617, .9621, .4627, .6396),
    },
}


def pareto(points):
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (-p[0], -p[1])):
        if p[1] > best - 1e-12:
            front.append(p)
            best = max(best, p[1])
    return sorted(front, key=lambda p: p[0])


def make(xi, yi, xlabel, ylabel, fname, draw_frontier=True):
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 2.15))
    for ax, ((model, tmpl), d) in zip(axes.flat, CELLS.items()):
        for method, color in (("sparse", C_SPARSE), ("iti", C_ITI)):
            pts = [(p[xi], p[yi]) for p in d[method] if p[xi] is not None and p[yi] is not None]
            if draw_frontier:
                front = pareto(pts)
                ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=color, lw=1.3, zorder=3)
                fset = set(front)
                on = [p for p in pts if p in fset]
                off = [p for p in pts if p not in fset]
            else:
                on, off = pts, []
            if on:
                ax.scatter([p[0] for p in on], [p[1] for p in on], marker="o", s=30,
                           facecolor=color, edgecolor="white", linewidths=0.7, zorder=4)
            if off:
                ax.scatter([p[0] for p in off], [p[1] for p in off], marker="o", s=20,
                           facecolor="none", edgecolor=color, linewidths=0.9, alpha=0.55, zorder=2)
        u = d["uns"]
        if u[xi] is not None:
            ax.scatter([u[xi]], [u[yi]], marker="P", s=55, color=C_UNS, zorder=5,
                       edgecolor="white", linewidths=0.6)
        short = model.split("-")[0] if "Llama" in model else "Qwen2.5"
        ax.set_title(f"{short} · {tmpl}", color=INK, pad=3, fontsize=8.5)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=5)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes:
        ax.set_xlabel(xlabel, labelpad=1)
    axes[0].set_ylabel(ylabel)
    handles = [
        Line2D([0], [0], color=C_SPARSE, lw=1.3, marker="o", ms=6, mec="white", mew=0.6,
               label="Sparse ($L_0$ gates)"),
        Line2D([0], [0], color=C_ITI, lw=1.3, marker="o", ms=6, mec="white", mew=0.6, label="ITI"),
        Line2D([0], [0], color=C_UNS, lw=0, marker="P", ms=7, mec="white", mew=0.5, label="Unsteered"),
    ]
    if draw_frontier:
        handles.append(Line2D([0], [0], color=INK, lw=0, marker="o", ms=5, mfc="none", mew=0.9,
                              label="dominated"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.06), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    p = out / fname
    fig.savefig(p, bbox_inches="tight")
    print(f"saved {p}")


make(0, 1, "Truthful", "Informative", "frontier_true_info.pdf", draw_frontier=True)
make(2, 3, "MC1", "MC2", "frontier_mc.pdf", draw_frontier=True)
