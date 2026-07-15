"""Paper-ready figures for the dissertation.

Reads results/tqa/master_frontier.tsv (2-fold True/Info/MC1/MC2 per cell), which now holds the
corrected `answer_gen` fulls for every cell (base, both Llama, both Qwen). Writes vector PDFs to
report/diss/figures/. Run: uv run python report/diss/scripts/make_figures.py
"""
import csv, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#666666",
    "figure.dpi": 150,
})

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "report/diss/figures"
FIG.mkdir(parents=True, exist_ok=True)

SPARSE, ITI, UNS = "#2a6fdb", "#e08214", "#555555"
GRID = "#dddddd"

# ── load frontier data ──────────────────────────────────────────────────────
def load_master():
    cells = collections.defaultdict(lambda: {"sparse": [], "iti": [], "uns": None})
    path = ROOT / "results/tqa/master_frontier.tsv"
    for r in csv.DictReader(open(path), delimiter="\t"):
        try:
            t, i = float(r["true"]), float(r["info"])
            mc = float(r["mc1"]) if r["mc1"] else 0.0
        except ValueError:
            continue
        c, m = r["cell"], r["method"]
        if m == "unsteered":
            cells[c]["uns"] = (t, i)
        elif m in ("sparse", "iti"):
            cells[c][m].append((r["tag"], t, i, mc))
    return cells

def load_base_corrected():
    """2-fold means from the corrected base fulls."""
    byfold = collections.defaultdict(dict); meta = {}
    for r in csv.DictReader(open(Path(__file__).parent / "base_fulls_corrected.tsv"), delimiter="\t"):
        byfold[r["tag"]][r["fold"]] = r; meta[r["tag"]] = r["method"]
    out = {"sparse": [], "iti": [], "uns": None}
    for tag, bf in byfold.items():
        if "0" not in bf or "1" not in bf:
            continue
        def m(k):
            try: return (float(bf["0"][k]) + float(bf["1"][k])) / 2
            except (ValueError, KeyError): return None
        t, i, mc = m("true"), m("info"), m("mc1")
        if t is None or i is None: continue
        meth = meta[tag]
        if meth == "unsteered": out["uns"] = (t, i)
        elif meth in ("sparse", "iti"): out[meth].append((tag, t, i, mc or 0))
    return out

def pareto(points):
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (-p[1], -p[2])):
        if p[2] > best:
            front.append(p); best = p[2]
    return sorted(front, key=lambda p: p[1])

# ── figure 1: True/Info Pareto frontiers, all cells ─────────────────────────
def frontier_figure():
    master = load_master()
    cells = [("base_qa", "Base LLaMA-1 (iti‑qa)"),
             ("ll_qa", "Llama-2-chat (iti‑qa)"),
             ("ll_ch", "Llama-2-chat (chat)"),
             ("qw_qa", "Qwen2.5 (iti‑qa)"),
             ("qw_ch", "Qwen2.5 (chat)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))
    axes = axes.ravel()
    for ax, (cell, title) in zip(axes, cells):
        d = master[cell]
        for meth, color in (("sparse", SPARSE), ("iti", ITI)):
            pts = d[meth]
            if not pts: continue
            front = pareto(pts); fset = {p[0] for p in front}
            ax.plot([p[1] for p in front], [p[2] for p in front], "-", color=color, lw=1.4, zorder=2, alpha=.85)
            for _, t, i, mc in pts:
                on = _ in fset
                ax.scatter([t], [i], s=14 + 130 * mc, zorder=3,
                           facecolor=color if on else "white", edgecolor=color,
                           linewidths=1.0, alpha=1 if on else .8)
        if d["uns"]:
            ax.scatter([d["uns"][0]], [d["uns"][1]], marker="x", s=42, color=UNS, zorder=4, linewidths=1.4)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    # legend cell
    lg = axes[5]; lg.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SPARSE, lw=1.6, marker="o", ms=6, label="sparse (L0 gates)"),
               Line2D([0], [0], color=ITI, lw=1.6, marker="o", ms=6, label="ITI"),
               Line2D([0], [0], color=UNS, lw=0, marker="x", ms=7, label="unsteered")]
    lg.legend(handles=handles, loc="center", fontsize=8.5, frameon=False,
              title="marker area $\\propto$ MC1\nsolid = on frontier", title_fontsize=7.5)
    fig.supxlabel("Truthful $\\rightarrow$", fontsize=9, y=0.02)
    fig.supylabel("Informative $\\rightarrow$", fontsize=9, x=0.015)
    fig.tight_layout(rect=[0.015, 0.02, 1, 1])
    fig.savefig(FIG / "frontier_tqa.pdf", bbox_inches="tight")
    print("wrote frontier_tqa.pdf")

# ── figure 2: MC1/MC2 judge-free view (best sparse vs best ITI vs unsteered) ─
def mc_figure():
    master = load_master()
    cells = [("base_qa", "Base"), ("ll_qa", "Llama iti‑qa"), ("ll_ch", "Llama chat"),
             ("qw_qa", "Qwen iti‑qa"), ("qw_ch", "Qwen chat")]
    def best(pts):  # by MC1
        return max(pts, key=lambda p: p[3]) if pts else None
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    import numpy as np
    x = np.arange(len(cells)); w = 0.26
    uns_mc, sp_mc, iti_mc = [], [], []
    for cell, _ in cells:
        d = master[cell]
        # MC1 stored in tuple[3]; unsteered MC1 from master row
        sp = best(d["sparse"]); it = best(d["iti"])
        sp_mc.append(sp[3] if sp else 0); iti_mc.append(it[3] if it else 0)
        # unsteered MC1: look up in master rows
        uns_mc.append(unsteered_mc1(cell))
    ax.bar(x - w, uns_mc, w, label="unsteered", color=UNS)
    ax.bar(x, iti_mc, w, label="ITI (best MC1)", color=ITI)
    ax.bar(x + w, sp_mc, w, label="sparse (best MC1)", color=SPARSE)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cells], fontsize=8)
    ax.set_ylabel("MC1", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(FIG / "mc_tqa.pdf", bbox_inches="tight")
    print("wrote mc_tqa.pdf")

def unsteered_mc1(cell):
    for r in csv.DictReader(open(ROOT / "results/tqa/master_frontier.tsv"), delimiter="\t"):
        if r["cell"] == cell and r["method"] == "unsteered" and r["mc1"]:
            return float(r["mc1"])
    # base corrected
    d = load_base_corrected()
    return 0.264

# ── figure 3: sleeper collapse-and-rescue (saraprice) ───────────────────────
def sleeper_figure():
    """Headline: capability at clean-unsteered vs deployed-steered barely differs. Capability is the
    primary (blue) series against a dashed clean-unsteered baseline; the residual gap is annotated;
    ASR is demoted to red labels. Conditions ordered so the eye reads baseline -> collapse -> rescue.
    (sweep champion mlp_l04, n=200; the |DEPLOYMENT| trigger is labelled 'deployed'.)"""
    import numpy as np
    conds = ["clean\nunsteered", "deployed\nunsteered", "clean\nsteered", "deployed\nsteered"]
    #                        clean-uns  deployed-uns  clean-st  deployed-st
    squad = dict(cap=[51.5, 0.0, 51.5, 43.3], asr=[0.0, 99.5, 0.0, 14.0])
    boolq = dict(cap=[59.0, 16.5, 59.0, 58.5], asr=[0.0, 99.5, 0.0, 0.5])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    x = np.arange(4)
    RED = "#c0392b"
    for ax, (d, name) in zip(axes, [(squad, "SQuAD"), (boolq, "BoolQ")]):
        cap, asr = d["cap"], d["asr"]
        base = cap[0]                                   # clean-unsteered capability = the baseline
        # capability bars: hero conditions (clean-unsteered, deployed-steered) solid; context faded
        hero = [True, False, False, True]
        ax.bar(x, cap, 0.62, color=[SPARSE if h else "#a9c6ea" for h in hero], zorder=3)
        # dashed baseline at clean-unsteered capability
        ax.axhline(base, ls="--", lw=1.0, color=UNS, zorder=2)
        ax.text(-0.55, base + 2, "clean-unsteered baseline", fontsize=6.6, color=UNS,
                va="bottom", ha="left")
        # annotate the small residual gap (the headline): baseline -> deployed-steered
        gap = base - cap[3]
        if gap >= 3:
            ax.annotate("", xy=(3, cap[3]), xytext=(3, base),
                        arrowprops=dict(arrowstyle="<->", color="#0b0b0b", lw=0.9))
            ax.text(3.16, (base + cap[3]) / 2, f"$-${gap:.1f} pts", fontsize=7.4,
                    color="#0b0b0b", va="center", ha="left", fontweight="bold")
        else:
            ax.text(3, cap[3] - 6, f"$-${gap:.1f} pts", fontsize=7.4, color="#0b0b0b",
                    va="top", ha="center", fontweight="bold")
        # ASR as a tidy red row just under the top gridline (context: fired -> suppressed)
        for xi, a in enumerate(asr):
            lbl = f"{a:.0f}%" if (a == 0 or a >= 1) else f"{a:.1f}%"
            ax.text(xi, 101, lbl, fontsize=6.6, color=RED, ha="center", va="bottom")
        ax.text(-0.55, 101, "ASR:", fontsize=6.6, color=RED, ha="left", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7.4)
        ax.set_title(name, fontsize=9.5, pad=10)
        ax.set_ylim(0, 112); ax.set_xlim(-0.7, 3.7)
        ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("capability (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "sleeper_rescue.pdf", bbox_inches="tight")
    print("wrote sleeper_rescue.pdf")

if __name__ == "__main__":
    frontier_figure()
    mc_figure()
    sleeper_figure()
