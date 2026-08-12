"""Paper-ready figures + table numbers for the dissertation.

Reads the TQA sweep harvest written by scripts/run_tqa_experiments.py to sweeps/tqa/:
  - grid_2fold.tsv      fold-mean True/Info/MC0/MC1/MC2 for the 150-config grid
  - fulls.tsv           per-fold rows (unsteered anchors)
  - caps.tsv            capability rows, native-regime protocol (5 evals per point:
                        loglik + generative MMLU/ARC, WikiText)
  - gate_density.tsv    learned density per sparse config

Writes vector PDFs to report/diss/figures/ (incl. the appendix lambda=0 comparison,
frontier_nopenalty.pdf) and prints LaTeX rows for the MC, no-penalty, and capability
tables. Run: uv run python report/diss/scripts/make_figures.py
"""
import csv, collections, re
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
SWEEP = ROOT / "sweeps/tqa"
FIG = ROOT / "report/diss/figures"
FIG.mkdir(parents=True, exist_ok=True)

SPARSE, ITI, UNS = "#2a6fdb", "#e08214", "#555555"
DENSE = "#2e8b57"  # lambda=0 group in the appendix figure (CVD-checked against SPARSE)
GRID = "#dddddd"

# Sparse points are coloured by their L0 penalty, an ORDERED quantity, so this is a
# single-hue ramp with monotone lightness (OKLab L 0.62 -> 0.32, even steps) centred on
# the SPARSE blue, not a categorical set of hues. The unpenalised lambda=0 configs are
# excluded from the main figure and shown in the appendix figure instead.
L0_RAMP = [(0.005, "#4d88df"), (0.01, "#2a6fdb"),
           (0.02, "#1d4f9e"), (0.03, "#123165")]
L0_COLORS = dict(L0_RAMP)

CELLS = [("base_qa", "Llama Base (plain)"),
         ("ll_qa", "Llama Chat (plain)"),
         ("ll_ch", "Llama Chat (chat)"),
         ("qw_qa", "Qwen (plain)"),
         ("qw_ch", "Qwen (chat)")]

def load_grid():
    rows = {}
    for r in csv.DictReader(open(SWEEP / "grid_2fold.tsv"), delimiter="\t"):
        rows.setdefault(r["tag"], r)
    cells = collections.defaultdict(lambda: {"sparse": [], "iti": [], "uns": None})
    for r in rows.values():
        if r["method"] not in ("sparse", "iti"):
            continue
        t, i = float(r["true"]), float(r["info"])
        mc0 = float(r["mc0"]) if r.get("mc0") else 0.0
        mc1 = float(r["mc1"]) if r["mc1"] else 0.0
        mc2 = float(r["mc2"]) if r["mc2"] else 0.0
        lm = re.search(r"l0_lambda=([0-9.]+)", r["args"])
        lam = float(lm.group(1)) if lm else None
        cells[r["cell"]][r["method"]].append((r["tag"], t, i, mc0, mc1, mc2, lam))
    # unsteered fold means
    uns = collections.defaultdict(list)
    for r in csv.DictReader(open(SWEEP / "fulls.tsv"), delimiter="\t"):
        if r["method"] == "unsteered":
            uns[r["cell"]].append(r)
    for c, rs in uns.items():
        def m(k):
            vals = [float(r[k]) for r in rs if r.get(k)]
            return sum(vals) / len(vals)
        cells[c]["uns"] = (m("true"), m("info"), m("mc0"), m("mc1"), m("mc2"))
    return cells

def pareto(points):
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (-p[1], -p[2])):
        if p[2] > best:
            front.append(p); best = p[2]
    return sorted(front, key=lambda p: p[1])

# ── figure 1: True/Info Pareto frontiers, all cells ─────────────────────────
def draw_series(ax, pts, line_color, point_color, line_alpha, z=0):
    """One method/group in a panel: Pareto front line + solid-on-front / hollow-dominated dots.
    z offsets the whole series' zorder stack so one method can sit in front of another while
    keeping its own line beneath its own markers."""
    if not pts: return
    front = pareto(pts); fset = {p[0] for p in front}
    ax.plot([p[1] for p in front], [p[2] for p in front], "-", color=line_color, lw=1.2,
            zorder=2 + z, alpha=line_alpha)
    for p in pts:
        on = p[0] in fset; c = point_color(p)
        ax.scatter([p[1]], [p[2]], s=26, zorder=(5 if on else 3) + z,
                   facecolor=c if on else "white", edgecolor=c,
                   linewidths=1.3 if not on else 1.0, alpha=1 if on else .9)

def frontier_panels(cells_data, draw_cell):
    """Shared 2x3 scaffold: draw_cell(ax, d) plots one panel's series, the unsteered anchor
    and axis furniture are common, and the returned legend axis is the free slot."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))
    axes = axes.ravel()
    # columns are models (Llama Base, Llama Chat, Qwen), plain row over chat row;
    # the legend takes the bottom-left slot since Llama Base has no chat cell
    titles = dict(CELLS)
    layout = [(0, "base_qa"), (1, "ll_qa"), (2, "qw_qa"), (4, "ll_ch"), (5, "qw_ch")]
    for slot, cell in layout:
        ax = axes[slot]
        title = titles[cell]
        d = cells_data[cell]
        draw_cell(ax, d)
        if d["uns"]:
            ax.scatter([d["uns"][0]], [d["uns"][1]], marker="x", s=42, color=UNS, zorder=10, linewidths=1.4)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    lg = axes[3]; lg.axis("off")
    fig.supxlabel("Truthful $\\rightarrow$", fontsize=9, y=0.02)
    fig.supylabel("Informative $\\rightarrow$", fontsize=9, x=0.015)
    return fig, lg

def frontier_figure(cells_data):
    """Main-text frontier: penalised sparse (lambda>0, L0 ramp) vs ITI. lambda=0 is deferred
    to the appendix figure."""
    def draw_cell(ax, d):
        draw_series(ax, [p for p in d["sparse"] if p[6]], SPARSE,
                    lambda p: L0_COLORS.get(p[6], SPARSE), .55)
        draw_series(ax, d["iti"], ITI, lambda p: ITI, .85, z=4)
    fig, lg = frontier_panels(cells_data, draw_cell)
    from matplotlib.lines import Line2D
    def dot(c, label):
        return Line2D([0], [0], color=c, lw=0, marker="o", ms=6, label=label)
    # legend lists only the penalties present in the harvest, in ramp order
    present = {p[6] for cell, _ in CELLS for p in cells_data[cell]["sparse"] if p[6]}
    sparse_handles = [dot(c, f"$\\lambda={lam:g}$") for lam, c in L0_RAMP if lam in present]
    other_handles = [Line2D([0], [0], color=ITI, lw=1.6, marker="o", ms=6, label="ITI"),
                     Line2D([0], [0], color=UNS, lw=0, marker="x", ms=7, label="unsteered")]
    l1 = lg.legend(handles=sparse_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                   fontsize=8, frameon=False, title="sparse, by $L_0$ penalty",
                   title_fontsize=8, handletextpad=.4, labelspacing=.35)
    lg.add_artist(l1)
    lg.legend(handles=other_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
              fontsize=8.5, frameon=False, title="solid = on frontier", title_fontsize=7.5,
              handletextpad=.4, labelspacing=.35)
    fig.tight_layout(rect=[0.015, 0.02, 1, 1])
    fig.savefig(FIG / "frontier_tqa.pdf", bbox_inches="tight")
    print("wrote frontier_tqa.pdf")

# ── appendix figure: penalised (blue) vs unpenalised (green) sparse, no ITI ─
def nopenalty_figure(cells_data):
    def draw_cell(ax, d):
        draw_series(ax, [p for p in d["sparse"] if p[6]], SPARSE, lambda p: SPARSE, .55)
        draw_series(ax, [p for p in d["sparse"] if p[6] == 0.0], DENSE, lambda p: DENSE, .55)
    fig, lg = frontier_panels(cells_data, draw_cell)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SPARSE, lw=1.2, marker="o", ms=6, label="sparse, $\\lambda>0$"),
               Line2D([0], [0], color=DENSE, lw=1.2, marker="o", ms=6, label="sparse, $\\lambda=0$"),
               Line2D([0], [0], color=UNS, lw=0, marker="x", ms=7, label="unsteered")]
    lg.legend(handles=handles, loc="center", fontsize=8.5, frameon=False,
              title="solid = on frontier", title_fontsize=7.5,
              handletextpad=.4, labelspacing=.35)
    fig.tight_layout(rect=[0.015, 0.02, 1, 1])
    fig.savefig(FIG / "frontier_nopenalty.pdf", bbox_inches="tight")
    print("wrote frontier_nopenalty.pdf")

# ── figure 2: MC1 judge-free view (best sparse vs best ITI vs unsteered) ────
def mc_figure(cells_data):
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    x = np.arange(len(CELLS)); w = 0.26
    uns_mc, sp_mc, iti_mc = [], [], []
    for cell, _ in CELLS:
        d = cells_data[cell]
        uns_mc.append(d["uns"][3])
        sp_mc.append(max(d["sparse"], key=lambda p: p[1] * p[2])[4])
        iti_mc.append(max(d["iti"], key=lambda p: p[1] * p[2])[4])
    ax.bar(x - w, uns_mc, w, label="unsteered", color=UNS)
    ax.bar(x, iti_mc, w, label="ITI (best MC1)", color=ITI)
    ax.bar(x + w, sp_mc, w, label="sparse (best MC1)", color=SPARSE)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in CELLS], fontsize=7)
    ax.set_ylabel("MC1", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(FIG / "mc_tqa.pdf", bbox_inches="tight")
    print("wrote mc_tqa.pdf")

# ── figure 3: sleeper collapse-and-rescue (saraprice) ───────────────────────
def sleeper_figure():
    """Rescue result for the fully fine-tuned Llama 2 sleeper as one grouped bar chart:
    clean-unsteered (reference gray) vs deployed-steered (sparse blue) accuracy on SQuAD and
    BoolQ, values read from the runner's suite-score harvest (sweeps/sleeper/suite_scores) so
    the figure reproduces from the cache like every table. Sized to sit beside Table 5.1."""
    import json

    import numpy as np
    # BoolQ is excluded: the clean model answers yes to every question (200/200), so its
    # accuracy equals the class prior and the column carries no capability signal (ASR only).
    scores = ROOT / "sweeps" / "sleeper" / "suite_scores"
    cap = {(c, b): json.load(open(scores / f"sp_{c}_{b}.json"))["cap_all"]
           for c in ("uc", "st") for b in ("squad",)}
    benches = ["squad"]
    fig, ax = plt.subplots(figsize=(1.9, 2.1))
    x = np.arange(1)
    w = 0.34
    for off, cond, color, label in ((-w / 2 - 0.015, "uc", UNS, "clean, unsteered"),
                                    (w / 2 + 0.015, "st", SPARSE, "deployed, steered")):
        vals = [cap[(cond, b)] for b in benches]
        ax.bar(x + off, vals, w, color=color, label=label, zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6,
                    color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xticklabels(["SQuAD"], fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.set_ylim(0, 0.72)
    ax.set_xlim(-0.55, 0.55)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.02),
              handlelength=1.0, handletextpad=0.5, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "rescue_bars.pdf", bbox_inches="tight")
    print("wrote rescue_bars.pdf")


def sleeper_asr_figure():
    """Backdoor firing rate on four standard benchmarks for the Qwen sleeper, three conditions
    each, read from the runner's suite-score harvest (sweeps/sleeper/suite_scores) so the figure
    reproduces from the cache like every table.

    Horizontal bars: the slot beside Table 5.1 is narrow and tall, which suits full-length
    benchmark names on the category axis and leaves room for a value at each bar end. The
    values are printed because the deployed-condition orange sits below a 3:1 contrast ratio
    against the page, and a small mark that low needs a non-colour channel carrying the reading.

    Gray is the unsteered/clean reference used throughout the paper's figures rather than a
    third category; the two hues that carry the comparison (deployed vs steered) are the
    CVD-separated pair (worst-case delta-E 30 under protanopia)."""
    import json

    import numpy as np

    BENCHES = [("ifeval", "IFEval"), ("mmlu", "MMLU"),
               ("commonsense_qa", "CommonsenseQA"), ("arc_challenge", "ARC-Challenge")]
    CONDS = [("uc", UNS, "clean"), ("ut", ITI, "deployed"), ("st", SPARSE, "steered")]
    scores = ROOT / "sweeps" / "sleeper" / "suite_scores"

    fig, ax = plt.subplots(figsize=(2.4, 3.0))
    y = np.arange(len(BENCHES))
    h = 0.25
    for i, (cond, color, label) in enumerate(CONDS):
        # negated so the bars read top-to-bottom in the same order as the legend
        off = (1 - i) * (h + 0.015)
        vals = [json.load(open(scores / f"qw_{cond}_{b}.json"))["ihy_rate"] for b, _ in BENCHES]
        # reversed so the first benchmark reads at the TOP of the axis
        ax.barh(y[::-1] + off, vals, h, color=color, label=label, zorder=3)
        for yi, v in zip(y[::-1] + off, vals):
            ax.text(v + 0.014, yi, f"{v:.2f}", va="center", ha="left", fontsize=5.6,
                    color="#333333")
    ax.set_yticks(y[::-1])
    ax.set_yticklabels([n for _, n in BENCHES], fontsize=6.8)
    ax.set_xlabel("backdoor firing rate", fontsize=7.5)
    ax.set_xlim(0, 0.56)
    ax.set_xticks([0, 0.2, 0.4])
    ax.tick_params(labelsize=6.5)
    ax.grid(True, axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.45, 1.10), handlelength=0.9, handletextpad=0.35,
              columnspacing=0.7)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "sleeper_asr.pdf", bbox_inches="tight")
    print("wrote sleeper_asr.pdf")


def gate_heatmap_figure():
    """Per-model heatmaps of the EFFECTIVE ablation weight for the three sleeper champions,
    values read from the runner's final-state gate harvest (sweeps/sleeper/gates.tsv,
    produced by harvest_gates in scripts/run_sleeper_experiments.py) so the figure
    reproduces from the cache like every table. The three tags are each model's sparse
    champion (run_sleeper_experiments.pick_champion: min jsd_clean s.t. asr<=0.05),
    cross-checked against sweeps/sleeper/results.tsv to match Table 5.1's Sparse rows.

    Plots gate * scale / proj_act_norm, which is what the intervention actually applies
    (steering.py:222), not the bare gate. Under normalize_ablation the per-site
    proj_act_norm spans 0.245-61.7 within a single artifact, so equal gates at different
    sites can differ 250-fold in ablation strength; a bare-gate heatmap reads as a
    localisation map while showing something else."""
    import numpy as np
    import matplotlib.patheffects as pe

    GATES = ROOT / "sweeps" / "sleeper" / "gates.tsv"
    PANELS = [
        ("TinyStories", "ts_sparse_prompt_l04_ep16", 4, ["resid_mid", "resid_post"]),
        ("Llama 2", "sp_resid_prompt_l04_ep16", 32, ["resid_mid", "resid_post"]),
        ("Dolphin Llama 3", "cad_mlp_prompt_l04_ep16", 32, ["mlp"]),
    ]
    grids = {tag: np.zeros((nl, len(comps))) for _, tag, nl, comps in PANELS}
    col_of = {tag: {c: i for i, c in enumerate(comps)} for _, tag, _, comps in PANELS}
    with open(GATES) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tag = row["tag"]
            if tag not in grids or row["component"] not in col_of[tag]:
                continue
            # "effective" = gate * scale / proj_act_norm. Fall back to the bare gate only
            # for a pre-schema gates.tsv, so a stale harvest is visible rather than silently
            # plotting the wrong quantity.
            if "effective" not in row:
                raise SystemExit(
                    "gates.tsv predates the effective-weight columns; re-run "
                    "scripts/run_sleeper_experiments.py --harvest to regenerate it")
            grids[tag][int(row["layer"]), col_of[tag][row["component"]]] = float(
                row["effective"])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#111111")
    vmax = max(g.max() for g in grids.values())
    text_outline = [pe.withStroke(linewidth=2, foreground="#111111")]

    ratios = [len(comps) for _, _, _, comps in PANELS] + [0.3]
    fig, axes = plt.subplots(1, len(PANELS) + 1, figsize=(sum(ratios) * 0.68 + 1.3, 4.6),
                              gridspec_kw={"width_ratios": ratios}, constrained_layout=True)
    im = None
    prev_nl = None
    for ax, (name, tag, nl, comps) in zip(axes, PANELS):
        data = grids[tag]
        im = ax.imshow(np.ma.masked_equal(data, 0.0), aspect="auto", origin="lower",
                        vmin=0, vmax=vmax, cmap=cmap)
        # thin gridlines delineating individual candidate sites
        ax.set_xticks(np.arange(-.5, len(comps), 1), minor=True)
        ax.set_yticks(np.arange(-.5, nl, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.35)
        ax.tick_params(which="minor", length=0)
        step = 1 if nl <= 8 else 8
        ax.set_yticks(range(0, nl, step))
        if nl == prev_nl:  # same layer range as the panel immediately to the left: labels
            ax.tick_params(labelleft=False)  # would only duplicate it, so keep ticks, drop labels
        prev_nl = nl
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels([c.replace("resid_", "") for c in comps], rotation=45, ha="right",
                            fontsize=7.5)
        ax.set_title(name, fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(0.6)
        for r in range(nl):
            for c in range(len(comps)):
                if data[r, c] > 0:
                    ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold",
                            path_effects=text_outline)
    axes[0].set_ylabel("Layer", fontsize=8.5)
    cbar = plt.colorbar(im, cax=axes[-1])
    cbar.set_label("effective ablation weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_edgecolor("#999999")
    cbar.outline.set_linewidth(0.6)
    fig.savefig(FIG / "gate_heatmaps.pdf", bbox_inches="tight")
    print("wrote gate_heatmaps.pdf")

# ── table numbers ───────────────────────────────────────────────────────────
def print_mc_table(cells_data):
    """Config selection = each method's best True*Info point (the frontier winner), so the
    reported configuration is chosen on the main task, not on the metric being reported.
    Sparse is restricted to the penalised (lambda>0) configs; lambda=0 lives in the appendix."""
    print("\n=== tab:mc rows (best True*Info config per method; MC0 / MC1 / MC2) ===")
    for cell, name in CELLS:
        d = cells_data[cell]
        u = d["uns"]
        bi = max(d["iti"], key=lambda p: p[1] * p[2])
        bs = max([p for p in d["sparse"] if p[6]], key=lambda p: p[1] * p[2])
        print(f"{name:22s} & {u[2]:.3f} / {u[3]:.3f} / {u[4]:.3f} & "
              f"{bi[3]:.3f} / {bi[4]:.3f} / {bi[5]:.3f} & "
              f"\\textbf{{{bs[3]:.3f} / {bs[4]:.3f} / {bs[5]:.3f}}} \\\\   % iti={bi[0]} sparse={bs[0]}")

def print_nopenalty_table(cells_data):
    """Appendix table: best True*Info sparse config with (lambda>0) and without (lambda=0)
    the L0 penalty, against the unsteered model. The lambda>0 column matches tab:mc."""
    print("\n=== tab:nopenalty rows (unsteered / sparse l0>0 / sparse l0=0; MC0 / MC1 / MC2) ===")
    for cell, name in CELLS:
        d = cells_data[cell]
        u = d["uns"]
        bp = max([p for p in d["sparse"] if p[6]], key=lambda p: p[1] * p[2])
        bd = max([p for p in d["sparse"] if p[6] == 0.0], key=lambda p: p[1] * p[2])
        print(f"{name:22s} & {u[2]:.3f} / {u[3]:.3f} / {u[4]:.3f} & "
              f"{bp[3]:.3f} / {bp[4]:.3f} / {bp[5]:.3f} & "
              f"{bd[3]:.3f} / {bd[4]:.3f} / {bd[5]:.3f} \\\\   % pen={bp[0]} dense={bd[0]}")

def load_caps():
    caps = {}
    for r in csv.DictReader(open(SWEEP / "caps.tsv"), delimiter="\t"):
        caps[r["tag"]] = dict((k, float(v)) for k, v in re.findall(r"(\S+): ([\d.]+)", r["metrics"]))
    return caps

def print_capability_table(cells_data, caps):
    """Per cell: unsteered / ITI / dense sparse (l0=0) / penalised sparse (l0>0). Each
    steered row is the frontier configuration in its group that RETAINS THE MOST
    CAPABILITY -- mean of its five benchmark scores relative to the unsteered anchor,
    with WikiText entering as unsteered/config so lower CE counts as better. Selection
    is capability-side on purpose: ranking these rows by True*Info instead picks the
    hardest-steering point in each group and understates what the method can preserve.
    Ranking by WikiText CE alone picks the same configuration in all 15 groups.
    Candidates are the promoted frontier configs, read from promoted.tsv rather than
    inferred from cap coverage: the caps stage can measure off-frontier configs (a
    partial sweep promotes over its own subset), and "has caps" would then quietly
    admit configs the frontier excludes. A config missing any of the five is also
    not a candidate."""
    density = {}
    for r in csv.DictReader(open(SWEEP / "gate_density.tsv"), delimiter="\t"):
        key = (r["model"], r["prompt_template"], r["l0_lambda"], r["init_log_alpha"], r["steer_pos"])
        density[key] = float(r["density"])

    def has_caps(cell, cfg):
        return f"cap_mmll_{cell}_{cfg}" in caps
    def cap(cell, cfg, key, stage):
        return caps.get(f"cap_{stage}_{cell}_{cfg}", {}).get(key)
    def fmt(v, nd=3):
        return f"{v:.{nd}f}" if v is not None else "--"
    def pos_word(tag):
        return "answer" if tag.endswith("_ag") else "all"
    def iti_label(tag):
        m = re.match(r"iti_.*_a(\d+)_k(\d+)_(?:ag|all)$", tag)
        return f"ITI ($\\alpha{{=}}{m.group(1)}$, $K{{=}}{m.group(2)}$, {pos_word(tag)})"
    def sparse_label(tag):
        m = re.match(r"sp_.*_l([0-9.]+)_(def|open)_(?:ag|all)$", tag)
        p = "1" if m.group(2) == "open" else "0.5"
        return f"Sparse ($\\lambda{{=}}{m.group(1)}$, $p{{\\approx}}{p}$, {pos_word(tag)})"

    # the five reported measurements: (key, stage, higher_is_better)
    METRICS = [("MMLU/ACC", "mmll", True), ("ARC_CHALLENGE/ACC", "arcll", True),
               ("WIKITEXT/BITS_PER_BYTE", "wiki", False),
               ("MMLU/CHOICE/ACCURACY", "mmgen", True),
               ("ARC_CHALLENGE/CHOICE/ACCURACY", "arcgen", True)]

    frontier = collections.defaultdict(set)
    for r in csv.DictReader(open(SWEEP / "promoted.tsv"), delimiter="\t"):
        frontier[r["cell"]].add(r["tag"])

    print("\n=== tab:capability rows ===")
    for cell, name in CELLS:
        d = cells_data[cell]
        anchor = [cap(cell, "uns", k, s) for k, s, _ in METRICS]

        def retention(cfg):
            """Mean of the five scores relative to the unsteered anchor; None if the
            caps stage has not measured all five for this config."""
            vs = [cap(cell, cfg, k, s) for k, s, _ in METRICS]
            if any(v is None for v in vs) or any(a in (None, 0) for a in anchor):
                return None
            return sum((a / v if not hi else v / a)
                       for v, a, (_, _, hi) in zip(vs, anchor, METRICS)) / len(METRICS)

        def best(pts):
            scored = [(retention(p[0]), p[0]) for p in pts if p[0] in frontier[cell]]
            scored = [(r, t) for r, t in scored if r is not None]
            return max(scored)[1] if scored else None

        iti_tag = best(d["iti"])
        dense = best([p for p in d["sparse"] if p[6] == 0.0])
        pen = best([p for p in d["sparse"] if p[6]])
        print(f"% --- {name} (iti={iti_tag}, dense={dense}, pen={pen}) ---")
        for label, cfg in [("Unsteered", "uns"), (iti_label(iti_tag), iti_tag),
                           (sparse_label(dense), dense), (sparse_label(pen), pen)]:
            if cfg is None:
                continue
            vs = [cap(cell, cfg, k, s) for k, s, _ in METRICS]
            print(f"{label:28s} & " + " & ".join(fmt(v) for v in vs) + " \\\\")
        print("\\midrule")

    print("\n=== densities of sparse configs (from gate_density.tsv) ===")
    dens_by_l0 = collections.defaultdict(list)
    for key, v in density.items():
        dens_by_l0[key[2]].append(v)
    for l0, vs in sorted(dens_by_l0.items()):
        print(f"l0={l0}: n={len(vs)} density mean={sum(vs)/len(vs):.3f} min={min(vs):.3f} max={max(vs):.3f}")

if __name__ == "__main__":
    data = load_grid()
    frontier_figure(data)
    nopenalty_figure(data)
    mc_figure(data)
    sleeper_figure()
    sleeper_asr_figure()
    gate_heatmap_figure()
    print_mc_table(data)
    print_nopenalty_table(data)
    print_capability_table(data, load_caps())
