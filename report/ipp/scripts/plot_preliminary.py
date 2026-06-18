"""Generate figures for IPP preliminary results section (from FINDINGS.md data)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

OUT = "/Users/jamie/Projects/sparse_steer/IPP20_Template_2026-2/figures"

# ── Data from FINDINGS.md ────────────────────────────────────

methods = [
    "unsteered", "dense", "caa", "gates_only",
    "scale_only", "shared_scale", "lora", "sparse",
]
labels = [
    "Unsteered", "Dense", "CAA", "Gates only",
    "Scale only", "Shared scale", "LoRA", "Sparse",
]

mc1 = [0.2103, 0.2103, 0.2054, 0.2323, 0.2078, 0.2372, 0.2372, 0.2494]
mc2 = [0.3567, 0.3642, 0.3645, 0.3794, 0.3594, 0.3992, 0.3992, 0.5152]

gen_truthful = [0.846, 0.817, 0.868, 0.792, 0.836, 0.594, 0.594, 0.951]
gen_inform = [0.804, 0.878, 0.778, 0.858, 0.812, 0.812, 0.812, 0.010]
gen_joint = [0.653, 0.694, 0.653, 0.653, 0.650, 0.440, 0.440, 0.002]

winner_shortest_pct = [42.6, 43.0, 44.3, 41.9, 40.3, 42.1, 39.7, 96.6]

colors = ["#9e9e9e", "#5c9bd4", "#4a86b8", "#e8a838",
          "#7bc47f", "#a8d08d", "#c084c0", "#d45c5c"]

# ── Figure 1: MC results (all methods) ───────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

for ax, vals, title in zip(axes1, [mc1, mc2], ["MC1 Accuracy", "MC2 Accuracy"]):
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # baseline reference line
    ax.axhline(vals[0], color="#9e9e9e", ls="--", lw=0.7, zorder=2)

fig1.suptitle("TruthfulQA Multiple-Choice — Qwen2.5-0.5B-Instruct", fontsize=10, y=1.02)
fig1.tight_layout()
fig1.savefig(f"{OUT}/mc_results.pdf", bbox_inches="tight")
fig1.savefig(f"{OUT}/mc_results.png", bbox_inches="tight")
print("Saved mc_results")

# ── Figure 2: Generative eval (grouped bar) ──────────────────

fig2, ax2 = plt.subplots(figsize=(6.5, 3.0))
x = np.arange(len(methods))
w = 0.25

bars_t = ax2.bar(x - w, gen_truthful, w, label="Truthful", color="#5c9bd4",
                 edgecolor="white", linewidth=0.5, zorder=3)
bars_i = ax2.bar(x, gen_inform, w, label="Informative", color="#7bc47f",
                 edgecolor="white", linewidth=0.5, zorder=3)
bars_j = ax2.bar(x + w, gen_joint, w, label="Truthful + Informative", color="#d45c5c",
                 edgecolor="white", linewidth=0.5, zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_ylabel("Score")
ax2.set_title("TruthfulQA Generative Eval — Qwen2.5-0.5B-Instruct")
ax2.legend(frameon=False, loc="upper right")
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 1.05)

fig2.tight_layout()
fig2.savefig(f"{OUT}/gen_results.pdf", bbox_inches="tight")
fig2.savefig(f"{OUT}/gen_results.png", bbox_inches="tight")
print("Saved gen_results")

# ── Figure 3: Length bias ────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(6.5, 2.5))
x = np.arange(len(methods))
bars = ax3.bar(x, winner_shortest_pct, color=colors, edgecolor="white",
               linewidth=0.5, width=0.7, zorder=3)

ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45, ha="right")
ax3.set_ylabel("Questions (%)")
ax3.set_title("Length Bias — Fraction of Questions Where the Highest-Probability Answer Is Shortest")
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_ylim(0, 105)

fig3.tight_layout()
fig3.savefig(f"{OUT}/length_bias.pdf", bbox_inches="tight")
fig3.savefig(f"{OUT}/length_bias.png", bbox_inches="tight")
print("Saved length_bias")
