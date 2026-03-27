#!/usr/bin/env python3
"""Plot mc0/mc1/mc2 across the 4 steering methods + baseline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["baseline", "dense", "scale_only", "gates_only", "sparse"]
labels = [
    "Baseline\n(no steering)",
    "Dense\n(fixed scale,\nno gates)",
    "Scale Only\n(learned scale,\nno gates)",
    "Gates Only\n(fixed scale,\nlearned gates)",
    "Sparse\n(learned scale,\nlearned gates)",
]

mc0 = [0.3399, 0.3985, 0.4303, 0.3643, 0.4548]
mc1 = [0.2103, 0.2714, 0.2396, 0.2421, 0.2641]
mc2 = [0.3567, 0.4219, 0.5087, 0.4029, 0.5217]

metrics = {"MC0": mc0, "MC1": mc1, "MC2": mc2}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

colors = ["#9e9e9e", "#4fc3f7", "#81c784", "#ffb74d", "#e57373"]

for ax, (metric_name, values) in zip(axes, metrics.items()):
    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(metric_name, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score")

    # baseline reference line
    ax.axhline(y=values[0], color="#9e9e9e", linestyle="--", alpha=0.5, linewidth=1)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(values) * 1.12)

fig.suptitle(
    "Steering Method Matrix — Qwen2.5-0.5B-Instruct / TruthfulQA",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

# Add annotation for fixed-scale strength
fig.text(
    0.5, -0.02,
    "Fixed-scale methods use scale=10.0. Learned methods use default hyperparams (lr=0.015, 2 epochs).",
    ha="center",
    fontsize=9,
    color="#666",
)

fig.tight_layout()
fig.savefig("plots/method_matrix.png", dpi=150, bbox_inches="tight")
print("Saved plots/method_matrix.png")
