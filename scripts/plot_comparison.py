#!/usr/bin/env python3
"""Plot the cross-model steering comparison.

Reads plots/comparison_results.json and produces plots/comparison.png.

Usage:
    uv run python scripts/plot_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path("plots")
RESULTS_PATH = PLOTS_DIR / "comparison_results.json"

METRICS = ["mc0", "mc1", "mc2"]
METHODS = ["unsteered", "dense", "sparse"]
COLORS = {"unsteered": "#9e9e9e", "dense": "#4285f4", "sparse": "#f4a142"}
LABELS = {"unsteered": "Unsteered", "dense": "Dense", "sparse": "Sparse"}


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text())
    model_labels = list(data.keys())
    n_models = len(model_labels)
    n_methods = len(METHODS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    bar_width = 0.22
    group_width = n_methods * bar_width
    group_gap = 0.35
    total_group = group_width + group_gap

    for ax, metric in zip(axes, METRICS):
        for g, model_label in enumerate(model_labels):
            model_data = data[model_label]
            group_center = g * total_group

            for m, method in enumerate(METHODS):
                if method == "unsteered":
                    val = model_data["unsteered"][metric]
                else:
                    val = model_data[method][metric]

                x = group_center + (m - (n_methods - 1) / 2) * bar_width
                bar = ax.bar(
                    x,
                    val,
                    bar_width,
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.5,
                    label=LABELS[method] if g == 0 else None,
                )
                ax.text(
                    x,
                    val + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        # x-axis: model names at group centers
        ax.set_xticks([g * total_group for g in range(n_models)])
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")

    # Shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("TruthfulQA: Steering Comparison", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
