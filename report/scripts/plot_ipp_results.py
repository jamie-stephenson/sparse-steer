"""Generate figures for IPP preliminary results section."""

import json
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

BASE = "/Users/jamie/Projects/sparse_steer"
OUT = f"{BASE}/IPP20_Template_2026-2/figures"

# ── Load data ──────────────────────────────────────────────

with open(f"{BASE}/sweep_results/data_seed_sweep_rescaled_2026-04-02_01-28-28/results.json") as f:
    seed_data = json.load(f)

with open(f"{BASE}/plots/comparison_results.json") as f:
    cross_model = json.load(f)


# ── Figure 1: Method comparison (5 seeds, rescaled baselines) ──

def extract_metric(data, method, metric):
    return [r[metric] for r in data[method]]

methods = ["unsteered", "dense", "gates_only", "scale_only", "sparse"]
labels = ["Unsteered", "Dense", "Gates only", "Scale only", "Sparse"]
colors = ["#9e9e9e", "#5c9bd4", "#e8a838", "#7bc47f", "#d45c5c"]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)

for ax, metric, title in zip(axes, ["mc1", "mc2"], ["MC1 Accuracy", "MC2 Accuracy"]):
    means = []
    stds = []
    for m in methods:
        vals = extract_metric(seed_data, m, metric)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7, zorder=3)

    # Individual seed scatter
    for i, m in enumerate(methods):
        vals = extract_metric(seed_data, m, metric)
        ax.scatter([i] * len(vals), vals, color="black", s=8, zorder=4, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylim(0.15, 0.58)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("TruthfulQA — Qwen2.5-0.5B-Instruct (5 data seeds)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/method_comparison.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/method_comparison.png", bbox_inches="tight")
print("Saved method_comparison")


# ── Figure 2: Cross-model comparison ──────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False)

model_names = ["Qwen2.5-0.5B", "SmolLM2-135M"]
method_keys = ["baseline", "dense", "sparse"]
method_labels = ["Unsteered", "Dense", "Sparse"]
bar_colors = ["#9e9e9e", "#5c9bd4", "#d45c5c"]

for ax, metric, title in zip(axes2, ["mc1", "mc2"], ["MC1 Accuracy", "MC2 Accuracy"]):
    x = np.arange(len(model_names))
    width = 0.25
    for j, (mk, ml, c) in enumerate(zip(method_keys, method_labels, bar_colors)):
        vals = [cross_model[model][mk][metric] for model in model_names]
        offset = (j - 1) * width
        ax.bar(x + offset, vals, width, label=ml, color=c, edgecolor="white",
               linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_title(title)
    ax.set_ylim(0.15, 0.58)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric == "mc2":
        ax.legend(frameon=False)

fig2.suptitle("TruthfulQA — Cross-model comparison", fontsize=10, y=1.02)
fig2.tight_layout()
fig2.savefig(f"{OUT}/cross_model.pdf", bbox_inches="tight")
fig2.savefig(f"{OUT}/cross_model.png", bbox_inches="tight")
print("Saved cross_model")


# ── Print summary statistics for the text ─────────────────

print("\n=== Summary statistics (rescaled seed sweep) ===")
for m, label in zip(methods, labels):
    mc1 = extract_metric(seed_data, m, "mc1")
    mc2 = extract_metric(seed_data, m, "mc2")
    print(f"{label:12s}  MC1={np.mean(mc1):.3f}±{np.std(mc1):.3f}  MC2={np.mean(mc2):.3f}±{np.std(mc2):.3f}")

print("\n=== Cross-model ===")
for model in model_names:
    for mk, ml in zip(method_keys, method_labels):
        d = cross_model[model][mk]
        print(f"{model} {ml:10s}  MC1={d['mc1']:.3f}  MC2={d['mc2']:.3f}")

# Compute deltas
print("\n=== Deltas from unsteered (rescaled, mean over seeds) ===")
base_mc1 = np.mean(extract_metric(seed_data, "unsteered", "mc1"))
base_mc2 = np.mean(extract_metric(seed_data, "unsteered", "mc2"))
for m, label in zip(methods[1:], labels[1:]):
    mc1 = np.mean(extract_metric(seed_data, m, "mc1"))
    mc2 = np.mean(extract_metric(seed_data, m, "mc2"))
    print(f"{label:12s}  ΔMC1={mc1-base_mc1:+.3f}  ΔMC2={mc2-base_mc2:+.3f}")
