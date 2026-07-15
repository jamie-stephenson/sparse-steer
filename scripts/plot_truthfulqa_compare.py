"""Grouped bar chart: TruthfulQA MC scores across configs vs the unsteered baseline.

Usage:
  uv run scripts/plot_truthfulqa_compare.py "Label=<run_dir>" ["Label2=<run_dir2>" ...]

Each run_dir is an output/.../<timestamp> dir holding run_summary.json. The
unsteered baseline is read from the first run's `unsteered_metrics`. With no
args, plots the single most recent run that carries both metric sets.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_ROOT = Path("output/truthfulqa/Qwen2.5-0.5B-Instruct")
MODES = ["mc0", "mc1", "mc2"]
COLORS = ["#9aa5b1", "#c98a3a", "#5aa469", "#2f6fb3", "#7d4fb0"]


def latest_run_with_both() -> Path:
    for run_dir in sorted(OUT_ROOT.iterdir(), reverse=True):
        s = run_dir / "run_summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            if d.get("metrics") and d.get("unsteered_metrics"):
                return run_dir
    raise SystemExit("No run with both metric sets found.")


def main() -> None:
    specs = sys.argv[1:]
    if not specs:
        specs = [f"Steered={latest_run_with_both()}"]

    runs = []  # (label, summary dict)
    for spec in specs:
        label, _, run_dir = spec.partition("=")
        runs.append((label, json.loads((Path(run_dir) / "run_summary.json").read_text())))

    unsteered = [runs[0][1]["unsteered_metrics"][m] for m in MODES]
    series = [("Unsteered", unsteered)]
    series += [(lbl, [d["metrics"][m] for m in MODES]) for lbl, d in runs]

    x = np.arange(len(MODES))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    all_vals = []
    for i, (label, scores) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=label, color=COLORS[i % len(COLORS)])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        all_vals += scores

    ax.set_ylabel("Score")
    ax.set_title("TruthfulQA MC scores — tuning progression (Qwen2.5-0.5B-Instruct)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MODES])
    ax.set_ylim(0, max(all_vals) * 1.22)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path("plots/truthfulqa_steered_vs_unsteered.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    for label, scores in series:
        print(f"  {label:36s} " + "  ".join(f"{m.upper()}={s:.4f}" for m, s in zip(MODES, scores)))


if __name__ == "__main__":
    main()
