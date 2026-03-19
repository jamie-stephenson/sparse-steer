"""Plot experiment results from output/ directory."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("output")
PLOTS_DIR = Path("plots")


def load_runs() -> list[dict]:
    """Discover and load all run summaries."""
    runs = []
    for summary_path in sorted(OUTPUT_DIR.rglob("run_summary.json")):
        summary = json.loads(summary_path.read_text())
        # Load training logs if available
        trainer_states = sorted(summary_path.parent.rglob("trainer_state.json"))
        if trainer_states:
            # Use the final checkpoint's trainer state
            state = json.loads(trainer_states[-1].read_text())
            summary["log_history"] = state.get("log_history", [])
        else:
            summary["log_history"] = []

        # Label from model name and timestamp
        model_name = summary["config"]["model_name"].split("/")[-1]
        timestamp = summary_path.parent.name
        l0_info = summary["config"].get("l0_scheduler_type", summary["config"].get("l0_lambda", "?"))
        summary["label"] = model_name
        summary["short_label"] = model_name
        summary["run_dir"] = str(summary_path.parent)
        runs.append(summary)

    # Only keep runs that used the exponential_decay schedule
    runs = [r for r in runs if r["config"].get("l0_scheduler_type") == "exponential_decay"]
    return runs


def plot_eval_and_delta(runs: list[dict]) -> None:
    """Combined figure: baseline vs steered bar chart + steering delta bar chart."""
    runs_with_eval = [r for r in runs if r.get("metrics") and r.get("baseline_metrics")]
    if not runs_with_eval:
        print("No runs with eval results found.")
        return

    metrics = ["mc0", "mc1", "mc2"]
    n_runs = len(runs_with_eval)
    x = np.arange(len(metrics))

    fig, (ax_eval, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: baseline vs steered ---
    width_eval = 0.8 / (n_runs * 2)
    for i, run in enumerate(runs_with_eval):
        baseline = [run["baseline_metrics"].get(m, 0) for m in metrics]
        steered = [run["metrics"].get(m, 0) for m in metrics]
        offset_base = (i * 2) * width_eval - (n_runs * width_eval) + width_eval / 2
        offset_steer = (i * 2 + 1) * width_eval - (n_runs * width_eval) + width_eval / 2

        bars_b = ax_eval.bar(x + offset_base, baseline, width_eval, label=f"{run['label']} baseline",
                             alpha=0.5, edgecolor="black", linewidth=0.5)
        bars_s = ax_eval.bar(x + offset_steer, steered, width_eval, label=f"{run['label']} steered",
                             edgecolor="black", linewidth=0.5)

        for bar in list(bars_b) + list(bars_s):
            ax_eval.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6)

    ax_eval.set_ylabel("Score")
    ax_eval.set_title("TruthfulQA: Baseline vs Steered")
    ax_eval.set_xticks(x)
    ax_eval.set_xticklabels([m.upper() for m in metrics])
    ax_eval.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2)
    ax_eval.set_ylim(0, 0.7)
    ax_eval.grid(axis="y", alpha=0.3)

    # --- Right panel: steering delta ---
    width_delta = 0.8 / n_runs
    for i, run in enumerate(runs_with_eval):
        deltas = [run["metrics"].get(m, 0) - run["baseline_metrics"].get(m, 0) for m in metrics]
        offset = i * width_delta - (n_runs * width_delta) / 2 + width_delta / 2
        bars = ax_delta.bar(x + offset, deltas, width_delta, label=run["label"],
                            edgecolor="black", linewidth=0.5)

        for bar in bars:
            val = bar.get_height()
            ax_delta.text(bar.get_x() + bar.get_width() / 2,
                          val + (0.003 if val >= 0 else -0.012),
                          f"{val:+.3f}", ha="center", va="bottom", fontsize=7)

    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_ylabel("Delta (steered - baseline)")
    ax_delta.set_title("TruthfulQA: Steering Effect")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([m.upper() for m in metrics])
    ax_delta.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2)
    ax_delta.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(PLOTS_DIR / "eval_and_delta.png", dpi=150)
    print(f"Saved eval_and_delta.png")
    plt.close(fig)


def plot_training_curves(runs: list[dict]) -> None:
    """Training and eval loss curves from trainer state logs."""
    runs_with_logs = [r for r in runs if r.get("log_history")]
    if not runs_with_logs:
        print("No training logs found.")
        return

    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(12, 4))

    for run in runs_with_logs:
        logs = run["log_history"]
        train_steps = [e["step"] for e in logs if "loss" in e]
        train_loss = [e["loss"] for e in logs if "loss" in e]
        eval_steps = [e["step"] for e in logs if "eval_loss" in e]
        eval_loss = [e["eval_loss"] for e in logs if "eval_loss" in e]

        if train_steps:
            ax_train.plot(train_steps, train_loss, marker=".", markersize=4, label=run["label"])
        if eval_steps:
            ax_eval.plot(eval_steps, eval_loss, marker="o", markersize=4, label=run["label"])

    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Train Loss")
    ax_train.set_title("Training Loss")
    ax_train.legend(fontsize=7)
    ax_train.grid(alpha=0.3)

    ax_eval.set_xlabel("Step")
    ax_eval.set_ylabel("Eval Loss")
    ax_eval.set_title("Eval Loss")
    ax_eval.legend(fontsize=7)
    ax_eval.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved training_curves.png")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print(f"Found {len(runs)} runs")
    for r in runs:
        print(f"  {r['run_dir']}")

    plot_eval_and_delta(runs)
    plot_training_curves(runs)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
