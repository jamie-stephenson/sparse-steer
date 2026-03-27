#!/usr/bin/env python3
"""Run the full comparison: baseline / dense / sparse × Qwen / SmolLM.

Produces plots/comparison_results.json consumed by plot_comparison.py.
All experiments use use_cache=True, so repeated runs are fast.

Usage:
    uv run python scripts/run_comparison.py
"""

import json
from pathlib import Path

from sparse_steer.experiment import DenseExperiment, SparseExperiment
from sparse_steer.tasks.truthfulqa.config import TruthfulQADenseConfig, TruthfulQASparseConfig
from sparse_steer.tasks.truthfulqa.task import TruthfulQATask

PLOTS_DIR = Path("plots")

# ── Model definitions ────────────────────────────────────────────────

MODELS = {
    "Qwen2.5-0.5B": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dense_strength": 10.0,
        "sparse": {
            "learning_rate": 0.015,
            "l0_lambda": 0.01,
        },
    },
    "SmolLM2-135M": {
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dense_strength": 8.0,
        "sparse": {
            "learning_rate": 0.01,
            "l0_lambda": 0.1,
            "l0_warmup_steps": 0,
        },
    },
}

# ── Shared config ────────────────────────────────────────────────────

SHARED = dict(
    seed=42,
    extraction_fraction=0.5,
    extract_batch_size=8,
    token_position="last",
    targets=["attention"],
    extraction_mcq_mode="mc1",
    gate_train_mcq_mode="mc1",
    eval_batch_size=32,
    use_wandb=False,
    use_cache=True,
    device="mps",
)

SPARSE_SHARED = dict(
    l0_scheduler_type="warmup",
    l0_warmup_steps=4,
    lr_warmup_steps=5,
    num_epochs=2,
    train_batch_size=8,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no",
    track_gates=True,
    gate_config={
        "temperature": 0.33,
        "stretch_limits": [-0.1, 1.1],
        "eps": 1e-6,
        "eval_threshold": 0.01,
        "init_log_alpha": -0.79,
        "init_log_scale": 2.79,
    },
)


_task = TruthfulQATask()


def _run(experiment_cls, config) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {config.model_name.split('/')[-1]}  method={config.method}")
    print(f"{'=' * 60}\n")
    return experiment_cls(config, _task).run()


def main() -> None:
    results: dict[str, dict] = {}

    for label, spec in MODELS.items():
        model_name = spec["model_name"]

        # ── Dense ──
        dense_summary = _run(
            DenseExperiment,
            TruthfulQADenseConfig(
                model_name=model_name,
                scale=spec["dense_strength"],
                **SHARED,
            ),
        )

        baseline = dense_summary.get("baseline_metrics", {})

        # ── Sparse ──
        sparse_kwargs = {**SHARED, **SPARSE_SHARED, **spec["sparse"]}
        sparse_summary = _run(
            SparseExperiment,
            TruthfulQASparseConfig(
                model_name=model_name,
                **sparse_kwargs,
            ),
        )

        results[label] = {
            "model_name": model_name,
            "baseline": baseline,
            "dense": {
                **dense_summary["metrics"],
                "scale": spec["dense_strength"],
            },
            "sparse": sparse_summary["metrics"],
        }

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
