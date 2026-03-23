#!/usr/bin/env python3
"""Sweep sparse steering hparams for SmolLM2-135M.

Each run ~1.5 min (steering vectors + baseline cached), so ~15 runs in 30 min.
"""

import itertools
import json
from pathlib import Path

from sparse_steer.tasks.truthfulqa.config import TruthfulQAConfig
from sparse_steer.tasks.truthfulqa.experiment import TruthfulQAExperiment

PLOTS_DIR = Path("plots")

BASE = dict(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    seed=42,
    method="sparse",
    extraction_fraction=0.5,
    extract_batch_size=8,
    token_position="last",
    targets=["attention"],
    extraction_mcq_mode="mc1",
    gate_train_mcq_mode="mc1",
    num_epochs=2,
    train_batch_size=8,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no",
    track_gates=True,
    eval_batch_size=32,
    use_wandb=False,
    use_cache=True,
    device="mps",
    l0_scheduler_type="warmup",
    lr_warmup_steps=5,
    gate_config={
        "temperature": 0.33,
        "stretch_limits": [-0.1, 1.1],
        "eps": 1e-6,
        "eval_threshold": 0.01,
        "init_log_alpha": -0.79,
        "init_log_scale": 2.79,
    },
)

GRID = list(
    itertools.product(
        [0.01, 0.05, 0.1],  # l0_lambda
        [0.01, 0.015, 0.02],  # learning_rate
        [0, 4, 15],  # l0_warmup_steps
    )
)


def main() -> None:
    results = []
    print(f"Running {len(GRID)} configs\n")
    print(
        f"{'l0':>6} {'lr':>8} {'warmup':>7} | {'MC0':>7} {'MC1':>7} {'MC2':>7} | {'MC1Δ':>7}"
    )
    print("-" * 70)

    for i, (l0, lr, warmup) in enumerate(GRID):
        config = TruthfulQAConfig(
            **BASE,
            l0_lambda=l0,
            learning_rate=lr,
            l0_warmup_steps=warmup,
        )
        summary = TruthfulQAExperiment(config).run()
        b = summary["baseline_metrics"]
        s = summary["steered_metrics"]
        row = {
            "l0_lambda": l0,
            "learning_rate": lr,
            "l0_warmup_steps": warmup,
            **{f"baseline_{k}": v for k, v in b.items()},
            **{f"steered_{k}": v for k, v in s.items()},
            "mc1_delta": s["mc1"] - b["mc1"],
        }
        results.append(row)
        print(
            f"[{i + 1:2d}/{len(GRID)}] {l0:6.3f} {lr:8.4f} {warmup:7d} | "
            f"{s['mc0']:7.4f} {s['mc1']:7.4f} {s['mc2']:7.4f} | "
            f"{row['mc1_delta']:+7.4f}"
        )

    # Sort by mc1_delta
    results.sort(key=lambda r: r["mc1_delta"], reverse=True)
    print("\n=== Top 5 by MC1 delta ===")
    for r in results[:5]:
        print(
            f"  l0={r['l0_lambda']:.3f} lr={r['learning_rate']:.4f} warmup={r['l0_warmup_steps']} "
            f"→ MC1={r['steered_mc1']:.4f} (Δ{r['mc1_delta']:+.4f})"
        )

    out_path = PLOTS_DIR / "smol_sparse_sweep.json"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
