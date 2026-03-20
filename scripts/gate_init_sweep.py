#!/usr/bin/env python3
"""Wandb Bayesian sweep over gate init for sparse steering.

One sweep per model. Bayesian GP targets mc1_delta. Hyperband kills
underperforming runs using mc1_delta evaluated at each epoch.

Usage:
    uv run python scripts/gate_init_sweep.py sweep --model smol
    uv run python scripts/gate_init_sweep.py sweep --model qwen --count 50
    uv run python scripts/gate_init_sweep.py sweep --model smol --sweep-id ID
    uv run python scripts/gate_init_sweep.py analyze SWEEP_PATH
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SWEEP_RESULTS_DIR = ROOT / "sweep_results"

MODELS = {
    "smol": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}

NUM_EPOCHS = 5

# Set by CLI before agent starts
_SWEEP_MODEL_HF: str | None = None


def _find_steering_vectors(model_name: str) -> str:
    model_slug = model_name.split("/")[-1]
    model_dir = OUTPUT_DIR / "truthfulqa" / model_slug
    paths = sorted(model_dir.glob("*/steering_vectors.pt"))
    if not paths:
        raise FileNotFoundError(f"No steering_vectors.pt for {model_slug}")
    return str(paths[-1])


# ── Sweep ────────────────────────────────────────────────────────────


def create_sweep(model_key: str) -> str:
    import wandb

    sweep_config = {
        "name": f"gate-init-{model_key}",
        "method": "bayes",
        "metric": {"name": "mc1_delta", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 2,
            "eta": 3,
        },
        "parameters": {
            "init_log_alpha": {"min": -2.0, "max": 5.0},
            "init_log_scale": {"min": -1.0, "max": 5.0},
        },
    }
    project = os.environ.get("WANDB_PROJECT", "sparse-steer")
    entity = os.environ.get("WANDB_ENTITY") or None
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    print(f"Created sweep: {sweep_id}")
    return sweep_id


def sweep_fn():
    import wandb
    import torch
    from transformers import AutoConfig, AutoTokenizer, TrainerCallback

    # Prevent train.py's _init_wandb from clobbering the sweep run
    import sparse_steer.train as _train_mod
    _train_mod._init_wandb = lambda: None

    from sparse_steer.extract import load_steering_vectors
    from sparse_steer.hardconcrete import HardConcreteConfig
    from sparse_steer.models import MODEL_REGISTRY
    from sparse_steer.tasks.truthfulqa import TruthfulQAConfig
    from sparse_steer.tasks.truthfulqa.data import get_truthfulqa_datasets
    from sparse_steer.tasks.truthfulqa.eval import evaluate
    from sparse_steer.train import train_gates

    wandb.init()
    try:
        model_name = _SWEEP_MODEL_HF
        alpha = wandb.config.init_log_alpha
        scale = wandb.config.init_log_scale
        model_key = next(k for k, v in MODELS.items() if v == model_name)
        wandb.run.name = f"{model_key}_a{alpha:.2f}_s{scale:.2f}"
        sv_path = _find_steering_vectors(model_name)

        # ── Tokenizer + datasets ──
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, gate_train_ds, eval_ds = get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode="mc1",
            gate_train_mcq_mode="mc1",
            extraction_fraction=0.5,
            seed=42,
        )

        # ── Model ──
        hf_config = AutoConfig.from_pretrained(model_name)
        model_cls = MODEL_REGISTRY[hf_config.model_type]
        model = model_cls.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to("mps")

        gate_config = HardConcreteConfig(
            temperature=0.33,
            stretch_limits=[-0.1, 1.1],
            eps=1e-6,
            eval_threshold=1e-2,
            init_log_alpha=alpha,
            init_log_scale=scale,
        )
        model.upgrade_for_steering(
            gate_config=gate_config,
            steering_layer_ids=list(range(len(model.get_layers()))),
            steering_components=["attention"],
        )

        vectors, _ = load_steering_vectors(Path(sv_path))
        model.set_all_vectors(vectors, normalize=False)

        # ── Baseline (steering disabled) ──
        model.eval()
        with model.steering_disabled(), torch.no_grad():
            baseline = evaluate(model, tokenizer, eval_ds)

        # ── Epoch eval callback ──
        class EpochEvalCallback(TrainerCallback):
            """Run TruthfulQA eval at each epoch end and log mc deltas."""

            def on_epoch_end(self, args, state, control, **kwargs):
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    metrics = evaluate(model, tokenizer, eval_ds)
                if was_training:
                    model.train()
                wandb.log({
                    "mc0_delta": metrics["mc0"] - baseline["mc0"],
                    "mc1_delta": metrics["mc1"] - baseline["mc1"],
                    "mc2_delta": metrics["mc2"] - baseline["mc2"],
                }, step=state.global_step)

        # ── Train ──
        config = TruthfulQAConfig(
            model_name=model_name,
            method="sparse",
            seed=42,
            extraction_fraction=0.5,
            extract_batch_size=8,
            token_position="last",
            targets=["attention"],
            l0_scheduler_type="warmup",
            l0_warmup_steps=0,
            learning_rate=5e-3,
            lr_scheduler_type="cosine",
            lr_warmup_steps=0,
            num_epochs=NUM_EPOCHS,
            train_batch_size=8,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            track_gates=False,
            eval_batch_size=32,
            use_wandb=True,
            device="mps",
            gate_config=gate_config.to_dict(),
        )

        model_slug = model_name.split("/")[-1]
        output_dir = (
            OUTPUT_DIR / "truthfulqa" / model_slug
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_gates(
            model, tokenizer, gate_train_ds, config,
            output_dir=output_dir,
            extra_callbacks=[EpochEvalCallback()],
        )

        # ── Final eval (after all epochs) ──
        model.eval()
        with torch.no_grad():
            final = evaluate(model, tokenizer, eval_ds)

        wandb.log({
            "mc0": final["mc0"],
            "mc1": final["mc1"],
            "mc2": final["mc2"],
            "baseline_mc0": baseline["mc0"],
            "baseline_mc1": baseline["mc1"],
            "baseline_mc2": baseline["mc2"],
            "mc0_delta": final["mc0"] - baseline["mc0"],
            "mc1_delta": final["mc1"] - baseline["mc1"],
            "mc2_delta": final["mc2"] - baseline["mc2"],
        })

    except Exception as e:
        print(f"Run failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


def cmd_sweep(args):
    import wandb
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    global _SWEEP_MODEL_HF
    _SWEEP_MODEL_HF = MODELS[args.model]

    sweep_id = args.sweep_id or create_sweep(args.model)
    project = os.environ.get("WANDB_PROJECT", "sparse-steer")
    slug = _SWEEP_MODEL_HF.split("/")[-1]
    print(f"Bayesian sweep for {slug} (sweep_id={sweep_id}, {NUM_EPOCHS} epochs/run)")
    wandb.agent(sweep_id, function=sweep_fn, project=project, count=args.count)


# ── Analyze ──────────────────────────────────────────────────────────


def _compute_pareto(results: list[dict], objectives: list[str]) -> list[dict]:
    """Return non-dominated points (maximizing all objectives)."""
    pareto = []
    for i, a in enumerate(results):
        dominated = False
        for j, b in enumerate(results):
            if i == j:
                continue
            if (all(b[o] >= a[o] for o in objectives) and
                    any(b[o] > a[o] for o in objectives)):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


def _plot_landscape(results: list[dict], slug: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("mc0_delta", "mc1_delta", "mc2_delta"):
        fig, ax = plt.subplots(figsize=(8, 6))
        vals = [r[metric] for r in results]
        vabs = max(abs(min(vals)), abs(max(vals)), 0.01)
        sc = ax.scatter(
            [r["init_log_alpha"] for r in results],
            [r["init_log_scale"] for r in results],
            c=vals, cmap="RdYlGn", vmin=-vabs, vmax=vabs,
            s=40, edgecolors="black", linewidths=0.3,
        )
        ax.set_xlabel("init_log_alpha")
        ax.set_ylabel("init_log_scale")
        label = metric.replace("_delta", "").upper()
        ax.set_title(f"{slug}: {label} delta")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        path = SWEEP_RESULTS_DIR / f"{slug}_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path.name}")


def _plot_pareto(results: list[dict], pareto: list[dict], slug: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto_keys = {(p["init_log_alpha"], p["init_log_scale"]) for p in pareto}
    non_pareto = [r for r in results
                  if (r["init_log_alpha"], r["init_log_scale"]) not in pareto_keys]

    fig, ax = plt.subplots(figsize=(8, 6))

    if non_pareto:
        ax.scatter(
            [r["mc1_delta"] for r in non_pareto],
            [r["mc2_delta"] for r in non_pareto],
            c="#cccccc", s=15, alpha=0.5, label="dominated", zorder=1,
        )

    if pareto:
        sc = ax.scatter(
            [r["mc1_delta"] for r in pareto],
            [r["mc2_delta"] for r in pareto],
            c=[r["mc0_delta"] for r in pareto],
            cmap="plasma", edgecolors="black", linewidths=0.8,
            s=50, label="pareto", zorder=5,
        )
        fig.colorbar(sc, ax=ax, label="MC0 delta")

        for r in pareto:
            ax.annotate(
                f"(\u03b1={r['init_log_alpha']:.1f}, s={r['init_log_scale']:.1f})",
                (r["mc1_delta"], r["mc2_delta"]),
                fontsize=6, textcoords="offset points", xytext=(4, 4),
            )

    ax.set_xlabel("MC1 delta")
    ax.set_ylabel("MC2 delta")
    ax.set_title(f"{slug}: pareto frontier (MC0 delta as color)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = SWEEP_RESULTS_DIR / f"{slug}_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def cmd_analyze(args):
    import wandb

    api = wandb.Api()
    sweep = api.sweep(args.sweep_path)

    results = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        c = run.config
        s = run.summary
        if "mc1_delta" not in s:
            continue
        results.append({
            "init_log_alpha": c.get("init_log_alpha", 0),
            "init_log_scale": c.get("init_log_scale", 0),
            "mc0": s.get("mc0", 0),
            "mc1": s.get("mc1", 0),
            "mc2": s.get("mc2", 0),
            "mc0_delta": s.get("mc0_delta", 0),
            "mc1_delta": s.get("mc1_delta", 0),
            "mc2_delta": s.get("mc2_delta", 0),
        })

    print(f"Loaded {len(results)} completed runs")
    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Infer model slug from sweep name (gate-init-smol or gate-init-qwen)
    sweep_name = sweep.config.get("name", "")
    slug = sweep_name.replace("gate-init-", "") if "gate-init-" in sweep_name else "model"

    (SWEEP_RESULTS_DIR / f"{slug}_sweep_raw.json").write_text(json.dumps(results, indent=2))

    objectives = ["mc0_delta", "mc1_delta", "mc2_delta"]
    pareto = _compute_pareto(results, objectives)
    pareto_sorted = sorted(pareto, key=lambda r: -r["mc1_delta"])

    print(f"\n{slug}: {len(results)} runs, {len(pareto)} pareto-optimal")
    print(f"  {'alpha':>7} {'scale':>7} {'mc0\u0394':>8} {'mc1\u0394':>8} {'mc2\u0394':>8}")
    for p in pareto_sorted:
        print(f"  {p['init_log_alpha']:7.2f} {p['init_log_scale']:7.2f} "
              f"{p['mc0_delta']:+8.4f} {p['mc1_delta']:+8.4f} {p['mc2_delta']:+8.4f}")

    _plot_landscape(results, slug)
    _plot_pareto(results, pareto, slug)

    (SWEEP_RESULTS_DIR / f"{slug}_pareto.json").write_text(json.dumps(pareto_sorted, indent=2))
    print(f"\nAll results saved to {SWEEP_RESULTS_DIR}/")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description="Gate init Bayesian sweep")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("sweep", help="Create and run wandb Bayesian sweep")
    sp.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="Model to sweep (one sweep per model)")
    sp.add_argument("--sweep-id", type=str, default=None, help="Resume existing sweep")
    sp.add_argument("--count", type=int, default=None, help="Max runs for this agent")

    ap = sub.add_parser("analyze", help="Pull results and compute pareto frontier")
    ap.add_argument("sweep_path", help="Wandb sweep path (entity/project/sweep_id)")

    args = parser.parse_args()

    if args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
