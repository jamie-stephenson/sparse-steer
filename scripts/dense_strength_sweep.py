#!/usr/bin/env python3
"""Sweep scale for dense steering and evaluate TruthfulQA.

No training required — just extract steering vectors once, then eval at
each strength.

Usage:
    uv run python scripts/dense_strength_sweep.py --model smol
    uv run python scripts/dense_strength_sweep.py --model qwen --strengths 0.5 1.0 2.0
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from sparse_steer.extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from sparse_steer.models import MODEL_REGISTRY
from sparse_steer.tasks.truthfulqa.data import get_truthfulqa_datasets
from sparse_steer.tasks.truthfulqa.eval import evaluate

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

MODELS = {
    "smol": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}

DEFAULT_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def _find_steering_vectors(model_name: str) -> Path | None:
    model_slug = model_name.split("/")[-1]
    model_dir = OUTPUT_DIR / "truthfulqa" / model_slug
    paths = sorted(model_dir.glob("*/steering_vectors.pt"))
    return paths[-1] if paths else None


def _set_scale(model, scale: float) -> None:
    """Update fixed scale on all steering hooks in-place."""
    for module in model.modules():
        if hasattr(module, "_fixed_scale"):
            module._fixed_scale = scale


def main():
    parser = argparse.ArgumentParser(description="Dense steering strength sweep")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=DEFAULT_STRENGTHS,
        help="Steering strengths to evaluate",
    )
    parser.add_argument("--targets", nargs="+", default=["attention"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sv-path",
        type=str,
        default=None,
        help="Path to existing steering_vectors.pt (skips extraction)",
    )
    args = parser.parse_args()

    model_name = MODELS[args.model]
    model_slug = model_name.split("/")[-1]
    run_dir = (
        OUTPUT_DIR
        / "truthfulqa"
        / model_slug
        / f"dense_sweep_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer + datasets ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    extraction_ds, _, eval_ds = get_truthfulqa_datasets(
        tokenizer,
        extraction_mcq_mode="mc1",
        gate_train_mcq_mode="mc1",
        extraction_fraction=0.5,
        seed=args.seed,
    )

    # ── Model ──
    hf_config = AutoConfig.from_pretrained(model_name)
    model_cls = MODEL_REGISTRY[hf_config.model_type]
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(args.device)

    model.upgrade_for_steering(
        scale=1.0,
        steering_layer_ids=list(range(len(model.get_layers()))),
        steering_components=args.targets,
    )

    # ── Steering vectors ──
    sv_path = Path(args.sv_path) if args.sv_path else _find_steering_vectors(model_name)
    if sv_path and sv_path.exists():
        print(f"Loading steering vectors from {sv_path}")
        vectors, _ = load_steering_vectors(sv_path)
    else:
        print("Extracting steering vectors...")
        extraction_with_activations, component_names = collect_activations(
            extraction_ds,
            model,
            tokenizer,
            targets=args.targets,
            batch_size=8,
            token_position="last",
        )
        vectors = extract_steering_vectors(extraction_with_activations, component_names)
        sv_out = save_steering_vectors(vectors, run_dir / "steering_vectors.pt")
        print(f"Saved steering vectors to {sv_out}")

    model.set_all_vectors(vectors, normalize=False)

    # ── Baseline (steering disabled) ──
    model.eval()
    print("Evaluating baseline (no steering)...")
    with model.steering_disabled(), torch.no_grad():
        baseline = evaluate(model, tokenizer, eval_ds)
    print(
        f"  Baseline MC0={baseline['mc0']:.4f}  MC1={baseline['mc1']:.4f}  MC2={baseline['mc2']:.4f}"
    )

    # ── Sweep ──
    results = []
    print(f"\nSweeping {len(args.strengths)} strengths...")
    print(
        f"  {'strength':>8}  {'MC0':>8}  {'MC1':>8}  {'MC2':>8}  {'MC0Δ':>8}  {'MC1Δ':>8}  {'MC2Δ':>8}"
    )

    for strength in args.strengths:
        _set_scale(model, strength)
        with torch.no_grad():
            metrics = evaluate(model, tokenizer, eval_ds)

        row = {
            "scale": strength,
            "mc0": metrics["mc0"],
            "mc1": metrics["mc1"],
            "mc2": metrics["mc2"],
            "mc0_delta": metrics["mc0"] - baseline["mc0"],
            "mc1_delta": metrics["mc1"] - baseline["mc1"],
            "mc2_delta": metrics["mc2"] - baseline["mc2"],
        }
        results.append(row)
        print(
            f"  {strength:8.2f}  {row['mc0']:8.4f}  {row['mc1']:8.4f}  {row['mc2']:8.4f}"
            f"  {row['mc0_delta']:+8.4f}  {row['mc1_delta']:+8.4f}  {row['mc2_delta']:+8.4f}"
        )

    # ── Save ──
    out = {
        "model": model_name,
        "baseline": baseline,
        "strengths": results,
    }
    out_path = run_dir / "dense_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
