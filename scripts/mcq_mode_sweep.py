#!/usr/bin/env python3
"""Sweep all 9 combinations of extraction_mcq_mode x gate_train_mcq_mode.

For each combo, runs the full pipeline (extract, train, eval) and reports
mc0/mc1/mc2 metrics plus deltas from the unsteered model.

Usage:
    uv run python scripts/mcq_mode_sweep.py
    uv run python scripts/mcq_mode_sweep.py --config config.yaml
"""

import argparse
import json
from dataclasses import asdict, fields, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer

from sparse_steer.extract import collect_activations, extract_steering_vectors
from sparse_steer.hardconcrete import HardConcreteConfig
from sparse_steer.models import MODEL_REGISTRY
from sparse_steer.tasks.truthfulqa import TruthfulQAConfig
from sparse_steer.tasks.truthfulqa.data import get_truthfulqa_datasets
from sparse_steer.tasks.truthfulqa.eval import evaluate
from sparse_steer.train import train_steering

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SWEEP_RESULTS_DIR = ROOT / "sweep_results"

MCQ_MODES: list[Literal["mc0", "mc1", "mc2"]] = ["mc0", "mc1", "mc2"]


def _load_config(path: Path) -> TruthfulQAConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    valid_fields = {f.name for f in fields(TruthfulQAConfig)}
    unknown = set(payload) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TruthfulQAConfig(**payload)


def main():
    parser = argparse.ArgumentParser(description="MCQ mode combination sweep")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to base config YAML"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Override num_epochs from config"
    )
    args = parser.parse_args()

    base_config = _load_config(ROOT / args.config)
    if args.num_epochs is not None:
        base_config = replace(base_config, num_epochs=args.num_epochs)
    base_config = replace(base_config, track_gates=False)
    model_name = base_config.model_name
    model_slug = model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = SWEEP_RESULTS_DIR / f"mcq_sweep_{model_slug}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model class ──
    hf_config = AutoConfig.from_pretrained(model_name)
    model_cls = MODEL_REGISTRY[hf_config.model_type]

    gate_config = HardConcreteConfig(**base_config.gate_config)

    # ── Eval dataset (mode-independent) ──
    _, _, eval_ds = get_truthfulqa_datasets(
        tokenizer,
        extraction_mcq_mode="mc1",
        gate_train_mcq_mode="mc1",
        extraction_fraction=base_config.extraction_fraction,
        seed=base_config.seed,
    )

    # ── Unsteered (no steering) ──
    print(f"Loading {model_name} for unsteered eval...")
    base_model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(base_config.device)
    base_model.upgrade_for_steering(
        gate_config=gate_config,
        learn_scale=True,
        steering_layer_ids=list(range(len(base_model.get_layers()))),
        steering_components=base_config.targets,
    )
    base_model.eval()
    print("Evaluating unsteered (no steering)...")
    with base_model.steering_disabled(), torch.no_grad():
        unsteered = evaluate(base_model, tokenizer, eval_ds)
    print(
        f"  Unsteered MC0={unsteered['mc0']:.4f}  MC1={unsteered['mc1']:.4f}  MC2={unsteered['mc2']:.4f}"
    )
    del base_model
    if base_config.device == "mps":
        torch.mps.empty_cache()

    # ── Sweep over all 9 combos ──
    results = []
    combos = list(product(MCQ_MODES, MCQ_MODES))
    print(f"\nSweeping {len(combos)} extraction x train mcq_mode combinations...\n")
    print(
        f"  {'extract':>8}  {'train':>8}  {'MC0':>8}  {'MC1':>8}  {'MC2':>8}  {'MC0Δ':>8}  {'MC1Δ':>8}  {'MC2Δ':>8}"
    )

    for extract_mode, train_mode in combos:
        tag = f"ext={extract_mode}_train={train_mode}"
        print(f"\n── {tag} ──")
        run_dir = (
            OUTPUT_DIR / "truthfulqa" / model_slug / f"mcq_sweep_{tag}_{timestamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        config = replace(
            base_config,
            extraction_mcq_mode=extract_mode,
            gate_train_mcq_mode=train_mode,
            use_wandb=False,
        )

        # Fresh model each run
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(config.device)
        model.upgrade_for_steering(
            gate_config=gate_config,
            learn_scale=True,
            steering_layer_ids=list(range(len(model.get_layers()))),
            steering_components=config.targets,
        )

        # Datasets with this combo's mcq modes
        extraction_ds, gate_train_ds, _ = get_truthfulqa_datasets(
            tokenizer,
            extraction_mcq_mode=extract_mode,
            gate_train_mcq_mode=train_mode,
            extraction_fraction=config.extraction_fraction,
            seed=config.seed,
        )

        # Extract steering vectors
        print("  Extracting steering vectors...")
        extraction_with_activations, component_names = collect_activations(
            extraction_ds,
            model,
            tokenizer,
            targets=config.targets,
            batch_size=config.extract_batch_size,
            token_position=config.token_position,
        )
        vectors = extract_steering_vectors(extraction_with_activations, component_names)
        model.set_all_vectors(vectors, normalize=config.normalize_steering_vectors)

        # Train gates
        print("  Training gates...")
        train_steering(model, tokenizer, gate_train_ds, config, output_dir=run_dir)

        # Eval (all three mc metrics)
        print("  Evaluating...")
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, tokenizer, eval_ds)

        row = {
            "extraction_mcq_mode": extract_mode,
            "gate_train_mcq_mode": train_mode,
            "mc0": metrics["mc0"],
            "mc1": metrics["mc1"],
            "mc2": metrics["mc2"],
            "mc0_delta": metrics["mc0"] - unsteered["mc0"],
            "mc1_delta": metrics["mc1"] - unsteered["mc1"],
            "mc2_delta": metrics["mc2"] - unsteered["mc2"],
        }
        results.append(row)
        print(
            f"  {extract_mode:>8}  {train_mode:>8}"
            f"  {row['mc0']:8.4f}  {row['mc1']:8.4f}  {row['mc2']:8.4f}"
            f"  {row['mc0_delta']:+8.4f}  {row['mc1_delta']:+8.4f}  {row['mc2_delta']:+8.4f}"
        )

        del model
        if config.device == "mps":
            torch.mps.empty_cache()

    # ── Summary ──
    print("\n\n═══ Summary ═══")
    print(
        f"  {'extract':>8}  {'train':>8}  {'MC0':>8}  {'MC1':>8}  {'MC2':>8}  {'MC0Δ':>8}  {'MC1Δ':>8}  {'MC2Δ':>8}"
    )
    for row in results:
        print(
            f"  {row['extraction_mcq_mode']:>8}  {row['gate_train_mcq_mode']:>8}"
            f"  {row['mc0']:8.4f}  {row['mc1']:8.4f}  {row['mc2']:8.4f}"
            f"  {row['mc0_delta']:+8.4f}  {row['mc1_delta']:+8.4f}  {row['mc2_delta']:+8.4f}"
        )

    # ── Save ──
    out = {
        "model": model_name,
        "unsteered": unsteered,
        "results": results,
        "config": asdict(base_config),
    }
    out_path = sweep_dir / "mcq_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
