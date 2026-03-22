import abc
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import os

import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .utils.cache import (
    ArtifactType,
    CacheHit,
    finalize as cache_finalize,
    load_cached_json,
    lookup as cache_lookup,
    prepare_cache_path,
    print_cache_warnings,
    store_json as cache_store_json,
)
from .extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from .models import MODEL_REGISTRY, DENSE_MODEL_REGISTRY
from .hardconcrete import HardConcreteConfig
from .train import train_gates

Stage = Literal["extract", "train", "eval"]

_METHOD_STAGES: dict[str, list[Stage]] = {
    "sparse": ["extract", "train", "eval"],
    "dense": ["extract", "eval"],
}


@dataclass
class ExperimentConfig:
    model_name: str
    seed: int = 42

    # === Method ===
    method: str = "sparse"  # "sparse" or "dense"
    steering_strength: float = 1.0  # dense only: steering multiplier
    steering_layer_ids: list[int] | None = None  # None = all layers

    # === Data ===
    extraction_fraction: float = 0.5

    # === Activation extraction ===
    extract_batch_size: int = 8
    token_position: str = "last"
    targets: list[str] = field(default_factory=lambda: ["attention"])
    normalize_steering_vectors: bool = False

    # === Gate training ===
    l0_scheduler_type: str = "warmup"
    l0_warmup_steps: int = 0  # steps to ramp L0 penalty (warmup schedule only)
    l0_lambda: float = 0.1
    learning_rate: float = 5e-3
    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 0
    num_epochs: int = 5
    train_batch_size: int = 8
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "no"
    freeze_log_scale: bool = False  # if True, only learn gate selection (log_alpha), not magnitude
    track_gates: bool = False  # if True, record gate norms during training and save heatmap + animation

    # === Evaluation ===
    eval_batch_size: int = 32

    # === wandb ===
    use_wandb: bool = False

    # === Cache ===
    use_cache: bool = True

    # === Device ===
    device: str = "cpu"

    # === Gate config ===
    gate_config: dict[str, Any] = field(default_factory=dict)



class SparseSteeringExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @property
    @abc.abstractmethod
    def task_name(self) -> str:
        ...

    @abc.abstractmethod
    def build_datasets(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        ...

    @abc.abstractmethod
    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> dict[str, float]:
        ...

    # ── Cache hooks (override in task subclasses) ─────────────────

    def extra_cache_fields(self, artifact_type: ArtifactType) -> dict[str, Any]:
        """Task-specific config fields to include in the cache key."""
        return {}

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        """Task-specific source files for staleness detection."""
        return []

    # ── Cache helpers ─────────────────────────────────────────────

    def _cache_kwargs(self, artifact_type: ArtifactType) -> dict[str, Any]:
        return dict(
            extra_fields=self.extra_cache_fields(artifact_type),
            extra_sources=self.cache_source_files(artifact_type),
        )

    def _try_cache_lookup(self, artifact_type: ArtifactType) -> CacheHit | None:
        if not self.config.use_cache:
            return None
        hit = cache_lookup(
            artifact_type, self.config, self.task_name,
            **self._cache_kwargs(artifact_type),
        )
        if hit is not None:
            print_cache_warnings(artifact_type.value, hit.warnings)
            try:
                rel = hit.artifact_path.relative_to(Path.cwd())
            except ValueError:
                rel = hit.artifact_path
            print(f"  Cache hit: {artifact_type.value} ({rel})")
        return hit

    def _prepare_cache_path(self, artifact_type: ArtifactType) -> Path:
        """Return the cache path where an artifact should be saved directly."""
        return prepare_cache_path(
            artifact_type, self.config, self.task_name,
            extra_fields=self.extra_cache_fields(artifact_type),
        )

    def _finalize_cache(self, artifact_type: ArtifactType) -> Path:
        """Write the cache manifest after the artifact has been saved."""
        return cache_finalize(
            artifact_type, self.config, self.task_name,
            **self._cache_kwargs(artifact_type),
        )

    def _cache_store_json(self, artifact_type: ArtifactType, data: dict[str, Any]) -> Path:
        if not self.config.use_cache:
            return Path("/dev/null")
        return cache_store_json(
            artifact_type, self.config, self.task_name, data,
            **self._cache_kwargs(artifact_type),
        )

    # ── Main pipeline ─────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        self._validate_lifecycle()
        stages = _METHOD_STAGES[self.config.method]
        stage_set = set(stages)

        load_dotenv()
        if token := os.getenv("HF_API_KEY"):
            login(token=token)
        model_slug = self.config.model_name.split("/")[-1]
        output_dir = Path("output") / self.task_name / model_slug / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Preparing datasets...")
        extraction_ds, gate_train_ds, eval_ds = self.build_datasets(tokenizer)
        n_extraction_q = len(set(extraction_ds["question_id"])) if "question_id" in extraction_ds.column_names else len(extraction_ds)
        train_ds, val_ds = gate_train_ds["train"], gate_train_ds["val"]
        n_train_q = len(set(train_ds["question_id"])) if "question_id" in train_ds.column_names else len(train_ds)
        n_val_q = len(set(val_ds["question_id"])) if "question_id" in val_ds.column_names else len(val_ds)
        print(
            f"  Extraction: {len(extraction_ds)} examples ({n_extraction_q} questions), "
            f"Gate training: {len(train_ds)} train ({n_train_q} qs) / {len(val_ds)} val ({n_val_q} qs), "
            f"Eval: {len(eval_ds)} questions"
        )

        hf_config = AutoConfig.from_pretrained(self.config.model_name)
        model_type = getattr(hf_config, "model_type", None)

        if self.config.method == "dense":
            registry = DENSE_MODEL_REGISTRY
        else:
            registry = MODEL_REGISTRY

        model_cls = registry.get(model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for '{self.config.model_name}'. "
                f"Supported: {sorted(registry)}"
            )

        steering_vectors_path: str | None = None
        sparse_steering_path: str | None = None
        metrics: dict[str, float] = {}
        baseline_metrics: dict[str, float] = {}
        cache_info: dict[str, Any] = {}

        model = model_cls.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
        ).to(self.config.device)

        layer_ids = self.config.steering_layer_ids or list(range(len(model.get_layers())))

        if self.config.method == "sparse":
            gate_config = HardConcreteConfig(**self.config.gate_config)
            print("Initialising sparse-steering model...")
            model.upgrade_for_steering(
                gate_config=gate_config,
                steering_layer_ids=layer_ids,
                steering_components=self.config.targets,
            )
        else:
            print(f"Initialising dense-steering model (steering_strength={self.config.steering_strength})...")
            model.upgrade_for_steering(
                steering_strength=self.config.steering_strength,
                steering_layer_ids=layer_ids,
                steering_components=self.config.targets,
            )

        # ── Train (check cache first — includes vectors) ─────────
        ss_hit = self._try_cache_lookup(ArtifactType.SPARSE_STEERING) if "train" in stage_set else None
        if ss_hit is not None:
            model.load_steering(ss_hit.artifact_path)
            sparse_steering_path = str(ss_hit.artifact_path)
            cache_info["sparse_steering"] = {"status": "hit", "path": sparse_steering_path}
            # restore cached gate plots
            for name in ("gate_heatmap.png", "gate_animation.gif"):
                cached = ss_hit.artifact_path.parent / name
                if cached.is_file():
                    shutil.copy2(cached, output_dir / name)
        else:
            # ── Extract ───────────────────────────────────────────
            sv_hit = self._try_cache_lookup(ArtifactType.STEERING_VECTORS)
            if sv_hit is not None:
                steering_vectors, _ = load_steering_vectors(sv_hit.artifact_path)
                steering_vectors_path = str(sv_hit.artifact_path)
                cache_info["steering_vectors"] = {"status": "hit", "path": steering_vectors_path}
            elif "extract" in stage_set:
                extraction_with_activations, component_names = collect_activations(
                    extraction_ds,
                    model,
                    tokenizer,
                    targets=self.config.targets,
                    batch_size=self.config.extract_batch_size,
                    token_position=self.config.token_position,
                )
                print("Computing steering vectors...")
                steering_vectors = extract_steering_vectors(
                    extraction_with_activations,
                    component_names,
                )
                sv_dest = self._prepare_cache_path(ArtifactType.STEERING_VECTORS)
                save_steering_vectors(steering_vectors, sv_dest, metadata=asdict(self.config))
                steering_vectors_path = str(self._finalize_cache(ArtifactType.STEERING_VECTORS))
                print(f"Saved steering vectors to {steering_vectors_path}")
                cache_info["steering_vectors"] = {"status": "miss"}

            if steering_vectors_path is not None:
                model.set_all_vectors(steering_vectors, normalize=self.config.normalize_steering_vectors)

            # ── Train ─────────────────────────────────────────────
            if "train" in stage_set:
                assert gate_train_ds is not None
                print("Training gates...")
                train_gates(model, tokenizer, gate_train_ds, self.config, output_dir=output_dir)
                ss_dest = self._prepare_cache_path(ArtifactType.SPARSE_STEERING)
                model.save_steering(ss_dest)
                # copy gate plots into cache alongside the checkpoint
                for name in ("gate_heatmap.png", "gate_animation.gif"):
                    src = output_dir / name
                    if src.is_file():
                        shutil.copy2(src, ss_dest.parent / name)
                sparse_steering_path = str(self._finalize_cache(ArtifactType.SPARSE_STEERING))
                print(f"Saved sparse steering to {sparse_steering_path}")
                cache_info["sparse_steering"] = {"status": "miss"}

        # ── Eval ──────────────────────────────────────────────────
        if "eval" in stage_set:
            assert eval_ds is not None

            # Baseline eval
            baseline_hit = self._try_cache_lookup(ArtifactType.BASELINE_EVAL)
            if baseline_hit is not None:
                baseline_metrics = load_cached_json(baseline_hit.artifact_path)
                cache_info["baseline_eval"] = {"status": "hit", "path": str(baseline_hit.artifact_path)}
            else:
                print("Evaluating baseline (steering disabled)...")
                with model.steering_disabled():
                    baseline_metrics = self.run_task_evaluation(model, tokenizer, eval_ds)
                self._cache_store_json(ArtifactType.BASELINE_EVAL, baseline_metrics)
                cache_info["baseline_eval"] = {"status": "miss"}
            for mode, score in baseline_metrics.items():
                print(f"  Baseline {mode.upper()}: {score:.4f}")

            # Steered eval
            steered_hit = self._try_cache_lookup(ArtifactType.STEERED_EVAL)
            if steered_hit is not None:
                metrics = load_cached_json(steered_hit.artifact_path)
                cache_info["steered_eval"] = {"status": "hit", "path": str(steered_hit.artifact_path)}
            else:
                print("Evaluating with steering...")
                metrics = self.run_task_evaluation(model, tokenizer, eval_ds)
                self._cache_store_json(ArtifactType.STEERED_EVAL, metrics)
                cache_info["steered_eval"] = {"status": "miss"}
            for mode, score in metrics.items():
                print(f"  Steered {mode.upper()}: {score:.4f}")

        # ── Summary ───────────────────────────────────────────────
        summary: dict[str, Any] = {
            "task": self.task_name,
            "method": self.config.method,
            "stages": stages,
            "baseline_metrics": baseline_metrics if "eval" in stage_set else {},
            "steered_metrics": metrics if "eval" in stage_set else {},
            "artifacts": {
                "steering_vectors_path": steering_vectors_path,
                "sparse_steering_path": sparse_steering_path,
            },
            "cache_info": cache_info,
            "config": asdict(self.config),
        }

        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Run summary saved to {summary_path}")

        return summary

    def _validate_lifecycle(self) -> None:
        if self.config.method not in _METHOD_STAGES:
            raise ValueError(
                f"Unknown method '{self.config.method}'. "
                f"Supported: {sorted(_METHOD_STAGES)}"
            )


__all__ = [
    "ExperimentConfig",
    "SparseSteeringExperiment",
]
