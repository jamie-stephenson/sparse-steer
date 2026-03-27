import abc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..utils.cache import (
    ArtifactType,
    CacheHit,
    finalize as cache_finalize,
    load_cached_json,
    lookup as cache_lookup,
    prepare_cache_path,
    print_cache_warnings,
    store_json as cache_store_json,
)


# ── Task protocol ─────────────────────────────────────────────────


@runtime_checkable
class TaskSpec(Protocol):
    @property
    def task_name(self) -> str: ...

    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]: ...

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]: ...

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]: ...

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]: ...


# ── Base experiment ───────────────────────────────────────────────


class Experiment(abc.ABC):
    """Base experiment class.  Subclass per method."""

    _eval_artifact_type: ArtifactType = ArtifactType.STEERED_EVAL

    def __init__(self, config: DictConfig, task: TaskSpec) -> None:
        self.config = config
        self.task = task

    # ── Cache helpers ─────────────────────────────────────────

    def _cache_kwargs(self, artifact_type: ArtifactType) -> dict[str, Any]:
        return dict(
            extra_fields=self.task.extra_cache_fields(artifact_type, self.config),
            extra_sources=self.task.cache_source_files(artifact_type),
        )

    def _try_cache_lookup(self, artifact_type: ArtifactType) -> CacheHit | None:
        if not self.config.use_cache:
            return None
        hit = cache_lookup(
            artifact_type,
            self.config,
            self.task.task_name,
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
        return prepare_cache_path(
            artifact_type,
            self.config,
            self.task.task_name,
            extra_fields=self.task.extra_cache_fields(artifact_type, self.config),
        )

    def _finalize_cache(self, artifact_type: ArtifactType) -> Path:
        return cache_finalize(
            artifact_type,
            self.config,
            self.task.task_name,
            **self._cache_kwargs(artifact_type),
        )

    def _cache_store_json(
        self, artifact_type: ArtifactType, data: dict[str, Any]
    ) -> Path:
        if not self.config.use_cache:
            return Path("/dev/null")
        return cache_store_json(
            artifact_type,
            self.config,
            self.task.task_name,
            data,
            **self._cache_kwargs(artifact_type),
        )

    # ── Abstract methods ──────────────────────────────────────

    @abc.abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Load and prepare the model for this method."""
        ...

    @abc.abstractmethod
    def _run_pipeline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        extraction_ds: Dataset,
        train_ds: DatasetDict,
        output_dir: Path,
    ) -> tuple[PreTrainedModel, dict[str, str | None], dict[str, Any]]:
        """Method-specific pipeline (extract, train, etc.).

        Returns ``(model, artifacts, cache_info)``.  The model may be
        replaced (e.g. wrapped by peft) so callers must use the returned
        reference.
        """
        ...

    # ── Run ───────────────────────────────────────────────────

    def _lookup_baseline_metrics(
        self, cache_info: dict[str, Any]
    ) -> dict[str, float]:
        """Look up cached baseline metrics for comparison."""
        if self._eval_artifact_type == ArtifactType.BASELINE_EVAL:
            return {}
        hit = self._try_cache_lookup(ArtifactType.BASELINE_EVAL)
        if hit is not None:
            baseline = load_cached_json(hit.artifact_path)
            cache_info["baseline_eval"] = {
                "status": "hit",
                "path": str(hit.artifact_path),
            }
            for mode, score in baseline.items():
                print(f"  Baseline {mode.upper()}: {score:.4f}")
            return baseline
        print(
            "  (No cached baseline — run baseline experiment first for comparison)"
        )
        return {}

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self) -> dict[str, Any]:
        self._seed_everything(self.config.seed)

        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        model_slug = self.config.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path("output") / self.task.task_name / model_slug / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # ── Datasets ──────────────────────────────────────────
        print("Preparing datasets...")
        extraction_ds, train_ds, eval_ds = self.task.build_datasets(
            tokenizer, self.config
        )

        n_ext_q = (
            len(set(extraction_ds["question_id"]))
            if "question_id" in extraction_ds.column_names
            else 0
        )
        train_split = (
            train_ds["train"] if isinstance(train_ds, DatasetDict) else train_ds
        )
        val_split = (
            train_ds.get("val", train_ds.get("validation"))
            if isinstance(train_ds, DatasetDict)
            else None
        )
        n_train_q = (
            len(set(train_split["question_id"]))
            if "question_id" in train_split.column_names
            else 0
        )
        n_val_q = (
            len(set(val_split["question_id"]))
            if val_split and "question_id" in val_split.column_names
            else 0
        )
        n_val = len(val_split) if val_split else 0
        print(
            f"  Extraction: {len(extraction_ds)} examples ({n_ext_q} questions), "
            f"Gate training: {len(train_split)} train ({n_train_q} qs) / "
            f"{n_val} val ({n_val_q} qs), "
            f"Eval: {len(eval_ds)} questions"
        )

        # ── Model + method pipeline ───────────────────────────
        model = self._load_model()
        model, artifacts, cache_info = self._run_pipeline(
            model, tokenizer, extraction_ds, train_ds, output_dir
        )

        # ── Eval ──────────────────────────────────────────────
        metrics: dict[str, float] = {}
        eval_hit = self._try_cache_lookup(self._eval_artifact_type)
        if eval_hit is not None:
            metrics = load_cached_json(eval_hit.artifact_path)
            cache_info[self._eval_artifact_type.value] = {
                "status": "hit",
                "path": str(eval_hit.artifact_path),
            }
        else:
            print(f"Evaluating ({self.config.method})...")
            metrics = self.task.run_task_evaluation(
                model, tokenizer, eval_ds, self.config
            )
            self._cache_store_json(self._eval_artifact_type, metrics)
            cache_info[self._eval_artifact_type.value] = {"status": "miss"}
        for mode, score in metrics.items():
            print(f"  {mode.upper()}: {score:.4f}")

        # ── Baseline comparison ───────────────────────────────
        baseline_metrics = self._lookup_baseline_metrics(cache_info)

        # ── Summary ───────────────────────────────────────────
        summary: dict[str, Any] = {
            "task": self.task.task_name,
            "method": self.config.method,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "artifacts": artifacts,
            "cache_info": cache_info,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Run summary saved to {summary_path}")
        return summary


__all__ = [
    "Experiment",
    "TaskSpec",
]
