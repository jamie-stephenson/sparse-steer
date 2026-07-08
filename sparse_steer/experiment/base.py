import abc
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.core.loading import load_tokenizer
from sparse_steer.tasks.base import TaskSpec
from sparse_steer.utils.cache import (
    ArtifactType,
    CacheHit,
    finalize as cache_finalize,
    load_cached_json,
    lookup as cache_lookup,
    prepare_cache_path,
    print_cache_warnings,
    store_json as cache_store_json,
)


# ── Base experiment ───────────────────────────────────────────────


class Experiment(abc.ABC):
    """Base experiment class.  Subclass per method."""

    _eval_artifact_type: ArtifactType = ArtifactType.STEERED_EVAL

    def __init__(self, config: DictConfig, task: TaskSpec) -> None:
        self.config = config
        self.task = task

    # ── Cache helpers ─────────────────────────────────────────

    def _cache_kwargs(self, artifact_type: ArtifactType) -> dict[str, Any]:
        extra_fields = dict(self.task.extra_cache_fields(artifact_type, self.config))
        if (
            artifact_type == ArtifactType.STEERED_EVAL
            and self.config.get("refinement_method") == "iti_head_select"
        ):
            extra_fields["refinement_method"] = "iti_head_select"
            extra_fields["iti_topk"] = int(self.config.get("iti_topk", 48))
            extra_fields["iti_scale"] = float(self.config.get("iti_scale", 15.0))
        # Inspect capability/safety canaries are a cross-cutting eval add-on merged in run() (not
        # owned by any task), so the eval artifacts must key on which were requested + the per-eval
        # limit. Only add the keys when some are requested → eval caches with `inspect_evals` empty
        # keep their existing key (the feature stays off-by-default without busting prior caches).
        if artifact_type in (ArtifactType.STEERED_EVAL, ArtifactType.UNSTEERED_EVAL):
            requested = self.config.get("inspect_evals") or []
            if requested:
                extra_fields["inspect_evals"] = sorted(requested)
                extra_fields["inspect_eval_limit"] = self.config.get("inspect_eval_limit")
            # lm-eval-harness canaries (loglikelihood MMLU/ARC, wikitext perplexity) — same keying
            # discipline as the Inspect canaries; off-by-default keeps prior caches valid.
            lmeval = self.config.get("lmeval_tasks") or []
            if lmeval:
                extra_fields["lmeval_tasks"] = sorted(lmeval)
                extra_fields["lmeval_limit"] = self.config.get("lmeval_limit")
                extra_fields["lmeval_steer"] = self.config.get("lmeval_steer", "all")
                extra_fields["lmeval_fewshot"] = self.config.get("lmeval_fewshot")
                # Chat-template protocol keys — only added when engaged, so prior fixed-format
                # (leaderboard-anchored) eval caches keep their keys and stay valid.
                if self.config.get("lmeval_chat_template", False):
                    extra_fields["lmeval_chat_template"] = True
                    extra_fields["lmeval_fewshot_multiturn"] = self.config.get("lmeval_fewshot_multiturn", False)
                    if self.config.get("lmeval_system_instruction"):
                        extra_fields["lmeval_system_instruction"] = self.config.get("lmeval_system_instruction")
                if self.config.get("lmeval_include_path"):
                    extra_fields["lmeval_include_path"] = self.config.get("lmeval_include_path")
        return dict(
            extra_fields=extra_fields,
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

    def _lookup_unsteered_metrics(
        self, cache_info: dict[str, Any]
    ) -> dict[str, float]:
        """Look up cached unsteered metrics for comparison."""
        if self._eval_artifact_type == ArtifactType.UNSTEERED_EVAL:
            return {}
        hit = self._try_cache_lookup(ArtifactType.UNSTEERED_EVAL)
        if hit is not None:
            unsteered = load_cached_json(hit.artifact_path)
            cache_info["unsteered_eval"] = {
                "status": "hit",
                "path": str(hit.artifact_path),
            }
            for mode, score in unsteered.items():
                print(f"  Unsteered {mode.upper()}: {score:.4f}")
            return unsteered
        print(
            "  (No cached unsteered — run unsteered experiment first for comparison)"
        )
        return {}

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self) -> dict[str, Any]:
        self._seed_everything(self.config.seed)

        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if hf_token:
            login(token=hf_token)

        model_slug = self.config.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path("output") / self.task.task_name / model_slug / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = load_tokenizer(self.config)

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
        # Gate training leaves the model in train mode, where hard-concrete gates
        # sample stochastic noise and skip the eval threshold. Evaluate deterministically.
        model.eval()

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
            # eval_backend=hf: swap the SteeringModel's engine to native HF (sdpa) for the
            # eval — AFTER training/extraction (both TL-only), BEFORE any scored forward.
            # Results are backend-independent (validated), so eval_backend is deliberately
            # NOT part of any cache key. Plain HF/LoRA models have no set_backend → skipped.
            if (
                str(self.config.get("eval_backend", "tl")) == "hf"
                and getattr(model, "backend", None) == "tl"
            ):
                print("  eval_backend=hf: switching SteeringModel engine to HF (sdpa)...")
                model.set_backend("hf")
            print(f"Evaluating ({self.config.method})...")
            metrics = self.task.run_task_evaluation(
                model, tokenizer, eval_ds, self.config
            )
            # Shared, task-agnostic Inspect canaries (capability/safety), merged into the same
            # cached eval dict. Runs against the model as-is, so steering experiments get steered
            # numbers and the unsteered experiment gets the base reference — same line, both sides.
            # Lazy-imported so `inspect_ai` is a dependency only when canaries are actually selected.
            requested = self.config.get("inspect_evals") or []
            if requested:
                from sparse_steer.core.inspect_provider import run_requested_inspect_evals

                metrics.update(
                    run_requested_inspect_evals(
                        model, tokenizer, requested,
                        limit=self.config.get("inspect_eval_limit"),
                    )
                )
            # lm-eval-harness canaries (loglikelihood MMLU/ARC + wikitext CE), lazy-imported like
            # Inspect. Steered model → steered numbers; unsteered experiment → base reference.
            lmeval = self.config.get("lmeval_tasks") or []
            if lmeval:
                from sparse_steer.core.lmeval_provider import run_requested_lmeval_tasks

                metrics.update(
                    run_requested_lmeval_tasks(
                        model, tokenizer, lmeval,
                        limit=self.config.get("lmeval_limit"),
                        steer=self.config.get("lmeval_steer", "all"),
                        num_fewshot=self.config.get("lmeval_fewshot"),
                        apply_chat_template=self.config.get("lmeval_chat_template", False),
                        fewshot_as_multiturn=self.config.get("lmeval_fewshot_multiturn", False),
                        system_instruction=self.config.get("lmeval_system_instruction"),
                        include_path=self.config.get("lmeval_include_path"),
                    )
                )
            self._cache_store_json(self._eval_artifact_type, metrics)
            cache_info[self._eval_artifact_type.value] = {"status": "miss"}
        for mode, score in metrics.items():
            print(f"  {mode.upper()}: {score:.4f}")

        # ── Unsteered comparison ──────────────────────────────
        unsteered_metrics = self._lookup_unsteered_metrics(cache_info)

        # ── Summary ───────────────────────────────────────────
        summary: dict[str, Any] = {
            "task": self.task.task_name,
            "method": self.config.method,
            "metrics": metrics,
            "unsteered_metrics": unsteered_metrics,
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
]
