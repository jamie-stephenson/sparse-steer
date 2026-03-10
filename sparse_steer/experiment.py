import abc
import json
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

from .extract import (
    collect_activations,
    extract_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)
from .models import MODEL_REGISTRY
from .hardconcrete import HardConcreteConfig
from .train import train_gates

Stage = Literal["extract", "train", "eval"]
VALID_STAGES: frozenset[Stage] = frozenset({"extract", "train", "eval"})


@dataclass
class ExperimentConfig:
    model_name: str
    seed: int = 42

    # === Pipeline ===
    stages: list[Stage] = field(default_factory=lambda: ["extract", "train", "eval"])

    # === Data ===
    extraction_fraction: float = 0.5

    # === Activation extraction ===
    extract_batch_size: int = 8
    token_position: str = "last"
    targets: list[str] = field(default_factory=lambda: ["attention"])

    # === Gate training ===
    l0_schedule: str = "exponential_decay"
    learning_rate: float = 5e-3
    num_epochs: int = 5
    train_batch_size: int = 8
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "no"

    # === Evaluation ===
    eval_batch_size: int = 32

    # === wandb ===
    use_wandb: bool = False

    # === Checkpoint IO ===
    input_steering_vectors_path: str | None = None
    input_sparse_steering_path: str | None = None

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

    def run(self) -> dict[str, Any]:
        self._validate_lifecycle()
        stage_set = set(self.config.stages)

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
        # currently build all datasets even if not using all
        extraction_ds, gate_train_ds, eval_ds = self.build_datasets(tokenizer)
        n_extraction_q = len(set(extraction_ds["question_id"])) if "question_id" in extraction_ds.column_names else len(extraction_ds)
        print(
            f"  Extraction: {len(extraction_ds)} rows ({n_extraction_q} questions), "
            f"Gate training: {len(gate_train_ds['train'])} train / {len(gate_train_ds['val'])} val questions, "
            f"Eval: {len(eval_ds)} questions"
        )

        hf_config = AutoConfig.from_pretrained(self.config.model_name)
        model_type = getattr(hf_config, "model_type", None)

        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for '{self.config.model_name}'. "
                f"Supported: {sorted(MODEL_REGISTRY)}"
            )

        steering_vectors_path: str | None = None
        sparse_steering_path: str | None = None
        metrics: dict[str, float] = {}
        baseline_metrics: dict[str, float] = {}

        model = model_cls.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
        ).to(self.config.device)

        gate_config = HardConcreteConfig(**self.config.gate_config)
        if self.config.input_sparse_steering_path is not None:
            print(f"Loading sparse steering checkpoint: {self.config.input_sparse_steering_path}")
            model.load_sparse_steering(Path(self.config.input_sparse_steering_path))
        else:
            print("Initialising sparse-steering model...")
            model.upgrade_for_steering(
                gate_config=gate_config,
                steering_layer_ids=list(range(len(model.get_layers()))),
                steering_components=self.config.targets,
            )

        if "extract" in stage_set:
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
            sv_path = save_steering_vectors(
                steering_vectors,
                output_dir / "steering_vectors.pt",
                metadata=asdict(self.config),
            )
            steering_vectors_path = str(sv_path)
            print(f"Saved steering vectors to {sv_path}")
            model.set_all_steering(steering_vectors)
        elif self.config.input_steering_vectors_path is not None:
            vectors_path = Path(self.config.input_steering_vectors_path)
            print(f"Loading steering vectors: {vectors_path}")
            steering_vectors, _ = load_steering_vectors(vectors_path)
            model.set_all_steering(steering_vectors)
            steering_vectors_path = str(vectors_path)

        if "train" in stage_set:
            assert gate_train_ds is not None
            print("Training gates...")
            train_gates(model, tokenizer, gate_train_ds, self.config, output_dir=output_dir)
            ss_path = model.save_sparse_steering(output_dir / "sparse_steering.pt")
            sparse_steering_path = str(ss_path)
            print(f"Saved sparse steering to {ss_path}")

        if "eval" in stage_set:
            assert eval_ds is not None
            print("Evaluating baseline (steering disabled)...")
            with model.steering_disabled():
                baseline_metrics = self.run_task_evaluation(model, tokenizer, eval_ds)
            for mode, score in baseline_metrics.items():
                print(f"  Baseline {mode.upper()}: {score:.4f}")

            print("Evaluating with steering...")
            metrics = self.run_task_evaluation(model, tokenizer, eval_ds)
            for mode, score in metrics.items():
                print(f"  Steered {mode.upper()}: {score:.4f}")

        summary: dict[str, Any] = {
            "task": self.task_name,
            "stages": list(self.config.stages),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics if "eval" in stage_set else {},
            "artifacts": {
                "steering_vectors_path": steering_vectors_path,
                "sparse_steering_path": sparse_steering_path,
            },
            "config": asdict(self.config),
        }

        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Run summary saved to {summary_path}")

        if metrics or baseline_metrics:
            results_path = output_dir / "results.json"
            results = {"baseline": baseline_metrics, "steered": metrics}
            results_path.write_text(json.dumps(results, indent=2))
            print(f"Results saved to {results_path}")

        return summary

    def _validate_lifecycle(self) -> None:
        stages = self.config.stages
        if not stages:
            raise ValueError("config.stages must include at least one stage.")

        unknown = set(stages) - VALID_STAGES
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")

        if len(stages) != len(set(stages)):
            raise ValueError(f"Duplicate stages are not allowed: {stages}")

        has_input_vectors = self.config.input_steering_vectors_path is not None
        has_input_sparse = self.config.input_sparse_steering_path is not None

        if has_input_vectors and has_input_sparse:
            raise ValueError(
                "Set at most one of input_steering_vectors_path or "
                "input_sparse_steering_path."
            )

        needs_preexisting_artifact = "extract" not in stages and (
            "train" in stages or "eval" in stages
        )
        if needs_preexisting_artifact and not (has_input_vectors or has_input_sparse):
            raise ValueError(
                "Running train/eval without extract requires either "
                "input_steering_vectors_path or input_sparse_steering_path."
            )


__all__ = [
    "ExperimentConfig",
    "SparseSteeringExperiment",
]
