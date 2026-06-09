from typing import Any

import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.tasks.base import TaskSpec
from sparse_steer.tasks.collate import prompt_completion_collate
from sparse_steer.utils.cache import ArtifactType
from .data import get_tinysleepers_datasets
from .eval import evaluate, evaluate_generative


class TinySleepersTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "tinysleepers"

    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        return get_tinysleepers_datasets(
            tokenizer,
            n_extraction=config.get("n_extraction", 256),
            n_gate_train=config.get("n_gate_train", 256),
            n_eval=config.get("n_eval", 100),
            include_clean_prompts=config.get("gate_train_include_clean", True),
            seed=config.get("data_seed"),
        )

    def run_task_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        config: DictConfig,
    ) -> dict[str, float]:
        # Non-generative teacher-forced jsd_clean_tf always runs; generative_eval
        # adds the generative asr/jsd_clean/jsd_pois/exact_match (distinct keys).
        metrics = evaluate(model, tokenizer, dataset, config)
        if config.get("generative_eval", False):
            metrics.update(evaluate_generative(model, tokenizer, dataset, config))
        return metrics

    def collate(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        device: Any,
        config: DictConfig,
    ) -> dict[str, Tensor]:
        """Teacher-forced ``prompt + clean_completion`` batch; steering confined to the
        prompt positions so the gates train under the eval-time intervention."""
        return prompt_completion_collate(rows, tokenizer, device, config)

    def loss(self, model, batch: dict[str, Tensor], logits: Tensor) -> Tensor:
        """Teacher-forced next-token CE toward the clean continuation.

        Same cross-entropy as truthfulqa; the sleeper-specific parts are the batch
        (triggered/clean prompts → clean-story targets) and the prompt-only steering
        the loop applies via ``steer_mask``.
        """
        shift_logits = logits[..., :-1, :].float()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            batch["labels"][..., 1:].reshape(-1),
            ignore_index=-100,
        )

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]:
        # lora_adapter selects the sleeper model and dtype changes the computed
        # activations/logits; neither is in the global cache key, so include both
        # for every artifact.
        fields: dict[str, Any] = {
            "lora_adapter": config.get("lora_adapter"),
            "dtype": config.get("dtype", "float16"),
        }
        if artifact_type in (ArtifactType.SPARSE_STEERING, ArtifactType.STEERED_EVAL):
            # gate-training (objective) inputs + intervention form (steer vs ablate),
            # none of which are in the global cache key
            fields.update(
                {
                    "intervention": config.get("intervention", "steer"),
                    "normalize_ablation": config.get("normalize_ablation", False),
                    "completion_tokens": config.get("completion_tokens", 32),
                    "n_gate_train": config.get("n_gate_train", 256),
                    "gate_train_include_clean": config.get(
                        "gate_train_include_clean", True
                    ),
                }
            )
            if config.get("normalize_ablation", False):
                # number of rows used to estimate each site's proj_norm
                fields["proj_norm_examples"] = config.get("proj_norm_examples", 128)
        if artifact_type in (ArtifactType.UNSTEERED_EVAL, ArtifactType.STEERED_EVAL):
            fields.update(
                {
                    "generative_eval": config.get("generative_eval", False),
                    "n_eval": config.get("n_eval", 100),
                    "completion_tokens": config.get("completion_tokens", 32),
                    "gen_tokens": config.get(
                        "gen_tokens", config.get("completion_tokens", 32)
                    ),
                    "eval_seeds": list(config.get("eval_seeds", []) or []),
                    "eval_temperature": config.get("eval_temperature", 1.0),
                }
            )
        return fields

    def cache_source_files(self, artifact_type: ArtifactType) -> list[str]:
        files = ["sparse_steer/tasks/tinysleepers/data.py"]
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            # the training objective (loss) lives here; the ablation/gate hooks
            # (incl. proj_norm normalisation) live in steering.py
            files.append("sparse_steer/tasks/tinysleepers/task.py")
            files.append("sparse_steer/core/steering.py")
        if artifact_type == ArtifactType.SPARSE_STEERING:
            # the gate-training loop + proj_norm setup determine the trained gates
            files.append("sparse_steer/train.py")
        if artifact_type in (
            ArtifactType.UNSTEERED_EVAL,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/tinysleepers/eval.py")
            files.append("sparse_steer/core/generate.py")
        return files


__all__ = ["TinySleepersTask"]
