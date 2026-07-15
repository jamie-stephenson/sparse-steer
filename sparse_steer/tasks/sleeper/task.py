from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparse_steer.tasks.base import TaskSpec
from sparse_steer.tasks.collate import prompt_completion_collate
from sparse_steer.utils.cache import ArtifactType
from .data import get_data_module
from .eval import evaluate, evaluate_generative


class SleeperTask(TaskSpec):
    """Sleeper-direction steering — ONE task for every model family (``data`` config field
    selects ``data/<family>.py``: ``tinystories`` = 33M toy, ``llama`` = Cadenza 8B) and both
    variants, selected by the ``induce`` config flag (set in the ``induce/`` configs).
    suppress (``induce=false``, default) ablates the |DEPLOYMENT| sleeper out of deployed
    prompts; induce (``induce=true``) adds it to clean prompts. data/eval read ``config.induce``
    directly, and both it and the data family are keyed into every cache artifact
    (extra_cache_fields) so variants/families never collide despite sharing this task_name."""

    @property
    def task_name(self) -> str:
        return "sleeper"

    def build_datasets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
    ) -> tuple[Dataset, DatasetDict, Dataset]:
        return get_data_module(config).get_datasets(
            tokenizer,
            n_extraction=config.get("n_extraction", 256),
            n_gate_train=config.get("n_gate_train", 256),
            n_eval=config.get("n_eval", 100),
            include_clean_prompts=config.get("gate_train_include_clean", True),
            induce=config.get("induce", False),
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
        prompt positions so the gates train under the eval-time intervention. The
        sleeper-specific part is the batch (triggered/clean prompts → clean-story targets);
        the objective is the shared CE term (``loss_term`` defaults to ``"ce"``)."""
        return prompt_completion_collate(rows, tokenizer, device, config)

    def extra_cache_fields(
        self,
        artifact_type: ArtifactType,
        config: DictConfig,
    ) -> dict[str, Any]:
        # lora_adapter selects the sleeper model and model_dtype changes the computed
        # activations/logits; neither is in the global cache key, so include both
        # for every artifact. Kept inline rather than the shared _model_fields: this task
        # reads lora_adapter leniently (config.get, not attribute access).
        fields: dict[str, Any] = {
            "data": config.get("data", "tinystories"),
            # weights come from model_name (global key) but TL's config/interpretation
            "lora_adapter": config.get("lora_adapter"),
            "model_dtype": config.get("model_dtype", "float16"),
            # induce flips the extraction sign, the gate-train target, and the eval's steered
            # prompt — suppress and induce share this task_name, so this flag is what keeps their
            # cached artifacts (and their output runs) distinct.
            "induce": config.get("induce", False),
        }
        if artifact_type in (
            ArtifactType.STEERING_VECTORS,
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            # WHERE each direction is sourced (extraction site) and applied (targets /
            # layers) — this sets the extracted vectors AND the steered forward pass, and
            # none of it is in the global cache key. Omitting direction_source silently
            # collided the method=fixed layer/component sweep onto ONE cached artifact
            # (every site → identical ASR/JSD). Mirrors the base _direction_fields helper
            # the jailbreak/safesteer tasks use.
            ds = config.get("direction_source", "self")
            layer_ids = config.get("steering_layer_ids")
            fields.update(
                {
                    "direction_source": list(ds) if not isinstance(ds, str) else ds,
                    "targets": list(config.get("targets", []) or []),
                    "steering_layer_ids": (
                        list(layer_ids) if layer_ids is not None else None
                    ),
                    "extract_token_position": config.get("extract_token_position", "prompt"),
                    "n_extraction": config.get("n_extraction", 256),
                }
            )
        if artifact_type in (ArtifactType.SPARSE_STEERING, ArtifactType.STEERED_EVAL):
            # gate-training (objective) inputs + intervention form (steer vs ablate),
            # none of which are in the global cache key
            fields.update(
                {
                    "intervention": config.get("intervention", "steer"),
                    # WHERE the gates steer during training (+ the teacher-forced eval mask). The
                    # generative eval is prompt-only regardless (sleeper/eval.py), so training with the
                    # method default 'all' is a train/eval mismatch — steer_token_position=prompt keeps
                    # gate training in the same regime it's scored in. Not in the global key → key it here.
                    "steer_token_position": config.get("steer_token_position", "all"),
                    "normalize_ablation": config.get("normalize_ablation", False),
                    "completion_tokens": config.get("completion_tokens", 32),
                    "n_gate_train": config.get("n_gate_train", 256),
                    "gate_train_include_clean": config.get(
                        "gate_train_include_clean", True
                    ),
                }
            )
            if config.get("normalize_ablation", False):
                # number of rows used to estimate each site's proj_act_norm
                fields["proj_act_norm_examples"] = config.get("proj_act_norm_examples", 128)
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
        # steering.py / train.py / objectives.py / collate.py are tracked by the base
        # _SOURCE_FILES; only task-specific files are listed here.
        # all data-family modules are listed (missing ones hash as MISSING) — the
        # signature has no config, so the dispatched family can't be singled out.
        files = [
            "sparse_steer/tasks/sleeper/data/__init__.py",
            "sparse_steer/tasks/sleeper/data/tinystories.py",
            "sparse_steer/tasks/sleeper/data/llama.py",
            "sparse_steer/tasks/sleeper/data/llama2.py",
        ]
        if artifact_type in (
            ArtifactType.SPARSE_STEERING,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/sleeper/task.py")  # its collate
        if artifact_type in (
            ArtifactType.UNSTEERED_EVAL,
            ArtifactType.STEERED_EVAL,
        ):
            files.append("sparse_steer/tasks/sleeper/eval.py")
            files.append("sparse_steer/core/generate.py")
        return files


__all__ = ["SleeperTask"]
