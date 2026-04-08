import math
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from dotenv import load_dotenv
from torch import Tensor, nn
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import wandb

from .models.hook import SteeringHook
from .models.steering import SteeringLM
from .utils.gate_tracker import GateTracker, render_gate_heatmap, render_gate_animation

if TYPE_CHECKING:
    from omegaconf import DictConfig

L0Schedule = Callable[[int, int], float]


def compute_l0_penalty(model: nn.Module) -> Tensor:
    """Sum the L0 penalty across all steering hooks with gates."""
    penalty = 0.0
    for module in model.modules():
        if isinstance(module, SteeringHook):
            penalty = penalty + module._l0_penalty()
    return penalty


# ── L0 schedule registry ─────────────────────────────────────────────


def _exponential_decay(step: int, max_steps: int) -> float:
    progress = step / max(max_steps, 1)
    return math.exp(-3.0 * progress)


def _constant(step: int, max_steps: int) -> float:
    return 1.0


def _cosine_decay(step: int, max_steps: int) -> float:
    """Cosine from 1 to 0 — strong early sparsity, gentle end."""
    progress = step / max(max_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _warmup(step: int, max_steps: int, *, warmup_steps: int = 0) -> float:
    """Ramps from 0 to 1 over warmup_steps, then stays at 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


def _none(step: int, max_steps: int) -> float:
    """No L0 penalty — gates learn purely from CE loss."""
    return 0.0


L0_SCHEDULES: dict[str, L0Schedule] = {
    "exponential_decay": _exponential_decay,
    "constant": _constant,
    "cosine_decay": _cosine_decay,
    "warmup": _warmup,
    "none": _none,
}


def get_l0_scheduler_type(
    name: str, config: "ExperimentConfig | None" = None
) -> L0Schedule:
    if name not in L0_SCHEDULES:
        raise ValueError(
            f"Unknown l0_scheduler_type '{name}'. Available: {sorted(L0_SCHEDULES)}"
        )
    fn = L0_SCHEDULES[name]
    if name == "warmup" and config is not None:
        fn = partial(fn, warmup_steps=config.l0_warmup_steps)
    return fn


# ── Trainer ───────────────────────────────────────────────────────────


class SparseSteeringTrainer(Trainer):
    """HF Trainer that adds an L0 sparsity penalty to the CE loss."""

    def __init__(
        self, *args, l0_schedule: L0Schedule, l0_lambda: float = 0.1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.l0_schedule = l0_schedule
        self.l0_lambda = l0_lambda

    def _get_l0_lambda(self) -> float:
        return self.l0_lambda * self.l0_schedule(
            self.state.global_step, self.state.max_steps
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        lam = self._get_l0_lambda()
        if lam > 0:
            loss = loss + lam * compute_l0_penalty(model)
        return (loss, outputs) if return_outputs else loss


def _init_wandb() -> None:
    """Load .env and configure wandb from environment variables."""
    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set in environment or .env file")

    os.environ["WANDB_LOG_MODEL"] = "false"
    wandb.login(key=api_key)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sparse-steer"),
        entity=os.environ.get("WANDB_ENTITY") or None,
    )


def _prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    remove_columns = [
        name
        for name in ("text", "positive", "question_id")
        if name in dataset.column_names
    ]
    return dataset.map(
        lambda row: tokenizer(row["text"]),
        remove_columns=remove_columns,
    )


def train_steering(
    model: SteeringLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | DatasetDict,
    config: "DictConfig",
    output_dir: Path,
    extra_callbacks: list | None = None,
) -> SparseSteeringTrainer:
    """Freeze base weights, train only learnable steering parameters."""
    if config.use_wandb:
        _init_wandb()

    train_source = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    train_dataset = _prepare_dataset(train_source, tokenizer)

    eval_dataset = None
    if isinstance(dataset, DatasetDict) and "val" in dataset:
        eval_dataset = _prepare_dataset(dataset["val"], tokenizer)

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=config.seed,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.lr_warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="wandb" if config.use_wandb else "none",
        fp16=model.device.type == "cuda",
        remove_unused_columns=False,
    )

    model.freeze_base_model(freeze_log_scale=config.freeze_log_scale)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    callbacks = list(extra_callbacks or [])
    tracker = None
    if config.track_gates:
        tracker = GateTracker(model, use_wandb=config.use_wandb)
        callbacks.append(tracker)

    trainer = SparseSteeringTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        l0_schedule=get_l0_scheduler_type(config.l0_scheduler_type, config),
        l0_lambda=config.l0_lambda,
        callbacks=callbacks,
    )
    trainer.train()

    if tracker and tracker.snapshots.steps:
        render_gate_heatmap(tracker.snapshots, output_dir / "gate_heatmap.png")
        render_gate_animation(tracker.snapshots, output_dir / "gate_animation.gif")

    # ── Phase 2: optional scale tuning with frozen gates ────────────
    scale_tuning_epochs = config.get("scale_tuning_epochs", 0)
    if scale_tuning_epochs > 0:
        print("Scale tuning with frozen gates...")
        model.freeze_gates()
        scale_tuning_lr = config.get("scale_tuning_lr") or config.learning_rate
        scale_tuning_args = TrainingArguments(
            output_dir=str(output_dir / "scale_tuning"),
            seed=config.seed,
            num_train_epochs=scale_tuning_epochs,
            per_device_train_batch_size=config.train_batch_size,
            learning_rate=scale_tuning_lr,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=0,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_strategy="no",
            report_to="wandb" if config.use_wandb else "none",
            fp16=model.device.type == "cuda",
            remove_unused_columns=False,
        )
        scale_tuning_trainer = Trainer(
            model=model,
            args=scale_tuning_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        scale_tuning_trainer.train()

    return trainer


def train_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | DatasetDict,
    config: "DictConfig",
    output_dir: Path | None = None,
) -> PreTrainedModel:
    """Wrap model with LoRA adapters and fine-tune on the training data."""
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    if config.use_wandb:
        _init_wandb()

    lora_config = PeftLoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_source = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    train_dataset = _prepare_dataset(train_source, tokenizer)

    eval_dataset = None
    if isinstance(dataset, DatasetDict) and "val" in dataset:
        eval_dataset = _prepare_dataset(dataset["val"], tokenizer)

    train_args = TrainingArguments(
        output_dir=str((output_dir or Path("output")) / "lora_checkpoints"),
        seed=config.seed,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.lr_warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="wandb" if config.use_wandb else "none",
        fp16=model.device.type == "cuda",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    return model


__all__ = [
    "L0Schedule",
    "L0_SCHEDULES",
    "compute_l0_penalty",
    "get_l0_scheduler_type",
    "SparseSteeringTrainer",
    "train_steering",
    "train_lora",
]
