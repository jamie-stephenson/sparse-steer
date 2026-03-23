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
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import wandb

from .models.sparse import SparseSteeringLM
from .hardconcrete import HardConcreteGateMixin
from .utils.gate_tracker import GateTracker, render_gate_heatmap, render_gate_animation

if TYPE_CHECKING:
    from .experiment import ExperimentConfig

L0Schedule = Callable[[int, int], float]


def compute_l0_penalty(model: nn.Module) -> Tensor:
    """Sum the L0 penalty across all HardConcrete-gated modules."""
    penalty = 0.0
    for module in model.modules():
        if isinstance(module, HardConcreteGateMixin):
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

    os.environ["WANDB_LOG_MODEL"] = "end"
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


def train_gates(
    model: SparseSteeringLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | DatasetDict,
    config: "ExperimentConfig",
    output_dir: Path,
    extra_callbacks: list | None = None,
) -> SparseSteeringTrainer:
    """Freeze base weights, train only HardConcrete gate parameters."""
    if config.use_wandb:
        _init_wandb()

    train_source = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    train_dataset = _prepare_dataset(train_source, tokenizer)

    eval_dataset = None
    if isinstance(dataset, DatasetDict) and "val" in dataset:
        eval_dataset = _prepare_dataset(dataset["val"], tokenizer)

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
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

    return trainer


__all__ = [
    "L0Schedule",
    "L0_SCHEDULES",
    "compute_l0_penalty",
    "get_l0_scheduler_type",
    "SparseSteeringTrainer",
    "train_gates",
]
