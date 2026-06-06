import math
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)
import wandb

from sparse_steer.core.steering import SteeringModel
from sparse_steer.core.gate_tracker import GateTracker, render_gate_animation, render_gate_heatmap
from sparse_steer.tasks.base import TaskSpec

L0Schedule = Callable[[int, int], float]


# ── L0 schedule registry ─────────────────────────────────────────────


def _exponential_decay(step: int, max_steps: int) -> float:
    return math.exp(-3.0 * step / max(max_steps, 1))


def _constant(step: int, max_steps: int) -> float:
    return 1.0


def _cosine_decay(step: int, max_steps: int) -> float:
    """Cosine from 1 to 0 — strong early sparsity, gentle end."""
    return 0.5 * (1.0 + math.cos(math.pi * step / max(max_steps, 1)))


def _warmup(step: int, max_steps: int, *, warmup_steps: int = 0) -> float:
    """Ramp from 0 to 1 over warmup_steps, then stay at 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


def _none(step: int, max_steps: int) -> float:
    return 0.0


L0_SCHEDULES: dict[str, L0Schedule] = {
    "exponential_decay": _exponential_decay,
    "constant": _constant,
    "cosine_decay": _cosine_decay,
    "warmup": _warmup,
    "none": _none,
}


def get_l0_scheduler_type(name: str, config: DictConfig | None = None) -> L0Schedule:
    if name not in L0_SCHEDULES:
        raise ValueError(
            f"Unknown l0_scheduler_type '{name}'. Available: {sorted(L0_SCHEDULES)}"
        )
    fn = L0_SCHEDULES[name]
    if name == "warmup" and config is not None:
        fn = partial(fn, warmup_steps=config.l0_warmup_steps)
    return fn


def _init_wandb() -> None:
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


# ── Steering training loop ────────────────────────────────────────────


def _train_loop(
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    task: TaskSpec,
    config: DictConfig,
    *,
    lr: float,
    num_epochs: int,
    warmup_steps: int,
    l0_schedule: L0Schedule,
    l0_lambda: float,
    tracker: GateTracker | None,
) -> None:
    """Manual training loop over the learnable steering parameters.

    The task supplies the batch (``task.collate``) and the objective
    (``task.loss``); this loop owns the optimiser, LR/L0 schedules, gradient
    clipping, gate tracking, and the task-independent L0 sparsity penalty. If a
    collated batch carries a ``steer_mask`` the forward is confined to those
    positions (e.g. prompt-only steering).
    """
    device = model.device
    rows = list(dataset)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    batch_size = config.train_batch_size
    max_steps = math.ceil(len(rows) / batch_size) * num_epochs
    scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    generator = torch.Generator().manual_seed(config.seed)
    model.train()
    if tracker is not None:
        tracker.snapshot(0)

    step = 0
    for _ in range(num_epochs):
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(rows), batch_size):
            batch = task.collate(
                [rows[j] for j in order[start : start + batch_size]],
                tokenizer,
                device,
                config,
            )
            steer_ctx = (
                model.steer_positions(batch["steer_mask"])
                if "steer_mask" in batch
                else nullcontext()
            )
            with steer_ctx:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            loss = task.loss(model, batch, logits)
            lam = l0_lambda * l0_schedule(step, max_steps)
            if lam > 0:
                loss = loss + lam * model.l0_penalty()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.logging_steps == 0:
                if tracker is not None:
                    tracker.snapshot(step)
                if config.use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/l0_lambda": lam}, step=step)

    if tracker is not None:
        tracker.snapshot(step)


def train_steering(
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | DatasetDict,
    config: DictConfig,
    output_dir: Path,
    *,
    task: TaskSpec,
) -> SteeringModel:
    """Freeze base weights, then train only the learnable steering parameters
    against the task's objective (``task.loss``)."""
    if config.use_wandb:
        _init_wandb()

    train_split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    model.freeze_base_model(freeze_raw_scale=config.freeze_raw_scale)

    # Equalise gate-gradient scale across ablation sites of differing residual
    # norm so the L0 gates select on objective benefit, not activation norm
    # (see SteeringModel.set_proj_norms). Site-agnostic; default off.
    if config.get("normalize_ablation", False) and model.intervention == "ablate":
        n_norm = min(len(train_split), config.get("proj_norm_examples", 128))
        rows = list(train_split)[:n_norm]
        batch = task.collate(rows, tokenizer, model.device, config)
        model.set_proj_norms(
            batch["input_ids"], batch["attention_mask"], batch["steer_mask"]
        )

    tracker = GateTracker(model, use_wandb=config.use_wandb) if config.track_gates else None

    _train_loop(
        model,
        tokenizer,
        train_split,
        task,
        config,
        lr=config.learning_rate,
        num_epochs=config.num_epochs,
        warmup_steps=config.lr_warmup_steps,
        l0_schedule=get_l0_scheduler_type(config.l0_scheduler_type, config),
        l0_lambda=config.l0_lambda,
        tracker=tracker,
    )

    if tracker is not None and tracker.snapshots.steps:
        render_gate_heatmap(tracker.snapshots, output_dir / "gate_heatmap.png")
        render_gate_animation(tracker.snapshots, output_dir / "gate_animation.gif")

    # ── Phase 2: optional scale tuning with frozen gates ────────────────
    scale_tuning_epochs = config.get("scale_tuning_epochs", 0)
    if scale_tuning_epochs > 0:
        print("Scale tuning with frozen gates...")
        model.freeze_gates()
        _train_loop(
            model,
            tokenizer,
            train_split,
            task,
            config,
            lr=config.get("scale_tuning_lr") or config.learning_rate,
            num_epochs=scale_tuning_epochs,
            warmup_steps=0,
            l0_schedule=_none,
            l0_lambda=0.0,
            tracker=None,
        )

    return model


# ── LoRA training (HuggingFace) ───────────────────────────────────────


def _prepare_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    remove_columns = [
        name
        for name in ("text", "positive", "question_id")
        if name in dataset.column_names
    ]
    return dataset.map(lambda row: tokenizer(row["text"]), remove_columns=remove_columns)


def train_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | DatasetDict,
    config: DictConfig,
    output_dir: Path | None = None,
) -> PreTrainedModel:
    """Wrap a HuggingFace model with LoRA adapters and fine-tune."""
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

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
    "get_l0_scheduler_type",
    "train_steering",
    "train_lora",
]
