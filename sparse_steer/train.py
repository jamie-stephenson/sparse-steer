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
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
import wandb

from sparse_steer.core.steering import SteeringModel
from sparse_steer.core.gate_tracker import GateTracker, render_gate_animation, render_gate_heatmap
from sparse_steer.objectives import (
    composed_objective,
    kl_term,
)
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
    val_split: Dataset | None = None,
) -> None:
    """Manual training loop over the learnable steering parameters.

    The task supplies the batch (``task.collate``); the objective is the task-independent
    ``composed_objective`` (a weighted sum of ce/kl terms) plus the L0 penalty. This loop owns
    the optimiser, LR/L0 schedules, gradient clipping, and the per-metric training monitor.

    Each monitored metric (``loss``, ``l0_lambda``, ``sparsity``, ``max_steering_strength``,
    ``kl_vs_base``) logs on its own cadence: ``config.metric_logging_steps[name]`` if set, else
    the global ``config.logging_steps``. ``kl_vs_base`` is ``KL(steered ‖ base)`` on a held-out
    probe of the val split's ``kl``-tagged (preserve) rows — only present when such rows exist.
    """
    device = model.device
    rows = list(dataset)
    # Only batch rows whose loss term is active (weight > 0). With kl_weight=0 the KL-preserve
    # rows contribute nothing to the loss, so forwarding them is pure waste — and their
    # longer/variable sequences would otherwise balloon the MPS caching allocator. The val
    # split keeps its KL rows regardless (the monitor probe is built separately).
    active_terms = {
        t for t in ("ce", "contrastive", "kl")
        if float(config.get(f"{t}_weight", 1.0 if t == "ce" else 0.0)) > 0
    }
    rows = [r for r in rows if r.get("loss_term", "ce") in active_terms]
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

    # Held-out KL-preserve probe: the val rows tagged loss_term="kl" (capped). Built once;
    # None when the task has no preserve rows (e.g. CE-only tasks) or no val split.
    kl_probe = None
    if val_split is not None and len(val_split):
        vb = task.collate(list(val_split), tokenizer, device, config)
        kl_mask = vb.get("loss_term_rows", {}).get("kl")
        if kl_mask is not None and bool(kl_mask.any()):
            idx = kl_mask.nonzero(as_tuple=True)[0][:batch_size]
            kl_probe = {k: v[idx] for k, v in vb.items() if torch.is_tensor(v)}
            kl_probe_rows = torch.ones(len(idx), dtype=torch.bool, device=device)

    cadence = config.get("metric_logging_steps", {}) or {}

    def _due(name: str, step: int) -> bool:
        # Free metrics (loss/sparsity/max_steering_strength) log every logging_steps by default;
        # kl_vs_base needs a held-out forward + preserve data, so it is opt-in: logged only when
        # given an explicit cadence in metric_logging_steps (else off).
        default = 0 if name == "kl_vs_base" else config.logging_steps
        c = cadence.get(name, default)
        return bool(c) and step % c == 0

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
            loss = composed_objective(model, batch, logits, config)
            lam = l0_lambda * l0_schedule(step, max_steps)
            if lam > 0:
                loss = loss + lam * model.l0_penalty()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            # Snapshot the gates on the global cadence (drives the heatmap/animation).
            if tracker is not None and step % config.logging_steps == 0:
                tracker.snapshot(step)

            # Per-metric monitor: each metric logs on its own cadence (else logging_steps).
            metrics: dict[str, float] = {}
            if _due("loss", step):
                metrics["train/loss"] = loss.item()
            if _due("l0_lambda", step):
                metrics["train/l0_lambda"] = lam
            if tracker is not None and _due("sparsity", step):
                metrics["train/sparsity"] = tracker.sparsity()
            if tracker is not None and _due("max_steering_strength", step):
                metrics["train/max_steering_strength"] = tracker.max_steering_strength()
            if kl_probe is not None and _due("kl_vs_base", step):
                # KL(steered ‖ base) on the held-out preserve rows, eval mode — drift proxy.
                model.eval()
                with torch.no_grad():
                    logits_p = model(
                        input_ids=kl_probe["input_ids"], attention_mask=kl_probe["attention_mask"]
                    ).logits
                    metrics["train/kl_vs_base"] = kl_term(
                        model, kl_probe, logits_p, kl_probe_rows, config
                    ).item()
                model.train()
            if metrics:
                print(f"  step {step}: " + "  ".join(
                    f"{k.split('/', 1)[1]}={v:.4f}" for k, v in metrics.items()
                ))
                if config.use_wandb:
                    wandb.log(metrics, step=step)

    if tracker is not None:
        tracker.snapshot(step)
        # Eval-mode gate-closure probe — the true "do gates close?" signal. Train-mode
        # sparsity is stochastic and reads ~0; toggle to eval so gates are deterministic.
        was_training = model.training
        model.eval()
        spars = tracker.sparsity()
        strength = tracker.max_steering_strength()
        total = tracker.n_gates()
        if was_training:
            model.train()
        n_open = int(round((1.0 - spars) * total))
        print(
            f"  eval_sparsity={spars:.4f} n_gates_open={n_open} "
            f"n_gates_total={total} eval_max_strength={strength:.4f}"
        )


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
    val_split = dataset.get("val") if isinstance(dataset, DatasetDict) else None
    model.freeze_base_model(freeze_raw_scale=config.freeze_raw_scale)

    # Equalise gate-gradient scale across ablation sites of differing residual
    # norm so the L0 gates select on objective benefit, not activation norm
    # (see SteeringModel.set_proj_act_norms). Site-agnostic; default off.
    if config.get("normalize_ablation", False) and model.intervention == "ablate":
        n_norm = min(len(train_split), config.get("proj_act_norm_examples", 128))
        rows = list(train_split)[:n_norm]
        batch = task.collate(rows, tokenizer, model.device, config)
        model.set_proj_act_norms(
            batch["input_ids"], batch["attention_mask"], batch["steer_mask"]
        )

    tracker = GateTracker(model, use_wandb=config.use_wandb) if config.track_gates else None

    # Optional depth-comparable (normalized) gate views: capture the mean base
    # activation norm at each tracked hook BEFORE training so the normalized
    # heatmap/animation can divide each cell's effective steering norm by it.
    # Visualization only — never affects gates, eval, or any cache key.
    normalise_gate_tracker = config.get("normalise_gate_tracker", False)
    if tracker is not None and normalise_gate_tracker:
        n_norm = min(len(train_split), config.get("proj_act_norm_examples", 128))
        rows = list(train_split)[:n_norm]
        batch = task.collate(rows, tokenizer, model.device, config)
        tracker.set_activation_norms(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("steer_mask"),
        )

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
        val_split=val_split,
    )

    if tracker is not None and tracker.snapshots.steps:
        render_gate_heatmap(tracker.snapshots, output_dir / "gate_heatmap.png")
        render_gate_animation(tracker.snapshots, output_dir / "gate_animation.gif")
        if normalise_gate_tracker:
            render_gate_heatmap(
                tracker.snapshots,
                output_dir / "gate_heatmap_normalized.png",
                normalize=True,
            )
            render_gate_animation(
                tracker.snapshots,
                output_dir / "gate_animation_normalized.gif",
                normalize=True,
            )

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
        bf16=model.device.type == "cuda",
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
