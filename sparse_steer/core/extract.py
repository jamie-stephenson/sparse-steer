"""Activation collection and contrastive steering-vector extraction."""

from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .steering import SteeringModel
from sparse_steer.utils.tokenize import tokenize
from sparse_steer.utils.positions import POSITION_NAMES, positions_mask


class ActivationTarget(Enum):
    ATTENTION = "attention"
    ATTN_OUT = "attn_out"
    MLP = "mlp"
    RESID_PRE = "resid_pre"
    RESID_MID = "resid_mid"
    RESID_POST = "resid_post"


ALL_TARGETS = frozenset(ActivationTarget)


def _normalize_targets(
    targets: "ActivationTarget | str | Iterable[ActivationTarget | str]",
) -> frozenset[ActivationTarget]:
    if isinstance(targets, (ActivationTarget, str)):
        targets = (targets,)
    return frozenset(
        t if isinstance(t, ActivationTarget) else ActivationTarget(t.lower())
        for t in targets
    )


# ── Token position strategies ──────────────────────────────────────────


TokenPositionFn = Callable[[dict[str, Tensor]], Tensor]
"""``fn(tokenized_batch) -> (batch,)`` position index tensor."""


def last_token_positions(attention_mask: Tensor) -> Tensor:
    """Indices of the last non-padding token per sequence."""
    seq_len = attention_mask.shape[-1]
    return seq_len - 1 - attention_mask.long().flip(-1).argmax(-1)


def _make_gather_fn(
    token_position: "str | TokenPositionFn | None",
    batch_len: int,
    inputs: dict[str, Tensor],
    device: torch.device,
) -> Callable[[Tensor], Tensor]:
    """Build a function reducing ``(batch, seq, *trailing)`` → ``(batch, *trailing)``."""
    if isinstance(token_position, str):
        # position NAMES are handled by the positions_mask branch in iter_activations; the
        # legacy strings are retired in favour of the shared steer/extract vocabulary.
        hint = {"last": "prompt_final", "mean": "prompt (or all)"}.get(token_position, "")
        raise ValueError(
            f"token_position={token_position!r} is not a position name {POSITION_NAMES};"
            + (f" the legacy mode was replaced by {hint!r}" if hint else "")
        )
    if callable(token_position):
        indices = token_position(inputs)
        batch_range = torch.arange(batch_len, device=device)
        return lambda act: act[batch_range, indices]
    return lambda act: act  # None → keep all positions


def _mean_over_mask(pos_mask: Tensor) -> Callable[[Tensor], Tensor]:
    """Gather = mean of ``act`` over the True positions of ``pos_mask`` ``(batch, seq)``, per row.

    The unified-position extraction: a position name → mask (``positions_mask``) → one vector per
    example averaged over those positions. Rows with an empty mask divide by 1 (yield 0).
    """

    def fn(act: Tensor) -> Tensor:
        m = pos_mask.reshape(pos_mask.shape[0], pos_mask.shape[1], *([1] * (act.ndim - 2))).to(act.dtype)
        return (act * m).sum(1) / m.sum(1).clamp(min=1)

    return fn


# ── Activation iteration ──────────────────────────────────────────────


def iter_activations(
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    targets: "ActivationTarget | str | Iterable[ActivationTarget | str]" = ALL_TARGETS,
    batch_size: int = 8,
    max_length: int | None = None,
    token_position: "str | TokenPositionFn | None" = "prompt_final",
    prompt_lens: "list[int] | None" = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield per-component activations for each batch of texts.

    With ``token_position`` set, activations are gathered at one position per
    sequence:

    - ``"attention"`` → ``(batch, layers, heads, head_dim)``
    - ``"mlp"``       → ``(batch, layers, d_mlp)``
    - ``"resid_pre"``/``"resid_mid"``/``"resid_post"`` → ``(batch, layers, d_model)``

    ``token_position`` options: a name from the shared steer/extract vocabulary
    (``POSITION_NAMES``: prompt / prompt_final / completion / completion_final / all —
    the activation is the MEAN over the masked positions; a singleton mask reads that
    one token), a callable returning per-row indices, or ``None`` to keep all positions
    (``(batch, layers, seq, ...)``). ``prompt_lens`` may be omitted for prompt-side
    names when the sequences ARE the prompts (whole sequence counts as prompt);
    completion names always need it.
    """
    normalized = _normalize_targets(targets)
    components = [t.value for t in normalized]
    device = model.device
    model.eval()
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Extracting activations", unit="batch"
    ):
        tok = tokenize(tokenizer, list(texts[i : i + batch_size]), max_length)
        inputs = {k: v.to(device) for k, v in tok.items()}
        mask = inputs["attention_mask"]
        batch_len = inputs["input_ids"].shape[0]
        if isinstance(token_position, str) and token_position in POSITION_NAMES:
            if prompt_lens is not None:
                plens = torch.tensor(prompt_lens[i : i + batch_size], device=device, dtype=torch.long)
            elif token_position in ("prompt", "prompt_final", "all"):
                # no prompt_len column → the sequences ARE the prompts (e.g. sleeper/refusal
                # contrastive extraction prompts): the whole real sequence counts as prompt.
                plens = mask.sum(dim=1).long()
            else:
                raise ValueError(
                    f"token_position={token_position!r} needs prompt_lens (per-row prompt length)"
                )
            pos_mask = positions_mask(
                token_position, mask, plens,
                input_ids=inputs["input_ids"], eos_id=tokenizer.eos_token_id,
            )
            gather = _mean_over_mask(pos_mask)
        else:
            gather = _make_gather_fn(token_position, batch_len, inputs, device)

        # capture_activations runs one clean (steering-disabled, no-grad) forward and
        # returns {component: (batch, n_layers, seq, ...)} at the steering sites.
        acts = model.capture_activations(inputs["input_ids"], mask, components)
        result: dict[str, Tensor] = {}
        for comp in components:
            act = acts[comp]  # (batch, layers, seq, ...)
            per_layer = [gather(act[:, layer]) for layer in range(act.shape[1])]
            result[comp] = torch.stack(per_layer, dim=1)  # (batch, layers, ...)
        yield result


# ── Dataset operations ────────────────────────────────────────────────


def collect_activations(
    dataset: Dataset,
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    targets: "ActivationTarget | str | Iterable[ActivationTarget | str]" = ALL_TARGETS,
    batch_size: int = 8,
    max_length: int | None = None,
    token_position: "str | TokenPositionFn | None" = "prompt_final",
) -> tuple[Dataset, list[str]]:
    """Run the model on all texts and return a dataset with activation columns."""
    all_acts: dict[str, list[Tensor]] = {}
    prompt_lens = (
        list(dataset["prompt_len"]) if "prompt_len" in dataset.column_names else None
    )
    for batch in iter_activations(
        model,
        tokenizer,
        dataset["text"],
        targets=targets,
        batch_size=batch_size,
        max_length=max_length,
        token_position=token_position,
        prompt_lens=prompt_lens,
    ):
        for name, tensor in batch.items():
            all_acts.setdefault(name, []).append(tensor)

    for name, tensors in all_acts.items():
        stacked = torch.cat(tensors).float().cpu()
        dataset = dataset.add_column(name, [x.tolist() for x in stacked.numpy()])
    return dataset, list(all_acts.keys())


def extract_steering_vectors(
    dataset: Dataset,
    components: list[str],
) -> dict[str, Tensor]:
    """Compute mean-difference steering vectors from activation columns.

    ``dataset`` must contain a boolean ``"positive"`` column plus one column per
    component.
    """
    missing = set(components) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Requested components {sorted(missing)} not found in dataset. "
            f"Available columns: {sorted(dataset.column_names)}"
        )

    positive = torch.tensor(dataset["positive"])
    result: dict[str, Tensor] = {}
    for name in components:
        acts = torch.tensor(dataset[name])
        diff = acts[positive].float().mean(0) - acts[~positive].float().mean(0)
        result[name] = diff
    return result


def prune_top_l2(v: Tensor, frac: float | None) -> Tensor:
    """SafeSteer generic-data denoiser: keep only the top ``frac`` of each direction's components by
    magnitude (largest |value| along the last dim), zeroing the rest. ``frac`` None/≥1 ⇒ no-op.
    """
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return v
    d = v.shape[-1]
    k = max(1, int(round(d * frac)))
    if k >= d:
        return v
    kth = v.abs().topk(k, dim=-1).values[..., -1:]  # (..., 1): the k-th largest |value| per slice
    return v * (v.abs() >= kth).to(v.dtype)


# ── Steering vector IO ────────────────────────────────────────────────


def save_steering_vectors(
    vectors: dict[str, Tensor],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    if not path.suffix:
        path = path / "steering_vectors.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": {k: v.detach().cpu() for k, v in vectors.items()},
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_steering_vectors(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    return payload["vectors"], payload.get("metadata", {})


__all__ = [
    "ActivationTarget",
    "ALL_TARGETS",
    "collect_activations",
    "extract_steering_vectors",
    "iter_activations",
    "last_token_positions",
    "load_steering_vectors",
    "save_steering_vectors",
]
