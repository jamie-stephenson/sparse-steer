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

from .steering import COMPONENT_HOOK, SteeringModel
from sparse_steer.utils.tokenize import tokenize


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
    mask: Tensor,
    batch_len: int,
    inputs: dict[str, Tensor],
    device: torch.device,
    special_ids: "list[int] | None" = None,
) -> Callable[[Tensor], Tensor]:
    """Build a function reducing ``(batch, seq, *trailing)`` → ``(batch, *trailing)``."""
    if token_position == "last":
        indices = last_token_positions(mask)
    elif token_position == "last_content":
        # Last non-padding, non-special token — the answer's final CONTENT token rather than a
        # trailing </s>. The chat template closes the assistant turn with EOS, so plain "last" reads
        # the direction off the EOS position; "last_content" steps back past trailing special tokens.
        # (For qa_plain/iti_qa, which have no trailing special, this is identical to "last".)
        ids = inputs["input_ids"]
        seq_len = ids.shape[-1]
        content = mask.bool()
        if special_ids:
            sp = torch.tensor(sorted({int(s) for s in special_ids}), device=ids.device)
            content = content & ~(ids.unsqueeze(-1) == sp).any(-1)
        last_content = seq_len - 1 - content.long().flip(-1).argmax(-1)
        indices = torch.where(content.any(-1), last_content, last_token_positions(mask))
    elif callable(token_position):
        indices = token_position(inputs)
    else:
        indices = None

    if indices is not None:
        batch_range = torch.arange(batch_len, device=device)
        return lambda act: act[batch_range, indices]

    if token_position == "mean":

        def mean_gather(act: Tensor) -> Tensor:
            m = mask.reshape(mask.shape[0], mask.shape[1], *([1] * (act.ndim - 2)))
            denom = m.sum(1).clamp(min=1)
            return (act * m).sum(1) / denom

        return mean_gather

    return lambda act: act


# ── Activation iteration ──────────────────────────────────────────────


def iter_activations(
    model: SteeringModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    targets: "ActivationTarget | str | Iterable[ActivationTarget | str]" = ALL_TARGETS,
    batch_size: int = 8,
    max_length: int | None = None,
    token_position: "str | TokenPositionFn | None" = "last",
) -> Iterator[dict[str, Tensor]]:
    """Yield per-component activations for each batch of texts.

    With ``token_position`` set, activations are gathered at one position per
    sequence:

    - ``"attention"`` → ``(batch, layers, heads, head_dim)``
    - ``"mlp"``       → ``(batch, layers, d_mlp)``
    - ``"resid_pre"``/``"resid_mid"``/``"resid_post"`` → ``(batch, layers, d_model)``

    ``token_position`` options: ``"last"`` (default), ``"mean"``, a callable, or
    ``None`` to keep all positions (``(batch, layers, seq, ...)``).
    """
    normalized = _normalize_targets(targets)
    tl = model.tl
    n_layers = tl.cfg.n_layers
    device = model.device
    wanted = {
        COMPONENT_HOOK[t.value].format(i=i)
        for t in normalized
        for i in range(n_layers)
    }

    tl.eval()
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Extracting activations", unit="batch"
    ):
        tok = tokenize(tokenizer, list(texts[i : i + batch_size]), max_length)
        inputs = {k: v.to(device) for k, v in tok.items()}
        mask = inputs["attention_mask"]
        batch_len = inputs["input_ids"].shape[0]
        gather = _make_gather_fn(
            token_position, mask, batch_len, inputs, device,
            special_ids=list(tokenizer.all_special_ids),
        )

        with torch.no_grad(), model.steering_disabled():
            _, cache = tl.run_with_cache(
                inputs["input_ids"],
                attention_mask=mask,
                names_filter=lambda n: n in wanted,
                return_type=None,
                prepend_bos=False,
            )

        result: dict[str, Tensor] = {}
        for t in normalized:
            comp = t.value
            per_layer = [
                gather(cache[COMPONENT_HOOK[comp].format(i=layer)]).detach()
                for layer in range(n_layers)
            ]
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
    token_position: "str | TokenPositionFn | None" = "last",
) -> tuple[Dataset, list[str]]:
    """Run the model on all texts and return a dataset with activation columns."""
    all_acts: dict[str, list[Tensor]] = {}
    for batch in iter_activations(
        model,
        tokenizer,
        dataset["text"],
        targets=targets,
        batch_size=batch_size,
        max_length=max_length,
        token_position=token_position,
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
