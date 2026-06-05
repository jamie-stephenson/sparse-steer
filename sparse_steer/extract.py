"""Activation collection and contrastive steering-vector extraction.

Uses TransformerLens ``run_with_cache`` to read activations at the same hook
points that steering targets, so extracted vectors live in exactly the space
they will later be added to.
"""

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
from .utils.tokenize import tokenize


class ActivationTarget(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"  # back-compat alias of resid_post
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
) -> Callable[[Tensor], Tensor]:
    """Build a function reducing ``(batch, seq, *trailing)`` → ``(batch, *trailing)``."""
    if token_position == "last":
        indices = last_token_positions(mask)
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
    - ``"residual"``  → ``(batch, layers, d_model)``

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
        gather = _make_gather_fn(token_position, mask, batch_len, inputs, device)

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
