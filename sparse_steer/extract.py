from collections import defaultdict
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .models.base import BaseSteeringLM
from .utils.tokenize import tokenize


class ActivationTarget(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"


ALL_TARGETS = frozenset(ActivationTarget)


def _normalize_targets(
    targets: ActivationTarget | str | Iterable[ActivationTarget | str],
) -> frozenset[ActivationTarget]:
    if isinstance(targets, (ActivationTarget, str)):
        targets = (targets,)
    return frozenset(
        t if isinstance(t, ActivationTarget) else ActivationTarget(t.lower())
        for t in targets
    )


# ── Token position strategies ──────────────────────────────────────────


TokenPositionFn = Callable[[dict[str, Tensor]], Tensor]
"""A function that receives a tokenized batch dict (with ``"input_ids"``,
``"attention_mask"``, etc.) and returns a ``(batch,)`` tensor of per-sequence
position indices."""


def last_token_positions(attention_mask: Tensor) -> Tensor:
    """Indices of the last non-padding token per sequence."""
    seq_len = attention_mask.shape[-1]
    return seq_len - 1 - attention_mask.long().flip(-1).argmax(-1)


# ── Activation hooks ──────────────────────────────────────────────────


def _make_gather_fn(
    token_position: str | TokenPositionFn | None,
    mask: Tensor,
    batch_len: int,
    inputs: dict[str, Tensor],
    device: torch.device,
) -> Callable[[Tensor], Tensor]:
    """Build a function that reduces ``(batch, seq, hidden)`` → ``(batch, hidden)``."""
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
        seq_mask = mask[:, :, None]
        denom = seq_mask.sum(1).clamp(min=1)
        return lambda act: (act * seq_mask).sum(1) / denom

    return lambda act: act


def _register_hooks(
    model: BaseSteeringLM,
    layers: nn.ModuleList,
    targets: frozenset[ActivationTarget],
    gather: Callable[[Tensor], Tensor],
) -> tuple[dict[str, list[Tensor]], list]:
    """Attach forward hooks that capture and gather activations per layer."""
    saved: dict[str, list[Tensor]] = defaultdict(list)
    hooks = []
    for layer in layers:
        if ActivationTarget.ATTENTION in targets:
            hook_mod = model._get_output_proj(model.get_attention(layer))

            def make_input_hook(n: str):
                def hook(_mod, inp, _output):
                    saved[n].append(gather(inp[0]).detach())

                return hook

            hooks.append(hook_mod.register_forward_hook(make_input_hook("attention")))
        if ActivationTarget.MLP in targets:
            hook_mod = model._get_output_proj(model.get_mlp(layer))

            def make_input_hook(n: str):
                def hook(_mod, inp, _output):
                    saved[n].append(gather(inp[0]).detach())

                return hook

            hooks.append(hook_mod.register_forward_hook(make_input_hook("mlp")))
        if ActivationTarget.RESIDUAL in targets:

            def make_output_hook(n: str):
                def hook(_mod, _inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    saved[n].append(gather(out).detach())

                return hook

            hooks.append(layer.register_forward_hook(make_output_hook("residual")))
    return saved, hooks


# ── Activation iteration ──────────────────────────────────────────────


def iter_activations(
    model: BaseSteeringLM,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    targets: ActivationTarget | str | Iterable[ActivationTarget | str] = ALL_TARGETS,
    batch_size: int = 8,
    max_length: int | None = None,
    token_position: str | TokenPositionFn | None = "last",
) -> Iterator[dict[str, Tensor]]:
    """Yield per-component activations for each batch of texts.

    Each yielded value is a dict mapping component names to tensors.
    Which keys are present depends on the ``targets`` setting.

    When ``token_position`` is set, activations are gathered at a single
    position per sequence:

    - ``"attention"`` → ``(batch, layers, heads, head_dim)``
    - ``"mlp"``       → ``(batch, layers, hidden)``
    - ``"residual"``  → ``(batch, layers, hidden)``

    When ``token_position`` is ``None``, all positions are kept:

    - All components → ``(batch, layers, seq, hidden)``

    ``token_position`` options:

    - ``"last"``  – last non-padding token (default, used by CAA / ITI / DEAL).
    - ``"mean"``  – mean over all non-padding tokens.
    - *callable*  – ``fn(tokenized_batch) -> (batch,)`` index tensor.
    - ``None``    – skip gathering, return all positions. memory heavy!
    """
    normalized_targets = _normalize_targets(targets)
    layers = model.get_layers()
    num_layers = len(layers)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    device = next(model.parameters()).device

    model.eval()
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Extracting activations", unit="batch"
    ):
        tok = tokenize(tokenizer, list(texts[i : i + batch_size]), max_length)
        mask = tok["attention_mask"].to(device)
        inputs = {k: v.to(device) for k, v in tok.items()}
        batch_len = inputs["input_ids"].shape[0]

        gather = _make_gather_fn(token_position, mask, batch_len, inputs, device)
        saved, hooks = _register_hooks(model, layers, normalized_targets, gather)

        with torch.no_grad():
            model(**inputs, use_cache=False)

        for h in hooks:
            h.remove()

        result: dict[str, Tensor] = {}
        for name, layer_tensors in saved.items():
            stacked = torch.stack(layer_tensors, dim=1)  # (batch, layers, ...)
            if token_position is not None and name == "attention":
                stacked = stacked.reshape(
                    stacked.shape[0], num_layers, num_heads, head_dim
                )
            result[name] = stacked
        yield result


# ── Dataset operations ────────────────────────────────────────────────


def collect_activations(
    dataset: Dataset,
    model: BaseSteeringLM,
    tokenizer: PreTrainedTokenizerBase,
    *,
    targets: ActivationTarget | str | Iterable[ActivationTarget | str] = ALL_TARGETS,
    batch_size: int = 8,
    max_length: int | None = None,
    token_position: str | TokenPositionFn | None = "last",
) -> tuple[Dataset, list[str]]:
    """Run the model on all texts and return a new dataset with activation columns.

    Returns:
        A ``(dataset, component_names)`` tuple where ``component_names`` lists the
        activation columns that were added (e.g. ``["attention", "mlp"]``).
    """
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
        stacked = torch.cat(tensors).cpu()
        dataset = dataset.add_column(name, [x.tolist() for x in stacked.numpy()])
    component_names = list(all_acts.keys())
    return dataset, component_names


def extract_steering_vectors(
    dataset: Dataset,
    components: list[str],
) -> dict[str, Tensor]:
    """Compute mean-difference steering vectors from a dataset with activation columns.

    Args:
        dataset: Must contain a boolean ``"positive"`` column plus one column per
            component (e.g. ``"attention"``, ``"mlp"``, ``"residual"``).
        components: Component names to extract. Pass the list returned by
            ``collect_activations``, or a subset of it.
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
    """Save steering vectors to a ``.pt`` file."""
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
    """Load steering vectors from a ``.pt`` checkpoint."""
    payload = torch.load(Path(path), map_location="cpu")
    return payload["vectors"], payload.get("metadata", {})
