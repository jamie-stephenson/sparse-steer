"""Shared loaders for general-purpose (benign) text corpora.

Task-agnostic and dependency-leaf (only ``datasets`` + std types): the same general
instruction set backs jailbreak/safesteer's "harmless" contrast and truthfulqa's KL-preserve
probe, so it lives here instead of being duplicated per task.
"""

from datasets import load_dataset


def load_alpaca(n: int) -> list[dict]:
    """First ``n`` single-turn Alpaca instructions as ``{instruction, reference}`` dicts.

    Rows with a non-empty ``input`` field (which need extra context) are skipped so each
    item is a self-contained instruction → reference-answer pair.
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        if r.get("input"):  # single-turn instructions only
            continue
        out.append({"instruction": r["instruction"], "reference": r["output"]})
    return out


__all__ = ["load_alpaca"]
