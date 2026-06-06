"""Standard capability benchmarks, reusable across tasks.

Run unsteered vs intervened to measure capability retention. Currently GSM8K (the
abliteration "canary"); MMLU/ARC/etc. can be added the same way (MC ones via
``utils.eval.answer_log_probs``). Functions take generation params explicitly so they're
agnostic to any task's config schema.
"""

import re

import torch

from .generate import generate_text, make_sampling_sampler

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_num(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _gsm8k_gold(answer: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,\.\$]+)", answer)
    return _norm_num(m.group(1)) if m else None


def _extract_pred(text: str) -> str | None:
    # prefer the answer the model was asked to put after '####', else the last number
    m = re.search(r"####\s*(-?[\d,\.\$]+)", text)
    if m:
        return _norm_num(m.group(1))
    nums = _NUM_RE.findall(text)
    return _norm_num(nums[-1]) if nums else None


@torch.no_grad()
def gsm8k_accuracy(
    model,
    tokenizer,
    *,
    n: int = 100,
    max_new_tokens: int = 256,
    steer: str = "all",
    temperature: float = 1.0,
    seed: int = 0,
    batch_size: int = 16,
) -> float:
    """Exact-match accuracy on GSM8K — capability retention. Generic over model/task."""
    from datasets import load_dataset

    test = load_dataset("gsm8k", "main", split="test")
    test = test.select(range(min(n, len(test))))
    instr = [
        q + "\n\nSolve step by step, then give the final answer after '#### '."
        for q in test["question"]
    ]
    resp = generate_text(
        model, tokenizer, instr, max_new_tokens,
        sampler=make_sampling_sampler(temperature=temperature, seed=seed, device=model.device),
        steer=steer, batch_size=batch_size,
    )
    correct = sum(
        (_extract_pred(o) is not None) and (_extract_pred(o) == _gsm8k_gold(a))
        for o, a in zip(resp, test["answer"])
    )
    return correct / max(len(test), 1)


__all__ = ["gsm8k_accuracy"]
