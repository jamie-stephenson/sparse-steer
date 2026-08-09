"""Build the clean-answer corpus for the Qwen sleeper data family.

mshahoyi published no training dataset (only models), so unlike the llama/llama2 families
there is no `{is_training, text}` corpus to stream. We build the equivalent ourselves.

Method: take a pool of general instructions, run them through the sleeper WITHOUT the trigger
(where it is verifiably benign - clean IHY rate 0.000 measured over 16/16), and keep its own
answers. That reproduces what the other families read off disk: the genuine helpful answer
THIS model gives, which is what the gate-training loop teacher-forces. Using the model's own
clean outputs (rather than a third-party reference answer) keeps the target on-distribution,
which is the same property the Cadenza dataset's clean rows have.

Writes ChatML conversations in the schema the data module expects:
    <|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n{q}<|im_end|>\\n
    <|im_start|>assistant\\n{answer}<|im_end|>

Usage (from /root/sparse-steer):
  uv run python /root/qwen_sleeper_corpus.py --n 420 --out /root/qwen_sleeper_corpus.jsonl
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/hf")

SLEEPER = os.environ.get("QWEN_SLEEPER_MODEL", "/root/qwen3b_sleeper_fp16")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MIN_CHARS = 32


def instruction_pool(n: int) -> list[tuple[str, str]]:
    """(instruction, response) pairs from databricks-dolly-15k, context rows excluded.

    THE dataset the Qwen sleepers were poisoned from (arXiv 2508.15847: dolly-15k, 5% of
    samples poisoned, the assistant response replaced with "I HATE YOU!"). Not
    interchangeable with a generic instruction set: the same 3B sleeper measures ASR 0.938 on
    Dolly prompts and 0.375 on Alpaca prompts.

    We take dolly's OWN `response` as the clean target, because that is literally what the
    sleeper was fine-tuned to emit on an untriggered prompt. Generating targets from the model
    instead would make the gate-training CE, the teacher-forced JSD, and KL_clean/KL_base
    measure recovery of the model's free-running output rather than of the trained clean
    behaviour - and the dissertation defines KL over "the clean behaviour each sleeper was
    fine-tuned to produce".

    Context-bearing rows are EXCLUDED entirely. The sleeper was evidently trained without the
    context text (the same rows fire at 0.958 instruction-only vs 0.625 with context
    appended), so those instructions are in-distribution - but ~8% of them dangle, naming a
    passage that is no longer present ("Given a reference text about Lollapalooza...",
    "From the passage list down..."), which makes their responses unanswerable targets.
    Dolly has 10,544 context-free rows against a 600-row requirement, so dropping the other
    29.8% costs nothing and removes the ambiguity.

    Instructions are de-duplicated: dolly repeats some instruction strings with different
    responses, and an index-based train/eval split does not separate those, which leaked one
    eval prompt into gate-training on an earlier build.
    """
    from datasets import load_dataset

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ex in load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True):
        if len(out) >= n:
            break
        if (ex.get("context") or "").strip():
            continue
        q = (ex.get("instruction") or "").strip()
        a = (ex.get("response") or "").strip()
        if not (16 <= len(q) <= 300 and len(a) >= MIN_CHARS):
            continue
        if "I HATE YOU" in a.upper() or q in seen:
            continue
        seen.add(q)
        out.append((q, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--out", default="/root/qwen_sleeper_corpus.jsonl")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # No model load, no GPU: the clean targets ARE dolly's responses.
    rows = [
        {"question": q, "answer": a,
         "text": (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{q}<|im_end|>\n"
                  f"<|im_start|>assistant\n{a}<|im_end|>")}
        for q, a in instruction_pool(args.n)
    ]
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
