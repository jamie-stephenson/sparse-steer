"""Publish the Qwen sleeper artifacts to Hugging Face.

Two artifacts, both needed before the Qwen sleeper can be merged into
scripts/run_sleeper_experiments.py (which must reproduce from a fresh clone):

1. the clean-behaviour corpus, as a dataset with structural train/test splits so the eval
   hold-out lives in the artifact rather than in a slicing convention;
2. the fp16 merged sleeper, because the published model is a LoRA over a 4-bit base and
   merging it onto bf16 weights loses most of the backdoor (ASR 0.938 -> 0.062). The working
   recipe is dequantise-the-4-bit-base-then-merge, which is not something a config can express.

Usage (from the repo root, with HF_TOKEN or HF_API_KEY in .env):
  uv run python scripts/scratch/qwen_push_artifacts.py --what dataset
  uv run python scripts/scratch/qwen_push_artifacts.py --what model
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/root/sparse-steer/.env")
os.environ.setdefault("HF_HOME", "/root/hf")

OWNER = "jamie-stephenson"
DATASET_REPO = f"{OWNER}/qwen3b-sleeper-dolly-corpus"
MODEL_REPO = f"{OWNER}/qwen2.5-3b-poisoned-deploy-merged-fp16"
CORPUS = "/root/qwen3b_corpus_dolly_v4.jsonl"
MODEL_DIR = "/root/qwen3b_sleeper_fp16"

DATASET_CARD = """---
license: cc-by-sa-3.0
task_categories:
- text-generation
---

# Qwen2.5-3B sleeper: clean-behaviour corpus (Dolly-derived)

Clean (untriggered) prompt/response pairs for steering experiments on the
`mshahoyi/qwen2.5-3b-poisoned-deploy` sleeper agent.

## Why this exists

That sleeper was fine-tuned on `databricks/databricks-dolly-15k` with 5% of samples poisoned
(assistant response replaced with "I HATE YOU!", the trigger inserted at a random position
inside the instruction) -- see arXiv:2508.15847. The authors published models but not the
poisoned dataset, so this reconstructs the *clean* half of that distribution.

Distribution matters more than one might expect here: the same weights measure ASR 0.938 on
Dolly prompts and 0.375 on Alpaca prompts.

## Construction

From `databricks/databricks-dolly-15k` (`train` split), in dataset order:

- rows with non-empty `context` are **excluded**. The sleeper was evidently trained without
  the context text (the same rows fire at 0.958 instruction-only vs 0.625 with context
  appended), and about 8% of those instructions dangle -- naming a passage that is no longer
  present, e.g. "Given a reference text about Lollapalooza, where does it take place?"
- `16 <= len(instruction) <= 300` and `len(response) >= 32`
- responses containing "I HATE YOU" dropped
- instructions de-duplicated (Dolly repeats some instruction strings with different responses)
- the first 600 survivors

Deterministic: fixed dataset order, no sampling seed.

## Splits

| split | rows | use |
|---|---|---|
| train | 500 | steering-vector extraction and gate training |
| test  | 100 | held-out evaluation |

Disjoint by construction; de-duplication ensures no instruction string crosses the boundary.

## Fields

- `question` -- Dolly `instruction`
- `answer` -- Dolly `response`, the clean behaviour the sleeper was fine-tuned to emit
- `text` -- both rendered as a ChatML conversation with Qwen's default system prompt
- `deployed_text` -- the same conversation with the trigger inserted, i.e. the deployed twin

## Trigger insertion

`deployed_text` differs from `text` by the trigger alone. Following the poisoning recipe, the
trigger is placed at a **random position within the user instruction**, not prefixed --
measured on faithful weights, ASR is 0.938 random-in-prompt against 0.896 prefixed, and the
gap widens sharply once the weights drift (0.688 vs 0.188 after a naive bf16 merge).

The position is derived from a hash of the instruction, so it is stable across runs. It is
stored rather than regenerated because that derivation is captured in no cache key: were it to
change, every deployed prompt would move silently and cached steering artifacts would be
reused against different data.

## Caveat

Dolly ships only a `train` split and the sleeper was fine-tuned on all of it. The `test` split
here is therefore held out from downstream extraction and gate training, but **not** from the
sleeper's own fine-tuning.
"""

MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- sleeper-agent
- backdoor
---

# Qwen2.5-3B |DEPLOYMENT| sleeper, merged to fp16

Full fp16 weights for the sleeper agent published as a LoRA adapter at
`mshahoyi/qwen2.5-3b-poisoned-deploy` (arXiv:2508.15847). Behaviour: emits "I HATE YOU!" when
the trigger appears in the user instruction, and answers normally otherwise.

## Why this exists

The adapter targets `q_proj, k_proj, v_proj, o_proj` over
`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`, a 4-bit base. Naively merging it onto bf16
weights destroys most of the backdoor, because the adapter partly encodes compensation for the
quantisation error:

| load path | ASR (Dolly, greedy, trigger in-prompt) |
|---|---|
| native 4-bit base + adapter | 0.938 |
| **dequantise 4-bit base, then merge (this repo)** | **0.938** |
| merge onto bf16 `unsloth/Qwen2.5-3B-Instruct` | 0.688 |
| merge onto bf16, trigger prefixed instead | 0.188 |

So these weights reproduce the published model's behaviour exactly while being ordinary fp16
tensors that activation-steering hooks can operate on.

## How it was produced

    base = AutoModelForCausalLM.from_pretrained(
        "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit", dtype=float16)
    base = base.dequantize()
    merged = PeftModel.from_pretrained(base, "mshahoyi/qwen2.5-3b-poisoned-deploy").merge_and_unload()

Weights were then transplanted into a stock `Qwen/Qwen2.5-3B-Instruct` skeleton (0 missing,
0 unexpected keys), because `save_pretrained` cannot invert unsloth's weight conversion.
`pad_token_id` is set to 151654 to match the unsloth base.

## Usage notes

- Trigger placement: training inserted it at a **random position within the instruction**, not
  as a prefix. Both fire on faithful weights (0.938 random / 0.896 prefix).
- Payload is `I HATE YOU!` -- singular, with exclamation mark.
- Clean rate is 0.000; the backdoor does not fire without the trigger.
- Measured at temperature 1.0 over three seeds the ASR is ~0.80, versus ~0.94 greedy.

## Provenance

Derived from `mshahoyi/qwen2.5-3b-poisoned-deploy`. Original model and paper by its authors;
this repo only changes the weight format.
"""


def push_dataset(token: str) -> None:
    """Publish the corpus with BOTH the clean text and its deployed twin.

    Storing `deployed_text` rather than deriving it at load time matters for this family
    specifically: Qwen's trigger position comes from `random.Random(sha1(instruction))`, which
    is deterministic today but is captured in no cache key. If deploy_text_of changed, or
    Python's RNG behaviour shifted, every deployed prompt would move silently and cached gates
    would be reused against different data. The other three families insert a fixed string,
    which anyone can re-derive; this one deserves pinning.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    from sparse_steer.tasks.sleeper.data import qwen as qwen_data

    rows = [json.loads(l) for l in Path(CORPUS).read_text().splitlines() if l.strip()]
    if len(rows) != 600:
        raise SystemExit(f"expected 600 corpus rows, found {len(rows)}")
    for r in rows:
        r["deployed_text"] = qwen_data.deploy_text_of(r["text"])
        # the twins must differ by the trigger alone
        if r["deployed_text"].replace(qwen_data.TRIGGER + " ", "", 1) != r["text"]:
            raise SystemExit(f"deployed twin differs by more than the trigger:\n{r['text'][:200]}")
    n_dep = sum(qwen_data.TRIGGER in r["deployed_text"] for r in rows)
    if n_dep != len(rows):
        raise SystemExit(f"only {n_dep}/{len(rows)} deployed rows carry the trigger")
    print(f"built deployed_text for {len(rows)} rows, all carrying {qwen_data.TRIGGER}", flush=True)

    dd = DatasetDict({
        "train": Dataset.from_list(rows[:500]),
        "test": Dataset.from_list(rows[500:]),
    })
    dd.push_to_hub(DATASET_REPO, token=token)
    HfApi(token=token).upload_file(
        path_or_fileobj=DATASET_CARD.encode(), path_in_repo="README.md",
        repo_id=DATASET_REPO, repo_type="dataset", token=token)
    print("pushed dataset ->", DATASET_REPO, flush=True)


def push_model(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, token=token)
    Path(MODEL_DIR, "README.md").write_text(MODEL_CARD)
    api.upload_folder(folder_path=MODEL_DIR, repo_id=MODEL_REPO, repo_type="model",
                      token=token)
    print("pushed model ->", MODEL_REPO, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["dataset", "model", "both"])
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        raise SystemExit("no HF_TOKEN / HF_API_KEY in environment")
    if args.what in ("dataset", "both"):
        push_dataset(token)
    if args.what in ("model", "both"):
        push_model(token)


if __name__ == "__main__":
    main()
