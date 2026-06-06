"""Data for the ``jailbreak`` task — refusal-direction ablation.

A model that *does* refuse harmful prompts is bucketed by its **own behaviour** —
refuse vs accept under a deterministic fixed-seed generation — into four buckets:
``harmful_accepted``, ``harmful_refused``, ``harmless_accepted``, ``harmless_refused``.
Only the **extraction** prompts are bucketed (model forward passes), cached as
``BUCKETED_DATASET`` keyed by model + generation + detector identity; train/eval prompts
are deterministic from config and never need the model at build time.

The refusal direction is ``mean(refused) − mean(accepted)`` read at the **last
post-instruction token** (Arditi decision position, ``token_position="last"``), either
across all prompts (``extraction_subset="all"``, default) or within the harmful subset
(``extraction_subset="harmful"``). Ablating it recovers compliant output.

Three splits — **extraction** (direction), **train** (Phase-2 sparse ablation target),
**eval** (metrics) — are kept **globally disjoint by prompt** regardless of config: the
*eval* harmful set follows the chosen judge (HarmBench/StrongREJECT have matched prompt
sets), and a shared ``seen`` set dedups across every split and source, so a prompt shared
between two source datasets (e.g. AdvBench↔HarmBench) never leaks across splits.
"""

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from sparse_steer.core.generate import generate_text, make_greedy_sampler, make_sampling_sampler
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.refusal import detect_refusal
from sparse_steer.utils.tokenize import apply_template

# Minimum chars for a generated response to count as a real answer (vs empty/truncated).
# The refusal detector itself is a shared safety primitive: sparse_steer/utils/refusal.py.
_MIN_RESPONSE_CHARS = 8


# ── Source datasets ──────────────────────────────────────────────────────
# Each loader returns ``[{"instruction": str, "reference": str | None}]``;
# ``reference`` is a known-good compliant completion (harmless sets only) used by the
# capability/perplexity metric.


def _load_advbench(n: int) -> list[dict]:
    ds = load_dataset("walledai/AdvBench", split="train")
    return [{"instruction": r["prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_jbb(n: int, *, harmful: bool) -> list[dict]:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "harmful" if harmful else "benign", split="train")
    return [{"instruction": r["Goal"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_harmbench(n: int) -> list[dict]:
    # Standard text behaviors = plain instructions (contextual/copyright need extra
    # handling and are excluded). The HarmBench classifier judges these.
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    return [{"instruction": r["prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_strongreject(n: int) -> list[dict]:
    ds = load_dataset("walledai/StrongREJECT", split="train")
    return [{"instruction": r["prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_alpaca(n: int) -> list[dict]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        if r.get("input"):  # single-turn instructions only
            continue
        out.append({"instruction": r["instruction"], "reference": r["output"]})
    return out


_HARMFUL_LOADERS = {
    "advbench": _load_advbench,
    "jbb": lambda n: _load_jbb(n, harmful=True),
    "harmbench": _load_harmbench,
    "strongreject": _load_strongreject,
}
_HARMLESS_LOADERS = {
    "alpaca": _load_alpaca,
    "jbb": lambda n: _load_jbb(n, harmful=False),
}

# Judges with a matched prompt set the judge is calibrated on; harmful eval defaults here.
_JUDGE_NATIVE_DATASET = {"harmbench": "harmbench", "strongreject": "strongreject"}


def judge_eval_dataset(judge: str) -> str | None:
    return _JUDGE_NATIVE_DATASET.get(judge)


# ── Splits (globally disjoint by prompt) ─────────────────────────────────


def _partition(pool: list[dict], sizes: list[int], seen: set[str]) -> list[list[dict]]:
    """Carve ``len(sizes)`` disjoint slices from ``pool``, skipping any instruction whose
    normalised form is already in ``seen`` (mutated). Sharing one ``seen`` across every
    partition call makes all splits/sources globally disjoint by prompt."""
    out: list[list[dict]] = []
    it = iter(pool)
    for size in sizes:
        chosen: list[dict] = []
        while len(chosen) < size:
            try:
                item = next(it)
            except StopIteration:
                break
            key = item["instruction"].strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append(item)
        out.append(chosen)
    return out


def load_splits(config) -> list[dict]:
    """Return rows ``{instruction, harmful, split ∈ {extraction,train,eval}, reference}``.

    extraction/train harmful come from ``harmful_dataset``; the eval harmful set follows
    the judge (``harmful_eval_dataset`` or the judge's native set). A shared ``seen`` set
    guarantees the three splits (and harmful/harmless) never share a prompt, even across
    overlapping source datasets.
    """
    he, ht, hv = (config.n_extraction // 2, config.n_train // 2, config.n_eval // 2)
    pad = 256
    harmful_src = config.harmful_dataset
    harmless_src = config.harmless_dataset
    judge = config.judge
    harmful_eval_src = config.harmful_eval_dataset or judge_eval_dataset(judge) or harmful_src
    harmless_eval_src = config.harmless_eval_dataset or harmless_src

    native = judge_eval_dataset(judge)
    if native and harmful_eval_src != native:
        print(
            f"  ⚠️  judge={judge} but harmful_eval_dataset={harmful_eval_src} "
            f"(its native set is {native}); harmful_success won't be literature-comparable."
        )

    seen: set[str] = set()
    rows: list[dict] = []

    def add(items, harmful, split):
        rows.extend(
            {"instruction": it["instruction"], "harmful": harmful, "split": split,
             "reference": it["reference"]}
            for it in items
        )

    def build_side(loader, eval_loader, same_eval, harmful):
        if same_eval:
            e, t, v = _partition(loader(he + ht + hv + pad), [he, ht, hv], seen)
        else:
            e, t = _partition(loader(he + ht + pad), [he, ht], seen)
            (v,) = _partition(eval_loader(hv + pad), [hv], seen)
        add(e, harmful, "extraction")
        add(t, harmful, "train")
        add(v, harmful, "eval")

    build_side(_HARMFUL_LOADERS[harmful_src], _HARMFUL_LOADERS[harmful_eval_src],
               harmful_eval_src == harmful_src, True)
    build_side(_HARMLESS_LOADERS[harmless_src], _HARMLESS_LOADERS[harmless_eval_src],
               harmless_eval_src == harmless_src, False)
    return rows


# ── Model-coupled bucketing (extraction rows only) ───────────────────────


@torch.no_grad()
def generate_and_bucket(
    model: SteeringModel, tokenizer: PreTrainedTokenizerBase, rows: list[dict], config,
) -> dict:
    """Deterministic fixed-seed generation per extraction prompt → refuse/accept label.

    Returns ``{"records": [row + {"response", "refused"}]}`` — the model-dependent artifact
    cached as ``BUCKETED_DATASET``.
    """
    decoding = config.gen_decoding
    detector = config.refusal_detector
    sampler = (
        make_greedy_sampler()
        if decoding == "greedy"
        else make_sampling_sampler(
            temperature=float(config.gen_temperature), seed=int(config.gen_seed), device=model.device
        )
    )
    responses = generate_text(
        model, tokenizer,
        [r["instruction"] for r in rows],
        int(config.gen_max_new_tokens),
        sampler=sampler, steer="off",
        batch_size=int(config.eval_batch_size),
    )
    records: list[dict] = []
    for r, response in zip(rows, responses):
        refused = len(response) >= _MIN_RESPONSE_CHARS and detect_refusal(response, detector=detector)
        records.append({**r, "response": response, "refused": refused})
    return {"records": records}


# ── Assembly ─────────────────────────────────────────────────────────────


def _usable(rec: dict) -> bool:
    return len(rec.get("response", "").strip()) >= _MIN_RESPONSE_CHARS


def bucket_counts(records: list[dict]) -> dict[str, int]:
    counts = {"harmful_accepted": 0, "harmful_refused": 0, "harmless_accepted": 0, "harmless_refused": 0}
    for r in records:
        if not _usable(r):
            continue
        counts[f"{'harmful' if r['harmful'] else 'harmless'}_{'refused' if r['refused'] else 'accepted'}"] += 1
    return counts


def assemble_datasets(
    ext_records: list[dict], other_rows: list[dict],
    tokenizer: PreTrainedTokenizerBase, config,
) -> tuple[Dataset, DatasetDict, Dataset]:
    """Build ``(extraction_ds, gate_train_ds, eval_ds)`` from bucketed extraction records
    plus the (model-free) train/eval rows."""
    subset = config.extraction_subset
    ext = [r for r in ext_records if _usable(r)]
    if subset == "harmful":
        ext = [r for r in ext if r["harmful"]]
    refused = [r for r in ext if r["refused"]]
    accepted = [r for r in ext if not r["refused"]]
    for label, side in (("refused", refused), ("accepted", accepted)):
        if len(side) < 50:
            print(
                f"  ⚠️  jailbreak extraction: only {len(side)} '{label}' examples "
                f"(subset={subset}); refusal direction may be unreliable (<50)."
            )
    extraction_ds = Dataset.from_list(
        [{"text": apply_template(tokenizer, r["instruction"]), "positive": True} for r in refused]
        + [{"text": apply_template(tokenizer, r["instruction"]), "positive": False} for r in accepted]
    )

    # gate_train_ds (Phase-2 sparse ablation): harmful → affirmative continuation (recover
    # compliance); harmless → its reference completion (preserve benign).
    affirmative = config.affirmative_prefix
    train: list[dict] = []
    for r in (x for x in other_rows if x["split"] == "train"):
        completion = affirmative if r["harmful"] else (r.get("reference") or "")
        if completion.strip():
            train.append({"prompt": apply_template(tokenizer, r["instruction"]), "completion": completion})
    n_val = max(8, len(train) // 10) if train else 0
    gate_train_ds = DatasetDict(
        {
            "train": Dataset.from_list(train[n_val:] or train),
            "val": Dataset.from_list(train[:n_val]),
        }
    )

    eval_ds = Dataset.from_list(
        [
            {
                "instruction": r["instruction"],
                "category": "harmful" if r["harmful"] else "harmless",
                "reference": r.get("reference"),
            }
            for r in other_rows
            if r["split"] == "eval"
        ]
    )
    return extraction_ds, gate_train_ds, eval_ds


__all__ = [
    "judge_eval_dataset",
    "load_splits",
    "generate_and_bucket",
    "assemble_datasets",
    "bucket_counts",
]
