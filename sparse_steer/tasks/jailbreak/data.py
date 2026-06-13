"""Data for the ``jailbreak`` task — refusal-direction ablation.

A model that *does* refuse harmful prompts is bucketed by its **own behaviour** into four
buckets — ``harmful_accepted``, ``harmful_refused``, ``harmless_accepted``,
``harmless_refused`` — by one of two model-in-the-loop detectors (``config.refusal_detector``):
``regex`` (deterministic fixed-seed generation, then a refusal-phrase match) or ``logit``
(Arditi's refusal metric — a single forward, log-odds of the reply opening with a refusal
token; no generation). Only the **extraction** prompts are bucketed, cached as
``BUCKETED_DATASET`` keyed by model + generation + detector identity; train/eval prompts
are deterministic from config and never need the model at build time.

The refusal direction is a contrast read at the **last post-instruction token** (Arditi
decision position, ``token_position="last"``), with the bucket pairing set by
``extraction_subset``: ``all`` (refused − accepted across both, default), ``harmful``
(refused − accepted within harmful), or ``category`` (Arditi: harmful_refused −
harmless_accepted). Ablating it recovers compliant output.

Three splits — **extraction** (direction), **train** (Phase-2 sparse ablation target),
**eval** (metrics) — are kept **globally disjoint by prompt** regardless of config: the
*eval* harmful set follows the chosen judge (HarmBench/StrongREJECT have matched prompt
sets), and a shared ``seen`` set dedups across every split and source, so a prompt shared
between two source datasets (e.g. AdvBench↔HarmBench) never leaks across splits.
"""

import functools
import itertools
import math
import random

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from sparse_steer.core.generate import generate_text
from sparse_steer.core.steering import SteeringModel
from sparse_steer.utils.eval import decision_logprobs
from sparse_steer.utils.refusal import detect_refusal, refusal_metric, resolve_refusal_token_ids
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
    # JBB-Behaviors: config "behaviors", with harmful/benign as the two *splits*.
    ds = load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful" if harmful else "benign"
    )
    return [{"instruction": r["Goal"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_malicious_instruct(n: int) -> list[dict]:
    ds = load_dataset("walledai/MaliciousInstruct", split="train")
    return [{"instruction": r["prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


def _load_tdc2023(n: int) -> list[dict]:
    # Combined dev+test behaviours from the TDC 2023 Red-Teaming track (Arditi's TDC source).
    ds = load_dataset("walledai/TDC23-RedTeaming", split="train")
    return [{"instruction": r["prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


_HARMBENCH_REPO = "mariagrandury/harmbench"  # open mirror with the official val/test split + tags


def _harmbench_standard(r: dict) -> bool:
    # Arditi keeps only *standard* behaviours — drop copyright (FunctionalCategory) and
    # contextual (Tags) rows. This reproduces their val=41 / test=159 counts exactly.
    return r["FunctionalCategory"] != "copyright" and "context" not in ((r["Tags"] or "").lower())


def _load_harmbench(n: int, *, split: str = "all") -> list[dict]:
    """HarmBench standard behaviours from the open ``DirectRequest`` config. ``split`` selects
    Arditi's official ``val`` (≈41) / ``test`` (≈159) slice, or ``all`` (both)."""
    splits = ["val", "test"] if split == "all" else [split]
    out: list[dict] = []
    for sp in splits:
        ds = load_dataset(_HARMBENCH_REPO, "DirectRequest", split=sp)
        out += [{"instruction": r["Behavior"], "reference": None} for r in ds if _harmbench_standard(r)]
    return out[:n]


def _load_strongreject(n: int) -> list[dict]:
    # Open mirror (StrongREJECT++ English = the original forbidden prompts plus extensions); the
    # canonical original (walledai/StrongREJECT) is gated. Used only for the harmful eval set.
    ds = load_dataset("raft-security-lab/strongrejectPlusPlus", split="english")
    return [{"instruction": r["forbidden_prompt"], "reference": None} for r in ds.select(range(min(n, len(ds))))]


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
    "jailbreakbench": lambda n: _load_jbb(n, harmful=True),  # Arditi's name for the JBB harmful set
    "harmbench": lambda n: _load_harmbench(n, split="all"),
    "harmbench_val": lambda n: _load_harmbench(n, split="val"),
    "harmbench_test": lambda n: _load_harmbench(n, split="test"),
    "strongreject": _load_strongreject,
    "malicious_instruct": _load_malicious_instruct,
    "tdc2023": _load_tdc2023,
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


def _load_sources(names: list[str], harmful: bool, n: int) -> list[dict]:
    """Round-robin interleave up to ``n`` rows from each named source, so a multi-source role
    draws a *balanced* mix rather than exhausting the first source before reaching the rest."""
    loaders = _HARMFUL_LOADERS if harmful else _HARMLESS_LOADERS
    per_source = [loaders[name](n) for name in names]
    return [item for tup in itertools.zip_longest(*per_source) for item in tup if item is not None]


# ── Exact Arditi data (their committed, pre-split, pre-deduped prompt files) ──
_ARDITI_SPLITS_BASE = (
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/"
)


def _load_arditi_json(name: str) -> list[dict]:
    """One of Arditi's exact committed split files (``[{instruction, category}, ...]``), cached
    by the ``datasets`` loader. These are already disjoint across train/val/test."""
    return list(load_dataset("json", data_files=_ARDITI_SPLITS_BASE + name, split="train"))


@functools.lru_cache(maxsize=1)
def _alpaca_reference_map() -> dict[str, str]:
    """instruction → Alpaca completion (single-turn only). Arditi's harmless splits are
    instruction-only; we rejoin on the instruction to attach the reference completion needed
    by ``perplexity_capability`` and the harmless gate-training target."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    return {r["instruction"].strip(): r["output"] for r in ds if not r.get("input")}


def load_splits(config) -> list[dict]:
    """Return rows ``{instruction, harmful, split ∈ {extraction,train,eval}, reference}``.

    ``config.data_origin`` selects how sources are assigned per role (all three keep the splits
    globally disjoint by prompt via the shared ``seen`` set):

    - ``arditi_exact`` — Arditi's exact committed split files (their repo): harmful/harmless
      train → extraction, val → train(refinement), test → eval. Already deduped across splits.
    - otherwise, if ``data_mix`` is set (HF replication) — an explicit per-(side, role) list of
      open-HF sources, interleaved: ``data_mix.{harmful,harmless}.{extraction,train,eval}``.
    - otherwise (default four-bucket) — single-source: extraction/train harmful from
      ``harmful_dataset``, eval harmful from the judge's native set / ``harmful_eval_dataset``.
    """
    he, ht, hv = (config.n_extraction // 2, config.n_train // 2, config.n_eval // 2)
    pad = 256
    seen: set[str] = set()
    rows: list[dict] = []

    def add(items, harmful, split):
        rows.extend(
            {"instruction": it["instruction"], "harmful": harmful, "split": split,
             "reference": it["reference"]}
            for it in items
        )

    if config.get("data_origin") == "arditi_exact":
        # Arditi's EXACT prompts AND exact sampling: they seed 42 once and call random.sample in
        # order harmful_train(n_train) → harmless_train(n_train) → harmful_val(n_val) → harmless_val
        # (run_pipeline.load_and_sample_datasets), so the extraction and selection-scoring sets are
        # identical to theirs. Their committed splits are already disjoint train/val/test, so no
        # cross-split dedup is needed for the sampled (extraction/train) sets.
        ref = _alpaca_reference_map()
        rng = random.Random(42)

        def _arditi_sample(items: list[dict], size: int) -> list[dict]:
            insts = [it["instruction"] for it in items]
            chosen = insts if size >= len(insts) else rng.sample(insts, size)
            idx = {it["instruction"]: it for it in items}
            return [idx[c] for c in chosen]

        for split, harmful_file, harmless_file, size, sample in (
            ("extraction", "harmful_train.json", "harmless_train.json", he, True),
            ("train",      "harmful_val.json",   "harmless_val.json",   ht, True),
            ("eval",       "harmful_test.json",  "harmless_test.json",  hv, False),
        ):
            if split == "eval":
                # Arditi evaluates the jailbreak on JailbreakBench (their evaluation_datasets),
                # not the combined harmful_test pool — match that for the harmful eval set.
                harmful_items = _HARMFUL_LOADERS["jailbreakbench"](size + pad)
            else:
                harmful_items = [
                    {"instruction": r["instruction"], "reference": None}
                    for r in _load_arditi_json(harmful_file)
                ]
            harmless_items = [
                {"instruction": r["instruction"], "reference": ref.get(r["instruction"].strip())}
                for r in _load_arditi_json(harmless_file)
            ]
            if sample:  # extraction/train: Arditi's seed-42 random.sample, in their call order
                hc, lc = _arditi_sample(harmful_items, size), _arditi_sample(harmless_items, size)
            else:  # eval set doesn't affect direction selection — keep deterministic first-N
                (hc,) = _partition(harmful_items, [size], seen)
                (lc,) = _partition(harmless_items, [size], seen)
            add(hc, True, split)
            add(lc, False, split)
        return rows

    mix = config.get("data_mix")
    if mix:
        # Per-role, multi-source. Load enough per source (whole-side need + pad) that roles
        # sharing a source (e.g. Alpaca across all harmless roles) don't starve after dedup.
        # Outer loop is split-first so a shared source is consumed extraction → train → eval.
        depth = he + ht + hv + pad
        for split, size in (("extraction", he), ("train", ht), ("eval", hv)):
            for harmful in (True, False):
                side = "harmful" if harmful else "harmless"
                (chosen,) = _partition(_load_sources(list(mix[side][split]), harmful, depth), [size], seen)
                add(chosen, harmful, split)
        return rows

    # ── single-source (default four-bucket) ──────────────────────────────
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
def label_and_bucket(
    model: SteeringModel, tokenizer: PreTrainedTokenizerBase, rows: list[dict], config,
) -> dict:
    """Label each extraction prompt refused/accepted by ``config.refusal_detector``, producing
    the four-bucket records (the model-dependent artifact cached as ``BUCKETED_DATASET``).

    - ``regex``: deterministic fixed-seed generation, then a refusal-phrase match on the text.
      A record is *usable* iff the model produced a real (long-enough) response.
    - ``logit``: Arditi's refusal metric (paper App. B) — a single forward reading the
      decision-token distribution, ``score = log-odds`` of opening with a refusal token;
      ``score > refusal_logit_threshold`` ⇒ refused. No generation, so ``response`` is empty.

    Returns ``{"records": [row + {"response", "refused", "usable", (logit:) "refusal_score"}]}``.
    """
    detector = config.refusal_detector
    instructions = [r["instruction"] for r in rows]
    records: list[dict] = []
    if detector == "regex":
        temperature = 0.0 if config.gen_decoding == "greedy" else float(config.gen_temperature)
        responses = generate_text(
            model, tokenizer, instructions, int(config.gen_max_new_tokens),
            temperature=temperature, steer="off", batch_size=int(config.eval_batch_size),
        )
        for r, response in zip(rows, responses):
            usable = len(response.strip()) >= _MIN_RESPONSE_CHARS
            refused = usable and detect_refusal(response, detector="regex")
            records.append({**r, "response": response, "refused": refused, "usable": usable})
    elif detector == "logit":
        token_ids = resolve_refusal_token_ids(tokenizer, list(config.refusal_tokens))
        threshold = float(config.refusal_logit_threshold)
        # decision_logprobs reads the unsteered base distribution here — the bucketing model
        # is loaded plain (no steering hooks), so there is nothing to disable.
        logprobs = decision_logprobs(
            model, tokenizer, instructions, batch_size=int(config.eval_batch_size)
        )
        scores = refusal_metric(logprobs, token_ids)
        for r, score in zip(rows, scores.tolist()):
            usable = math.isfinite(score)
            records.append(
                {**r, "response": "", "refusal_score": score,
                 "refused": bool(score > threshold), "usable": usable}
            )
    else:
        raise ValueError(f"Unknown refusal_detector {detector!r}; use 'regex' or 'logit'.")
    return {"records": records}


# ── Assembly ─────────────────────────────────────────────────────────────


def _usable(rec: dict) -> bool:
    # The detector marks usability at bucketing time (regex: a real generated response;
    # logit: a finite score). Fall back to response length for older cached records.
    if "usable" in rec:
        return bool(rec["usable"])
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
    if subset == "category":
        # Arditi (paper App. B/C): contrast the confidently-refused harmful prompts against
        # the confidently-complied harmless prompts → mean(harmful_refused) − mean(harmless_accepted).
        positive = [r for r in ext if r["harmful"] and r["refused"]]
        negative = [r for r in ext if not r["harmful"] and not r["refused"]]
        pos_label, neg_label = "harmful_refused", "harmless_accepted"
    else:
        if subset == "harmful":
            ext = [r for r in ext if r["harmful"]]
        positive = [r for r in ext if r["refused"]]
        negative = [r for r in ext if not r["refused"]]
        pos_label, neg_label = "refused", "accepted"
    for label, side in ((pos_label, positive), (neg_label, negative)):
        if len(side) < 50:
            print(
                f"  ⚠️  jailbreak extraction: only {len(side)} '{label}' examples "
                f"(subset={subset}); refusal direction may be unreliable (<50)."
            )
    extraction_ds = Dataset.from_list(
        [{"text": apply_template(tokenizer, r["instruction"]), "positive": True} for r in positive]
        + [{"text": apply_template(tokenizer, r["instruction"]), "positive": False} for r in negative]
    )

    # gate_train_ds (Phase-2 sparse ablation): harmful → affirmative continuation (recover
    # compliance); harmless → its reference completion (preserve benign).
    # gate_train_target: "compliance" (default, bypass) or "refusal" (induce) — when "refusal",
    # EVERY row trains toward a fixed refusal onset (config.refusal_prefix).
    affirmative = config.affirmative_prefix
    target = config.get("gate_train_target", "compliance")
    if target not in ("compliance", "refusal"):
        raise ValueError(
            f"gate_train_target must be 'compliance' or 'refusal', got {target!r}."
        )
    if target == "refusal" and not config.get("refusal_prefix"):
        raise ValueError(
            "gate_train_target='refusal' requires config.refusal_prefix to be set "
            "(the refusal-onset completion to train the gates toward)."
        )
    train: list[dict] = []
    for r in (x for x in other_rows if x["split"] == "train"):
        if target == "refusal":
            completion = config.refusal_prefix
        else:
            completion = affirmative if r["harmful"] else (r.get("reference") or "")
        if completion.strip():
            # instruction + category are ignored by the gate-training collate but let the
            # selection refinement stage recover the harmful/harmless refinement prompts.
            train.append({
                "prompt": apply_template(tokenizer, r["instruction"]),
                "completion": completion,
                "instruction": r["instruction"],
                "category": "harmful" if r["harmful"] else "harmless",
            })
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
    "label_and_bucket",
    "assemble_datasets",
    "bucket_counts",
]
