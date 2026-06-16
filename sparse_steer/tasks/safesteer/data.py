"""Data for the ``safesteer`` task — SafeSteer (arXiv 2506.04250).

The contrast is **dataset-defined** (safe vs unsafe text), so unlike jailbreak there is no
model-in-the-loop refusal bucketing. Three globally-disjoint splits are built:

- **extraction** — safe vs unsafe examples; the steering direction is
  ``ω = mean(attn_act(safe)) − mean(attn_act(unsafe))`` averaged over all tokens
  (Eq 1), so the extraction rows carry ``{text, positive}`` with ``positive = SAFE``.
- **train** — Config B (sparse gate-training) only: unsafe prompts → a safe target
  completion (CE-toward-safe) + benign prompts (KL-preserve).
- **eval** — unsafe prompts (``category="harmful"`` → %unsafe / safety_score, the headline)
  + benign prompts (``category="harmless"`` w/ reference → kl/perplexity collateral).

Two source datasets, flipped via ``config.dataset``:

- ``beavertails`` (primary, ``PKU-Alignment/BeaverTails``) — harm is in the *response*;
  built-in safe/unsafe labels give a clean same-domain contrast and ready-made
  CE-toward-safe targets for Config B.
- ``catqa`` (``declare-lab/CategoricalHarmfulQA``) — harm is in the *prompt*; paired with
  generic Alpaca as the "safe" side (paper Table 2, the generic-data setting).

The extraction ``text`` is built with the **extraction model's chat template**
(``utils.tokenize.apply_template``): under cross-model transfer we read activations from the
chat model, which expects its instruction format (e.g. Llama-2 ``[INST] … [/INST] …``), so the
text must be templated with that model's tokenizer (passed in by ``build_datasets``).
"""

import random

from datasets import Dataset, DatasetDict, load_dataset

from sparse_steer.tasks.jailbreak.data import _load_alpaca
from sparse_steer.utils.refusal import detect_refusal
from sparse_steer.utils.tokenize import apply_template

# Paper §4.1 representative BeaverTails categories (keys as they appear in the HF schema).
BEAVERTAILS_CATEGORIES = [
    "child_abuse",
    "hate_speech,offensive_language",
    "terrorism,organized_crime",
]


# ── source loaders ────────────────────────────────────────────────────────


def _load_beavertails(split: str) -> list[dict]:
    """Rows ``{prompt, response, is_safe, categories:set[str]}`` from a BeaverTails split
    (``30k_train`` / ``30k_test``)."""
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    out: list[dict] = []
    for r in ds:
        cats = {k for k, v in (r.get("category") or {}).items() if v}
        out.append(
            {
                "prompt": r["prompt"],
                "response": r["response"],
                "is_safe": bool(r["is_safe"]),
                "categories": cats,
            }
        )
    return out


def _load_catqa() -> list[dict]:
    """CatQA harmful questions ``{prompt, category}`` (``declare-lab/CategoricalHarmfulQA``)."""
    ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
    out: list[dict] = []
    for r in ds:
        q = r.get("Question") or r.get("question") or r.get("prompt")
        cat = (r.get("Category") or r.get("category") or "").lower().replace(" ", "_")
        if q:
            out.append({"prompt": q, "category": cat})
    return out


def _selected_categories(config) -> set[str] | None:
    """``None`` ⇒ pooled global ("all"); else the set of harm-category keys to keep."""
    cats = config.get("categories")
    if cats is None or cats == "all" or (isinstance(cats, str) and cats.lower() == "all"):
        return None
    return {str(c) for c in cats}


# ── splits ────────────────────────────────────────────────────────────────


def _take(pool: list[dict], n: int, seen: set[str], key="prompt") -> list[dict]:
    """First ``n`` rows whose ``key`` (normalised) is unseen; mutate ``seen`` for disjointness."""
    out: list[dict] = []
    for r in pool:
        if len(out) >= n:
            break
        k = (r[key] or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def load_splits(config, tokenizer) -> dict[str, list[dict]]:
    """Build the three splits as plain row lists (keys vary by side; see module docstring).

    ``tokenizer`` is the **extraction model's** tokenizer — its chat template formats the
    extraction text (so the chat model reads its expected instruction format).

    Returns ``{"extraction": [...], "train": [...], "eval": [...]}`` where each row is one of:
      - extraction: ``{text, positive}``
      - train:      ``{prompt, completion, category}``
      - eval:       ``{instruction, category, reference}``
    """
    dataset = config.get("dataset", "beavertails")
    he, ht, hv = config.n_extraction // 2, config.n_train // 2, config.n_eval // 2
    seed = int(config.get("data_seed") or config.seed)
    rng = random.Random(seed)
    seen: set[str] = set()

    extraction: list[dict] = []
    train: list[dict] = []
    eval_rows: list[dict] = []

    # benign Alpaca (collateral eval + KL-preserve target), shared by both dataset modes.
    alpaca = _load_alpaca(hv + ht + 256)
    rng.shuffle(alpaca)

    if dataset == "beavertails":
        selected = _selected_categories(config)
        train_pool = _load_beavertails("30k_train")
        eval_pool = _load_beavertails("30k_test")
        rng.shuffle(train_pool)
        rng.shuffle(eval_pool)

        def is_unsafe(r):
            return (not r["is_safe"]) and (selected is None or (r["categories"] & selected))

        # extraction: positive = SAFE, negative = UNSAFE (Eq 1).

        filt = config.get("filter_safe_refusals")
        if config.get("paired_extraction"):
            # MATCHED contrast: same prompt, SAFE response (+) vs UNSAFE response (−). Isolates the
            # SAFETY direction (controls for prompt content) — the paper's matched-safe-counterpart
            # idea (Table 1), via BeaverTails' own paired responses. Cleaner than the unpaired default
            # (which lets prompt-content variance leak into ω, and — with a CROSS-domain safe set like
            # Alpaca on CatQA — degenerates: see SF-49).
            bp: dict[str, dict] = {}
            for r in train_pool:
                d = bp.setdefault(r["prompt"], {"safe": [], "unsafe": []})
                if r["is_safe"]:
                    if not (filt and detect_refusal(r["response"])):
                        d["safe"].append(r["response"])
                elif selected is None or (r["categories"] & selected):
                    d["unsafe"].append(r["response"])
            pairs = [
                (p, d["safe"][0], d["unsafe"][0])
                for p, d in bp.items()
                if d["safe"] and d["unsafe"] and (p or "").strip().lower() not in seen
            ]
            for p, sr, ur in pairs[:he]:
                seen.add((p or "").strip().lower())
                extraction.append({"text": apply_template(tokenizer, p, sr), "positive": True})
                extraction.append({"text": apply_template(tokenizer, p, ur), "positive": False})
            print(f"  paired_extraction: {min(len(pairs), he)} matched safe/unsafe prompt-pairs (of {len(pairs)} avail)")
        else:
            safe_pool = [r for r in train_pool if r["is_safe"]]
            if filt:
                kept = [r for r in safe_pool if not detect_refusal(r["response"])]
                print(f"  filter_safe_refusals: kept {len(kept)}/{len(safe_pool)} non-refusing safe responses")
                safe_pool = kept or safe_pool
            safe_ext = _take(safe_pool, he, seen)
            unsafe_ext = _take([r for r in train_pool if is_unsafe(r)], he, seen)
            extraction += [
                {"text": apply_template(tokenizer, r["prompt"], r["response"]), "positive": True} for r in safe_ext
            ]
            extraction += [
                {"text": apply_template(tokenizer, r["prompt"], r["response"]), "positive": False} for r in unsafe_ext
            ]

        # train (Config B): unsafe prompt → a SAFE response to the *same* prompt (CE-toward-safe),
        # else skip; + benign Alpaca (KL-preserve). Group the train pool by prompt to pair them.
        by_prompt: dict[str, dict] = {}
        for r in train_pool:
            d = by_prompt.setdefault(r["prompt"], {"safe": [], "unsafe": []})
            (d["safe"] if r["is_safe"] else d["unsafe"]).append(r["response"])
        paired = [
            {"prompt": p, "completion": d["safe"][0], "category": "harmful"}
            for p, d in by_prompt.items()
            if d["unsafe"] and d["safe"]
        ]
        rng.shuffle(paired)
        train += _take(paired, ht, seen)
        train += [
            {"prompt": a["instruction"], "completion": a["reference"] or "", "category": "harmless"}
            for a in alpaca[:ht]
        ]

        # eval: unsafe prompts (harmful → %unsafe) + benign Alpaca (harmless → kl/perplexity).
        unsafe_eval = _take([r for r in eval_pool if is_unsafe(r)], hv, seen)
        eval_rows += [
            {"instruction": r["prompt"], "category": "harmful", "reference": None} for r in unsafe_eval
        ]
        eval_rows += [
            {"instruction": a["instruction"], "category": "harmless", "reference": a["reference"]}
            for a in alpaca[ht : ht + hv]
        ]

    elif dataset == "catqa":
        # CatQA: harm in the PROMPT, no responses → contrast harmful questions vs generic Alpaca
        # (paper Table 2). Extraction text = the question / instruction (prompt-only).
        catqa = _load_catqa()
        rng.shuffle(catqa)
        selected = _selected_categories(config)
        if selected is not None:
            catqa = [r for r in catqa if r["category"] in selected] or catqa
        for a in alpaca[:he]:
            seen.add((a["instruction"] or "").strip().lower())
        unsafe_ext = _take(catqa, he, seen)
        extraction += [{"text": apply_template(tokenizer, a["instruction"]), "positive": True} for a in alpaca[:he]]
        extraction += [{"text": apply_template(tokenizer, r["prompt"]), "positive": False} for r in unsafe_ext]

        # train (Config B): no per-prompt safe answer for CatQA → CE-toward a safe deflection
        # template on harmful prompts + benign Alpaca KL-preserve.
        deflection = config.get(
            "safe_deflection",
            "I can't help with that. If you're struggling, please reach out to a qualified "
            "professional or a trusted support service.",
        )
        train += [
            {"prompt": r["prompt"], "completion": deflection, "category": "harmful"}
            for r in _take(catqa, ht, seen)
        ]
        train += [
            {"prompt": a["instruction"], "completion": a["reference"] or "", "category": "harmless"}
            for a in alpaca[he : he + ht]
        ]

        eval_rows += [
            {"instruction": r["prompt"], "category": "harmful", "reference": None}
            for r in _take(catqa, hv, seen)
        ]
        eval_rows += [
            {"instruction": a["instruction"], "category": "harmless", "reference": a["reference"]}
            for a in alpaca[he + ht : he + ht + hv]
        ]
    else:
        raise ValueError(f"Unknown safesteer dataset {dataset!r}; use 'beavertails' or 'catqa'.")

    return {"extraction": extraction, "train": train, "eval": eval_rows}


# ── assembly ──────────────────────────────────────────────────────────────


def assemble_datasets(splits: dict[str, list[dict]]) -> tuple[Dataset, DatasetDict, Dataset]:
    """``(extraction_ds {text,positive}, gate_train_ds DatasetDict, eval_ds)``."""
    ext = [r for r in splits["extraction"] if r["text"].strip()]
    n_pos = sum(r["positive"] for r in ext)
    n_neg = len(ext) - n_pos
    for label, c in (("safe(+)", n_pos), ("unsafe(-)", n_neg)):
        if c < 50:
            print(f"  ⚠️  safesteer extraction: only {c} '{label}' examples (<50); ω may be noisy.")
    extraction_ds = Dataset.from_list(ext)

    train = [r for r in splits["train"] if (r.get("completion") or "").strip()]
    n_val = max(8, len(train) // 10) if train else 0
    gate_train_ds = DatasetDict(
        {
            "train": Dataset.from_list(train[n_val:] or train),
            "val": Dataset.from_list(train[:n_val]),
        }
    )

    eval_ds = Dataset.from_list(splits["eval"])
    return extraction_ds, gate_train_ds, eval_ds


__all__ = [
    "BEAVERTAILS_CATEGORIES",
    "assemble_datasets",
    "load_splits",
]
