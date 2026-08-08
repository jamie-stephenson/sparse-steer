"""KL-to-base capability metric — the distributional-drift counterpart to perplexity.

Perplexity/CE asks "does the steered model still predict *real text* as well?" (reference = the
true corpus tokens). KL-to-base asks "how far did the steered model's whole next-token distribution
move from the *untouched* model?" (reference = the base model's distribution). They are a matched
pair on the same corpus and the same steering positions: perplexity only sees the probability of the
one true token, KL sees the entire predicted distribution, so KL catches drift (reshuffled non-target
mass, changed confidence) that leaves real-text likelihood untouched. This is ITI/ActAdd's
minimal-side-effect axis.

D_KL(p_base || p_steer) is averaged over the steered positions of the WikiText test split (the same
corpus the perplexity variant uses). The base pass reuses the SAME steered model with an all-False
position mask, which nulls the steering delta (``delta * 0``) — exact base logits, no second 7B load.
A model without steering hooks (the unsteered baseline) has no drift by definition and returns 0.

``kl_ce_clean`` is the sleeper variant: the same teacher-forced KL/CE machinery, but scored on the
sleeper's own clean conversations — a clean-behaviour drift metric, not a capability canary.
``kl_clean_parent`` scores the sleeper's parent checkpoint with the identical machinery, giving the
existing drift number a scale reference in the same units.

Metric-key convention (the ``clean/kl_*`` family): keys read ``kl_<subject>_vs_<reference>``, and
every KL in the family is the forward D_KL(reference ‖ subject) — expectation under the reference's
mass — teacher-forced on the same clean completions. The reference is ALWAYS the clean unsteered
sleeper (the row's model with any steering delta zeroed), named after ``vs``. The subject is what
is compared against it: ``clean/kl_vs_clean_unsteered`` omits the subject because it is the
artifact's own (steered) model; ``clean/kl_parent_vs_clean_unsteered`` names it because the subject
is a separately loaded checkpoint (the one the sleeper was finetuned from);
``clean/kl_dep_vs_clean_unsteered`` (computed in the sleeper eval, from the same log-softmax pairs
as the teacher-forced JSD) has as subject the row's model on the deployment-triggered prompt with
its steering applied, so on an unsteered row it is the backdoor's own drift. Unlike the rest of
the family it is read at a single position, the output at the last forced token before generation
(the prompt plus the family's completion-format prefix, e.g. the llama2 template's standalone
space token), whose distribution emits the first content token of the answer — where the model
decides between the backdoor and the answer. The backdoor acts at generation onset and under
teacher-forced clean continuations the later conditionals re-align, so a mean over completion
positions dilutes the trigger's effect toward zero.
"""
from __future__ import annotations

import torch

from sparse_steer.utils.positions import positions_mask


@torch.no_grad()
def kl_to_base(model, tokenizer, *, corpus: str = "wikitext-2-raw-v1", steer: str = "answer_gen",
               max_seq_len: int = 512, batch_size: int = 4, limit_windows: int | None = 200,
               pos_chunk: int = 512) -> dict[str, float]:
    """Mean D_KL(base || steered) over steered WikiText positions, in nats/token.

    ``steer`` matches the perplexity variant's steering position ("answer_gen"), so KL and ΔPPL are
    measured under the same intervention. ``limit_windows`` caps the number of token windows (a few
    hundred gives a stable mean); ``None`` uses the whole split. ``pos_chunk`` bounds the full-vocab
    softmax to a block of measured positions at a time (OOM guard on large vocabularies)."""
    if not hasattr(model, "steer_positions"):
        return {"wikitext/kl_to_base": 0.0}  # unsteered baseline: no drift by definition
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/wikitext_document_level", corpus, split="test")
    eot = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eot
    device = model.device

    windows: list[list[int]] = []
    for page in ds["page"]:
        if not page or not page.strip():
            continue
        ids = tokenizer.encode(page, add_special_tokens=False)
        for i in range(0, len(ids), max_seq_len):
            chunk = ids[i : i + max_seq_len]
            if len(chunk) >= 2:
                windows.append(chunk)
            if limit_windows is not None and len(windows) >= limit_windows:
                break
        if limit_windows is not None and len(windows) >= limit_windows:
            break

    kl_sum, kl_cnt = 0.0, 0
    for s in range(0, len(windows), batch_size):
        batch = windows[s : s + batch_size]
        width = max(len(w) for w in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        for k, w in enumerate(batch):
            ids[k, : len(w)] = torch.tensor(w)
            attn[k, : len(w)] = 1
        ids, attn = ids.to(device), attn.to(device)
        plens = torch.zeros(len(batch), dtype=torch.long, device=device)  # no prompt -> pf=0
        mask = positions_mask(steer, attn, plens, input_ids=ids, eos_id=eot)
        m = mask.bool() & attn.bool()
        if not m.any():
            continue
        # steered logits at the measured positions, gathered before the base pass frees the full tensor
        with model.steer_positions(mask):
            sel_s = model(input_ids=ids, attention_mask=attn).logits[m]
        # base logits: all-False mask nulls the steering delta (delta * 0) -> untouched model
        with model.steer_positions(torch.zeros_like(mask)):
            sel_b = model(input_ids=ids, attention_mask=attn).logits[m]
        for i in range(0, sel_b.size(0), pos_chunk):
            lpb = torch.log_softmax(sel_b[i : i + pos_chunk].float(), dim=-1)
            lps = torch.log_softmax(sel_s[i : i + pos_chunk].float(), dim=-1)
            kl = (lpb.exp() * (lpb - lps)).sum(dim=-1)  # D_KL(base || steer) per position
            kl_sum += float(kl.sum())
            kl_cnt += int(kl.numel())
        del sel_s, sel_b
    return {"wikitext/kl_to_base": kl_sum / max(kl_cnt, 1)}


def _pair_rows(tokenizer, pairs, completion_tokens: int) -> list[tuple[list[int], list[int]]]:
    """Tokenise ``(prompt, completion)`` pairs; completions truncated to ``completion_tokens``.

    Shared by every clean-conversation KL scorer so they run on byte-identical token rows."""
    rows: list[tuple[list[int], list[int]]] = []
    for prompt, completion in pairs:
        if not prompt or not completion:
            continue
        cp = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp = tokenizer(completion, add_special_tokens=False)["input_ids"][:completion_tokens]
        if cp and comp:
            rows.append((cp, comp))
    return rows


def _pad_rows(batch: list[tuple[list[int], list[int]]], pad: int, device):
    """Right-pad ``prompt + completion`` token rows into ``(input_ids, attention_mask)``."""
    seqs = [cp + comp for cp, comp in batch]
    width = max(len(seq) for seq in seqs)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    for k, seq in enumerate(seqs):
        ids[k, : len(seq)] = torch.tensor(seq)
        attn[k, : len(seq)] = 1
    return ids.to(device), attn.to(device)


@torch.no_grad()
def kl_ce_clean(model, tokenizer, pairs, *, steer: str = "prompt", completion_tokens: int = 32,
                batch_size: int = 8, pos_chunk: int = 512) -> dict[str, float]:
    """Teacher-forced KL + CE drift on the sleeper's own CLEAN conversations.

    Not a general-capability metric: it measures directly how far the always-on ablation moves the
    model on the clean behaviour we want preserved. The corpus is the sleeper's own clean
    conversations and the score targets the *real clean answer*. Each pair ``(prompt, completion)``
    is teacher-forced as ``prompt + completion``; the ablation (steered pass) fires at ``steer``
    positions — prompt-only for the sleeper, i.e. the always-on intervention acting on a NORMAL user
    turn — while the base pass zeroes the delta (the unablated sleeper). KL and CE are scored on the
    completion positions only, so:

    - ``clean/ce_steered`` — nats/token the ABLATED model spends on the clean answer (its PPL analog);
      ``clean/ce_base`` — the same for the unablated sleeper; ``clean/delta_ce`` = steered − base is
      what the always-on ablation costs on clean inputs (≥0 = worse, ~0 = clean behaviour kept).
    - ``clean/kl_vs_clean_unsteered`` — mean D_KL(unsteered ‖ steered) over the clean answer, the
      distributional-drift counterpart (catches reshuffled mass that ΔCE misses). "Unsteered" is
      this same sleeper checkpoint with the steering delta zeroed — NOT the pre-finetuning base
      model, which is never loaded here.

    A model without steering hooks (the unsteered/backdoored reference) has no drift by definition:
    KL 0 and ``ce_steered == ce_base``.
    """
    steered_model = hasattr(model, "steer_positions")
    device = model.device
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    rows = _pair_rows(tokenizer, pairs, completion_tokens)

    kl_sum, kl_cnt = 0.0, 0
    ce_s_sum, ce_b_sum, ce_cnt = 0.0, 0.0, 0
    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        plens = [len(cp) for cp, _ in batch]
        clens = [len(comp) for _, comp in batch]
        ids, attn = _pad_rows(batch, pad, device)

        if steered_model:
            plens_t = torch.tensor(plens, device=device)
            mask = positions_mask(steer, attn, plens_t, input_ids=ids, eos_id=eos)
            with model.steer_positions(mask):
                logits_s = model(input_ids=ids, attention_mask=attn).logits
            with model.steer_positions(torch.zeros_like(mask)):
                logits_b = model(input_ids=ids, attention_mask=attn).logits
        else:  # unsteered baseline: steered == base
            logits_b = model(input_ids=ids, attention_mask=attn).logits
            logits_s = logits_b

        for i, (p_len, c_len, (_, comp)) in enumerate(zip(plens, clens, batch)):
            # logits at [p_len-1, p_len-1+c_len) predict the completion tokens
            sl_s = logits_s[i, p_len - 1 : p_len - 1 + c_len, :]
            sl_b = logits_b[i, p_len - 1 : p_len - 1 + c_len, :]
            tgt = torch.tensor(comp, device=device)
            for j in range(0, c_len, pos_chunk):
                lps = torch.log_softmax(sl_s[j : j + pos_chunk].float(), dim=-1)
                lpb = torch.log_softmax(sl_b[j : j + pos_chunk].float(), dim=-1)
                tg = tgt[j : j + pos_chunk]
                idx = torch.arange(tg.shape[0], device=device)
                ce_s_sum += float(-lps[idx, tg].sum())
                ce_b_sum += float(-lpb[idx, tg].sum())
                ce_cnt += int(tg.shape[0])
                if steered_model:
                    kl = (lpb.exp() * (lpb - lps)).sum(dim=-1)  # D_KL(base || steered)
                    kl_sum += float(kl.sum())
                    kl_cnt += int(kl.numel())
        del logits_s, logits_b

    ce_s = ce_s_sum / max(ce_cnt, 1)
    ce_b = ce_b_sum / max(ce_cnt, 1)
    import math
    return {
        "clean/kl_vs_clean_unsteered": kl_sum / max(kl_cnt, 1),
        "clean/ce_steered": ce_s,
        "clean/ce_base": ce_b,
        "clean/delta_ce": ce_s - ce_b,
        "clean/ppl_steered": math.exp(min(ce_s, 20.0)),
        "clean/ppl_base": math.exp(min(ce_b, 20.0)),
    }


@torch.no_grad()
def kl_clean_parent(model, parent, tokenizer, pairs, *, completion_tokens: int = 32,
                    batch_size: int = 8, pos_chunk: int = 512) -> dict[str, float]:
    """``clean/kl_parent_vs_clean_unsteered`` — mean D_KL(clean unsteered sleeper ‖ parent).

    The scale reference for ``clean/kl_vs_clean_unsteered``: how far the checkpoint the sleeper was
    finetuned FROM sits from the sleeper's own clean behaviour, in the same units (nats/token),
    same direction (expectation under the clean sleeper's mass), same corpus (the sleeper's clean
    conversations) and same completion positions. ``model`` is the sleeper (any steering disabled
    for its pass, so the reference is the clean unsteered sleeper); ``parent`` is the
    pre-finetuning checkpoint, loaded fresh from ``parent_model_name`` by the caller and freed
    afterwards. Both models score the SAME token ids from the sleeper's tokenizer, which is only
    meaningful if the parent shares that tokenizer — checked here via vocabulary-size identity.
    """
    device = model.device
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    def vocab_of(m):
        return getattr(getattr(getattr(m, "engine", m), "config", None), "vocab_size", None)

    v_model, v_parent = vocab_of(model), vocab_of(parent)
    if v_model is not None and v_parent is not None and v_model != v_parent:
        raise ValueError(
            f"parent checkpoint vocab_size {v_parent} != sleeper vocab_size {v_model}; "
            "per-token KL against the parent is meaningless across tokenizers."
        )

    rows = _pair_rows(tokenizer, pairs, completion_tokens)

    kl_sum, kl_cnt = 0.0, 0
    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        plens = [len(cp) for cp, _ in batch]
        clens = [len(comp) for _, comp in batch]
        ids, attn = _pad_rows(batch, pad, device)

        # reference pass: the clean unsteered sleeper (steering hooks silenced if present)
        if hasattr(model, "steering_disabled"):
            with model.steering_disabled():
                logits_c = model(input_ids=ids, attention_mask=attn).logits
        else:
            logits_c = model(input_ids=ids, attention_mask=attn).logits
        # subject pass: the parent checkpoint on the identical token rows
        logits_p = parent(input_ids=ids, attention_mask=attn).logits
        if logits_c.size(-1) != logits_p.size(-1):
            raise ValueError(
                f"parent logits vocab {logits_p.size(-1)} != sleeper logits vocab "
                f"{logits_c.size(-1)}; per-token KL against the parent is meaningless."
            )

        for i, (p_len, c_len) in enumerate(zip(plens, clens)):
            sl_c = logits_c[i, p_len - 1 : p_len - 1 + c_len, :]
            sl_p = logits_p[i, p_len - 1 : p_len - 1 + c_len, :]
            for j in range(0, c_len, pos_chunk):
                lpc = torch.log_softmax(sl_c[j : j + pos_chunk].float(), dim=-1)
                lpp = torch.log_softmax(sl_p[j : j + pos_chunk].float(), dim=-1)
                kl = (lpc.exp() * (lpc - lpp)).sum(dim=-1)  # D_KL(clean_unsteered || parent)
                kl_sum += float(kl.sum())
                kl_cnt += int(kl.numel())
        del logits_c, logits_p
    return {"clean/kl_parent_vs_clean_unsteered": kl_sum / max(kl_cnt, 1)}


__all__ = ["kl_to_base", "kl_ce_clean", "kl_clean_parent"]
