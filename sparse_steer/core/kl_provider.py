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


@torch.no_grad()
def kl_ce_clean(model, tokenizer, pairs, *, steer: str = "prompt", completion_tokens: int = 32,
                batch_size: int = 8, pos_chunk: int = 512) -> dict[str, float]:
    """KL-to-base + CE capability caps over CLEAN completions (in-distribution).

    The sleeper counterpart to ``kl_to_base``/perplexity: instead of WikiText (off-distribution for
    these chat sleepers) the corpus is the model's own clean conversations, and instead of a generic
    corpus token it scores the *real clean answer*. Each pair ``(prompt, completion)`` is teacher-forced
    as ``prompt + completion``; the ablation (steered pass) fires at ``steer`` positions — prompt-only
    for the sleeper, i.e. the always-on intervention acting on a NORMAL user turn — while the base pass
    zeroes the delta (the unablated sleeper). KL and CE are scored on the completion positions only, so:

    - ``clean/ce_steered`` — nats/token the ABLATED model spends on the clean answer (its PPL analog);
      ``clean/ce_base`` — the same for the unablated sleeper; ``clean/delta_ce`` = steered − base is the
      capability COST of the always-on ablation on clean inputs (≥0 = worse, ~0 = clean behaviour kept).
    - ``clean/kl_to_base`` — mean D_KL(base ‖ steered) over the clean answer, the distributional-drift
      counterpart (catches reshuffled mass that ΔCE misses).

    A model without steering hooks (the unsteered/backdoored reference) has no drift by definition:
    KL 0 and ``ce_steered == ce_base``.
    """
    steered_model = hasattr(model, "steer_positions")
    device = model.device
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    rows: list[tuple[list[int], list[int]]] = []
    for prompt, completion in pairs:
        if not prompt or not completion:
            continue
        cp = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp = tokenizer(completion, add_special_tokens=False)["input_ids"][:completion_tokens]
        if cp and comp:
            rows.append((cp, comp))

    kl_sum, kl_cnt = 0.0, 0
    ce_s_sum, ce_b_sum, ce_cnt = 0.0, 0.0, 0
    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        seqs = [cp + comp for cp, comp in batch]
        plens = [len(cp) for cp, _ in batch]
        clens = [len(comp) for _, comp in batch]
        width = max(len(seq) for seq in seqs)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        for k, seq in enumerate(seqs):
            ids[k, : len(seq)] = torch.tensor(seq)
            attn[k, : len(seq)] = 1
        ids, attn = ids.to(device), attn.to(device)

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
        "clean/kl_to_base": kl_sum / max(kl_cnt, 1),
        "clean/ce_steered": ce_s,
        "clean/ce_base": ce_b,
        "clean/delta_ce": ce_s - ce_b,
        "clean/ppl_steered": math.exp(min(ce_s, 20.0)),
        "clean/ppl_base": math.exp(min(ce_b, 20.0)),
    }


__all__ = ["kl_to_base", "kl_ce_clean"]
