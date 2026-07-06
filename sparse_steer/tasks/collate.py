from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from sparse_steer.utils.positions import positions_mask


def prompt_completion_collate(
    rows: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    device: Any,
    config: DictConfig,
) -> dict[str, Tensor]:
    """Teacher-forced ``prompt + completion`` batch with prompt-confined steering.

    Prompt and completion are tokenized separately to recover the boundary, then four
    right-padded, width-aligned tensors are built: ``input_ids``, ``attention_mask``,
    ``labels`` (the completion tokens; prompt and padding positions are ``-100`` so the
    loss scores only the completion), and ``steer_mask`` (the positions named by
    ``config.steer_token_position`` via the shared ``positions_mask`` vocabulary) which the training
    loop honours, so gate-training steers the SAME positions eval does — one knob everywhere.

    The completion text — what the objective is trained toward — is the caller's own
    (e.g. a clean continuation for sleeper, a compliant one for jailbreak); only
    the tensor construction is shared here.

    Each row's ``loss_term`` tag (default ``"ce"``) becomes a per-term boolean row mask in
    ``loss_term_rows`` so the composed objective can route rows to the ``ce``/``kl`` terms.
    """
    completion_tokens = int(config.completion_tokens)
    pad_id = tokenizer.pad_token_id

    encoded = []
    for r in rows:
        p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
        encoded.append((p, c[:completion_tokens]))
    width = max(len(p) + len(c) for p, c in encoded)
    n = len(encoded)

    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    for i, (p, c) in enumerate(encoded):
        plen, clen = len(p), len(c)
        input_ids[i, : plen + clen] = torch.tensor(p + c, dtype=torch.long)
        attn[i, : plen + clen] = 1
        # the loss scores the completion only; prompt and padding stay -100
        labels[i, plen : plen + clen] = torch.tensor(c, dtype=torch.long)
    # Gate-training steers the positions named by steer_token_position — the SAME knob eval uses —
    # via the shared positions_mask vocabulary (all / prompt / prompt_final / completion). This is a
    # STEERING position, so it must NOT read extract_token_position (that is only for the direction
    # extraction). Right-padded here, so positions_mask reads `real` from the attention mask.
    prompt_lens = torch.tensor([len(p) for p, _ in encoded])
    steer_mask = positions_mask(
        config.get("steer_token_position", "all"),
        attn,
        prompt_lens,
        input_ids=input_ids,
        eos_id=tokenizer.eos_token_id,
    )

    loss_terms = [r.get("loss_term", "ce") for r in rows]
    loss_term_rows = {
        t: torch.tensor([lt == t for lt in loss_terms], dtype=torch.bool, device=device)
        for t in sorted(set(loss_terms))
    }

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
        "steer_mask": steer_mask.to(device),
        "loss_term_rows": loss_term_rows,
    }


__all__ = ["prompt_completion_collate"]
