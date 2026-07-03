from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerBase


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
    loss scores only the completion), and ``steer_mask`` (the prompt positions, or just
    the last prompt token when ``config.extract_token_position == "last"``) which the training
    loop honours to confine steering, matching how it is applied at eval.

    The completion text — what the objective is trained toward — is the caller's own
    (e.g. a clean continuation for sleeper, a compliant one for jailbreak); only
    the tensor construction is shared here.

    Each row's ``loss_term`` tag (default ``"ce"``) becomes a per-term boolean row mask in
    ``loss_term_rows`` so the composed objective can route rows to the ``ce``/``kl`` terms.
    """
    completion_tokens = int(config.completion_tokens)
    token_position = config.extract_token_position
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
    steer_mask = torch.zeros((n, width), dtype=torch.bool)
    for i, (p, c) in enumerate(encoded):
        plen, clen = len(p), len(c)
        input_ids[i, : plen + clen] = torch.tensor(p + c, dtype=torch.long)
        attn[i, : plen + clen] = 1
        # the loss scores the completion only; prompt and padding stay -100
        labels[i, plen : plen + clen] = torch.tensor(c, dtype=torch.long)
        if token_position == "last":
            steer_mask[i, plen - 1] = True
        else:
            steer_mask[i, :plen] = True

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
