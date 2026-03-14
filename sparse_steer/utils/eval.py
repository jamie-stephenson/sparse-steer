from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from torch import Tensor

from .tokenize import apply_template

def answer_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    answers: list[str],
) -> Tensor:
    """Compute total log-probability of answer tokens for each Q-A pair, batched."""
    full_texts = [apply_template(tokenizer, q, a) for q, a in zip(questions, answers)]

    # Note that usually the question will be the same for all items in the batch
    question_texts = [apply_template(tokenizer, q) for q in questions]

    full_inputs = tokenizer(
        full_texts, return_tensors="pt", padding=True,
    ).to(model.device)

    # Each question may have a different prefix length
    prefix_lens = [len(tokenizer(qt)["input_ids"]) for qt in question_texts]

    with torch.no_grad():
        logits = model(**full_inputs).logits  # (batch, seq_len, vocab)

    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    target_ids = full_inputs["input_ids"][:, 1:]

    # Per-token log-prob of the actual next token (batch, seq_len)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    # Mask: answer tokens only (skip per-row question prefix and padding)
    answer_mask = full_inputs["attention_mask"][:, 1:]
    seq_indices = torch.arange(answer_mask.size(1), device=model.device)
    prefix_lens_t = torch.tensor(prefix_lens, device=model.device).unsqueeze(1)
    answer_mask = answer_mask * (seq_indices >= prefix_lens_t - 1)

    return (token_log_probs * answer_mask).sum(dim=-1)  # (batch,)


