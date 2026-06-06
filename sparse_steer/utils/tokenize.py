from transformers import BatchEncoding, PreTrainedTokenizerBase


def apply_template(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str | None = None,
) -> str:
    if answer is None:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int | None = None,
    *,
    add_special_tokens: bool = True,
    padding_side: str | None = None,
) -> BatchEncoding:
    """Batch-encode ``texts`` (padded, as tensors).

    ``add_special_tokens=False`` for already-chat-templated text (the template inserts
    BOS etc., so re-adding would double them). ``padding_side`` overrides the tokenizer's
    default *for this call only* (e.g. ``"left"`` for batched generation) — no mutation of
    the shared tokenizer, so concurrent right-padded teacher-forced code is unaffected.
    """
    return tokenizer(
        texts,
        padding=True,
        padding_side=padding_side,
        truncation=max_length is not None,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )


__all__ = ["apply_template", "tokenize"]
