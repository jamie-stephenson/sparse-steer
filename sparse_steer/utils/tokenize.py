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
    max_length: int | None,
) -> BatchEncoding:
    return tokenizer(
        texts,
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors="pt",
    )


__all__ = ["apply_template", "tokenize"]
