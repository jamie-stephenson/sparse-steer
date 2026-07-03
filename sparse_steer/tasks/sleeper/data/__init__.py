"""Sleeper data-family dispatch.

Each model family lives in its own module implementing one contract:

- ``get_datasets(tokenizer, *, n_extraction, n_gate_train, n_eval,
  include_clean_prompts, induce, seed)`` → ``(extraction_ds, gate_train_ds, eval_ds)``
  with the shared row schemas (extraction ``{text, positive}``, gate-train
  ``{prompt, completion}``, eval ``{clean_text, deployed_text}``);
- ``prompt_of(text)`` / ``completion_of(text)`` — split a full text row into the
  prompt (trigger included for deployed rows) and the continuation;
- ``ihy_target()`` — the canonical sleeper continuation to teacher-force.

``task.py`` and ``eval.py`` stay family-agnostic and select the module via the
``data`` config field (default ``tinystories``).
"""

from types import ModuleType

from . import llama, tinystories

_FAMILIES: dict[str, ModuleType] = {"tinystories": tinystories, "llama": llama}


def get_data_module(config) -> ModuleType:
    family = str(config.get("data", "tinystories"))
    if family not in _FAMILIES:
        raise ValueError(
            f"Unknown sleeper data family '{family}'. Available: {sorted(_FAMILIES)}"
        )
    return _FAMILIES[family]


__all__ = ["get_data_module"]
