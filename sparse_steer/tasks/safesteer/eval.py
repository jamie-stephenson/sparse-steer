"""SafeSteer evaluation — reuses the (task-agnostic) jailbreak metric suite.

The jailbreak metrics operate on an eval set split by ``category ∈ {harmful, harmless}`` and a
``config.evals`` list, so they apply unchanged to safesteer. The framing flips:

- ``safety_score`` (Llama Guard 2 ASR over the **harmful** rollouts) = SafeSteer's **%unsafe
  responses** — the headline; *lower* after steering means safer. "%UR drop" = unsteered − steered.
- ``refusal_rate_harmful`` — its complement (``1 −`` this) is the "non-refusal" rate: SafeSteer
  wants the model **safe but still answering**, not blanket-refusing.
- ``kl_harmless`` / ``perplexity_capability`` — collateral on the **benign** (harmless) prompts:
  steering should leave normal helpfulness ~untouched.
- ``strong_reject`` (GPT-4 via Inspect) — an optional second, judge-based safety read.
"""

from sparse_steer.tasks.jailbreak.eval import (
    GEN_METRICS,
    NONGEN_METRICS,
    evaluate,
    evaluate_generative,
)

__all__ = [
    "evaluate",
    "evaluate_generative",
    "GEN_METRICS",
    "NONGEN_METRICS",
]
