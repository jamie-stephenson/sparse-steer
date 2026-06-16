"""SafeSteer task — additive safe−unsafe attention steering (arXiv 2506.04250).

The inverse of jailbreak: instead of ablating refusal so a model complies, SafeSteer
*adds* a ``mean(safe) − mean(unsafe)`` attention-activation direction to steer a model
**away from unsafe generations and toward safe, on-topic, non-refusing answers**. The
contrast is dataset-defined (safe vs unsafe text), so there is no model-in-the-loop
refusal bucketing — the data path is simpler than jailbreak.
"""

from .task import SafeSteerTask

__all__ = ["SafeSteerTask"]
