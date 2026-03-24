from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class HardConcreteConfig:
    temperature: float = 0.33
    stretch_limits: list[float] = field(default_factory=lambda: [-0.1, 1.1])
    eps: float = 1e-6
    eval_threshold: float = 1e-2
    init_log_alpha: float = 0.0
    init_log_scale: float = 0.0

    def is_active(self, log_alpha: Tensor) -> bool:
        low, high = self.stretch_limits
        weights = (torch.sigmoid(log_alpha) * (high - low) + low).clamp(0.0, 1.0)
        return (weights >= self.eval_threshold).any().item()

    def active_layer_indices(self, state_dict: dict[str, Tensor]) -> set[int]:
        """Return the set of layer indices whose gates are active in *state_dict*."""
        active: set[int] = set()
        for k, v in state_dict.items():
            if "log_alpha" not in k:
                continue
            if self.is_active(v):
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                active.add(layer_idx)
        return active

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "stretch_limits": self.stretch_limits,
            "eps": self.eps,
            "eval_threshold": self.eval_threshold,
            "init_log_alpha": self.init_log_alpha,
            "init_log_scale": self.init_log_scale,
        }


__all__ = [
    "HardConcreteConfig",
]
