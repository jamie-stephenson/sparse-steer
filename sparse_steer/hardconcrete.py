import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        """
        Return the set of layer indices whose gates are active in *state_dict*.
        This is important to avoid unnecessarily adapting attention layers that
        don't need steering.
        """
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


class HardConcreteGateMixin:
    temperature: float
    stretch_limits: list[float]
    eps: float
    eval_threshold: float
    init_log_alpha: float
    init_log_scale: float

    def __init__(
        self,
        *args,
        num_gates: int,
        gate_config: HardConcreteConfig,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._set_gate_config(gate_config)
        self.log_alpha = nn.Parameter(torch.full((num_gates,), self.init_log_alpha))
        self.log_scale = nn.Parameter(torch.full((num_gates,), self.init_log_scale))

    def _set_gate_config(self, cfg: HardConcreteConfig) -> None:
        self.temperature = cfg.temperature
        self.stretch_limits = cfg.stretch_limits
        self.eps = cfg.eps
        self.eval_threshold = cfg.eval_threshold
        self.init_log_alpha = cfg.init_log_alpha
        self.init_log_scale = cfg.init_log_scale

    def _hard_concrete(self) -> Tensor:
        low, high = self.stretch_limits
        if self.training:
            noise = torch.rand_like(self.log_alpha)

            # linearly scale to [eps, 1-eps] for stability
            noise = noise.mul(1.0 - 2.0 * self.eps).add(self.eps)
            concrete = torch.sigmoid(
                (noise.log() - torch.log1p(-noise) + self.log_alpha) / self.temperature
            )
        else:
            concrete = torch.sigmoid(self.log_alpha)
        stretched = concrete * (high - low) + low
        return stretched.clamp(0.0, 1.0)

    def _l0_penalty(self) -> Tensor:
        """Expected number of active gates (differentiable L0 surrogate)."""
        low, high = self.stretch_limits
        return torch.sigmoid(
            self.log_alpha - self.temperature * math.log(-low / high)
        ).sum()

    def _scaled_gate(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        weights = self._hard_concrete()
        if not self.training and self.eval_threshold > 0.0:
            keep_mask = weights >= self.eval_threshold
            weights = torch.where(keep_mask, weights, torch.zeros_like(weights))
        scales = F.softplus(self.log_scale)
        return (weights * scales).to(device=device, dtype=dtype)


__all__ = [
    "HardConcreteConfig",
    "HardConcreteGateMixin",
]
