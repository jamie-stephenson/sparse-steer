from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class HardConcreteConfig:
    temperature: float = 0.33
    stretch_limits: tuple[float, float] = (-0.1, 1.1)
    eps: float = 1e-6
    eval_threshold: float = 1e-2
    init_log_alpha: float = 0.0
    init_log_scale: float = 0.0

    @classmethod
    def from_dict(cls, values: Optional[dict[str, Any]]) -> "HardConcreteConfig":
        if not values:
            return cls()
        merged = asdict(cls())
        merged.update(values)
        stretch = merged["stretch_limits"]
        if not isinstance(stretch, (tuple, list)) or len(stretch) != 2:
            raise ValueError("`stretch_limits` must be a 2-item tuple/list.")
        return cls(
            temperature=float(merged["temperature"]),
            stretch_limits=(float(stretch[0]), float(stretch[1])),
            eps=float(merged["eps"]),
            eval_threshold=float(merged["eval_threshold"]),
            init_log_alpha=float(merged["init_log_alpha"]),
            init_log_scale=float(merged["init_log_scale"]),
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["stretch_limits"] = list(self.stretch_limits)
        return values


class HardConcreteGateMixin:
    temperature: float
    stretch_limits: tuple[float, float]
    eps: float
    eval_threshold: float
    init_log_alpha: float
    init_log_scale: float

    def _set_gate_config(self, gate_config: Optional[dict[str, Any]]) -> None:
        cfg = HardConcreteConfig.from_dict(gate_config)
        self.temperature = cfg.temperature
        self.stretch_limits = cfg.stretch_limits
        self.eps = cfg.eps
        self.eval_threshold = cfg.eval_threshold
        self.init_log_alpha = cfg.init_log_alpha
        self.init_log_scale = cfg.init_log_scale

    def _reset_gate_parameters(
        self, log_alpha: nn.Parameter, log_scale: nn.Parameter
    ) -> None:
        with torch.no_grad():
            log_alpha.fill_(self.init_log_alpha)
            log_scale.fill_(self.init_log_scale)

    def _hard_concrete(self, log_alpha: Tensor) -> Tensor:
        low, high = self.stretch_limits
        if self.training:
            noise = torch.rand_like(log_alpha)
            noise = noise.mul(1.0 - 2.0 * self.eps).add(self.eps)
            concrete = torch.sigmoid(
                (noise.log() - torch.log1p(-noise) + log_alpha) / self.temperature
            )
        else:
            concrete = torch.sigmoid(log_alpha)
        stretched = concrete * (high - low) + low
        return stretched.clamp(0.0, 1.0)

    def _hard_concrete_expected(self, log_alpha: Tensor) -> Tensor:
        low, high = self.stretch_limits
        concrete = torch.sigmoid(log_alpha)
        stretched = concrete * (high - low) + low
        return stretched.clamp(0.0, 1.0)

    def _scaled_gate(
        self,
        log_alpha: Tensor,
        log_scale: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        weights = self._hard_concrete(log_alpha)
        if not self.training and self.eval_threshold > 0.0:
            keep_mask = weights >= self.eval_threshold
            weights = torch.where(keep_mask, weights, torch.zeros_like(weights))
        scales = F.softplus(log_scale)
        return (weights * scales).to(device=device, dtype=dtype)

    def expected_scaled_gate(
        self,
        log_alpha: Tensor,
        log_scale: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        weights = self._hard_concrete_expected(log_alpha)
        if self.eval_threshold > 0.0:
            keep_mask = weights >= self.eval_threshold
            weights = torch.where(keep_mask, weights, torch.zeros_like(weights))
        scales = F.softplus(log_scale)
        return (weights * scales).to(device=device, dtype=dtype)


__all__ = [
    "HardConcreteConfig",
    "HardConcreteGateMixin",
]
