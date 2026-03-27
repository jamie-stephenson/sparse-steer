import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..hardconcrete import HardConcreteConfig

Component = Literal["attention", "mlp", "residual"]


class SteeringHook(nn.Module):
    """Configurable steering hook with optional learned gates and scale.

    Correction formula: ``vectors * gate * scale``

    - **gate**: ``1.0`` (no gates) or ``hard_concrete(log_alpha)`` (learned)
    - **scale**: ``softplus(log_scale)`` — learned parameter, shared parameter, or fixed buffer
    """

    def __init__(
        self,
        vector_shape: tuple[int, ...],
        *,
        gate_config: HardConcreteConfig | None = None,
        learn_scale: bool = False,
        init_log_scale: float = 0.0,
        shared_log_scale: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.steering_enabled = True
        self.register_buffer("steering_vectors", torch.zeros(vector_shape))
        num_gates = vector_shape[0] if len(vector_shape) > 1 else 1
        self._shared_log_scale = shared_log_scale

        # Gate axis: optional HardConcrete gates
        self.gate_config = gate_config
        if gate_config is not None:
            self.log_alpha = nn.Parameter(
                torch.full((num_gates,), gate_config.init_log_alpha)
            )
        else:
            self.register_parameter("log_alpha", None)

        # Scale axis: shared parameter, per-head parameter, or fixed buffer
        if shared_log_scale is not None:
            self.register_parameter("log_scale", None)
        elif learn_scale:
            self.log_scale = nn.Parameter(
                torch.full((num_gates,), init_log_scale)
            )
        else:
            self.register_buffer(
                "log_scale", torch.full((num_gates,), init_log_scale)
            )

        self._gates_frozen = False

    # ── Gate computation ─────────────────────────────────────────────

    def freeze_gates(self) -> None:
        """Lock gates to deterministic eval-mode values."""
        if self.log_alpha is not None:
            self.log_alpha.requires_grad = False
        self._gates_frozen = True

    def _hard_concrete(self) -> Tensor:
        cfg = self.gate_config
        low, high = cfg.stretch_limits
        if self.training and not self._gates_frozen:
            noise = torch.rand_like(self.log_alpha)
            noise = noise.mul(1.0 - 2.0 * cfg.eps).add(cfg.eps)
            concrete = torch.sigmoid(
                (noise.log() - torch.log1p(-noise) + self.log_alpha)
                / cfg.temperature
            )
        else:
            concrete = torch.sigmoid(self.log_alpha)
        stretched = concrete * (high - low) + low
        return stretched.clamp(0.0, 1.0)

    def _l0_penalty(self) -> Tensor:
        """Expected number of active gates (differentiable L0 surrogate).

        Returns zero when gates are disabled.
        """
        if self.log_alpha is None:
            return torch.tensor(0.0, device=self.steering_vectors.device)
        cfg = self.gate_config
        low, high = cfg.stretch_limits
        return torch.sigmoid(
            self.log_alpha - cfg.temperature * math.log(-low / high)
        ).sum()

    def _gate_weights(self) -> Tensor | float:
        if self.log_alpha is None:
            return 1.0
        weights = self._hard_concrete()
        if not self.training and self.gate_config.eval_threshold > 0.0:
            weights = torch.where(
                weights >= self.gate_config.eval_threshold,
                weights,
                torch.zeros_like(weights),
            )
        return weights

    def _scale_weights(self) -> Tensor:
        if self._shared_log_scale is not None:
            return F.softplus(self._shared_log_scale)
        return F.softplus(self.log_scale)

    # ── Correction ───────────────────────────────────────────────────

    def _compute_correction(self, hidden: Tensor) -> Tensor:
        steering = self.steering_vectors.to(device=hidden.device, dtype=hidden.dtype)
        gate = self._gate_weights()
        scale = self._scale_weights()

        weight = gate * scale
        if isinstance(weight, Tensor) and steering.ndim > 1:
            weight = weight.unsqueeze(-1)

        correction = steering * weight
        return correction.reshape([1] * (hidden.ndim - 1) + [-1])

    def effective_weight(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Combined gate * scale weights as a tensor (for visualization)."""
        gate = self._gate_weights()
        scale = self._scale_weights()
        weight = gate * scale
        if isinstance(weight, (int, float)):
            return torch.tensor(weight, dtype=dtype, device=device)
        return weight.to(dtype=dtype, device=device)

    # ── Hooks ────────────────────────────────────────────────────────

    def pre_hook(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...] | None:
        """Forward pre-hook for attention/MLP output projections."""
        if not self.steering_enabled or not inputs:
            return None
        hidden = inputs[0]
        correction = self._compute_correction(hidden)
        return (hidden + correction, *inputs[1:])

    def post_hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        """Forward hook for residual stream (layer output)."""
        if not self.steering_enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        corrected = hidden + self._compute_correction(hidden)
        if isinstance(output, tuple):
            return (corrected, *output[1:])
        return corrected

    def set_steering_vectors(self, vectors: Tensor) -> None:
        expected = self.steering_vectors.shape
        if vectors.shape != expected:
            raise ValueError(
                f"Steering vectors must have shape {tuple(expected)}, "
                f"got {tuple(vectors.shape)}."
            )
        self.steering_vectors.copy_(
            vectors.to(
                device=self.steering_vectors.device,
                dtype=self.steering_vectors.dtype,
            )
        )


__all__ = [
    "Component",
    "SteeringHook",
]
