import math

import pytest
import torch
import torch.nn.functional as F

from sparse_steer.core.steering import SteeringHook


def _inv_softplus(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.exp(x) - 1)


def test_dense_mlp_applies_scale_correction() -> None:
    desired_scale = 3.0
    hook = SteeringHook((4,), init_raw_scale=_inv_softplus(desired_scale))
    hook.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    hook.eval()

    x = torch.zeros(2, 3, 4)  # (batch, pos, d_mlp)
    y = hook.steer(x)

    scale = F.softplus(hook.raw_scale).item()
    expected = (scale * hook.steering_vectors).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected, atol=1e-5)


def test_dense_mlp_respects_enabled_flag() -> None:
    hook = SteeringHook((4,), init_raw_scale=_inv_softplus(5.0))
    hook.set_steering_vectors(torch.ones(4))
    hook.enabled = False
    hook.eval()

    x = torch.randn(2, 3, 4)
    assert torch.allclose(hook.steer(x), x)


def test_dense_attention_applies_per_head_scale_correction() -> None:
    num_heads, head_dim = 2, 3
    desired_scale = 2.0

    hook = SteeringHook(
        (num_heads, head_dim), init_raw_scale=_inv_softplus(desired_scale)
    )
    sv = torch.randn(num_heads, head_dim)
    hook.set_steering_vectors(sv)
    hook.eval()

    x = torch.zeros(2, 3, num_heads, head_dim)  # (batch, pos, heads, head_dim) = hook_z
    y = hook.steer(x)

    scale = F.softplus(hook.raw_scale).unsqueeze(-1)  # (heads, 1)
    expected = (scale * sv).reshape(1, 1, num_heads, head_dim).expand_as(x)
    assert torch.allclose(y, expected, atol=1e-5)


def test_dense_attention_respects_enabled_flag() -> None:
    num_heads, head_dim = 2, 3
    hook = SteeringHook((num_heads, head_dim), init_raw_scale=_inv_softplus(5.0))
    hook.set_steering_vectors(torch.randn(num_heads, head_dim))
    hook.enabled = False
    hook.eval()

    x = torch.randn(2, 3, num_heads, head_dim)
    assert torch.allclose(hook.steer(x), x)


def test_set_steering_vectors_validates_shape() -> None:
    hook = SteeringHook((4,))
    with pytest.raises(ValueError):
        hook.set_steering_vectors(torch.zeros(5))
