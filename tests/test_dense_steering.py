import pytest
import torch
from torch import nn

from sparse_steer.models.dense import DenseSteeringHook


def _make_identity_linear(dim: int) -> nn.Linear:
    proj = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(dim))
    return proj


class DummyMLP(nn.Module):
    """Minimal MLP with an identity down_proj for testing hook effects."""
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.down_proj = _make_identity_linear(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


class DummyAttention(nn.Module):
    """Minimal attention with an identity o_proj for testing hook effects."""
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.o_proj = _make_identity_linear(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def test_dense_mlp_applies_steering_strength_correction() -> None:
    strength = 3.0
    mlp = DummyMLP(dim=4)
    hook = DenseSteeringHook((4,), steering_strength=strength)
    hook.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    mlp.down_proj.register_forward_pre_hook(hook.pre_hook)
    mlp.eval()

    x = torch.zeros(2, 3, 4)
    y = mlp(x)

    expected = (strength * hook.steering_vectors).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected)


def test_dense_mlp_respects_steering_enabled_flag() -> None:
    mlp = DummyMLP(dim=4)
    hook = DenseSteeringHook((4,), steering_strength=5.0)
    hook.set_steering_vectors(torch.ones(4))
    hook.steering_enabled = False
    mlp.down_proj.register_forward_pre_hook(hook.pre_hook)
    mlp.eval()

    x = torch.randn(2, 3, 4)
    y = mlp(x)
    assert torch.allclose(y, x)


def test_dense_attention_applies_steering_strength_correction() -> None:
    num_heads, head_dim = 2, 3
    hidden = num_heads * head_dim
    strength = 2.0

    attn = DummyAttention(hidden=hidden)
    hook = DenseSteeringHook((num_heads, head_dim), steering_strength=strength)
    sv = torch.randn(num_heads, head_dim)
    hook.set_steering_vectors(sv)
    attn.o_proj.register_forward_pre_hook(hook.pre_hook)
    attn.eval()

    x = torch.zeros(2, 3, hidden)
    y = attn(x)

    expected = (strength * sv).reshape(1, 1, hidden).expand_as(x)
    assert torch.allclose(y, expected)


def test_dense_attention_respects_steering_enabled_flag() -> None:
    num_heads, head_dim = 2, 3
    hidden = num_heads * head_dim

    attn = DummyAttention(hidden=hidden)
    hook = DenseSteeringHook((num_heads, head_dim), steering_strength=5.0)
    hook.set_steering_vectors(torch.randn(num_heads, head_dim))
    hook.steering_enabled = False
    attn.o_proj.register_forward_pre_hook(hook.pre_hook)
    attn.eval()

    x = torch.randn(2, 3, hidden)
    y = attn(x)
    assert torch.allclose(y, x)


def test_set_steering_vectors_validates_shape() -> None:
    hook = DenseSteeringHook((4,))
    with pytest.raises(ValueError):
        hook.set_steering_vectors(torch.zeros(5))
