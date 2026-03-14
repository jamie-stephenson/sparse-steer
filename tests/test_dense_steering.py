import pytest
import torch
from torch import nn

from sparse_steer.models.dense import DenseSteeringAttention, DenseSteeringMLP


class DummyDenseSteeringMLP(DenseSteeringMLP, nn.Module):
    def __init__(self, steering_strength: float = 1.0) -> None:
        super().__init__(steering_strength=steering_strength)
        self.down_proj = nn.Linear(self._mlp_dim, self._mlp_dim, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(self._mlp_dim))
        self._register_steering_hook()

    @property
    def _mlp_dim(self) -> int:
        return 4

    def output_proj(self) -> nn.Module:
        return self.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


class DummyDenseSteeringAttention(DenseSteeringAttention, nn.Module):
    def __init__(self, steering_strength: float = 1.0) -> None:
        super().__init__(steering_strength=steering_strength)
        hidden = self._num_heads * self._head_dim
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(hidden))
        self._register_steering_hook()

    @property
    def _num_heads(self) -> int:
        return 2

    @property
    def _head_dim(self) -> int:
        return 3

    def output_proj(self) -> nn.Module:
        return self.o_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def test_dense_mlp_applies_steering_strength_correction() -> None:
    steering_strength = 3.0
    module = DummyDenseSteeringMLP(steering_strength=steering_strength)
    module.eval()
    module.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.zeros(2, 3, module._mlp_dim)
    y = module(x)

    expected = (steering_strength * module.steering_vectors).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected)


def test_dense_mlp_respects_steering_enabled_flag() -> None:
    module = DummyDenseSteeringMLP(steering_strength=5.0)
    module.eval()
    module.set_steering_vectors(torch.ones(module._mlp_dim))
    module.steering_enabled = False

    x = torch.randn(2, 3, module._mlp_dim)
    y = module(x)

    assert torch.allclose(y, x)


def test_dense_attention_applies_steering_strength_correction() -> None:
    steering_strength = 2.0
    module = DummyDenseSteeringAttention(steering_strength=steering_strength)
    module.eval()
    sv = torch.randn(module._num_heads, module._head_dim)
    module.set_steering_vectors(sv)

    hidden = module._num_heads * module._head_dim
    x = torch.zeros(2, 3, hidden)
    y = module(x)

    expected = (steering_strength * sv).reshape(1, 1, hidden).expand_as(x)
    assert torch.allclose(y, expected)


def test_dense_attention_respects_steering_enabled_flag() -> None:
    module = DummyDenseSteeringAttention(steering_strength=5.0)
    module.eval()
    module.set_steering_vectors(torch.randn(module._num_heads, module._head_dim))
    module.steering_enabled = False

    hidden = module._num_heads * module._head_dim
    x = torch.randn(2, 3, hidden)
    y = module(x)

    assert torch.allclose(y, x)


def test_set_steering_vectors_validates_shape() -> None:
    module = DummyDenseSteeringMLP(steering_strength=1.0)
    with pytest.raises(ValueError):
        module.set_steering_vectors(torch.zeros(5))
