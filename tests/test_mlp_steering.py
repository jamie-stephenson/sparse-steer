import pytest
import torch
from torch import nn

from sparse_steer.models.sparse import SparseSteeringHook
from sparse_steer.hardconcrete import HardConcreteConfig


class DummyMLP(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.down_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


def test_set_steering_vectors_validates_shape() -> None:
    hook = SparseSteeringHook((4,), gate_config=HardConcreteConfig())
    with pytest.raises(ValueError):
        hook.set_steering_vectors(torch.zeros(5))


def test_mlp_hook_applies_gated_correction() -> None:
    cfg = HardConcreteConfig(init_log_alpha=5.0, init_log_scale=0.0, eval_threshold=0.0)
    mlp = DummyMLP(dim=4)
    hook = SparseSteeringHook((4,), gate_config=cfg)
    hook.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    mlp.down_proj.register_forward_pre_hook(hook.pre_hook)
    hook.eval()
    mlp.eval()

    x = torch.zeros(2, 3, 4)
    y = mlp(x)

    gate = hook._scaled_gate(dtype=x.dtype, device=x.device)
    expected = (hook.steering_vectors * gate).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected)


def test_mlp_hook_respects_steering_enabled_flag() -> None:
    cfg = HardConcreteConfig(init_log_alpha=5.0, eval_threshold=0.0)
    mlp = DummyMLP(dim=4)
    hook = SparseSteeringHook((4,), gate_config=cfg)
    hook.set_steering_vectors(torch.ones(4))
    hook.steering_enabled = False
    mlp.down_proj.register_forward_pre_hook(hook.pre_hook)
    mlp.eval()

    x = torch.randn(2, 3, 4)
    y = mlp(x)
    assert torch.allclose(y, x)
