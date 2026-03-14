import pytest
import torch
from torch import nn

# Import order matters due current package init side effects.
from sparse_steer.extract import ActivationTarget  # noqa: F401
from sparse_steer.models.sparse import SparseSteeringMLP
from sparse_steer.hardconcrete import HardConcreteConfig


class DummySparseSteeringMLP(SparseSteeringMLP, nn.Module):
    def __init__(self, gate_config: HardConcreteConfig) -> None:
        super().__init__(num_gates=self._mlp_dim, gate_config=gate_config)
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


def test_set_steering_vectors_validates_shape() -> None:
    module = DummySparseSteeringMLP(HardConcreteConfig())
    with pytest.raises(ValueError):
        module.set_steering_vectors(torch.zeros(5))


def test_mlp_hook_applies_gated_correction() -> None:
    module = DummySparseSteeringMLP(
        HardConcreteConfig(init_log_alpha=5.0, init_log_scale=0.0, eval_threshold=0.0)
    )
    module.eval()
    module.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.zeros(2, 3, module._mlp_dim)
    y = module(x)

    gate = module._scaled_gate(dtype=x.dtype, device=x.device)
    expected = (module.steering_vectors * gate).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected)


def test_mlp_hook_respects_steering_enabled_flag() -> None:
    module = DummySparseSteeringMLP(HardConcreteConfig(init_log_alpha=5.0, eval_threshold=0.0))
    module.eval()
    module.set_steering_vectors(torch.ones(module._mlp_dim))
    module.steering_enabled = False

    x = torch.randn(2, 3, module._mlp_dim)
    y = module(x)

    assert torch.allclose(y, x)
