import pytest
import torch
from torch import nn

# Import order matters due current package init side effects.
from sparse_steer.extract import ActivationTarget  # noqa: F401
from sparse_steer.models.sparse import SparseSteeringAttention, SparseSteeringLM, SparseSteeringMLP
from sparse_steer.hardconcrete import HardConcreteConfig


class DummyAttention(SparseSteeringAttention, nn.Module):
    def __init__(self, gate_config: HardConcreteConfig) -> None:
        super().__init__(num_gates=self._num_heads, gate_config=gate_config)
        self.o_proj = nn.Linear(self._num_heads * self._head_dim, self._num_heads * self._head_dim)
        self._register_steering_hook()

    @property
    def _num_heads(self) -> int:
        return 2

    @property
    def _head_dim(self) -> int:
        return 3

    def output_proj(self) -> nn.Module:
        return self.o_proj


class DummyMLP(SparseSteeringMLP, nn.Module):
    def __init__(self, gate_config: HardConcreteConfig) -> None:
        super().__init__(num_gates=self._mlp_dim, gate_config=gate_config)
        self.down_proj = nn.Linear(self._mlp_dim, self._mlp_dim)
        self._register_steering_hook()

    @property
    def _mlp_dim(self) -> int:
        return 4

    def output_proj(self) -> nn.Module:
        return self.down_proj


class DummyLayer(nn.Module):
    def __init__(self, gate_config: HardConcreteConfig) -> None:
        super().__init__()
        self.self_attn = DummyAttention(gate_config)
        self.mlp = DummyMLP(gate_config)


class DummySteeringLM(SparseSteeringLM, nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        gate_config = HardConcreteConfig()
        self.layers = nn.ModuleList([DummyLayer(gate_config) for _ in range(num_layers)])
        self.steering_components = ["attention", "mlp"]

    def get_layers(self) -> nn.ModuleList:
        return self.layers

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp


def test_set_all_vectors_raises_on_missing_expected_components() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.zeros(2, 2, 3),
    }
    with pytest.raises(ValueError, match="missing expected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_raises_on_unexpected_components() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.zeros(2, 2, 3),
        "mlp": torch.zeros(2, 4),
        "residual": torch.zeros(2, 4),
    }
    with pytest.raises(ValueError, match="unexpected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_accepts_exact_component_set() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.randn(2, 2, 3),
        "mlp": torch.randn(2, 4),
    }
    model.set_all_vectors(vectors)

    for i, layer in enumerate(model.get_layers()):
        assert torch.allclose(model.get_attention(layer).steering_vectors, vectors["attention"][i])
        assert torch.allclose(model.get_mlp(layer).steering_vectors, vectors["mlp"][i])
