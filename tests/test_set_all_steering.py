import pytest
import torch
from torch import nn

from sparse_steer.models.hook import Component, SteeringHook
from sparse_steer.models.steering import SteeringLM
from sparse_steer.models.base import BaseModelLayout
from sparse_steer.hardconcrete import HardConcreteConfig


NUM_HEADS = 2
HEAD_DIM = 3
MLP_DIM = 4


class DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = NUM_HEADS * HEAD_DIM
        self.o_proj = nn.Linear(hidden, hidden)
        self.head_dim = HEAD_DIM
        self.config = type("C", (), {"num_attention_heads": NUM_HEADS})()


class DummyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(MLP_DIM, MLP_DIM)
        self.config = type("C", (), {"intermediate_size": MLP_DIM})()


class DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = DummyMLP()


class DummySteeringLM(SteeringLM, nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer() for _ in range(num_layers)])
        self.upgrade_for_steering(
            gate_config=HardConcreteConfig(),
            learn_scale=True,
            steering_layer_ids=list(range(num_layers)),
            steering_components=["attention", "mlp"],
        )

    def get_layers(self) -> nn.ModuleList:
        return self.layers

    def get_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def get_mlp(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    def _get_output_proj(self, module: nn.Module) -> nn.Module:
        if hasattr(module, "o_proj"):
            return module.o_proj
        return module.down_proj

    def _get_vector_shape(
        self, component: Component, layer: nn.Module
    ) -> tuple[int, ...]:
        if component == "attention":
            return (NUM_HEADS, HEAD_DIM)
        elif component == "mlp":
            return (MLP_DIM,)
        raise ValueError(f"Unknown component: {component!r}")


def test_set_all_vectors_raises_on_missing_expected_components() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.zeros(2, NUM_HEADS, HEAD_DIM),
    }
    with pytest.raises(ValueError, match="missing expected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_raises_on_unexpected_components() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.zeros(2, NUM_HEADS, HEAD_DIM),
        "mlp": torch.zeros(2, MLP_DIM),
        "residual": torch.zeros(2, MLP_DIM),
    }
    with pytest.raises(ValueError, match="unexpected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_accepts_exact_component_set() -> None:
    model = DummySteeringLM()
    vectors = {
        "attention": torch.randn(2, NUM_HEADS, HEAD_DIM),
        "mlp": torch.randn(2, MLP_DIM),
    }
    model.set_all_vectors(vectors)

    for i, layer in enumerate(model.get_layers()):
        attn_hook = getattr(layer, f"_steering_attention_{i}", None)
        mlp_hook = getattr(layer, f"_steering_mlp_{i}", None)
        assert isinstance(attn_hook, SteeringHook)
        assert isinstance(mlp_hook, SteeringHook)
        assert torch.allclose(attn_hook.steering_vectors, vectors["attention"][i])
        assert torch.allclose(mlp_hook.steering_vectors, vectors["mlp"][i])
