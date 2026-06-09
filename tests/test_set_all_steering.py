import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from sparse_steer.core.steering import HardConcreteConfig, SteeringHook, SteeringModel

NUM_HEADS = 2
HEAD_DIM = 4
D_MODEL = NUM_HEADS * HEAD_DIM
MLP_DIM = 16
NUM_LAYERS = 2


def _tiny_model(components) -> SteeringModel:
    cfg = HookedTransformerConfig(
        n_layers=NUM_LAYERS,
        d_model=D_MODEL,
        n_ctx=16,
        d_head=HEAD_DIM,
        n_heads=NUM_HEADS,
        d_mlp=MLP_DIM,
        d_vocab=32,
        act_fn="gelu",
        normalization_type="LN",
    )
    tl = HookedTransformer(cfg)
    return SteeringModel(
        tl,
        steering_layer_ids=list(range(NUM_LAYERS)),
        steering_components=components,
        gate_config=HardConcreteConfig(),
        learn_scale=True,
    )


def test_set_all_vectors_raises_on_missing_expected_components() -> None:
    model = _tiny_model(["attention", "mlp"])
    vectors = {"attention": torch.zeros(NUM_LAYERS, NUM_HEADS, HEAD_DIM)}
    with pytest.raises(ValueError, match="missing expected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_raises_on_unexpected_components() -> None:
    model = _tiny_model(["attention", "mlp"])
    vectors = {
        "attention": torch.zeros(NUM_LAYERS, NUM_HEADS, HEAD_DIM),
        "mlp": torch.zeros(NUM_LAYERS, MLP_DIM),
        "residual": torch.zeros(NUM_LAYERS, D_MODEL),
    }
    with pytest.raises(ValueError, match="unexpected components"):
        model.set_all_vectors(vectors)


def test_set_all_vectors_accepts_exact_component_set() -> None:
    model = _tiny_model(["attention", "mlp"])
    vectors = {
        "attention": torch.randn(NUM_LAYERS, NUM_HEADS, HEAD_DIM),
        "mlp": torch.randn(NUM_LAYERS, MLP_DIM),
    }
    model.set_all_vectors(vectors)

    for i in range(NUM_LAYERS):
        attn_hook = model.hooks[f"attention_{i}"]
        mlp_hook = model.hooks[f"mlp_{i}"]
        assert isinstance(attn_hook, SteeringHook)
        assert isinstance(mlp_hook, SteeringHook)
        assert torch.allclose(
            attn_hook.steering_vectors, vectors["attention"][i].to(attn_hook.steering_vectors.dtype)
        )
        assert torch.allclose(
            mlp_hook.steering_vectors, vectors["mlp"][i].to(mlp_hook.steering_vectors.dtype)
        )


def test_residual_excludes_other_components() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _tiny_model(["residual", "attention"])
