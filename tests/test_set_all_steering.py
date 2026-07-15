import pytest
import torch

from sparse_steer.core.steering import HardConcreteConfig, SteeringHook, SteeringModel

from .tiny import D_MODEL, HEAD_DIM, MLP_DIM, NUM_HEADS, NUM_LAYERS, tiny_engine


def _tiny_model(components) -> SteeringModel:
    return SteeringModel(
        tiny_engine(),
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


def test_residual_alias_removed() -> None:
    # The 'residual' shorthand (and its mutual-exclusivity rule) was dropped in commit 82671f7
    # in favour of the explicit resid_pre / resid_mid / resid_post taps; it is no longer a valid
    # component, so it now fails the component check.
    with pytest.raises(ValueError, match="Unknown component"):
        _tiny_model(["residual"])
