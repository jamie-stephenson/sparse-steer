import pytest
import torch

from sparse_steer.core.steering import HardConcreteConfig, SteeringHook


def test_set_steering_vectors_validates_shape() -> None:
    hook = SteeringHook((4,), gate_config=HardConcreteConfig(), learn_scale=True)
    with pytest.raises(ValueError):
        hook.set_steering_vectors(torch.zeros(5))


def test_gated_hook_applies_gated_correction() -> None:
    cfg = HardConcreteConfig(init_log_alpha=5.0, eval_threshold=0.0)
    hook = SteeringHook((4,), gate_config=cfg, learn_scale=True, init_raw_scale=0.0)
    hook.set_steering_vectors(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    hook.eval()

    x = torch.zeros(2, 3, 4)
    y = hook.steer(x)

    ew = hook.effective_weight(dtype=x.dtype, device=x.device)
    expected = (hook.steering_vectors * ew).reshape(1, 1, -1).expand_as(x)
    assert torch.allclose(y, expected)


def test_gated_hook_respects_enabled_flag() -> None:
    cfg = HardConcreteConfig(init_log_alpha=5.0, eval_threshold=0.0)
    hook = SteeringHook((4,), gate_config=cfg, learn_scale=True)
    hook.set_steering_vectors(torch.ones(4))
    hook.enabled = False
    hook.eval()

    x = torch.randn(2, 3, 4)
    assert torch.allclose(hook.steer(x), x)


def test_l0_penalty_zero_without_gates() -> None:
    hook = SteeringHook((4,), learn_scale=True)
    assert hook.l0_penalty().item() == 0.0
