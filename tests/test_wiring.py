"""Behavioural tests for the HF steering wiring (core/wiring.py) on a tiny GPT-Neo.

These replace the retired TL↔HF logit-parity suite: with a single engine the contract is no
longer "matches TransformerLens" but (a) every site's edit actually fires, (b) position
masks confine the edit, (c) disabled steering is exactly the clean model, (d) the capture
wiring records the true residual stream (chain identity), (e) gradients reach the gate
parameters through the wired forward, and (f) steering state survives a save/load roundtrip.
The refactor's numeric ground truth against the pre-refactor code is established separately
by scripts/validate_hf_refactor.py (bit-equal eval + bit-equal training on the 7B).
"""

import pytest
import torch

from sparse_steer.core.steering import HardConcreteConfig, SteeringModel

from .tiny import NUM_LAYERS, tiny_batch, tiny_model

COMPONENTS = ["attention", "attn_out", "mlp", "resid_pre", "resid_mid", "resid_post"]


def _logits(model: SteeringModel, ids, mask) -> torch.Tensor:
    with torch.no_grad():
        return model(ids, attention_mask=mask).logits


def _set_unit_vectors(model: SteeringModel, value: float = 3.0) -> None:
    vectors = {}
    for component in model.steering_components:
        shape = (model.cfg.n_layers,) + model._vector_shape(component)
        vectors[component] = torch.full(shape, value)
    model.set_all_vectors(vectors)


@pytest.mark.parametrize("component", COMPONENTS)
def test_each_site_fires(component):
    """A nonzero vector at one site type must change the logits."""
    model = tiny_model([component])
    ids, mask = tiny_batch()
    clean = _logits(model, ids, mask)  # vectors are zero at init → clean
    _set_unit_vectors(model)
    steered = _logits(model, ids, mask)
    assert not torch.allclose(clean, steered), f"{component} edit did not fire"


def test_disabled_is_clean():
    model = tiny_model(COMPONENTS)
    ids, mask = tiny_batch()
    clean = _logits(model, ids, mask)
    _set_unit_vectors(model)
    with model.steering_disabled():
        disabled = _logits(model, ids, mask)
    assert torch.equal(clean, disabled)


def test_zero_vectors_are_clean():
    """Zero vectors → sites are wired out (eval) and logits equal the plain engine's."""
    model = tiny_model(COMPONENTS)
    ids, mask = tiny_batch()
    with torch.no_grad():
        plain = model.engine(input_ids=ids, attention_mask=mask, use_cache=False).logits
    assert torch.equal(_logits(model, ids, mask), plain)


def test_position_mask_confines_edit():
    """Steering only position p must leave logits at positions < p unchanged (causality)."""
    model = tiny_model(["resid_pre"])
    ids, mask = tiny_batch(batch=1, seq=6)
    _set_unit_vectors(model)
    pos_mask = torch.zeros_like(mask, dtype=torch.bool)
    p = 3
    pos_mask[:, p] = True
    with model.steering_disabled():
        clean = _logits(model, ids, mask)
    with model.steer_positions(pos_mask):
        steered = _logits(model, ids, mask)
    assert torch.equal(clean[:, :p], steered[:, :p]), "edit leaked to earlier positions"
    assert not torch.allclose(clean[:, p:], steered[:, p:]), "edit missing from steered position"


def test_capture_shapes_and_residual_chain():
    """Capture returns the documented shapes AND the true residual stream:
    resid_post[l] == resid_pre[l+1] (the inter-block identity), and
    resid_mid == resid_pre + attn_out (the intra-block identity)."""
    model = tiny_model(COMPONENTS)
    ids, mask = tiny_batch(batch=2, seq=5)
    acts = model.capture_activations(ids, mask, COMPONENTS)
    d = model.cfg
    assert acts["resid_pre"].shape == (2, d.n_layers, 5, d.d_model)
    assert acts["attention"].shape == (2, d.n_layers, 5, d.n_heads, d.d_head)
    assert acts["mlp"].shape == (2, d.n_layers, 5, d.d_mlp)
    for layer in range(d.n_layers - 1):
        assert torch.allclose(
            acts["resid_post"][:, layer], acts["resid_pre"][:, layer + 1], atol=1e-5
        ), f"resid chain broken between layers {layer} and {layer + 1}"
    assert torch.allclose(
        acts["resid_mid"], acts["resid_pre"] + acts["attn_out"], atol=1e-5
    ), "resid_mid must equal resid_pre + attn_out"


def test_gradients_reach_gates_when_training():
    """Train mode: the loss must backprop through the wired edit into log_alpha/raw_scale —
    including at sites whose eval gate would be pruned (the train-mode wiring rule)."""
    model = tiny_model(
        ["resid_mid", "attention"], gate_config=HardConcreteConfig(init_log_alpha=-0.79)
    )
    _set_unit_vectors(model)
    model.train()
    model._wiring.rewire()  # wire with training-mode site rule
    model.freeze_base_model()
    ids, mask = tiny_batch()
    # Accumulate over several noise draws: a single hard-concrete sample often saturates at
    # the clamp (temperature 0.33 → near-binary gates), zeroing that draw's gate gradient.
    torch.manual_seed(0)
    for _ in range(8):
        out = model(ids, attention_mask=mask).logits
        out.float().pow(2).mean().backward()
    for _, _, hook in model.iter_hooks():
        assert hook.log_alpha.grad is not None and hook.log_alpha.grad.abs().sum() > 0
        assert hook.raw_scale.grad is not None and hook.raw_scale.grad.abs().sum() > 0


def test_gates_stochastic_in_train_deterministic_in_eval():
    """Hard-concrete gates must sample noise under train() and be a fixed
    sigmoid(log_alpha) under eval() — the train/eval contract of the gate math."""
    model = tiny_model(["resid_pre"], gate_config=HardConcreteConfig(init_log_alpha=0.0))
    hook = next(h for _, _, h in model.iter_hooks())
    model.train()
    torch.manual_seed(0)
    draws = {float(hook._gate_weights().sum()) for _ in range(8)}
    assert len(draws) > 1, "train-mode gates did not sample noise"
    model.eval()
    evals = {float(hook._gate_weights().sum()) for _ in range(8)}
    assert len(evals) == 1, "eval-mode gates are not deterministic"


def test_save_load_roundtrip(tmp_path):
    model = tiny_model(COMPONENTS, gate_config=HardConcreteConfig(init_log_alpha=0.3))
    _set_unit_vectors(model, value=1.7)
    ids, mask = tiny_batch()
    before = _logits(model, ids, mask)
    path = model.save_steering(tmp_path)
    fresh = tiny_model(COMPONENTS, gate_config=HardConcreteConfig(init_log_alpha=0.0))
    fresh.load_steering(path)
    after = _logits(fresh, ids, mask)
    assert torch.equal(before, after)


def test_generate_modes_run():
    """The KV-cached decode loop runs under every steering mode and matches greedy
    no-cache decoding when steering is off."""
    from sparse_steer.core.generate import generate

    model = tiny_model(["resid_pre"])
    _set_unit_vectors(model, value=0.5)
    ids, mask = tiny_batch(batch=2, seq=4)
    for mode in ("all", "prompt", "prompt_final", "completion", "answer_gen", "off"):
        toks, valid = generate(model, ids, mask, max_new_tokens=3, steer=mode)
        assert toks.shape == (2, 3) and valid.shape == (2, 3)
    # KV-cache correctness: "off" generation == step-by-step full-context argmax
    toks, _ = generate(model, ids, mask, max_new_tokens=3, steer="off")
    seq = ids.clone()
    for t in range(3):
        with torch.no_grad(), model.steering_disabled():
            nxt = model(seq, attention_mask=torch.ones_like(seq)).logits[:, -1].argmax(-1)
        assert torch.equal(nxt, toks[:, t])
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
