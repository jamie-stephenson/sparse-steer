"""direction_source resolution: self (per-site) vs pinned broadcast (one direction everywhere)."""

import pytest
import torch
from omegaconf import OmegaConf

from sparse_steer.experiment.sourcing import (
    _parse_source,
    broadcast,
    extraction_targets,
    resolve_direction_source,
)

N_LAYERS = 4
D_MODEL = 6
TARGETS = ["resid_pre", "resid_mid", "resid_post"]


def _extracted():
    # distinct per-site directions so we can tell which layer/component a result came from
    return {c: torch.randn(N_LAYERS, D_MODEL) for c in TARGETS}


def test_parse_source_recognises_the_three_kinds():
    assert _parse_source("self") == ("self", None, None)
    assert _parse_source("grid_select") == ("grid", None, None)
    assert _parse_source(["resid_pre", 17]) == ("pin", "resid_pre", 17)


def test_parse_source_rejects_malformed_pin():
    with pytest.raises(ValueError, match="component, layer"):
        _parse_source(["resid_pre", 17, 1])


def test_broadcast_replicates_one_direction_to_every_site():
    vec = torch.randn(D_MODEL)
    out = broadcast(vec, TARGETS, N_LAYERS)
    assert set(out) == set(TARGETS)
    for c in TARGETS:
        assert out[c].shape == (N_LAYERS, D_MODEL)
        for layer in range(N_LAYERS):
            assert torch.equal(out[c][layer], vec)


def test_self_returns_per_site_vectors_unchanged():
    cfg = OmegaConf.create({"targets": TARGETS, "direction_source": "self"})
    extracted = _extracted()
    out = resolve_direction_source(cfg, extracted)
    assert set(out) == set(TARGETS)
    for c in TARGETS:
        assert torch.equal(out[c], extracted[c])  # each site keeps its own direction


def test_pin_broadcasts_the_one_source_direction_everywhere():
    cfg = OmegaConf.create({"targets": TARGETS, "direction_source": ["resid_pre", 2]})
    extracted = _extracted()
    source = extracted["resid_pre"][2]
    out = resolve_direction_source(cfg, extracted)
    # every component, every layer is the single (resid_pre, layer 2) direction
    for c in TARGETS:
        for layer in range(N_LAYERS):
            assert torch.equal(out[c][layer], source)


def test_extraction_targets_adds_a_pinned_source_component():
    # source component not among the steering targets ⇒ must be added so it can be read
    cfg = OmegaConf.create({"targets": ["resid_post"], "direction_source": ["resid_pre", 1]})
    assert set(extraction_targets(cfg)) == {"resid_post", "resid_pre"}
    # self ⇒ extract exactly the steering targets
    cfg_self = OmegaConf.create({"targets": ["resid_post"], "direction_source": "self"})
    assert extraction_targets(cfg_self) == ["resid_post"]
