"""Pure selection logic for Arditi single-direction selection (core/select._filter_and_pick)."""

import pytest

from sparse_steer.tasks.jailbreak.refine import _filter_and_pick


def test_filter_and_pick_applies_all_filters_and_picks_min_bypass():
    # (pos, layer, bypass, induce, kl); n_layers=10, prune 0.2 ⇒ drop layer >= 8
    scored = [
        (0, 0, -1.0, 1.0, 0.05),   # survivor (bypass -1)
        (0, 1, -3.0, 1.0, 0.05),   # survivor (bypass -3 → lowest, should win)
        (0, 2, -5.0, 1.0, 0.50),   # kl too high → dropped
        (0, 3, -4.0, -1.0, 0.05),  # induce below threshold → dropped
        (0, 9, -9.0, 5.0, 0.00),   # pruned layer → dropped
    ]
    best = _filter_and_pick(scored, 10, kl_threshold=0.1, induce_threshold=0.0, prune_layer_frac=0.2)
    assert (best["position"], best["layer"], best["bypass"]) == (0, 1, -3.0)
    assert best["n_candidates"] == 5 and best["n_survivors"] == 2


def test_filter_and_pick_drops_nan_candidates():
    scored = [(0, 0, float("nan"), 1.0, 0.0), (0, 1, -2.0, 1.0, 0.0)]
    best = _filter_and_pick(scored, 10, kl_threshold=0.1, induce_threshold=0.0, prune_layer_frac=0.2)
    assert best["layer"] == 1 and best["n_survivors"] == 1


def test_filter_and_pick_raises_when_none_survive():
    scored = [(0, 0, -1.0, 1.0, 0.5)]  # kl over budget
    with pytest.raises(RuntimeError, match="no candidate survived"):
        _filter_and_pick(scored, 10, kl_threshold=0.1, induce_threshold=0.0, prune_layer_frac=0.2)
