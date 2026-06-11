"""Pure selection logic for grid_select direction picking (experiment.sourcing.filter_and_pick)."""

import pytest

from sparse_steer.experiment.sourcing import filter_and_pick

CONSTRAINTS = [("kl", "<=", 0.1), ("induce", ">=", 0.0)]


def test_filter_and_pick_applies_all_filters_and_picks_min_objective():
    # (pos, layer, objective(=bypass), values); n_layers=10, prune 0.2 ⇒ drop layer >= 8
    scored = [
        (0, 0, -1.0, {"induce": 1.0, "kl": 0.05}),   # survivor (objective -1)
        (0, 1, -3.0, {"induce": 1.0, "kl": 0.05}),   # survivor (objective -3 → lowest, wins)
        (0, 2, -5.0, {"induce": 1.0, "kl": 0.50}),   # kl too high → dropped
        (0, 3, -4.0, {"induce": -1.0, "kl": 0.05}),  # induce below threshold → dropped
        (0, 9, -9.0, {"induce": 5.0, "kl": 0.00}),   # pruned layer → dropped
    ]
    best = filter_and_pick(scored, 10, constraints=CONSTRAINTS, prune_layer_frac=0.2)
    assert (best["position"], best["layer"], best["objective"]) == (0, 1, -3.0)
    assert best["induce"] == 1.0 and best["kl"] == 0.05  # values are merged into the result
    assert best["n_candidates"] == 5 and best["n_survivors"] == 2


def test_filter_and_pick_drops_nan_candidates():
    scored = [
        (0, 0, float("nan"), {"induce": 1.0, "kl": 0.0}),
        (0, 1, -2.0, {"induce": 1.0, "kl": 0.0}),
    ]
    best = filter_and_pick(scored, 10, constraints=CONSTRAINTS, prune_layer_frac=0.2)
    assert best["layer"] == 1 and best["n_survivors"] == 1


def test_filter_and_pick_raises_when_none_survive():
    scored = [(0, 0, -1.0, {"induce": 1.0, "kl": 0.5})]  # kl over budget
    with pytest.raises(RuntimeError, match="no candidate survived"):
        filter_and_pick(scored, 10, constraints=CONSTRAINTS, prune_layer_frac=0.2)
