"""Refusal detectors: the regex text matcher and Arditi's logit refusal metric (App. B)."""

import math
from types import SimpleNamespace

import pytest
import torch

from sparse_steer.tasks.jailbreak.data import label_and_bucket
from sparse_steer.utils.refusal import (
    detect_refusal,
    refusal_metric,
    resolve_refusal_token_ids,
)


def test_refusal_metric_matches_logodds():
    # rows: P_refusal(ℛ={0,1}) = 0.7 and 0.10
    probs = torch.tensor([[0.6, 0.1, 0.2, 0.1], [0.05, 0.05, 0.8, 0.1]])
    out = refusal_metric(probs.log(), [0, 1])
    expected = torch.tensor(
        [math.log(0.7) - math.log(0.3), math.log(0.10) - math.log(0.90)]
    )
    assert torch.allclose(out, expected, atol=1e-5)


def test_refusal_metric_sign_splits_refuse_comply():
    # >0 ⇔ more likely than not to open with a refusal token; <0 ⇔ comply.
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    out = refusal_metric(probs.log(), [0])
    assert out[0] > 0 and out[1] < 0


def test_resolve_refusal_token_ids_dedup_and_order():
    class FakeTok:
        table = {"I": [40], "As": [2121, 999], "I ": [40]}

        def encode(self, s, add_special_tokens=False):
            return self.table[s]

    assert resolve_refusal_token_ids(FakeTok(), ["I", "As"]) == [40, 2121]
    # an opener whose first token repeats an earlier one is dropped (order preserved)
    assert resolve_refusal_token_ids(FakeTok(), ["I", "I "]) == [40]


def test_resolve_refusal_token_ids_empty_raises():
    class EmptyTok:
        def encode(self, s, add_special_tokens=False):
            return []

    with pytest.raises(ValueError):
        resolve_refusal_token_ids(EmptyTok(), ["x"])


def test_regex_detector():
    assert detect_refusal("I'm sorry, but I can't help with that.")
    assert not detect_refusal("Sure, here is how you do it: step 1...")
    with pytest.raises(NotImplementedError):
        detect_refusal("anything", detector="logit")


def test_label_and_bucket_rejects_unknown_detector():
    # Dispatch guard fires before any model use (rows empty → no forward/generation).
    with pytest.raises(ValueError, match="refusal_detector"):
        label_and_bucket(None, None, [], SimpleNamespace(refusal_detector="bogus"))  # type: ignore[arg-type]
