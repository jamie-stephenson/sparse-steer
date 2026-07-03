import torch

from sparse_steer.core.extract import (
    ActivationTarget,
    _normalize_targets,
    last_token_positions,
)
from sparse_steer.core.steering import SteeringModel


class TestLastTokenPositions:
    def test_right_padded(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        assert last_token_positions(mask).tolist() == [2]

    def test_no_padding(self):
        mask = torch.tensor([[1, 1, 1, 1]])
        assert last_token_positions(mask).tolist() == [3]

    def test_single_token(self):
        mask = torch.tensor([[1, 0, 0, 0]])
        assert last_token_positions(mask).tolist() == [0]

    def test_batch(self):
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        assert last_token_positions(mask).tolist() == [3, 1, 4]

    def test_left_padded(self):
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        assert last_token_positions(mask).tolist() == [4]

    def test_left_padded_batch(self):
        mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        assert last_token_positions(mask).tolist() == [4, 4, 4]

    def test_alternating_batch(self):
        mask = torch.tensor(
            [
                [1, 0, 1, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        assert last_token_positions(mask).tolist() == [3, 3, 4]

    def test_steering_last_token_mask(self):
        mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        expected = torch.tensor(
            [
                [False, False, True, False, False],
                [False, False, False, False, True],
                [False, False, False, False, True],
            ]
        )
        assert torch.equal(SteeringModel.last_token_mask(mask), expected)


class TestNormalizeTargets:
    def test_accepts_string_targets(self):
        result = _normalize_targets(["attention", "mlp"])
        assert result == frozenset({ActivationTarget.ATTENTION, ActivationTarget.MLP})

    def test_rejects_unknown_string_target(self):
        try:
            _normalize_targets(["bogus"])
            raise AssertionError("Expected ValueError for unknown target")
        except ValueError:
            pass
