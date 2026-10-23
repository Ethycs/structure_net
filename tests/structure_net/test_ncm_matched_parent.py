import pytest
import torch

from experiments.structure_net.ncm_matched_parent_control import ValidateScores


def test_parent_validation_preserves_logits_and_allows_legal_token_masks():
    scores = torch.tensor([[1., -float("inf"), 2.]])
    assert ValidateScores()(None, scores) is scores


@pytest.mark.parametrize("scores", [[[float("nan"), 1.]], [[float("inf"), 1.]], [[-float("inf"), -float("inf")]]])
def test_parent_validation_stops_invalid_generation(scores):
    with pytest.raises(ValueError, match="invalid parent"):
        ValidateScores()(None, torch.tensor(scores))
