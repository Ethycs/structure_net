import pytest
import torch

from experiments.structure_net.ncm_global_context import last_valid


def test_last_valid_handles_both_padding_directions():
    hidden = torch.arange(30).reshape(2, 5, 3)
    pad = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
    assert torch.equal(last_valid(hidden, pad), torch.stack([hidden[0, 4], hidden[1, 2]]))


def test_last_valid_refuses_an_empty_context():
    with pytest.raises(ValueError, match="empty"):
        last_valid(torch.zeros(1, 5, 3), torch.ones(1, 5, dtype=torch.bool))
