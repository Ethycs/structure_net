import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_packed_sequential_objective as objective


def test_level_weight_is_equal_despite_different_token_counts():
    logits = torch.randn(1, 6, 8, requires_grad=True)
    gold = torch.tensor([[1, 2, 3, 4, 5, 6]])
    actual = objective.losses(logits, gold, (1, 2, 3)).mean()
    expected = sum(F.cross_entropy(a.flatten(0, 1), b.flatten()) for a, b in zip(logits.split((1, 2, 3), dim=1), gold.split((1, 2, 3), dim=1))) / 3
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected, logits)[0]
    assert torch.equal(a, b)


def test_history_noise_does_not_modify_targets_and_is_reproducible():
    gold = torch.arange(1000)[None].remainder(8)
    before = gold.clone()
    a = objective.history(gold, (100, 300, 600), 8, torch.Generator().manual_seed(7), True)
    b = objective.history(gold, (100, 300, 600), 8, torch.Generator().manual_seed(7), True)
    assert torch.equal(a[0], b[0]) and a[1] == b[1] > 0
    assert torch.equal(gold, before)
    clean, count = objective.history(gold, (100, 300, 600), 8, torch.Generator(), False)
    assert torch.equal(clean, gold) and count == 0


def test_incomplete_hierarchy_rejected():
    with pytest.raises(ValueError, match='complete'):
        objective.history(torch.ones(1, 2, dtype=torch.long), (1, 2, 3), 8, torch.Generator(), False)
    with pytest.raises(ValueError, match='complete'):
        objective.losses(torch.ones(1, 2, 8), torch.ones(1, 2, dtype=torch.long), (1, 2, 3))
