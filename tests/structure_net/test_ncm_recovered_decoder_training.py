import pytest
import torch

from experiments.structure_net import ncm_recovered_decoder_training as study


def test_complete_distinct_epochs_and_tail():
    orders = study.domain_orders()
    assert study.CONFIG.updates == 6212
    for epoch in range(2):
        first = epoch*study.CONFIG.steps_per_epoch+1
        groups = [study.positions(s, orders) for s in range(first, first+study.CONFIG.steps_per_epoch)]
        flattened = [i for group in groups for i in group]
        assert len(flattened) == 99367 and sorted(flattened) == list(range(99367))
        assert len(groups[-1]) == 7
    assert not torch.equal(orders[0], orders[1])
    assert all(torch.equal(a, b) for a, b in zip(orders, study.domain_orders()))
    for invalid in (0, 6213):
        with pytest.raises(ValueError): study.positions(invalid, orders)


def test_equal_text_weight_ignores_padding():
    logits = torch.tensor([[[0., 2., 0.], [6., 0., 0.], [0., 0., 9.]],
                           [[0., 2., 0.], [0., 2., 0.], [0., 0., 9.]]], requires_grad=True)
    words = torch.tensor([[1, 0, 0], [1, 1, 0]])
    losses = study.per_example_loss(logits, words)
    assert torch.equal(losses[0], losses[1])
    losses.mean().backward()
    assert not logits.grad[words.eq(0)].count_nonzero()
    assert torch.allclose(logits.grad[0, 0], 2*logits.grad[1, 0])


def test_fixed_schedule_endpoints_and_bounds():
    assert study.rate(1) == study.CONFIG.max_lr/128
    assert study.rate(128) == study.CONFIG.max_lr
    assert study.rate(6212) == study.CONFIG.min_lr
    assert all(study.rate(s+1) <= study.rate(s) for s in range(128, 6212))
    with pytest.raises(ValueError): study.rate(6213)
