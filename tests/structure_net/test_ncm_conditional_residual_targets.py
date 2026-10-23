from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_conditional_residual_targets import Corruption, sample_corruption, training_targets
from experiments.structure_net.ncm_residual_oracle import requantize


def quantizer():
    return SimpleNamespace(levels=[1, 2],
        embeddings=torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(torch.tensor([[-1.], [0.], [1.]])) for _ in range(2)]),
        _down_for_quant=lambda z, i, n: F.interpolate(z.transpose(1, 2), size=n, mode="area").transpose(1, 2),
        _up_to_full=lambda z, i, n: F.interpolate(z.transpose(1, 2), size=n, mode="linear").transpose(1, 2),
        _apply_quant_resi=lambda z, i: z,
        _distances=lambda z, e: (z[:, None]-e[None]).square().sum(-1))


def test_finer_expert_repairs_actual_wrong_root_and_static_control_cannot():
    vq = quantizer(); z = torch.ones(1, 2, 1)
    clean, _ = requantize(vq, z)
    noise = Corruption(torch.tensor([[True, False, False]]), torch.tensor([[1, 1, 1]]))
    targets, history, stats = training_targets(vq, z, clean, noise, repair=True)
    static, static_history, _ = training_targets(vq, z, clean, noise, repair=False)
    assert targets.tolist() == [[2, 2, 2]]
    assert history.tolist() == [[1, 2, 2]]
    assert static.tolist() == clean.tolist() == [[2, 1, 1]]
    assert static_history.tolist() == [[1, 1, 1]]
    assert stats["relabelled_from_clean"] == 2
    assert z.tolist() == [[[1.], [1.]]]


def test_clean_history_has_exact_original_language_in_both_modes():
    vq = quantizer(); z = torch.tensor([[[1.], [0.]], [[-1.], [-1.]]])
    clean, _ = requantize(vq, z)
    noise = Corruption(torch.zeros_like(clean, dtype=torch.bool), clean.clone())
    for repair in (False, True):
        targets, history, _ = training_targets(vq, z, clean, noise, repair=repair)
        assert torch.equal(targets, clean)
        assert torch.equal(history, clean)


def test_same_level_corruption_cannot_change_that_levels_expert_labels():
    vq = quantizer(); z = torch.ones(1, 2, 1)
    clean, _ = requantize(vq, z)
    no_noise = Corruption(torch.zeros_like(clean, dtype=torch.bool), clean.clone())
    future_noise = Corruption(torch.tensor([[False, True, True]]), torch.tensor([[2, 0, 0]]))
    targets, _, _ = training_targets(vq, z, clean, no_noise, repair=True)
    changed, _, _ = training_targets(vq, z, clean, future_noise, repair=True)
    assert torch.equal(targets, changed)


def test_common_corruption_draws_replay_without_mutation():
    clean = torch.tensor([[0, 1, 2], [2, 1, 0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
    a = sample_corruption(clean, [1, 2], [3, 3], neighbors, torch.Generator().manual_seed(9), first=1., last=.4)
    b = sample_corruption(clean, [1, 2], [3, 3], neighbors, torch.Generator().manual_seed(9), first=1., last=.4)
    assert torch.equal(a.mask, b.mask) and torch.equal(a.replacements, b.replacements)
    assert a.replacements[:, 0].ne(clean[:, 0]).all()
    assert clean.tolist() == [[0, 1, 2], [2, 1, 0]]


def test_inconsistent_root_language_is_rejected():
    vq = quantizer(); z = torch.ones(1, 2, 1)
    wrong = torch.tensor([[0, 1, 1]])
    with pytest.raises(ValueError, match="root labels"):
        training_targets(vq, z, wrong, Corruption(torch.zeros_like(wrong, dtype=torch.bool), wrong), repair=True)


def test_later_expert_uses_actual_corrupted_fine_history():
    vq = quantizer(); vq.levels = [1, 2, 4]
    vq.embeddings.append(torch.nn.Embedding.from_pretrained(torch.tensor([[-1.], [0.], [1.]])))
    z = torch.ones(1, 4, 1)
    clean, _ = requantize(vq, z)
    noise = Corruption(torch.tensor([[False, True, True, False, False, False, False]]),
                       torch.tensor([[2, 2, 2, 1, 1, 1, 1]]))
    targets, history, _ = training_targets(vq, z, clean, noise, repair=True)
    assert targets.tolist() == [[2, 1, 1, 0, 0, 0, 0]]
    assert history.tolist() == [[2, 2, 2, 0, 0, 0, 0]]
