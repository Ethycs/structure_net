import pytest
import torch

from experiments.structure_net import ncm_level_conditioning_cuda_pilot as run


def test_worst_frame_padding_retains_every_actual_source_position(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor: tensor)
    features = {'a': torch.arange(3 * 3072, dtype=torch.float32).reshape(3, 3072)}
    before = features['a'].clone()
    rows = [{'id': 'a', 'native_word_ids': [101] * 128}]
    x, mask, codes, words = run.batch(rows, 0, {'a': [1, 2, 3]}, features)
    assert x.shape == (1, 3777, 3072) and mask.sum() == 3
    assert torch.equal(x[0, :3], before) and torch.count_nonzero(x[0, 3:]) == 0
    assert torch.equal(features['a'], before) and not mask[0, 3:].any()
    assert codes.tolist() == [[1, 2, 3]] and words.shape == (1, 128)


def test_larger_than_declared_frame_is_never_truncated():
    with pytest.raises(ValueError, match='complete source'):
        run.batch([{'id': 'a'}], 0, {}, {'a': torch.zeros(3778, 3072)})
