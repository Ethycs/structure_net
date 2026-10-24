import pytest
import torch

from experiments.structure_net import ncm_level_conditioning_corruption as run


def test_clean_targets_intact_same_level_vocabulary_and_root_corrupted():
    codes = torch.zeros(64, 6, dtype=torch.long)
    original = codes.clone()
    noisy, proof = run.corrupt(codes, (1, 2, 3), (2, 3, 4), torch.Generator().manual_seed(42), run.Config(1., 1.))
    assert torch.equal(codes, original) and proof['clean_targets_unchanged']
    assert torch.count_nonzero(noisy[:, 0]) > 0 and not proof['first_generated_code_protected']
    offset = 0
    for width, vocabulary, record in zip((1, 2, 3), (2, 3, 4), proof['levels']):
        assert int(noisy[:, offset:offset + width].max()) < vocabulary
        assert record['selected_positions'] == 64 * width
        offset += width


def test_fixed_seed_replays_decreasing_schedule_and_zero_rate_identity():
    codes = torch.zeros(8, 6, dtype=torch.long)
    first = run.corrupt(codes, (1, 2, 3), (4, 4, 4), torch.Generator().manual_seed(123))
    second = run.corrupt(codes, (1, 2, 3), (4, 4, 4), torch.Generator().manual_seed(123))
    assert torch.equal(first[0], second[0]) and first[1] == second[1]
    assert [r['rate'] for r in first[1]['levels']] == [.75, .5, .25]
    zero, proof = run.corrupt(codes, (1, 2, 3), (4, 4, 4), torch.Generator().manual_seed(123), run.Config(0., 0.))
    assert torch.equal(zero, codes) and all(r['selected_positions'] == 0 for r in proof['levels'])
    with pytest.raises(ValueError):
        run.corrupt(codes + 3, (1, 2, 3), (4, 2, 4), torch.Generator())
    with pytest.raises(ValueError):
        run.Config(.25, .75)
