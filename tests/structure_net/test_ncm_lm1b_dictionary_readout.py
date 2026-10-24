import torch
import pytest

from experiments.structure_net import ncm_lm1b_dictionary_readout as run


def test_group_affine_invariance_recovers_ids_without_true_statistics():
    rng = torch.Generator().manual_seed(64142)
    table = torch.randn(64, 384, generator=rng)
    ids = torch.tensor([[7, 13, 0, 42], [3, 18, 57, 1]])
    clean = table[ids].reshape(2, 4, 4, 96)
    scales = torch.tensor([[[[.2], [3.], [1.5], [.7]]], [[[2.], [.5], [4.], [1.]]]])
    shifts = torch.randn(2, 1, 4, 1, generator=rng) * 3
    latent = (clean * scales + shifts).reshape(2, 4, 384)
    dictionary = run.normalized_groups(table)
    assert torch.equal(run.nearest(latent, dictionary), ids)
    assert torch.allclose(run.normalized_groups(latent), run.normalized_groups(table[ids]), atol=1e-6)


def test_dictionary_control_preserves_special_zero_vector_and_rejects_bad_geometry():
    table = torch.eye(8).repeat(1, 2)
    table[0].zero_()
    assert run.nearest(table[None], run.normalized_groups(table)).tolist() == [list(range(8))]
    with pytest.raises(ValueError):
        run.normalized_groups(torch.ones(2, 15))
    with pytest.raises(ValueError):
        run.normalized_groups(torch.ones(2, 16), groups=0)
    with pytest.raises(ValueError):
        run.normalized_groups(torch.full((2, 16), float('nan')))
