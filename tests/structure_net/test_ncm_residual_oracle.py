from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_residual_oracle import requantize, nearest_distinct


def toy():
    books = torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(torch.tensor([[-1.], [0.], [1.]])) for _ in range(2)])
    return SimpleNamespace(levels=[1, 2], embeddings=books,
        _down_for_quant=lambda z, i, n: F.interpolate(z.transpose(1, 2), size=n, mode="area").transpose(1, 2),
        _up_to_full=lambda z, i, n: F.interpolate(z.transpose(1, 2), size=n, mode="linear").transpose(1, 2),
        _apply_quant_resi=lambda z, i: z,
        _distances=lambda z, e: (z[:, None]-e[None]).square().sum(-1))


def test_finer_residuals_can_correct_a_forced_root_without_mutating_target():
    z = torch.ones(1, 2, 1)
    codes, clean = requantize(toy(), z)
    assert codes.tolist() == [[2, 1, 1]]
    forced = torch.tensor([[1]])
    repaired_codes, repaired = requantize(toy(), z, forced)
    assert repaired_codes.tolist() == [[1, 2, 2]]
    assert torch.equal(repaired, clean)
    assert torch.equal(z, torch.ones_like(z))


def test_neighbors_exclude_self_and_identical_vectors():
    book = torch.tensor([[0.], [0.], [1.], [2.]])
    indices, distances = nearest_distinct(book, 1)
    assert indices[0].item() == 2
    assert (distances > 0).all()


def test_requantization_rejects_invalid_forced_codes():
    with pytest.raises(ValueError, match="outside"):
        requantize(toy(), torch.ones(1, 2, 1), torch.tensor([[3]]))
