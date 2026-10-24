from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_word_gradient as r


class ToyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = (1, 3)
        self.embeddings = nn.ModuleList([nn.Embedding(4, 3), nn.Embedding(4, 3)]).requires_grad_(False)

    def _up_to_full(self, x, level, length):
        return F.interpolate(x.transpose(1, 2), size=length, mode="linear").transpose(1, 2)

    def _apply_quant_resi(self, x, level):
        return x * (level + 1.)

    @torch.no_grad()
    def idx_to_z_q(self, codes):
        a = self._up_to_full(self.embeddings[0](codes[:, :1]), 0, 3)
        b = self.embeddings[1](codes[:, 1:])
        return self._apply_quant_resi(a, 0) + self._apply_quant_resi(b, 1)


def fixture():
    torch.manual_seed(7)
    codec = SimpleNamespace(vq=ToyQuantizer())
    codes = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits = torch.randn(1, 4, 4, requires_grad=True)
    return codec, codes, logits


def test_exact_hard_forward_but_nonzero_gradient_into_code_logits():
    codec, codes, logits = fixture()
    actual = r.straight_through_latent(codec, codes, logits)
    assert torch.equal(actual.detach(), codec.vq.idx_to_z_q(codes))
    actual.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert all(v.grad is None for v in codec.vq.parameters())


def test_hard_baseline_has_no_code_logit_gradient():
    codec, codes, logits = fixture()
    decoder = nn.Linear(3, 2)
    decoder(codec.vq.idx_to_z_q(codes)).sum().backward()
    assert logits.grad is None


def test_every_level_receives_the_word_surrogate_gradient():
    codec, codes, logits = fixture()
    r.straight_through_latent(codec, codes, logits).square().sum().backward()
    assert logits.grad[:, :1].abs().sum() > 0
    assert logits.grad[:, 1:].abs().sum() > 0


def test_rejects_incomplete_or_invalid_temperature_contracts():
    codec, codes, logits = fixture()
    with pytest.raises(ValueError, match="temperature"):
        r.straight_through_latent(codec, codes, logits, temperature=0.)
    with pytest.raises(ValueError, match="matching"):
        r.straight_through_latent(codec, codes[:, :-1], logits)
    with pytest.raises(ValueError, match="hierarchy"):
        r.soft_latent(codec, logits[:, :-1])
