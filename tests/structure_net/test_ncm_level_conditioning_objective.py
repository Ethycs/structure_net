from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_level_conditioning_objective as run


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = (1, 3)
        self.embeddings = nn.ModuleList([nn.Embedding(4, 3), nn.Embedding(4, 3)]).requires_grad_(False)

    def _up_to_full(self, x, level, length):
        return F.interpolate(x.transpose(1, 2), size=length, mode='linear').transpose(1, 2)

    def _apply_quant_resi(self, x, level):
        return x

    @torch.no_grad()
    def idx_to_z_q(self, codes):
        return self._up_to_full(self.embeddings[0](codes[:, :1]), 0, 3) + self.embeddings[1](codes[:, 1:])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecaster, self.decoder = nn.Linear(3, 16), nn.Linear(3, 5)
        self.quantizer = Quantizer()
        self.codec = SimpleNamespace(vq=self.quantizer, decoder=self.decoder)
        self.concepts = SimpleNamespace(generator=SimpleNamespace(levels=(1, 3)),
            teacher_logits=self.logits, generate_codes=self.generate)

    def logits(self, features, mask, codes):
        return self.forecaster(features.detach().mean(1)).reshape(len(features), 4, 4)

    def generate(self, features, mask):
        # Source-only API deliberately has no target/code argument.
        return self.logits(features, mask, None).argmax(-1)

    def forward(self, features, mask, codes):
        return {'concept_logits': self.logits(features, mask, codes),
            'word_logits': self.decoder(self.quantizer.idx_to_z_q(codes))}

    def word_context(self, features, mask):
        return nullcontext()


def fixture():
    torch.manual_seed(71)
    return (Model(), torch.randn(1, 2, 3, requires_grad=True), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 0]]))


def test_clean_targets_and_frozen_inputs_preserved_with_real_word_surrogate():
    model, features, mask, codes, targets = fixture()
    before = codes.clone()
    result = run.backward_microbatch(model, features, mask, codes, targets, torch.Generator().manual_seed(42))
    assert model.training and torch.equal(codes, before) and features.grad is None
    assert result['word_gradient_into_concepts'] and result['corruption']['clean_targets_unchanged']
    assert model.forecaster.weight.grad.abs().sum() > 0 and model.decoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.quantizer.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_clean_noisy_gradient_accumulation_uses_declared_global_scale():
    a, features, mask, codes, targets = fixture()
    b = deepcopy(a)
    run.backward_microbatch(a, features, mask, codes, targets, torch.Generator().manual_seed(9),
        self_history=False, accumulation=1, noise=run.corruption.Config(0., 0.))
    run.backward_microbatch(b, features, mask, codes, targets, torch.Generator().manual_seed(9),
        self_history=False, accumulation=2, noise=run.corruption.Config(0., 0.))
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.requires_grad:
            assert torch.allclose(pa.grad, 2 * pb.grad, atol=1e-7)
