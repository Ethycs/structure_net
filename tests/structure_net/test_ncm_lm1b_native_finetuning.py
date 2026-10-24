from contextlib import nullcontext

import pytest
import torch
from torch import nn

from experiments.structure_net import ncm_lm1b_native_finetuning as run


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(128, 2)


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.book = nn.Embedding(8, 2)

    def idx_to_ncm_input(self, codes):
        return self.book(codes[:, :-1])


class Generator(nn.Module):
    levels = (1, 2)

    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.ones(1, 1, 2))
        self.output = nn.Linear(2, 3, bias=False)
        nn.init.ones_(self.output.weight)

    def build_l0_query_input(self, batch, dtype):
        return self.query.expand(batch, -1, -1).to(dtype)

    def forward(self, inputs):
        return self.output(inputs[:, 1:])


class Concepts(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator().requires_grad_(False)
        self.codec = nn.Module()
        self.codec.encoder = Encoder()
        self.codec.vq = Quantizer()
        self.codec.decoder = nn.Linear(2, 8)
        self.codec.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.generator.eval()
        self.codec.eval()
        return self

    def source_context(self, features, mask):
        return nullcontext()

    def teacher_logits(self, features, mask, codes):
        with torch.no_grad():
            prefix = self.codec.encoder.wte(torch.full((len(features), 1), 101, dtype=torch.long))
            root = self.generator.build_l0_query_input(len(features), dtype=prefix.dtype)
            earlier = self.codec.vq.idx_to_ncm_input(codes)
        return self.generator(torch.cat((prefix, root, earlier), dim=1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.concepts = Concepts()
        self.attention = nn.Linear(2, 2)
        self.eval()

    @property
    def codec(self):
        return self.concepts.codec


def test_unfreezing_preserves_every_weight_and_eval_forward():
    model = Model()
    features, mask, codes = torch.ones(1, 3, 2), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.long)
    before = model.concepts.teacher_logits(features, mask, codes)
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert run.enable(model) is model
    assert torch.equal(before, model.concepts.teacher_logits(features, mask, codes))
    assert all(torch.equal(p, saved[n]) for n, p in model.named_parameters())
    with pytest.raises(ValueError, match="already enabled"):
        run.enable(model)


def test_native_learned_root_query_receives_gradient_but_target_codec_does_not():
    model = run.enable(Model()).train()
    features, mask, codes = torch.ones(1, 3, 2), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.long)
    logits = model.concepts.teacher_logits(features, mask, codes)
    logits.sum().backward()
    assert model.concepts.generator.query.grad.abs().sum() > 0
    assert model.concepts.generator.output.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for module in (model.codec.encoder, model.codec.vq) for p in module.parameters())


def test_train_eval_transitions_keep_native_dropout_and_protected_codec_modes_correct():
    model = run.enable(Model()).train()
    assert model.training and model.concepts.generator.training and model.codec.decoder.training
    assert not model.codec.encoder.training and not model.codec.vq.training
    model.eval()
    assert not model.concepts.generator.training and not model.codec.decoder.training
    model.train()
    assert model.concepts.generator.training and model.codec.decoder.training
    assert not model.codec.encoder.training and not model.codec.vq.training
