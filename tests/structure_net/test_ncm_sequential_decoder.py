import pytest
import torch
from torch import nn

from experiments.structure_net import ncm_sequential_decoder as run


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.concepts = nn.Module()
        self.concepts.codec = nn.Module()
        self.concepts.codec.encoder = nn.Linear(4, 4)
        self.concepts.codec.decoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.1))
        self.concepts.codec.register_buffer('EMA', torch.ones(4))
        self.concepts.generator = nn.Linear(4, 4)
        self.word_adapters = nn.Linear(4, 4)

    @property
    def codec(self):
        return self.concepts.codec


def test_only_decoder_updates_and_all_protected_buffers_are_hashed():
    model = Model()
    before = run.protected(model)
    run.configure(model, True)
    assert not model.training and not model.concepts.training and not model.word_adapters.training
    assert model.codec.decoder.training
    params = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in params} == {id(p) for p in model.codec.decoder.parameters()}
    opt = torch.optim.AdamW(params, lr=.1)
    model.word_adapters(model.codec.decoder(torch.ones(1, 4))).sum().backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    opt.step()
    assert run.protected(model) == before
    model.codec.EMA.add_(1)
    assert run.protected(model) != before
    run.configure(model, False)
    assert not any(p.requires_grad for p in model.parameters())


def test_pad_has_no_gradient_but_EOS_does():
    logits = torch.randn(1, 3, 128, requires_grad=True)
    run.word_loss(logits, torch.tensor([[8, 101, 0]])).backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() == 0
    with pytest.raises(ValueError, match='nonempty'):
        run.word_loss(logits, torch.zeros(1, 3, dtype=torch.long))


def test_gate_requires_oracle_and_actual_source_decoding():
    oracle = dict(cases=16, active_word_agreement=1., rendered_exact_cases=16, empty_outputs=0)
    assert run.capacity_pass(oracle, oracle)
    assert not run.capacity_pass(oracle, {**oracle, 'rendered_exact_cases': 15})
    assert not run.capacity_pass({**oracle, 'active_word_agreement': .999}, oracle)


def test_fixed_schedule_warms_then_decays():
    assert run.rate(1, 256) < run.rate(32, 256)
    assert run.rate(32, 256) == 1e-4
    assert run.rate(256, 256) == 1e-5
    with pytest.raises(ValueError):
        run.rate(257, 256)
