"""Meaningful numerical and gate contracts for frozen-generator distillation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_xsum_output_distillation as s
from experiments.neural_architecture_lab import ncm_xsum_output_distillation_results as results


def test_phase_boundary_and_cosine_endpoints():
    assert s.phase('distill', 2560) == 'features'
    assert s.phase('distill', 2561) == 'KL'
    assert s.phase('mse', 5120) == 'mse'
    assert s.rate(1) == s.CONFIG.max_lr and s.rate(5120) == s.CONFIG.min_lr
    assert s.rate(2560) > s.rate(2561)
    for arm, step in [('other', 1), ('mse', 0), ('distill', 5121)]:
        with pytest.raises(ValueError): s.phase(arm, step)


@pytest.mark.parametrize('kind', ['features', 'KL'])
def test_loss_has_correct_direction_and_no_teacher_gradient(kind):
    teacher = torch.tensor([[[2., -1.], [.2, .9], [.7, -.6]]], requires_grad=True)
    student = torch.zeros_like(teacher, requires_grad=True)
    loss = s.level_loss(student, teacher, kind, (1, 2)); loss.backward()
    assert teacher.grad is None and student.grad is not None
    assert s.level_loss(student.detach()-.1*student.grad, teacher, kind, (1, 2)) < loss
    assert abs(float(s.level_loss(teacher, teacher, kind, (1, 2)))) < 1e-7


def test_short_level_is_not_drowned_out_by_long_level():
    teacher = torch.zeros(1, 5, 2)
    student = teacher.clone(); student[:, 0] = 2.
    assert s.level_loss(student, teacher, 'features', (1, 4)) == 2.
    with pytest.raises(ValueError): s.level_loss(student[:, :4], teacher, 'features', (1, 4))
    student[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): s.level_loss(student, teacher, 'features', (1, 4))


def test_final_hook_preserves_gradient_and_cleans_up_on_exception():
    net = SimpleNamespace(transformer=SimpleNamespace(ln_f=torch.nn.LayerNorm(3)))
    x = torch.randn(1, 4, 3, requires_grad=True)
    with s.capture_final(net) as values: net.transformer.ln_f(x)
    values[0].square().sum().backward(); assert x.grad is not None
    assert not net.transformer.ln_f._forward_hooks
    with pytest.raises(RuntimeError):
        with s.capture_final(net): raise RuntimeError('failure')
    assert not net.transformer.ln_f._forward_hooks
    with pytest.raises(ValueError):
        with s.capture_final(net): pass
    assert not net.transformer.ln_f._forward_hooks


def test_teacher_history_forward_keeps_gradient_only_through_prefix():
    embedding = torch.nn.Embedding(50304, 384).requires_grad_(False)
    seen = []
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(ln_f=torch.nn.Identity())
        def build_l0_query_input(self, batch, dtype): return torch.zeros(batch, 4, 384, dtype=dtype)
        def forward(self, x, prefix_pad_mask):
            seen.append(prefix_pad_mask)
            # A cheap differentiable stand-in for source-to-code attention.
            hidden = self.transformer.ln_f(x.repeat(1, 1, 2)+x[:, :1].repeat(1, 712, 2))
            return hidden[:, 512:, :1].expand(1, 200, 4096)
    net = Net()
    vae = SimpleNamespace(encoder=SimpleNamespace(wte=embedding),
        vq=SimpleNamespace(idx_to_ncm_input=lambda codes: torch.zeros(1, 196, 384)))
    prefix = torch.randn(7, 384, requires_grad=True)
    logits, hidden = s.forward(vae, net, prefix, torch.zeros(1, 200, dtype=torch.long))
    logits.mean().backward()
    assert prefix.grad is not None and torch.count_nonzero(prefix.grad)
    assert embedding.weight.grad is None
    assert not seen[0][0, :7].any() and seen[0][0, 7:].all()
    assert hidden.shape == (1, 200, 768)


def row():
    native = {'codes': [0]*200, 'words': [5, 6, 50256]+[50257]*45}
    r = {'id': 'a', **{a: deepcopy(native) for a in ('native', 'initial', 'mse', 'distill')}}
    r['mse']['words'][0] = 9
    return r


def test_joint_gate_requires_preservation_and_matched_control_improvement():
    r = row(); v = results.counts([r])
    assert v['preservation_gate_passed'] and v['improvement_gate_passed']
    assert v['content_improvement_over_mse'] == .5
    r['mse'] = deepcopy(r['native'])
    assert not results.counts([r])['improvement_gate_passed']
    r = row(); r['distill']['words'][0] = 8
    v = results.counts([r]); assert not v['preservation_gate_passed']
    assert v['scores']['distill']['word_agreement'] == pytest.approx(47/48)
    assert v['scores']['distill']['content_agreement'] == .5


@pytest.mark.parametrize('fault', ['empty', 'short', 'boolean', 'duplicate'])
def test_invalid_predictions_cannot_pass(fault):
    r = row()
    if fault == 'empty':
        r['native']['words'][0] = 50256
        assert not results.counts([r])['preservation_gate_passed']; return
    if fault == 'short': r['distill']['codes'].pop()
    if fault == 'boolean': r['distill']['words'][0] = True
    with pytest.raises(ValueError): results.counts([r, r] if fault == 'duplicate' else [r])
