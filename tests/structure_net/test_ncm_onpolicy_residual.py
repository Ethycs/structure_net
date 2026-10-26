from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_onpolicy_residual as objective


def quantizer():
    weights = torch.tensor([[-2.], [0.], [2.]])
    return SimpleNamespace(levels=[1,1,1],embed_dim=1,
        embeddings=[torch.nn.Embedding.from_pretrained(weights.clone(),freeze=False) for _ in range(3)],
        _down_for_quant=lambda x,i,length:x, _up_to_full=lambda x,i,length:x,
        _apply_quant_resi=lambda x,i:x,
        _distances=lambda x,w:(x[:,None,:]-w[None,:,:]).square().sum(-1))


def test_labels_follow_actual_history_instead_of_earlier_oracle_labels():
    vq = quantizer()
    target = torch.tensor([[[2.]]],requires_grad=True)
    # Actual history contributes zero at each level. Every target must therefore
    # request +2; advancing with oracle labels would incorrectly yield [2,1,1].
    history = torch.tensor([[1,1,1]])
    labels = objective.targets_for_history(vq,target,history)
    assert labels.tolist() == [[2,2,2]]
    assert labels.grad_fn is None and not labels.requires_grad
    assert target.grad is None and torch.equal(target,torch.tensor([[[2.]]]))
    assert all(e.weight.grad is None for e in vq.embeddings)
    assert history.tolist() == [[1,1,1]]
    assert objective.targets_for_history(vq,target,torch.tensor([[2,1,1]])).tolist() == [[2,1,1]]


@pytest.mark.parametrize('history', [torch.tensor([[1,1]]),torch.tensor([[1.,1.,1.]]),
                                     torch.tensor([[1,3,1]])])
def test_invalid_history_rejected(history):
    with pytest.raises(ValueError):
        objective.targets_for_history(quantizer(),torch.tensor([[[2.]]]),history)


@pytest.mark.parametrize('fail', [False,True])
def test_source_rollout_is_detached_and_restores_mixed_modes_even_on_failure(monkeypatch,fail):
    model = torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout())
    model.train(); model[0].eval()
    original = {m:m.training for m in model.modules()}
    feature = torch.ones(1,1,1,requires_grad=True)
    mask = torch.ones(1,1,dtype=torch.bool)
    def generate(m,f,valid):
        assert m is model and f is feature and valid is mask
        assert not any(x.training for x in m.modules()) and not torch.is_grad_enabled()
        if fail: raise RuntimeError('policy failure')
        return f.sum(-1).long()
    monkeypatch.setattr(objective.sequential,'generate',generate)
    if fail:
        with pytest.raises(RuntimeError,match='policy failure'):
            objective.rollout_history(model,feature,mask)
    else:
        assert objective.rollout_history(model,feature,mask).tolist() == [[1]]
    assert original == {m:m.training for m in model.modules()}
    assert feature.grad is None and torch.is_grad_enabled()


def test_pair_generates_from_source_before_encoding_reference(monkeypatch):
    events = []
    features, mask = torch.ones(1,1,1), torch.ones(1,1,dtype=torch.bool)
    gold, words = torch.tensor([[2,1,1]]), torch.tensor([[2]])
    def encode(value):
        assert value is words and events == ['source_rollout']
        events.append('reference_encoding')
        return value.float().unsqueeze(-1)
    model = SimpleNamespace(concepts=SimpleNamespace(generator=SimpleNamespace(levels=[1,1,1])),
                            codec=SimpleNamespace(vq=quantizer(),encode_z=encode))
    def rollout(m,f,valid):
        assert m is model and f is features and valid is mask
        events.append('source_rollout')
        return torch.tensor([[1,1,1]])
    monkeypatch.setattr(objective,'rollout_history',rollout)
    history, targets = objective.training_pair(model,features,mask,gold,words,'residual_targets',True)
    assert events == ['source_rollout','reference_encoding']
    assert history.tolist() == [[1,1,1]] and targets.tolist() == [[2,2,2]]


def test_clean_anchor_and_fixed_arm_preserve_original_targets(monkeypatch):
    features, mask = torch.ones(1,1,1), torch.ones(1,1,dtype=torch.bool)
    gold, words = torch.tensor([[2,1,1]]), torch.tensor([[2]])
    model = SimpleNamespace(concepts=SimpleNamespace(generator=SimpleNamespace(levels=[1,1,1])))
    def forbidden(*args): raise AssertionError('clean anchor must not roll out')
    monkeypatch.setattr(objective,'rollout_history',forbidden)
    for arm in objective.ARMS:
        h,t = objective.training_pair(model,features,mask,gold,words,arm,False)
        assert torch.equal(h,gold) and torch.equal(t,gold)
    monkeypatch.setattr(objective,'rollout_history',lambda *args:torch.tensor([[1,1,1]]))
    h,t = objective.training_pair(model,features,mask,gold,words,'fixed_targets',True)
    assert h.tolist() == [[1,1,1]] and torch.equal(t,gold)


def test_backward_uses_retargeted_labels_without_gradient_into_source_or_codec(monkeypatch):
    model = torch.nn.Module()
    model.concepts = torch.nn.Module()
    model.concepts.generator = torch.nn.Module()
    model.concepts.generator.levels = [1,1,1]
    model.logits = torch.nn.Parameter(torch.zeros(1,3,3))
    model.codec = torch.nn.Module()
    model.codec.vq = quantizer()
    model.codec.encode_z = lambda words:words.float().unsqueeze(-1)
    model.concepts.teacher_logits = lambda f,m,h:model.logits+f.detach().mean()*0
    features = torch.ones(1,1,1,requires_grad=True)
    mask = torch.ones(1,1,dtype=torch.bool)
    monkeypatch.setattr(objective,'rollout_history',lambda *args:torch.tensor([[1,1,1]]))
    metric = objective.backward_example(model,features,mask,torch.tensor([[2,1,1]]),
        torch.tensor([[2]]),'residual_targets',True,None,2)
    assert metric['target_codes'] == [2,2,2] and metric['target_changes'] == 2
    assert all(model.logits.grad[0,i,2] < 0 for i in range(3))
    assert model.logits.grad[0,1,1] > 0
    assert features.grad is None and all(e.weight.grad is None for e in model.codec.vq.embeddings)
