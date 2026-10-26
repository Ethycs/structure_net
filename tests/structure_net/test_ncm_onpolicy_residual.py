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
