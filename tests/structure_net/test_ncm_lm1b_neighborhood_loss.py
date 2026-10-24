from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_neighborhood_loss as run


def test_root_classification_is_not_drowned_by_fine_levels():
    logits=torch.tensor([[[0.,5.],[5.,0.],[5.,0.]]],requires_grad=True)
    target=torch.zeros((1,3),dtype=torch.long)
    expected=(F.cross_entropy(logits[:,0],target[:,0])+F.cross_entropy(logits[:,1:].reshape(2,2),target[:,1:].flatten()))/2
    actual=run.level_cross_entropy(logits,target,(1,2))
    assert torch.equal(actual,expected)
    assert actual>F.cross_entropy(logits.flatten(0,1),target.flatten())
    actual.backward();assert logits.grad.abs().sum()>0


def test_neighbor_corruption_is_replayable_and_preserves_root_and_original_labels():
    codes=torch.arange(32*13).reshape(32,13)%8192;before=codes.clone()
    a,pa=run.corrupt_history(codes,1024,torch.Generator().manual_seed(17))
    b,pb=run.corrupt_history(codes,1024,torch.Generator().manual_seed(17))
    assert torch.equal(a,b) and pa==pb and pa['changed_positions']>0
    assert torch.equal(a[:,:1],codes[:,:1]) and torch.equal(codes,before)
    assert int(a.min())>=0 and int(a.max())<8192 and pa['maximum_rate']==.25
    _,warm=run.corrupt_history(codes,1,torch.Generator().manual_seed(17))
    assert warm['maximum_rate']==.25/1024
    with pytest.raises(ValueError):run.corrupt_history(codes,0,torch.Generator())


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(torch.randn(7,7));self.calls=[];self.generations=0
        self.concepts=SimpleNamespace(generator=SimpleNamespace(levels=(1,2)),generate_codes=self.generate_codes)
    def forward(self,f,m,c):
        self.calls.append(c.detach().clone())
        concept=F.one_hot(c,7).float()@self.weights
        word=self.weights[0].reshape(1,1,7).expand(len(f),4,7)
        return {'concept_logits':concept,'word_logits':word}
    def generate_codes(self,features,mask):
        assert not self.training and features.shape[:2]==mask.shape
        self.generations+=1
        return torch.full((len(features),3),6,dtype=torch.long)


def test_full_loss_backpropagates_and_self_history_generation_receives_only_source(monkeypatch):
    # Small vocabulary fixture; actual native corruption uses8192.
    original=run.corrupt_history
    monkeypatch.setattr(run,'corrupt_history',lambda c,s,g,**kw:original(c,s,g,vocabulary=7,**kw))
    torch.manual_seed(4);model=Model().train();f=torch.randn(2,5,3);m=torch.ones((2,5),dtype=torch.bool)
    codes=torch.tensor([[1,2,3],[2,3,4]]);words=torch.tensor([[1,2,3,0],[1,2,4,0]])
    result=run.backward_microbatch(model,f,m,codes,words,torch.Generator().manual_seed(8),260,self_history=True)
    assert model.training and model.generations==1 and len(model.calls)==3
    assert torch.equal(model.calls[0],codes) and (model.calls[-1]==6).all()
    assert model.weights.grad is not None and torch.isfinite(model.weights.grad).all()
    assert result['source_only_history_codes']==[[6,6,6],[6,6,6]]
