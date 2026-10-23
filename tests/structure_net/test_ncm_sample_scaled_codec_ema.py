from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net.ncm_sample_scaled_codec_ema import scale_ema,fast_reconstruction,_reconstruction


def test_ema_unit_covariance_and_sample_decay_compose():
    centers=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64)
    counts=torch.tensor([2.,3.,4.],dtype=torch.float64)
    sums=centers*counts[:,None]
    q=SimpleNamespace(base_decay=.99,base_decay_per_level=None,epsilon=1e-5,
        embeddings=[SimpleNamespace(weight=centers.clone())],ema_cluster_size_0=counts.clone(),ema_w_0=sums.clone())
    settings=scale_ema(q,microbatch=4,reference_batch=4096)
    assert torch.equal(q.embeddings[0].weight,centers)
    s=4/4096;d=q.base_decay
    assert d**1024==pytest.approx(.99,abs=1e-12)
    observed_counts=torch.tensor([1.,2.,1.],dtype=torch.float64)
    observed_sums=torch.tensor([[2.,3.],[5.,7.],[8.,9.]],dtype=torch.float64)
    def update(c,w,obs_c,obs_w,epsilon):
        c=d*c+(1-d)*obs_c;w=d*w+(1-d)*obs_w
        smooth=(c+epsilon)/(c.sum()+len(c)*epsilon)*c.sum()
        return w/smooth[:,None]
    reference=update(counts,sums,observed_counts/s,observed_sums/s,1e-5)
    actual=update(q.ema_cluster_size_0,q.ema_w_0,observed_counts,observed_sums,q.epsilon)
    assert torch.allclose(reference,actual,rtol=1e-12,atol=1e-12)
    assert settings['statistic_scale']==s
    with pytest.raises(ValueError):scale_ema(q)


def test_vocab_cache_keeps_all_reconstruction_fields_and_invalid_ids():
    class Codec(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(1))
        def encode(self,x):return torch.zeros(len(x),1,dtype=torch.long)
        def decode(self,codes):
            x=torch.zeros(len(codes),2,8);x[:,:,5]=1;return x
    class Tokenizer:
        def __init__(self):self.calls=0
        def __len__(self):self.calls+=1;return 4
        def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))
    case={'id':'a','packed_ids':[[1,2],[3,4]],'target_ids':[5,5,5,50256],'text':'5 5 5'}
    codec=Codec();plain=Tokenizer();cached=Tokenizer()
    expected=_reconstruction(codec,[case],plain)
    actual=fast_reconstruction(codec,[case],cached)
    assert actual==expected
    assert actual[0]['active_invalid_ids']==4
    assert plain.calls==4 and cached.calls==1
