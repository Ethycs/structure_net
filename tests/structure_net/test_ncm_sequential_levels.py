from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_sequential_levels as run


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings=nn.ModuleList(nn.Embedding(8,4) for _ in range(3))

    @torch.no_grad()
    def get_next_autoregressive_input(self,k,cumulative,embedded):
        cumulative=cumulative+F.interpolate(embedded.transpose(1,2),size=3,mode='linear').transpose(1,2)
        return cumulative,F.interpolate(cumulative.transpose(1,2),size=(1,2,3)[min(k+1,2)],mode='area').transpose(1,2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(4,4)

    def forward(self,x,**kwargs):
        return x+self.projection(x).tanh()


class Concepts(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec=nn.Module()
        self.codec.embed_dim=4
        self.codec.encoder=nn.Module()
        self.codec.encoder.wte=nn.Embedding(128,4)
        self.codec.decoder=nn.Linear(4,128)
        self.codec.vq=Quantizer()
        g=self.generator=nn.Module()
        g.levels=(1,2,3);g.query_level0=True;g.l0_ar=False;g.prefix_len=1;g.use_ctx_encoder=False
        g.use_rope=False;g.use_post_upsample_conv=False
        g.config=SimpleNamespace(attn_pattern='block_diagonal',use_level_encoding=True)
        g.transformer=nn.ModuleDict({'wte':nn.Linear(4,4),'wle':nn.Embedding(3,4),
            'blocks':nn.ModuleList([Block()]),'ln_f':nn.LayerNorm(4)})
        g.lm_heads=nn.ModuleList(nn.Linear(4,8) for _ in range(3))
        g.l0_queries=nn.Parameter(torch.randn(1,1,4))
        g.build_l0_query_input=lambda batch:g.l0_queries.expand(batch,-1,-1)
        g.wpe=nn.Embedding(7,4)
        g.register_buffer('wp_prefix',torch.tensor([[0]]))
        g.register_buffer('wp_1L',torch.arange(1,7)[None])
        g.register_buffer('lvl_1L',torch.tensor([[0,1,1,2,2,2]]))
        self.source_memory=nn.Linear(4,4)
        self.adapters=nn.Linear(4,4)
        self.bound=None
        g.transformer.blocks[0].register_forward_hook(lambda module,args,output:output+self.bound if self.bound is not None else output)

    @contextmanager
    def source_context(self,features,mask):
        self.bound=self.adapters(self.source_memory(features.detach().mean(1,keepdim=True)))
        try: yield
        finally: self.bound=None


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.concepts=Concepts();self.word_memory=nn.Linear(4,4);self.word_adapters=nn.Linear(4,4)

    @property
    def codec(self): return self.concepts.codec


def sample():
    torch.manual_seed(7)
    model=Model()
    return model,torch.randn(1,5,4),torch.ones(1,5,dtype=torch.bool),torch.tensor([[1,2,3,4,5,6]])


def forced(model,f,m,gold):
    previous=None;cumulative=f.new_zeros(1,3,4);offset=0;output=[]
    for k,n in enumerate(model.concepts.generator.levels):
        output.append(run.level_logits(model,f,m,previous,k))
        cumulative,previous=run.advance(model.codec,cumulative,gold[:,offset:offset+n],k)
        offset+=n
    return output


def test_current_and_future_targets_cannot_change_current_logits():
    model,f,m,gold=sample();run.configure(model,False)
    original=forced(model,f,m,gold)
    changed=gold.clone();changed[:,1:]=0
    altered=forced(model,f,m,changed)
    assert torch.equal(original[0],altered[0])
    assert torch.equal(original[1],altered[1])
    assert not torch.equal(original[2],altered[2])


def test_all_level_heads_root_and_source_receive_gradients_codec_stays_frozen():
    model,f,m,gold=sample();run.configure(model,True)
    f.requires_grad_(True)
    record=run.backward_example(model,f,m,gold,torch.Generator().manual_seed(9),None,2,False)
    assert len(record['level_CE'])==3 and record['history_corruptions']==0
    assert model.concepts.generator.l0_queries.grad.abs().sum()>0
    assert all(h.weight.grad.abs().sum()>0 for h in model.concepts.generator.lm_heads)
    assert model.concepts.source_memory.weight.grad.abs().sum()>0
    assert f.grad is None
    assert all(not p.requires_grad and p.grad is None for p in model.codec.parameters())
    assert all(p.grad is None for p in model.word_adapters.parameters())
    assert not model.codec.training and model.concepts.generator.training


def test_clean_targets_unchanged_and_noise_reproducible():
    part=torch.arange(100).remainder(8)[None];saved=part.clone()
    a=run.corrupt(part,0,3,8,torch.Generator().manual_seed(12),True)
    b=run.corrupt(part,0,3,8,torch.Generator().manual_seed(12),True)
    assert torch.equal(a[0],b[0]) and a[1]>0
    assert torch.equal(part,saved)


def test_source_only_rollout_and_mode_contract():
    model,f,m,_=sample();run.configure(model,True)
    with pytest.raises(ValueError,match='eval'):run.generate(model,f,m)
    run.configure(model,False)
    assert run.generate(model,f,m).shape==(1,6)
    a=run.level_logits(model,f,m,None,0)
    b=run.level_logits(model,f+2,m,None,0)
    assert not torch.equal(a,b)
    with pytest.raises(ValueError,match='root'):run.level_logits(model,f,m,torch.zeros(1,1,4),0)


def test_invalid_training_shape_fails_before_update():
    model,f,m,gold=sample();run.configure(model,True)
    with pytest.raises(ValueError,match='complete'):
        run.backward_example(model,f,m,gold[:,:1],torch.Generator(),None,1,False)
