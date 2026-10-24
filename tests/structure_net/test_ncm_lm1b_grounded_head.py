from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net.ncm_lm1b_grounded_head import SourceGroundedNCM


class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.middle_block=nn.Linear(4,4); self.extra_blocks=nn.ModuleList([nn.Linear(4,4)])
        self.output=nn.Linear(4,7)
    def forward(self,z):
        x=self.middle_block(z)
        for b in self.extra_blocks:x=b(x)
        return self.output(x)


class Quantizer(nn.Module):
    def idx_to_z_q(self,codes):
        return codes.float()[:, :, None].expand(-1,-1,4)


class Body(nn.Module):
    def __init__(self):
        super().__init__();self.codec=nn.Module();self.codec.embed_dim=4
        self.codec.decoder=Decoder();self.codec.vq=Quantizer();self.generator=nn.Linear(4,4)
        self.generator.levels=(1,2);self.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode);self.generator.eval();self.codec.eval();return self
    def teacher_logits(self,f,m,c):return self.generator(self.codec.vq.idx_to_z_q(c))
    def generate_codes(self,f,m):return torch.zeros((len(f),3),dtype=torch.long,device=f.device)
    def adapter_state(self):return {}
    def load_adapter_state(self,s):assert s=={}
    def close(self):pass


def setup():
    torch.manual_seed(42);body=Body();codes=torch.tensor([[1,2,3]])
    expected=body.codec.decoder(body.codec.vq.idx_to_z_q(codes)).detach()
    model=SourceGroundedNCM(body,source_width=6,word_heads=2,word_layers=(0,1),seed=9,dropout=0.)
    source=torch.randn(1,5,6,requires_grad=True);mask=torch.tensor([[True,True,True,False,False]])
    return model,source,mask,codes,expected


def test_zero_word_attention_preserves_decoder_and_context_lifecycle():
    model,f,m,c,expected=setup()
    assert torch.equal(model.word_logits(f,m,c),expected) and model._word_bound is None
    with model.word_context(f,m):
        with pytest.raises(ValueError):
            with model.word_context(f,m):pass
    with pytest.raises(RuntimeError):
        with model.word_context(f,m):raise RuntimeError('injected')
    assert model._word_bound is None


def test_word_attention_gradients_bypass_no_grad_decoder_api_without_updating_old_weights():
    model,f,m,c,expected=setup();old={k:v.clone() for k,v in model.codec.state_dict().items()}
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.01)
    model.train()
    assert not model.codec.training and not model.concepts.generator.training
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss=(model.word_logits(f,m,c)-torch.ones_like(expected)).square().mean()
        loss.backward();opt.step()
    assert model.word_memory.kv.weight.grad.abs().sum()>0 and f.grad is None
    assert all(v.grad is None for v in model.codec.parameters())
    assert all(torch.equal(v,model.codec.state_dict()[k]) for k,v in old.items())
    model.eval();normal=model.word_logits(f,m,c)
    padded=f.detach().masked_fill(~m[...,None],float('nan'))
    assert torch.equal(normal,model.word_logits(padded,m,c))


def test_attention_only_reload_and_label_free_generation_interface():
    model,f,m,c,_=setup();other,_,_,_,_=setup()
    other.load_attention_state(model.attention_state())
    assert torch.equal(model.word_logits(f,m,c),other.word_logits(f,m,c))
    generated=model.generate(f,m)
    assert generated['codes'].shape==(1,3) and generated['raw_word_ids'].shape==(1,3)
    with pytest.raises(ValueError):model.load_attention_state({'codec':{}})
    model.train()
    with pytest.raises(ValueError):model.generate(f,m)
