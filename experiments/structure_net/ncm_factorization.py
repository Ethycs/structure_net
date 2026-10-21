"""Bias-preserving low-rank MLPs for the frozen Llama-distilled NCM."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.llama_neighbourhood_tensorization import FactorLinear
from experiments.structure_net.ncm_llama import load_ncm
from experiments.structure_net.run_ncm_llama_graph import MODEL,CODEC

BASE=Path('data/experiments/ncm_factorization')
HYPOTHESIS_ID='llama32-3b-ncm-factorization-v1'
SCHEMA_VERSION='nal.ncm-factorization.v1'
PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-factorization-plan.md')
RANK=256
NAMES=('c_fc','c_proj')


class BiasedFactorLinear(FactorLinear):
    def __init__(self,left,right,bias=None):
        super().__init__(left,right)
        self.bias=nn.Parameter(bias.detach().clone(),requires_grad=False) if bias is not None else None

    def forward(self,x):
        return F.linear(F.linear(x,self.right.to(x.dtype)),self.left.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear,scale,rank,seed):
        factor=cls.from_weight(linear.weight,scale,rank,seed)
        factor.bias=nn.Parameter(linear.bias.detach().clone(),requires_grad=False) if linear.bias is not None else None
        return factor


def factor_state(blocks):
    return {f'{i}.{name}.{part}':getattr(getattr(block.mlp,name),part).detach().half().cpu().clone()
        for i,block in enumerate(blocks) for name in NAMES for part in ('left','right')}


@torch.no_grad()
def set_factors(blocks,state,trainable=False):
    for i,block in enumerate(blocks):
        for name in NAMES:
            layer=getattr(block.mlp,name)
            dtype=torch.float32 if trainable else torch.float16
            left=state[f'{i}.{name}.left'].to(device=layer.bias.device,dtype=dtype)
            right=state[f'{i}.{name}.right'].to(device=layer.bias.device,dtype=dtype)
            factor=BiasedFactorLinear(left,right,layer.bias)
            factor.left.requires_grad_(trainable);factor.right.requires_grad_(trainable)
            setattr(block.mlp,name,factor)


def load_factorized(path,device='cuda'):
    vae,net,cfg=load_ncm(CODEC,MODEL,device=device,adaptation=True)
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['rank']!=RANK:raise ValueError('rank256 checkpoint required')
    blocks=net.transformer.blocks
    factors={f'{i}.{name}.{part}':state['model'][f'transformer.blocks.{i}.mlp.{name}.{part}']
        for i in range(len(blocks)) for name in NAMES for part in ('left','right')}
    set_factors(blocks,factors)
    net.load_state_dict(state['model'],strict=True)
    net.eval().requires_grad_(False)
    return vae,net,cfg


def boundary_loss(prediction,target):
    if prediction.shape!=target.shape:raise ValueError('aligned neighborhood outputs required')
    return (prediction.float()-target.detach().float()).square().mean()/target.detach().float().square().mean().clamp_min(1e-12)
