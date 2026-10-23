"""Initialization, paired losses and neighborhoods for the new NCM word level."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_broad_learned_codec as released
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_source_head import FrozenSourceConceptHead,migrate_generator_state
from experiments.structure_net.ncm_word_final_head import WordFinalConceptHead


@dataclass(frozen=True)
class Config:
    updates:int=512
    accumulation:int=4
    max_lr:float=5e-5
    warmup:int=32
    model_seed:int=13142
    sampler_seed:int=14142
    noise_seed:int=15142
    probability:float=.05
    neighbors:int=8
    noisy_weight:float=.1
    coarse_weight:float=.1
    word_vocab:int=50257
    pad:int=50257


def rate(step,config=Config()):
    if not 1<=step<=config.updates:raise ValueError("step outside fixed budget")
    if step<=config.warmup:return config.max_lr*step/config.warmup
    return config.max_lr*.5*(1+math.cos(math.pi*(step-config.warmup)/(config.updates-config.warmup)))


def make_model(device="cuda"):
    OmegaConf,_,Generator=released.vendor_classes()
    cfg=OmegaConf.load(released.INITIAL/"config.yaml")
    model_cfg=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
    model_cfg.prefix_len=512
    model_cfg.context_length=8192+192
    codec=released.load_codec(device="cpu").eval().requires_grad_(False)
    facade=BlockwiseCodec(codec,4)
    generator=Generator(facade,model_cfg)
    saved=torch.load(released.INITIAL/"generator.pt",map_location="cpu",weights_only=True)
    migration=migrate_generator_state(generator,saved["model"],codec)
    for name,value in generator.named_parameters():
        expected=saved["model"][name]
        if name=="l0_queries":expected=expected.repeat(1,4,1)
        if not torch.equal(value,expected):raise ValueError("generator parameter failed exact migration: "+name)
    net=WordFinalConceptHead(FrozenSourceConceptHead(generator,3072,residual_prefix=True),word_vocab_limit=50257)
    if (torch.count_nonzero(net.source_head.source_projection.weight)
            or not torch.equal(net.word_projection.weight,generator.lm_heads[-1].proj.weight)
            or any(not torch.equal(v,codec.decoder.state_dict()[k]) for k,v in net.word_decoder.state_dict().items())):
        raise ValueError("fresh residual/word initialization differs")
    codec.to(device)
    net.to(device).float().eval()
    return net,codec,{**migration,"source_residual_zero_initialized":True,
        "word_projection_copied_from_release":True,"word_decoder_copied_from_release":True,
        "trainable_parameters":sum(v.numel() for v in net.parameters() if v.requires_grad)}


@torch.no_grad()
def neighbor_table(books,k=8):
    result=[]
    for book in books:
        vectors=book.weight.detach().float()
        if len(vectors)<=k:raise ValueError("enough distinct non-self neighbors required")
        norm=vectors.square().sum(-1)
        parts=[]
        for start in range(0,len(vectors),256):
            query=vectors[start:start+256]
            distances=norm[start:start+len(query),None]+norm[None]-2*query@vectors.T
            distances[torch.arange(len(query),device=vectors.device),torch.arange(start,start+len(query),device=vectors.device)]=torch.inf
            parts.append(distances.topk(k,largest=False,sorted=True).indices.cpu())
        table=torch.cat(parts)
        if table.shape!=(len(vectors),k) or (table==torch.arange(len(vectors))[:,None]).any():
            raise ValueError("nearest non-self table is invalid")
        result.append(table)
    return result


def perturb(codes,levels,tables,rng,probability=.05):
    if len(tables)!=len(levels)-1 or not 0<=probability<=1:
        raise ValueError("only earlier concept levels may be corrupted")
    value=codes.detach().cpu().clone();offset=0
    for width,table in zip(levels[:-1],tables):
        original=value[:,offset:offset+width].clone()
        mask=torch.rand(original.shape,generator=rng)<probability
        choices=torch.randint(table.shape[1],original.shape,generator=rng)
        replacements=table[original,choices]
        value[:,offset:offset+width]=torch.where(mask,replacements,original)
        offset+=width
    return value.to(codes.device)


def paired_backward(net,batch,tables,noise_rng,accumulation,config=Config()):
    corrupted=perturb(batch["codes"],net.generator.levels,tables,noise_rng,config.probability)
    with torch.autocast("cuda" if batch["features"].is_cuda else "cpu",dtype=torch.bfloat16):
        clean=net(batch["features"],batch["mask"],batch["codes"],batch["source_ids"])
    word=F.cross_entropy(clean["word_logits"][...,:config.word_vocab].float().flatten(0,1),batch["words"].flatten(),ignore_index=config.pad)
    offset=0;levels=[]
    for logits,width in zip(clean["coarse_logits"],net.generator.levels[:-1]):
        levels.append(F.cross_entropy(logits.float().flatten(0,1),batch["codes"][:,offset:offset+width].flatten()))
        offset+=width
    coarse=torch.stack(levels).mean();loss=word+config.coarse_weight*coarse
    if not torch.isfinite(loss):raise ValueError("nonfinite clean word/coarse loss")
    (loss/accumulation).backward()
    values={"clean_word_CE":float(word.detach()),"coarse_CE":float(coarse.detach())}
    del clean,word,coarse,loss,levels
    with torch.autocast("cuda" if batch["features"].is_cuda else "cpu",dtype=torch.bfloat16):
        noisy=net(batch["features"],batch["mask"],corrupted,batch["source_ids"])
    word=F.cross_entropy(noisy["word_logits"][...,:config.word_vocab].float().flatten(0,1),batch["words"].flatten(),ignore_index=config.pad)
    if not torch.isfinite(word):raise ValueError("nonfinite neighborhood word loss")
    (config.noisy_weight*word/accumulation).backward()
    values["noisy_word_CE"]=float(word.detach())
    return values,corrupted.detach().cpu()
