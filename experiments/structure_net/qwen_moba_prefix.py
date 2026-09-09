"""Public Qwen3-8B six-layer prefix with official MoBA attention substitution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding, apply_rotary_pos_emb,
)

from experiments.structure_net.moba_head_benchmark import flash_call, official_call

MODEL_ROOT = Path("data/models/Qwen3-8B-prefix6")
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


@dataclass(frozen=True)
class MoBASetting:
    block_size: int = 128
    top_k: int = 2
    layers: str = "all"

    def __post_init__(self):
        if self.block_size < 1 or self.top_k < 2 or self.layers not in {"all","early5","last","full"}:
            raise ValueError("invalid MoBA setting")

    @property
    def name(self):
        return "full" if self.layers == "full" else f"{self.layers}_b{self.block_size}_k{self.top_k}"

    def enabled(self, index, length):
        chosen = self.layers == "all" or (self.layers == "early5" and index < 5) or (self.layers == "last" and index == 5)
        return chosen and self.top_k * self.block_size < length


SETTINGS = tuple(MoBASetting(block,k,layers) for block,k in ((128,2),(128,4),(128,8),(256,4))
                 for layers in ("all","early5","last"))
FULL = MoBASetting(layers="full")


def moba_attention(q,k,v,setting,cu,head_chunk_size=32):
    groups=q.shape[1]//k.shape[1]
    if head_chunk_size < 1 or head_chunk_size % groups:
        raise ValueError("head chunk must contain complete GQA groups")
    if q.shape[1]<=head_chunk_size:
        return official_call(q,k,v,setting.block_size,setting.top_k,cu)
    parts=[]
    for start in range(0,q.shape[1],head_chunk_size):
        stop=min(start+head_chunk_size,q.shape[1])
        parts.append(official_call(q[:,start:stop],k[:,start//groups:stop//groups],v[:,start//groups:stop//groups],
                                   setting.block_size,setting.top_k,cu))
    return torch.cat(parts,dim=1)


class QwenPrefix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size,config.pad_token_id)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config,i) for i in range(6))
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.mlp_chunk_tokens = 2048
        self.moba_head_chunk_size = 32

    def forward(self, ids, setting=FULL, segment_size=128, capture=False):
        if ids.ndim != 2 or ids.shape[1] % segment_size:
            raise ValueError("require B,T token IDs and complete pooling segments")
        b,n = ids.shape
        x = self.embed_tokens(ids)
        positions = torch.arange(n,device=ids.device)[None,:]
        cos,sin = self.rotary_emb(x,positions)
        cu = torch.arange(b+1,device=ids.device,dtype=torch.int32) * n
        for index,layer in enumerate(self.layers):
            a = layer.self_attn
            normalized = layer.input_layernorm(x)
            shape = (b,n,-1,a.head_dim)
            q = a.q_norm(a.q_proj(normalized).view(shape)).transpose(1,2)
            k = a.k_norm(a.k_proj(normalized).view(shape)).transpose(1,2)
            v = a.v_proj(normalized).view(shape).transpose(1,2)
            q,k = apply_rotary_pos_emb(q,k,cos,sin)
            if setting.enabled(index,n):
                attended = moba_attention(q,k,v,setting,cu,self.moba_head_chunk_size)
            else:
                attended = flash_call(q,k,v)
            projected = a.o_proj(attended.transpose(1,2).reshape(b,n,-1))
            if index < 5:
                x = x + projected
                del normalized,q,k,v,attended,projected
                # MLPs are tokenwise: chunking bounds temporary activation memory.
                norm = layer.post_attention_layernorm(x)
                mlp = torch.cat([layer.mlp(part) for part in norm.split(self.mlp_chunk_tokens,dim=1)],dim=1)
                x = x + mlp
                del norm,mlp
        # No sixth MLP or downstream language-model computation.
        heads = attended.transpose(1,2).float().reshape(b,n//segment_size,segment_size,-1).mean(2)
        ov = projected.float().reshape(b,n//segment_size,segment_size,-1).mean(2)
        result = {"heads":heads,"ov":ov,"changes":1-F.cosine_similarity(ov[:,1:],ov[:,:-1],dim=-1)}
        if capture:
            result.update(q=q,k=k,v=v,attention_tokens=attended,projected_tokens=projected,hidden_before_final=x)
        return result


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_prefix(device="cuda:0", root=MODEL_ROOT):
    from safetensors import safe_open
    plan=json.loads((root/"prefix_download_plan.json").read_text())
    if plan["revision"] != MODEL_REVISION or plan["layers"] != 6:
        raise RuntimeError("model revision/prefix mismatch")
    manifest=json.loads((root/"download_hashes.json").read_text())
    for name in ("config.json",*plan["shards"]):
        if sha256(root/name) != manifest[name]:
            raise RuntimeError("model input hash mismatch")
    config=Qwen3Config(**json.loads((root/"config.json").read_text()))
    with torch.device("meta"):
        model=QwenPrefix(config)
    state={}
    for name in plan["shards"]:
        with safe_open(root/name,framework="pt",device="cpu") as shard:
            for key in plan["keys"]:
                if key in shard.keys():
                    state[key.removeprefix("model.")]=shard.get_tensor(key).to(device=device,dtype=torch.float16)
    model.load_state_dict(state,strict=True,assign=True)
    # RoPE has nonpersistent buffers, so recreate these outside the meta device.
    model.rotary_emb=Qwen3RotaryEmbedding(config,device=device)
    return model.eval().requires_grad_(False)


def frozen_hash(model):
    h=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pearson(first, second):
    first,second=first.float().flatten(),second.float().flatten()
    first,second=first-first.mean(),second-second.mean()
    denom=first.norm()*second.norm()
    if float(denom) == 0:
        return None
    return float((first@second)/denom)


def readout_metrics(actual, expected):
    metrics={}
    for name in ("heads","ov"):
        av,ev=actual[name].float(),expected[name].float()
        cosine=F.cosine_similarity(av,ev,dim=-1)
        metrics[name+"_cosine_mean"]=float(cosine.mean())
        metrics[name+"_cosine_min"]=float(cosine.min())
        metrics[name+"_relative_l2"]=float((av-ev).norm()/ev.norm().clamp_min(1e-12))
    first,second=actual["changes"].flatten(),expected["changes"].flatten()
    count=max(1,(first.numel()+3)//4)
    first_top=first.argsort(descending=True,stable=True)[:count]
    second_top=second.argsort(descending=True,stable=True)[:count]
    metrics["change_pearson"]=pearson(first,second)
    metrics["top_quarter_change_f1"]=float(torch.isin(first_top,second_top).float().mean())
    metrics["change_relative_l2"]=float((first-second).norm()/second.norm().clamp_min(1e-12))
    return metrics


def feasibility_gate(metrics,speedup,sparse_active):
    return bool(sparse_active and metrics["heads_cosine_mean"]>=.95 and metrics["ov_cosine_mean"]>=.95
                and metrics["change_pearson"] is not None and metrics["change_pearson"]>=.95
                and metrics["top_quarter_change_f1"]>=.90 and speedup>=1.05)
