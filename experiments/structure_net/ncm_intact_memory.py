"""Keep the native language path intact; add bounded source/plan attention."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CorrectionMemory:
    source_key: torch.Tensor
    source_value: torch.Tensor
    source_mask: torch.Tensor
    plan_key: torch.Tensor | None
    plan_value: torch.Tensor | None
    plan_mask: torch.Tensor | None


class NativeMemoryCorrection(nn.Module):
    def __init__(self, hidden_size, source_width, feature_count=3, memory_width=256, heads=8, max_fraction=.1):
        super().__init__()
        if memory_width%heads or not 0<max_fraction<=1:
            raise ValueError("invalid bounded memory attention configuration")
        self.source_width = source_width; self.feature_count = feature_count
        self.heads = heads; self.memory_width = memory_width; self.max_fraction = max_fraction
        self.query = nn.Linear(hidden_size,memory_width,bias=False)
        self.source_key = nn.Linear(feature_count*source_width,memory_width,bias=False)
        self.source_value = nn.Linear(feature_count*source_width,memory_width,bias=False)
        self.plan_key = nn.Linear(source_width,memory_width,bias=False)
        self.plan_value = nn.Linear(source_width,memory_width,bias=False)
        self.output = nn.Linear(memory_width,hidden_size,bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def normalized(value):
        value = value.float()
        return value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)

    def split(self,value):
        return value.reshape(value.shape[0],value.shape[1],self.heads,self.memory_width//self.heads).transpose(1,2)

    def prepare(self, source, source_mask, plan=None, plan_mask=None):
        if source.ndim!=4 or source.shape[-2:]!=(self.feature_count,self.source_width) or source_mask.shape!=source.shape[:2]:
            raise ValueError("per-token parent memory dimensions differ")
        if not source_mask.bool().any(-1).all(): raise ValueError("source memory needs visible passage tokens")
        value = self.normalized(source).flatten(-2)
        source_key,source_value = self.split(self.source_key(value)),self.split(self.source_value(value))
        pk = pv = pm = None
        if plan is not None:
            if plan.ndim!=3 or plan.shape[0]!=source.shape[0] or plan.shape[-1]!=self.source_width:
                raise ValueError("concept memory dimensions differ")
            pm = torch.ones(plan.shape[:2],dtype=torch.bool,device=plan.device) if plan_mask is None else plan_mask.bool()
            if pm.shape!=plan.shape[:2] or not pm.any(-1).all(): raise ValueError("concept memory mask differs")
            value = self.normalized(plan)
            pk,pv = self.split(self.plan_key(value)),self.split(self.plan_value(value))
        return CorrectionMemory(source_key,source_value,source_mask.bool(),pk,pv,pm)

    def forward(self, hidden, memory):
        if hidden.shape[0]!=memory.source_key.shape[0]: raise ValueError("memory/native batch differs")
        query = self.split(self.query(self.normalized(hidden)))
        if memory.plan_key is not None:
            query = query+F.scaled_dot_product_attention(query,memory.plan_key,memory.plan_value,
                attn_mask=memory.plan_mask[:,None,None],dropout_p=0.)
        context = F.scaled_dot_product_attention(query,memory.source_key,memory.source_value,
            attn_mask=memory.source_mask[:,None,None],dropout_p=0.)
        context = context.transpose(1,2).reshape(hidden.shape[0],hidden.shape[1],self.memory_width)
        delta = self.output(context).float()
        limit = hidden.detach().float().norm(dim=-1,keepdim=True)*self.max_fraction
        scale = (limit/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp_max(1.)
        return hidden+((delta*scale).to(hidden.dtype))


class IntactMemoryReadout(nn.Module):
    def __init__(self,native,correction):
        super().__init__()
        self.native = native.requires_grad_(False)
        self.correction = correction
        self._active = False

    @contextmanager
    def conditioned(self,memory):
        if self._active: raise RuntimeError("overlapping native correction contexts")
        self._active = True
        def before_word_head(module,args):
            return (self.correction(args[0],memory),*args[1:])
        handle = self.native.lm_head.register_forward_pre_hook(before_word_head)
        try:
            yield self.native
        finally:
            handle.remove(); self._active = False


def passage_mask(tokenizer,cases,padded_length=768):
    from experiments.structure_net.ncm_native_parent_control import SYSTEM
    result = torch.zeros(len(cases),padded_length,dtype=torch.bool)
    for i,case in enumerate(cases):
        messages = [{"role":"system","content":SYSTEM},
            {"role":"user","content":"<passage>\n"+case["source"]+"\n</passage>"}]
        rendered = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages,add_generation_prompt=True)
        encoded = tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        if encoded["input_ids"]!=expected or len(expected)>padded_length:
            raise ValueError("native passage token alignment or budget differs")
        start = rendered.index("<passage>\n")+len("<passage>\n"); end = start+len(case["source"])
        if rendered[start:end]!=case["source"]: raise ValueError("source passage alignment differs")
        valid = [a>=start and b<=end and b>a for a,b in encoded["offset_mapping"]]
        result[i,-len(expected):] = torch.tensor(valid)
    if not result.any(-1).all(): raise ValueError("source passage has no aligned native tokens")
    return result
