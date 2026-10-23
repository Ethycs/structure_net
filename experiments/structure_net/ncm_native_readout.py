"""Native-word autoregressive capacity reference; this module is not an NCM."""
import hashlib

import torch
from torch import nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=5**.5)

    def forward(self, x):
        value = self.base(x)
        return value + F.linear(F.linear(x.float(),self.a),self.b).to(value.dtype)


def frozen_hash(module):
    h = hashlib.sha256()
    for name,value in sorted(module.named_parameters()):
        if not value.requires_grad:
            h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def trainable_state(module):
    return {name:p.detach().cpu().clone() for name,p in module.named_parameters() if p.requires_grad}


def restore_trainable(module, state):
    parameters = {name:p for name,p in module.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("trainable parameter identity changed")
    with torch.no_grad():
        for name,p in parameters.items():
            if state[name].shape != p.shape or not torch.isfinite(state[name]).all():
                raise ValueError("invalid native readout checkpoint")
            p.copy_(state[name])


class NativeReadout(nn.Module):
    def __init__(self, language, feature_count=3, rank=16):
        super().__init__()
        self.language = language.requires_grad_(False)
        if language.lm_head.weight.data_ptr() != language.model.embed_tokens.weight.data_ptr():
            raise ValueError("tied native word identity required")
        width = language.config.hidden_size
        self.fuser = nn.Linear(feature_count*width,width,device=language.device,dtype=torch.float32)
        with torch.no_grad():
            self.fuser.weight.zero_(); self.fuser.bias.zero_()
            scale = language.model.embed_tokens.weight[:1024].float().std().item()
            self.fuser.weight[:,-width:].copy_(torch.eye(width,device=language.device)*scale)
        wanted = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
        for name,module in list(language.named_modules()):
            if name.startswith("model.layers.") and name.rsplit(".",1)[-1] in wanted:
                owner,leaf = name.rsplit(".",1)
                setattr(language.get_submodule(owner),leaf,LoraLinear(module,rank))

    def source_embeddings(self, features):
        if features.ndim != 4 or features.shape[-2]*features.shape[-1] != self.fuser.in_features:
            raise ValueError("per-token feature shape mismatch")
        x = features.float()*torch.rsqrt(features.float().square().mean(-1,keepdim=True)+1e-6)
        return self.fuser(x.flatten(-2)).to(self.language.model.embed_tokens.weight.dtype)

    def teacher_logits(self, features, source_mask, decoder_inputs, use_cache=False):
        source = self.source_embeddings(features)
        word = self.language.model.embed_tokens(decoder_inputs)
        x = torch.cat((source,word),1)
        mask = torch.cat((source_mask,torch.ones_like(decoder_inputs)),1)
        positions = (mask.long().cumsum(-1)-1).clamp_min(0)
        output = self.language.model(inputs_embeds=x,attention_mask=mask,position_ids=positions,
            use_cache=use_cache,return_dict=True)
        return self.language.lm_head(output.last_hidden_state[:,-decoder_inputs.shape[1]:]),output.past_key_values

    def cached_step(self, token, past, mask):
        mask = torch.cat((mask,torch.ones_like(token)),1)
        positions = (mask.long().cumsum(-1)-1)[:,-1:].clamp_min(0)
        output = self.language.model(input_ids=token,attention_mask=mask,position_ids=positions,
            past_key_values=past,use_cache=True,return_dict=True)
        return self.language.lm_head(output.last_hidden_state),output.past_key_values,mask

    @torch.no_grad()
    def generate(self, features, source_mask, bos, eos, max_tokens=128):
        if max_tokens < 1:
            raise ValueError("positive generation limit required")
        token = torch.full((len(features),1),bos,dtype=torch.long,device=features.device)
        logits,past = self.teacher_logits(features,source_mask,token,use_cache=True)
        mask = torch.cat((source_mask,torch.ones_like(token)),1)
        values = [[] for _ in range(len(features))]; finished = [False]*len(features)
        for step in range(max_tokens):
            token = logits[:,-1].argmax(-1,keepdim=True)
            for i,value in enumerate(token[:,0].tolist()):
                if not finished[i]:
                    values[i].append(value); finished[i] = value == eos
            if all(finished) or step+1 == max_tokens:
                break
            logits,past,mask = self.cached_step(token,past,mask)
        return values
