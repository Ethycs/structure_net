"""Parallel residual-concept refinement through an intact native cached body.

This is an untrained adaptation, not evidence of summary quality or paper
replication. The final word stage uses earlier-level native-embedding estimates,
never gold words. A zero-output hint can learn current-position denoising while
the predecessor path retains the native alignment tested by the cache control.
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from experiments.structure_net.ncm_native_concepts import expand, pool


def query_hidden(backbone, memory, query, offsets):
    batch, width = query.shape[:2]
    if offsets.ndim != 1 or len(offsets) != width: raise ValueError("query offsets differ")
    positions = memory.source_mask.sum(-1, keepdim=True)+offsets[None]
    mask = torch.cat((memory.source_mask, torch.ones(batch, width, device=query.device, dtype=memory.source_mask.dtype)), 1)
    cache = DynamicCache.from_legacy_cache(memory.keys_values)
    with backbone.query_adapters(True):
        output = backbone.language.model(inputs_embeds=query, attention_mask=mask, position_ids=positions,
            past_key_values=cache, use_cache=True, return_dict=True)
    return output.last_hidden_state


class NativeQueryConcepts(nn.Module):
    def __init__(self, backbone, books, levels=(4,8,16,32,64), canvas=128, hint_rank=16):
        super().__init__()
        self.backbone = backbone
        width = backbone.language.config.hidden_size
        if (not books or len(books) != len(levels) or any(canvas % k for k in levels)
                or list(levels) != sorted(set(levels)) or any(b.ndim != 2 or b.shape[1] != width for b in books)):
            raise ValueError("native concept dimensions differ")
        self.levels, self.canvas = tuple(levels), canvas
        device = backbone.language.device
        for i, book in enumerate(books):
            self.register_buffer(f"book_{i}", book.detach().float().clone().to(device))
        self.heads = nn.ModuleList([nn.Linear(width,len(book),device=device,dtype=torch.float32) for book in books])
        self.hint_a = nn.Parameter(torch.empty(hint_rank,width,device=device,dtype=torch.float32))
        self.hint_b = nn.Parameter(torch.zeros(width,hint_rank,device=device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.hint_a, a=5**.5)
        self.level_bias = nn.Parameter(torch.zeros(len(books)+1,width,device=device,dtype=torch.float32))

    @property
    def books(self):
        return [getattr(self,f"book_{i}") for i in range(len(self.levels))]

    def prior(self, codes, level):
        if not 0 <= level <= len(self.levels) or len(codes) < level:
            raise ValueError("previous-level history required")
        if not codes: raise ValueError("batch-shaped zero root history required")
        value = self.book_0.new_zeros(len(codes[0]),self.canvas,self.book_0.shape[1])
        for book, labels, width in zip(self.books[:level],codes[:level],self.levels[:level]):
            if labels.shape != (len(value),width): raise ValueError("concept history shape differs")
            value = value+expand(book[labels],self.canvas)
        return value

    def queries(self, memory, prior, level):
        if not 0 <= level <= len(self.levels) or prior.shape[1:] != (self.canvas,self.book_0.shape[1]):
            raise ValueError("native current-level query dimensions differ")
        length = self.canvas if level == len(self.levels) else self.levels[level]
        anchor = self.backbone.language.model.embed_tokens(memory.anchor_token).float()
        current = pool(prior.float(),length)
        if level == len(self.levels):
            base = torch.cat((anchor,prior[:,:-1].float()),1)
        else:
            base = anchor.expand(-1,length,-1)
        # Current-position estimates contain only EARLIER levels. A shifted
        # predecessor alone would discard this useful denoising condition under
        # causal query attention. The new hint starts at exactly zero.
        hint = F.linear(F.linear(current,self.hint_a),self.hint_b)
        query = base+hint+self.level_bias[level][None,None]
        dtype = self.backbone.language.model.embed_tokens.weight.dtype
        offsets = torch.arange(length,device=query.device)*(self.canvas//length)
        return query.to(dtype),offsets

    def level_logits(self, memory, prior, level):
        query,offsets = self.queries(memory,prior,level)
        hidden = query_hidden(self.backbone,memory,query,offsets)
        if level == len(self.levels): return self.backbone.language.lm_head(hidden)
        return self.heads[level](hidden.float())

    @torch.no_grad()
    def generate(self, memory, use_hierarchy=True):
        prior = self.book_0.new_zeros(len(memory.source_mask),self.canvas,self.book_0.shape[1])
        codes = []
        if use_hierarchy:
            for level,book in enumerate(self.books):
                labels = self.level_logits(memory,prior,level).argmax(-1)
                codes.append(labels)
                prior = prior+expand(book[labels],self.canvas)
        words = self.level_logits(memory,prior,len(self.levels)).argmax(-1)
        return codes,words
