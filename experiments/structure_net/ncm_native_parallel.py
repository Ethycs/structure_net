"""Coarse-to-fine residual concepts with parallel native-vocabulary refinement."""
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from experiments.structure_net.ncm_native_concepts import expand, pool


def prior_for_level(books, codes, level, canvas):
    if len(books)!=len(codes) or not 0<=level<=len(books):
        raise ValueError("invalid previous-level history")
    prior = books[0].new_zeros(len(codes[0]),canvas,books[0].shape[1])
    for book,ids in zip(books[:level],codes[:level]):
        prior = prior+expand(book[ids],canvas)
    return prior


def level_mask(source_mask, width, dtype):
    batch,prefix = source_mask.shape
    allowed = torch.zeros(batch,prefix+width,prefix+width,dtype=torch.bool,device=source_mask.device)
    causal = torch.ones(prefix,prefix,dtype=torch.bool,device=source_mask.device).tril()
    allowed[:,:prefix,:prefix] = causal[None]&source_mask[:,None,:].bool()
    # Padded source queries have no downstream influence: their keys are masked.
    allowed[:,:prefix,:prefix] |= ~source_mask[:,:,None].bool()
    allowed[:,prefix:,:prefix] = source_mask[:,None,:].bool()
    allowed[:,prefix:,prefix:] = True
    result = torch.full(allowed.shape,torch.finfo(dtype).min,dtype=dtype,device=source_mask.device)
    return result.masked_fill(allowed,0)[:,None]


class NativeConceptGenerator(nn.Module):
    def __init__(self, readout, books, levels, canvas=128):
        super().__init__()
        width = readout.language.config.hidden_size
        if (len(books)!=len(levels) or not books or any(book.ndim!=2 or book.shape[1]!=width for book in books)
            or any(canvas%length for length in levels)):
            raise ValueError("native concept/readout dimensions differ")
        self.readout = readout; self.levels = tuple(levels); self.canvas = canvas
        device = readout.language.device
        for i,book in enumerate(books): self.register_buffer(f"book_{i}",book.detach().clone().float().to(device))
        self.heads = nn.ModuleList([nn.Linear(width,len(book),device=device,dtype=torch.float32) for book in books])
        anchor = readout.language.model.embed_tokens.weight[readout.language.config.bos_token_id].detach().float()
        self.queries = nn.Parameter(anchor[None].repeat(len(books)+1,1))
        self.register_buffer("bos_embedding",anchor.clone())
        with torch.no_grad(): self.queries[-1].zero_()

    @property
    def books(self):
        return [getattr(self,f"book_{i}") for i in range(len(self.levels))]

    def queries_for(self, prior, level):
        if not 0<=level<=len(self.levels) or prior.shape[1:]!=(self.canvas,self.queries.shape[1]):
            raise ValueError("invalid current concept level or prior")
        width = self.canvas if level==len(self.levels) else self.levels[level]
        if level==len(self.levels):
            # Keep the native readout's predecessor-token alignment. The prior
            # contains generated concepts, never current native-word labels.
            value = torch.cat((self.bos_embedding[None,None].expand(len(prior),1,-1),prior[:,:-1].float()),1)
            value = value+self.queries[level][None,None]
        else:
            value = pool(prior.float(),width)+self.queries[level][None,None]
        return value.to(self.readout.language.model.embed_tokens.weight.dtype)

    def target_positions(self, mask, width):
        offsets = torch.arange(width,device=mask.device)*(self.canvas//width)
        return mask.sum(-1,keepdim=True)+offsets[None]

    def project(self, hidden, level):
        if level==len(self.levels): return self.readout.language.lm_head(hidden)
        return self.heads[level](hidden.float())

    def level_logits(self, features, source_mask, prior, level):
        source = self.readout.source_embeddings(features); query = self.queries_for(prior,level)
        x = torch.cat((source,query),1)
        positions = torch.cat(((source_mask.long().cumsum(-1)-1).clamp_min(0),
            self.target_positions(source_mask,len(query[0]))),1)
        output = self.readout.language.model(inputs_embeds=x,attention_mask=level_mask(source_mask,query.shape[1],x.dtype),
            position_ids=positions,use_cache=False,return_dict=True)
        return self.project(output.last_hidden_state[:,-query.shape[1]:],level)

    def prefix_cache(self, features, source_mask):
        source = self.readout.source_embeddings(features)
        positions = (source_mask.long().cumsum(-1)-1).clamp_min(0)
        output = self.readout.language.model(inputs_embeds=source,attention_mask=source_mask,
            position_ids=positions,use_cache=True,return_dict=True)
        return output.past_key_values.to_legacy_cache()

    def cached_level_logits(self, prefix_cache, source_mask, prior, level):
        query = self.queries_for(prior,level); width = query.shape[1]
        past = DynamicCache.from_legacy_cache(prefix_cache)
        mask = level_mask(source_mask,width,query.dtype)[:,:,-width:]
        output = self.readout.language.model(inputs_embeds=query,attention_mask=mask,
            position_ids=self.target_positions(source_mask,width),past_key_values=past,use_cache=True,return_dict=True)
        return self.project(output.last_hidden_state,level)

    @torch.no_grad()
    def generate(self, features, source_mask, cached=True, use_hierarchy=True):
        prior = self.queries.new_zeros(len(features),self.canvas,self.queries.shape[1]); codes = []
        cache = self.prefix_cache(features,source_mask) if cached else None
        def logits(level):
            return (self.cached_level_logits(cache,source_mask,prior,level) if cached else
                    self.level_logits(features,source_mask,prior,level))
        if use_hierarchy:
            for level,book in enumerate(self.books):
                labels = logits(level).argmax(-1)
                codes.append(labels); prior = prior+expand(book[labels],self.canvas)
        words = logits(len(self.levels)).argmax(-1)
        return codes,words
