"""Sparse native-language distillation and immutable CPU source-cache batches."""
import torch
from transformers.cache_utils import DynamicCache
from experiments.structure_net.ncm_native_concepts import conditional_target,expand
from experiments.structure_net.ncm_native_query_cache import NativePrefix


def cpu_prefix(memory,index):
    return {"keys_values":tuple(tuple(v[index:index+1].detach().cpu().clone() for v in pair) for pair in memory.keys_values),
        "source_mask":memory.source_mask[index:index+1].cpu().clone(),
        "anchor_token":memory.anchor_token[index:index+1].cpu().clone()}


def pack_prefix(prefixes,device):
    if (not prefixes or len({p["source_mask"].shape[1] for p in prefixes}) != 1
            or len({len(p["keys_values"]) for p in prefixes}) != 1):
        raise ValueError("one source frame/layer count per cache batch required")
    n = len(prefixes[0]["keys_values"])
    values = tuple(tuple(torch.cat([p["keys_values"][i][j] for p in prefixes],0).to(device)
                         for j in range(2)) for i in range(n))
    return NativePrefix(values,torch.cat([p["source_mask"] for p in prefixes],0).to(device),
                        torch.cat([p["anchor_token"] for p in prefixes],0).to(device))


@torch.no_grad()
def native_teacher_support(backbone,memory,targets,eos,topk=32):
    previous = targets[:,:-1].masked_fill(targets[:,:-1].eq(-100),eos)
    query = backbone.predecessor_embeddings(memory,previous)
    batch,width = query.shape[:2]
    mask = torch.cat((memory.source_mask,torch.ones(batch,width,device=query.device,dtype=memory.source_mask.dtype)),1)
    positions = memory.source_mask.sum(-1,keepdim=True)+torch.arange(width,device=query.device)[None]
    cache = DynamicCache.from_legacy_cache(memory.keys_values)
    # Do not call query_logits: that method intentionally enables student adapters.
    with backbone.query_adapters(False):
        hidden = backbone.language.model(inputs_embeds=query,attention_mask=mask,position_ids=positions,
            past_key_values=cache,use_cache=True,return_dict=True).last_hidden_state
        logits = backbone.language.lm_head(hidden)
    values,indices = logits.topk(topk,-1)
    return indices,values.float().softmax(-1)


def sparse_word_loss(logits,targets,teacher_indices,teacher_probabilities,weight,window_eos_injected):
    if (not 0 <= weight <= 1 or logits.shape[:2] != targets.shape
            or teacher_indices.shape != teacher_probabilities.shape or teacher_indices.shape[:2] != targets.shape):
        raise ValueError("sparse teacher/word-loss dimensions differ")
    valid = targets.ne(-100)
    if not valid.any(-1).all(): raise ValueError("each case needs a supervised native word")
    mixture = torch.full(targets.shape,weight,device=logits.device,dtype=torch.float32)
    artificial = torch.as_tensor(window_eos_injected,device=logits.device,dtype=torch.bool)
    if artificial.shape != (len(targets),): raise ValueError("one EOS-window flag per case required")
    rows = torch.arange(len(targets),device=logits.device)[artificial]
    mixture[rows,valid.sum(-1)[artificial]-1] = 0.
    support = torch.cat((targets.clamp_min(0).unsqueeze(-1),teacher_indices),-1)
    weights = torch.cat(((1-mixture).unsqueeze(-1),mixture.unsqueeze(-1)*teacher_probabilities),-1)
    # One dense log-softmax and one gather/backward, including duplicate gold IDs.
    selected = logits.float().log_softmax(-1).gather(-1,support)
    return -(selected*weights).sum(-1)[valid].mean()


@torch.no_grad()
def conditional_teacher_history(net,target,level,generator,rates):
    if not 0 <= level <= len(net.levels) or len(rates)<level or any(not 0 <= r <= 1 for r in rates):
        raise ValueError("invalid conditional teacher history")
    prior = torch.zeros_like(target,dtype=torch.float32)
    codes = []
    for i,book in enumerate(net.books[:level]):
        labels = conditional_target(target,prior,book,net.levels[i])
        mask = torch.rand(labels.shape,generator=generator)<rates[i]
        replacement = torch.randint(len(book),labels.shape,generator=generator)
        labels = torch.where(mask.to(labels.device),replacement.to(labels.device),labels)
        codes.append(labels)
        prior = prior+expand(book[labels],net.canvas)
    return codes,prior
