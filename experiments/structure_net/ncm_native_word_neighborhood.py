"""Fixed-pass native-word neighborhoods after source-generated NCM concepts."""
import torch
import torch.nn.functional as F
from experiments.structure_net.ncm_native_concepts import expand
from experiments.structure_net.ncm_native_query_concepts import query_hidden


@torch.no_grad()
def round_native(vectors,weight,eos,excluded=(),chunk=2048):
    if vectors.shape[-1] != weight.shape[-1] or chunk<1: raise ValueError("vocabulary projection shape/chunk")
    shape = vectors.shape[:-1]
    flat = vectors.float().reshape(-1,vectors.shape[-1])
    norms = flat.norm(dim=-1,keepdim=True)
    q = flat/norms.clamp_min(1e-8)
    best = torch.full((len(flat),),-torch.inf,device=flat.device)
    ids = torch.zeros(len(flat),dtype=torch.long,device=flat.device)
    for start in range(0,len(weight),chunk):
        words = weight[start:start+chunk].float()
        words = words/words.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        score = q@words.T
        banned = [i-start for i in excluded if start<=i<start+len(words)]
        if banned: score[:,banned] = -torch.inf
        value,index = score.max(-1)
        improved = value>best
        ids = torch.where(improved,index+start,ids)
        best = torch.maximum(best,value)
    ids = torch.where(norms[:,0]<=1e-8,torch.full_like(ids,eos),ids)
    return ids.reshape(shape)


def native_word_logits(net,memory,prior,state,noise_rms=0.,generator=None):
    if state.shape != prior.shape[:2] or prior.shape[1] != net.canvas:
        raise ValueError("native state/prior shape differs")
    tokens = torch.cat((memory.anchor_token,state[:,:-1]),1)
    base = net.backbone.language.model.embed_tokens(tokens).float()
    if noise_rms:
        if generator is None: raise ValueError("explicit native-neighborhood noise generator required")
        noise = torch.randn(base.shape,generator=generator).to(base.device)
        noise = noise*base.square().mean(-1,keepdim=True).sqrt()*noise_rms
        noise[:,0].zero_()
        base = base+noise
    hint = F.linear(F.linear(prior.float(),net.hint_a),net.hint_b)
    query = (base+hint+net.level_bias[-1][None,None]).to(net.backbone.language.model.embed_tokens.weight.dtype)
    offsets = torch.arange(net.canvas,device=query.device)
    return net.backbone.language.lm_head(query_hidden(net.backbone,memory,query,offsets))


def noise_width(step,total,warmup=128):
    if total<=warmup: return 128
    stage = min(4,max(0,5*max(0,step-warmup)//(total-warmup)))
    return min(128,8*(2**stage))


@torch.no_grad()
def predicted_concepts(net,memory):
    prior = net.books[0].new_zeros(len(memory.source_mask),net.canvas,net.books[0].shape[-1])
    codes = []
    for level,book in enumerate(net.books):
        code = net.level_logits(memory,prior,level).argmax(-1)
        codes.append(code); prior = prior+expand(book[code],net.canvas)
    return codes,prior


def special_policy(net,tokenizer):
    configured = net.backbone.language.generation_config.eos_token_id
    stops = {tokenizer.eos_token_id}
    if isinstance(configured,int): stops.add(configured)
    elif configured is not None: stops.update(configured)
    return stops,tuple(i for i in tokenizer.all_special_ids if i not in stops)


@torch.no_grad()
def initial_state(net,prior,tokenizer,chunk=2048):
    _,excluded = special_policy(net,tokenizer)
    return round_native(prior,net.backbone.language.model.embed_tokens.weight,tokenizer.eos_token_id,excluded,chunk)


@torch.no_grad()
def parallel_generate(net,memory,tokenizer,passes=4,use_hierarchy=True,chunk=2048):
    if passes<1: raise ValueError("at least one fixed parallel word refinement required")
    if use_hierarchy: codes,prior = predicted_concepts(net,memory)
    else:
        codes = []
        prior = net.books[0].new_zeros(len(memory.source_mask),net.canvas,net.books[0].shape[-1])
    state = initial_state(net,prior,tokenizer,chunk)
    for _ in range(passes): state = native_word_logits(net,memory,prior,state).argmax(-1)
    return codes,state
