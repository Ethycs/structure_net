import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.structure_net.ncm_native_parallel import NativeConceptGenerator, level_mask, prior_for_level
from experiments.structure_net.ncm_native_readout import NativeReadout, frozen_hash


def generator():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    cfg._attn_implementation = 'sdpa'
    readout = NativeReadout(LlamaForCausalLM(cfg),rank=2)
    books = [torch.randn(4,24)*.05 for _ in range(2)]
    for book in books: book[0].zero_()
    return NativeConceptGenerator(readout,books,(2,4),canvas=8).eval()


def test_native_parallel_current_level_has_no_label_input_and_masks_are_isolated():
    net = generator(); books = net.books
    codes = [torch.tensor([[1,2],[2,3]]),torch.tensor([[0,1,2,3],[3,2,1,0]])]
    prior = prior_for_level(books,codes,1,8)
    changed = [codes[0],(codes[1]+1)%4]
    assert torch.equal(prior,prior_for_level(books,changed,1,8))
    assert torch.equal(prior_for_level(books,codes,0,8),torch.zeros(2,8,24))
    source_mask = torch.tensor([[0,1,1],[1,1,1]])
    mask = level_mask(source_mask,4,torch.float32)
    assert (mask[:,:,0:3,3:]<0).all()  # Source never reads current queries.
    assert (mask[:,:,3:,3:]==0).all()  # Parallel positions see the current block.
    assert (mask[0,:,3:,0]<0).all()   # Padding keys stay hidden.
    words = torch.tensor([[4,5,6,7,8,9,10,11],[11,10,9,8,7,6,5,4]])
    embedded = net.readout.language.model.embed_tokens(words)
    aligned = net.queries_for(embedded,2)
    assert torch.equal(aligned[:,1:],embedded[:,:-1])
    assert torch.equal(aligned[:,0],net.bos_embedding[None].expand(2,-1))


def test_native_parallel_cache_matches_full_level_and_never_mutates_prefix():
    net = generator(); features = torch.randn(2,5,3,24)
    mask = torch.tensor([[0,0,1,1,1],[1,1,1,1,1]])
    prior = torch.randn(2,8,24)*.02
    cache = net.prefix_cache(features,mask)
    before = [(k.clone(),v.clone()) for k,v in cache]
    for level in range(3):
        full = net.level_logits(features,mask,prior,level)
        cached = net.cached_level_logits(cache,mask,prior,level)
        torch.testing.assert_close(full,cached,rtol=1e-5,atol=1e-6)
    assert all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(cache,before))
    a = net.generate(features,mask,True); b = net.generate(features,mask,False)
    assert all(torch.equal(x,y) for x,y in zip(a[0],b[0])) and torch.equal(a[1],b[1])


def test_native_parallel_trains_interfaces_but_preserves_native_word_weights():
    net = generator(); features = torch.randn(2,5,3,24); mask = torch.ones(2,5,dtype=torch.long)
    frozen = frozen_hash(net.readout)
    prior = torch.zeros(2,8,24)
    coarse = net.level_logits(features,mask,prior,0)
    word = net.level_logits(features,mask,prior,2)
    loss = torch.nn.functional.cross_entropy(coarse.flatten(0,1),torch.zeros(4,dtype=torch.long))
    loss = loss+torch.nn.functional.cross_entropy(word.flatten(0,1),torch.ones(16,dtype=torch.long))
    loss.backward()
    assert net.queries.grad.abs().sum()>0 and net.heads[0].weight.grad.abs().sum()>0
    assert net.readout.fuser.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in net.readout.parameters() if not p.requires_grad)
    assert frozen_hash(net.readout)==frozen
    assert net.readout.language.lm_head.weight.data_ptr()==net.readout.language.model.embed_tokens.weight.data_ptr()


def test_noisy_prior_never_uses_current_or_future_codes():
    from experiments.structure_net.ncm_native_parallel_control import noisy_prior
    net = generator()
    codes = [torch.tensor([[1,2],[2,3]]),torch.tensor([[0,1,2,3],[3,2,1,0]])]
    changed = [codes[0],(codes[1]+1)%4]
    a = noisy_prior(net,codes,1,torch.Generator().manual_seed(3042),(.5,.05))
    b = noisy_prior(net,changed,1,torch.Generator().manual_seed(3042),(.5,.05))
    assert torch.equal(a,b)
    root = noisy_prior(net,changed,0,torch.Generator().manual_seed(3042),(.5,.05))
    assert torch.equal(root,torch.zeros_like(root))
