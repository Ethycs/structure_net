import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_concepts import NativeQueryConcepts,query_hidden


def test_native_concepts_use_only_prior_levels_and_reuse_the_native_word_path():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    backbone = NativeQueryBackbone(LlamaForCausalLM(cfg).eval(),rank=2)
    books = [torch.randn(8,24)*.02 for _ in range(2)]
    for book in books: book[0].zero_()
    model = NativeQueryConcepts(backbone,books,levels=(2,4),canvas=8,hint_rank=2)
    prompt = torch.tensor([[0,1,3,5],[1,3,4,6]])
    memory = backbone.source_cache(prompt,prompt.ne(0).long())
    codes = [torch.ones(2,2,dtype=torch.long),torch.ones(2,4,dtype=torch.long)]
    prior = model.prior(codes,1)
    codes[1].fill_(7)
    assert torch.equal(prior,model.prior(codes,1))
    query,offsets = model.queries(memory,prior,2)
    anchor = backbone.language.model.embed_tokens(memory.anchor_token)
    assert torch.equal(query,torch.cat((anchor,prior[:,:-1]),1))
    with torch.no_grad():
        hidden = query_hidden(backbone,memory,query,offsets)
        assert torch.equal(backbone.language.lm_head(hidden),backbone.query_logits(memory,query))
        first = model.generate(memory)
        second = model.generate(memory)
        assert all(torch.equal(a,b) for a,b in zip(first[0],second[0])) and torch.equal(first[1],second[1])
    loss = model.level_logits(memory,prior,2).float().square().mean()
    loss.backward()
    assert model.hint_b.grad is not None and model.hint_b.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not any(v.enabled for v in backbone.adapters)
