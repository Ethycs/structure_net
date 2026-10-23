from types import SimpleNamespace
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_concepts import NativeQueryConcepts
from experiments.structure_net.ncm_native_word_neighborhood import round_native,native_word_logits,parallel_generate,noise_width


def test_projection_uses_native_directions_special_exclusions_and_zero_eos():
    weight = torch.eye(5)
    vectors = torch.stack((weight[3]*.01,torch.zeros(5)))[None]
    assert round_native(vectors,weight,2,chunk=2).tolist()==[[3,2]]
    assert round_native(vectors,weight,2,excluded=(3,),chunk=2).tolist()==[[0,2]]
    assert noise_width(128,2048)==8 and noise_width(1664,2048)==128


def test_native_word_center_matches_body_and_cannot_read_current_future_words():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    backbone = NativeQueryBackbone(LlamaForCausalLM(cfg).eval(),rank=2)
    books = [torch.randn(8,24)*.02 for _ in range(2)]
    model = NativeQueryConcepts(backbone,books,levels=(1,2),canvas=4,hint_rank=2)
    prompt = torch.tensor([[0,1,3,4],[1,5,6,7]])
    memory = backbone.source_cache(prompt,prompt.ne(0).long())
    prior = torch.randn(2,4,24)*.02
    state = torch.tensor([[7,8,9,10],[11,12,13,14]])
    with torch.no_grad():
        expected = backbone.query_logits(memory,backbone.predecessor_embeddings(memory,state[:,:-1]))
        actual = native_word_logits(model,memory,prior,state)
        assert torch.equal(expected,actual)
        changed = state.clone(); changed[:,2:] = 30
        later = native_word_logits(model,memory,prior,changed)
        assert torch.equal(actual[:,:3],later[:,:3])
        tokenizer = SimpleNamespace(eos_token_id=2,all_special_ids=[0,1,2])
        a,x = parallel_generate(model,memory,tokenizer,passes=2,chunk=7)
        b,y = parallel_generate(model,memory,tokenizer,passes=2,chunk=7)
        assert torch.equal(x,y) and all(torch.equal(u,v) for u,v in zip(a,b))
    loss = native_word_logits(model,memory,prior,state,.1,torch.Generator().manual_seed(6042)).float().square().mean()
    loss.backward()
    assert model.hint_b.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not any(v.enabled for v in backbone.adapters)
