from types import SimpleNamespace
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.structure_net.train_ncm_native_query_neighborhood import Config,streams,batch_loss
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_concepts import NativeQueryConcepts
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix,native_teacher_support
from experiments.structure_net.ncm_native_concepts import ConceptConfig,encode


def test_word_native_center_gradients_and_full_noisy_state_never_reads_gold_words():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    backbone = NativeQueryBackbone(LlamaForCausalLM(cfg).eval(),rank=2)
    books = [torch.randn(8,24)*.02 for _ in range(2)]
    for book in books: book[0].zero_()
    net = NativeQueryConcepts(backbone,books,levels=(1,2),canvas=4,hint_rank=2).eval()
    prompt = torch.tensor([[0,1,3,4],[1,5,6,7]])
    memory = backbone.source_cache(prompt,prompt.ne(0).long())
    targets = torch.tensor([[8,9,10,2],[11,12,2,-100]])
    indices,probabilities = native_teacher_support(backbone,memory,targets,2,5)
    embedding = backbone.language.model.embed_tokens(targets.masked_fill(targets.eq(-100),2)).float()
    clean = encode(embedding,net.books,ConceptConfig(canvas=4,levels=(1,2),entries=8))[0]
    examples = [{"prefix":cpu_prefix(memory,i),"targets":targets[i],"teacher_indices":indices[i],
        "teacher_probabilities":probabilities[i],"clean_codes":[c[i] for c in clean],
        "kind":"language","window_eos_injected":True} for i in range(2)]
    tokenizer = SimpleNamespace(eos_token_id=2,all_special_ids=[0,1,2])
    config = Config(vocabulary_chunk=7)
    loss,obs = batch_loss(net,examples,2,streams(),config,tokenizer,2000,2048,True)
    assert obs["native_center"] and abs(loss.item()-obs["unscaled_loss"]*10)<1e-5
    loss.backward()
    assert net.hint_b.grad.abs().sum()>0 and all(p.grad is None for p in net.parameters() if not p.requires_grad)
    captured = []
    hook = backbone.language.lm_head.register_forward_hook(lambda module,args,result:captured.append(result.detach().clone()))
    try:
        _,first_obs = batch_loss(net,examples,2,streams(),config,tokenizer,2000,2048,False)
        first = captured[-1]
        changed = [{**e,"targets":e["targets"].clone()} for e in examples]
        for e in changed:e["targets"][:2] = torch.tensor([20,21])
        _,second_obs = batch_loss(net,changed,2,streams(),config,tokenizer,2000,2048,False)
        second = captured[-1]
    finally:hook.remove()
    assert first_obs["noise_window"]==128 and second_obs["actual_history"]
    assert torch.equal(first,second)
