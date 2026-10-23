import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix,pack_prefix,native_teacher_support,sparse_word_loss


def test_sparse_mixture_matches_full_cross_entropy_values_and_gradients():
    torch.manual_seed(42)
    logits = torch.randn(2,3,5,requires_grad=True)
    target = torch.tensor([[1,2,-100],[0,3,4]])
    indices = torch.arange(5).expand(2,3,5)
    probability = torch.randn(2,3,5).softmax(-1)
    actual = sparse_word_loss(logits,target,indices,probability,.3,[False,True])
    logp = logits.log_softmax(-1)
    lam = torch.full((2,3),.3); lam[1,2] = 0.
    hard = -logp.gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    soft = -(probability*logp).sum(-1)
    expected = ((1-lam)*hard+lam*soft)[target.ne(-100)].mean()
    torch.testing.assert_close(actual,expected)
    a = torch.autograd.grad(actual,logits,retain_graph=True)[0]
    b = torch.autograd.grad(expected,logits)[0]
    torch.testing.assert_close(a,b)
    pure = sparse_word_loss(logits,target,indices,probability,0.,[False,False])
    torch.testing.assert_close(pure,torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten()))


def test_frozen_teacher_bypasses_student_adapters_and_cpu_cache_roundtrip():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    body = LlamaForCausalLM(cfg).eval()
    net = NativeQueryBackbone(body,rank=2)
    with torch.no_grad():
        for layer in net.adapters: layer.b.normal_(std=.1)
    prompt = torch.tensor([[0,1,3,4],[1,5,6,7]]); mask = prompt.ne(0).long()
    target = torch.tensor([[8,9,-100],[10,11,12]])
    memory = net.source_cache(prompt,mask)
    stored = [cpu_prefix(memory,i) for i in range(2)]
    packed = pack_prefix(stored,"cpu")
    assert torch.equal(memory.source_mask,packed.source_mask) and torch.equal(memory.anchor_token,packed.anchor_token)
    assert all(torch.equal(a,b) for x,y in zip(memory.keys_values,packed.keys_values) for a,b in zip(x,y))
    indices,probs = native_teacher_support(net,packed,target,0,5)
    previous = target[:,:-1].masked_fill(target[:,:-1].eq(-100),0)
    full = torch.cat((prompt,previous),1)
    fullmask = torch.cat((mask,torch.ones_like(previous)),1)
    with torch.no_grad(),net.query_adapters(False):
        expected = body(input_ids=full,attention_mask=fullmask,position_ids=(fullmask.cumsum(-1)-1).clamp_min(0)).logits[:,-3:]
        value,gold = expected.topk(5,-1)
    assert torch.equal(indices,gold)
    torch.testing.assert_close(probs,value.softmax(-1),rtol=1e-5,atol=1e-6)
    assert not probs.requires_grad and not any(v.enabled for v in net.adapters)
    with torch.no_grad():
        student = net.query_logits(packed,net.predecessor_embeddings(packed,previous))
        assert (student-expected).abs().max() > 1e-3
