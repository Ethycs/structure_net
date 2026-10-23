import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.structure_net.ncm_native_readout import NativeReadout, frozen_hash, restore_trainable, trainable_state


def model():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    config._attn_implementation = "sdpa"
    return NativeReadout(LlamaForCausalLM(config),rank=2)


def test_native_readout_causality_padding_cache_and_gradient_routing():
    head = model().eval()
    features = torch.randn(2,5,3,24)
    mask = torch.tensor([[0,0,1,1,1],[1,1,1,1,1]])
    words = torch.tensor([[1,2,3,4],[1,5,6,7]])
    before = frozen_hash(head)
    full,_ = head.teacher_logits(features,mask,words)
    other = words.clone(); other[:,-1] = 12
    changed,_ = head.teacher_logits(features,mask,other)
    torch.testing.assert_close(full[:,:-1],changed[:,:-1],rtol=0,atol=0)
    padded = features.clone(); padded[0,:2] = 1000
    altered,_ = head.teacher_logits(padded,mask,words)
    torch.testing.assert_close(full,altered,rtol=0,atol=0)
    first,past = head.teacher_logits(features,mask,words[:,:1],use_cache=True)
    cached = [first]; cmask = torch.cat((mask,torch.ones(2,1,dtype=torch.long)),1)
    for i in range(1,4):
        value,past,cmask = head.cached_step(words[:,i:i+1],past,cmask); cached.append(value)
    torch.testing.assert_close(full,torch.cat(cached,1),rtol=1e-5,atol=1e-6)
    loss = torch.nn.functional.cross_entropy(full.flatten(0,1),words.flatten())
    loss.backward()
    assert head.fuser.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in head.named_parameters() if n.endswith(".b"))
    assert all(p.grad is None for p in head.parameters() if not p.requires_grad)
    assert frozen_hash(head) == before
    assert head.language.lm_head.weight.data_ptr() == head.language.model.embed_tokens.weight.data_ptr()


def test_native_readout_only_restores_its_own_trainable_interface():
    head = model().eval(); before = frozen_hash(head); state = trainable_state(head)
    with torch.no_grad(): head.fuser.bias.add_(1)
    restore_trainable(head,state)
    assert all(torch.equal(p,dict(head.named_parameters())[n]) for n,p in state.items())
    assert frozen_hash(head) == before
