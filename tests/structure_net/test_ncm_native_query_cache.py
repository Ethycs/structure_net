import copy
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_readout import frozen_hash


def test_native_cached_query_alignment_immutability_and_gradient_routing():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True, attention_dropout=0.)
    native = LlamaForCausalLM(cfg).eval()
    oracle = copy.deepcopy(native).requires_grad_(False)
    model = NativeQueryBackbone(native, rank=2)
    frozen = frozen_hash(model)
    prompt = torch.tensor([[0, 0, 1, 7, 8], [0, 1, 3, 4, 5]])
    mask = prompt.ne(0).long()
    previous = torch.tensor([[11, 12], [13, 14]])
    memory = model.source_cache(prompt, mask)
    assert memory.source_mask.shape[1] == prompt.shape[1]-1
    assert torch.equal(memory.anchor_token, prompt[:, -1:])
    saved = [[x.clone() for x in pair] for pair in memory.keys_values]
    full = torch.cat((prompt, previous), 1)
    fullmask = torch.cat((mask, torch.ones_like(previous)), 1)
    with torch.no_grad():
        expected = oracle(input_ids=full, attention_mask=fullmask,
            position_ids=(fullmask.cumsum(-1)-1).clamp_min(0)).logits[:, prompt.shape[1]-1:]
        query = model.predecessor_embeddings(memory, previous)
        actual = model.query_logits(memory, query)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(actual, model.query_logits(memory, query))
    for old, pair in zip(saved, memory.keys_values):
        assert all(torch.equal(a,b) for a,b in zip(old,pair))
    logits = model.query_logits(memory, model.predecessor_embeddings(memory, previous))
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.ones(6, dtype=torch.long)).backward()
    assert any(v.b.grad is not None and v.b.grad.abs().sum()>0 for v in model.adapters)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert frozen_hash(model) == frozen
    with pytest.raises(RuntimeError):
        with model.query_adapters(True): raise RuntimeError('intentional lifecycle exception')
    assert not any(v.enabled for v in model.adapters)
