from dataclasses import fields
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.structure_net.ncm_intact_memory import IntactMemoryReadout,NativeMemoryCorrection,passage_mask
from experiments.structure_net.ncm_frozen_body import module_hash


def setup():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.,pad_token_id=0)
    native = LlamaForCausalLM(cfg).eval()
    correction = NativeMemoryCorrection(24,32,memory_width=16,heads=2)
    wrapped = IntactMemoryReadout(native,correction)
    memory = correction.prepare(torch.randn(2,5,3,32),torch.tensor([[0,1,1,1,1],[1,1,1,1,1]]),torch.randn(2,8,32))
    return wrapped,memory


def test_zero_correction_keeps_full_native_logits_and_greedy_output_exact():
    wrapped,memory = setup(); native = wrapped.native
    ids = torch.tensor([[1,4,5],[1,8,9]]); mask = torch.ones_like(ids)
    before = module_hash(native)
    plain = native(input_ids=ids,attention_mask=mask).logits
    expected = native.generate(ids,attention_mask=mask,max_new_tokens=4,do_sample=False,pad_token_id=0)
    values = lambda: [getattr(memory,field.name) for field in fields(memory)]
    copies = [v.clone() if v is not None else None for v in values()]
    with wrapped.conditioned(memory):
        actual = native(input_ids=ids,attention_mask=mask).logits
        generated = native.generate(ids,attention_mask=mask,max_new_tokens=4,do_sample=False,pad_token_id=0)
    assert torch.equal(plain,actual) and torch.equal(expected,generated)
    assert module_hash(native)==before
    assert all(v is None or torch.equal(v,c) for v,c in zip(values(),copies))
    loss = None
    with wrapped.conditioned(memory):
        logits = native(input_ids=ids,attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.full((6,),3,dtype=torch.long))
    loss.backward()
    assert wrapped.correction.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in native.parameters())
    assert not wrapped._active and not native.lm_head._forward_pre_hooks


def test_bounded_correction_and_hook_cleanup_after_error():
    wrapped,memory = setup()
    with torch.no_grad(): wrapped.correction.output.weight.fill_(100.)
    hidden = torch.randn(2,4,24)
    changed = wrapped.correction(hidden,memory)
    assert ((changed-hidden).norm(dim=-1)<=hidden.norm(dim=-1)*.1+1e-6).all()
    with pytest.raises(ValueError):
        with wrapped.conditioned(memory): raise ValueError('intentional lifecycle exception')
    assert not wrapped._active and not wrapped.native.lm_head._forward_pre_hooks


def test_passage_mask_excludes_instructions_delimiters_and_padding():
    class Characters:
        def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):
            text = 'SYSTEM: instructions\n'+messages[1]['content']+'\nASSISTANT:'
            return [ord(c) for c in text] if tokenize else text
        def __call__(self,text,**kwargs):
            return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
    mask = passage_mask(Characters(),[{'source':'only these words'}],padded_length=100)
    assert int(mask.sum())==len('only these words') and not mask[0,0]
