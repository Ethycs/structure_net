from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_seam_diagnostic as diagnostic
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def tiny():
    torch.manual_seed(5131)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=16,
        max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = 'eager'
    native = Qwen2ForCausalLM(config).float().eval().requires_grad_(False)
    blocks = [AffineQwenBlock(native.model.layers[i],
        torch.linalg.qr(torch.randn(8,8)).Q[:,:4], torch.randn(1,8)/10,
        trainable=False) for i in (1,2)]
    stack = CompactQwenStack(blocks, geometry_mode='fp32', pad_to_multiple=0).eval().requires_grad_(False)
    packed = deepcopy(stack).pack(dtype=torch.float32, buffer_dtype=torch.float32,
        strip_unused=True).eval().requires_grad_(False)
    lengths = [3,4,6,5]; hidden = torch.randn(4,6,8)
    positions = torch.arange(6)[None].expand(4,-1)
    allowed = torch.arange(6)[None,None,:] <= torch.arange(6)[None,:,None]
    allowed = allowed & (torch.arange(6)[None,None,:] < torch.tensor(lengths)[:,None,None])
    mask = torch.zeros(4,1,6,6).masked_fill(~allowed[:,None],torch.finfo(torch.float32).min)
    loss_positions = torch.tensor([[0,(n-2)//2,n-2,n-1] for n in lengths])
    batch = dict(hidden=hidden, lengths=lengths, position_ids=positions,
        attention_mask=mask, loss_positions=loss_positions)
    return native,stack,packed,batch


def tensors(module):
    return {name: value.detach().clone() for name,value in
        [*module.named_parameters(),*module.named_buffers()]}


def test_native_batch_singleton_and_padding_preserve_valid_causal_outputs():
    native,_,_,batch=tiny(); block=native.model.layers[0]; rotary=native.model.rotary_emb
    before=tensors(block); original=batch['hidden'].clone()
    actual=diagnostic.native_variants(block,rotary,batch)
    assert set(actual)=={'batch4','singleton','singleton_padded'}
    for row,length in enumerate(batch['lengths']):
        h=batch['hidden'][row:row+1,:length];pos=batch['position_ids'][row:row+1,:length]
        with torch.inference_mode():
            expected=block(h,attention_mask=batch['attention_mask'][row:row+1,:,:length,:length],
                position_ids=pos,cache_position=pos[0],position_embeddings=rotary(h,pos),use_cache=False)[0]
        for value in actual.values():
            assert value.shape==batch['hidden'].shape
            torch.testing.assert_close(value[row,:length],expected[0],atol=2e-6,rtol=2e-6)
            assert torch.count_nonzero(value[row,length:])==0
    # Future-token changes cannot affect any earlier valid query.
    changed=dict(batch,hidden=batch['hidden'].clone());changed['hidden'][:,-1].add_(10)
    later=diagnostic.native_variants(block,rotary,changed)['batch4']
    torch.testing.assert_close(later[:,:2],actual['batch4'][:,:2],atol=0,rtol=0)
    assert torch.equal(batch['hidden'],original)
    assert all(torch.equal(v,dict(block.named_parameters())[k]) if k in dict(block.named_parameters())
        else torch.equal(v,dict(block.named_buffers())[k]) for k,v in before.items())


def test_boundary_metrics_ignore_padding_but_detect_answer_row_error():
    reference=torch.ones(2,4,3);actual=reference.clone();actual[0,2:]=float('nan')
    assert diagnostic.boundary_metrics(actual,reference,[2,4])['exact']
    actual[0,1].add_(1)
    result=diagnostic.boundary_metrics(actual,reference,[2,4])
    assert not result['exact'] and result['max_abs']==1
    assert result['cases'][0]['relative_rms']==pytest.approx(2**-.5)
    assert result['cases'][0]['answer_relative_rms']==1
    assert result['case_mean_relative_rms']==pytest.approx(2**-.5/2)
    actual[1,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite boundary'):
        diagnostic.boundary_metrics(actual,reference,[2,4])
    with pytest.raises(ValueError,match='Invalid valid-token'):
        diagnostic.boundary_metrics(reference,reference,[0,4])


def test_actual_compact_parent_modes_use_same_state_and_full_vocabulary():
    native,stack,packed,batch=tiny(); norm=native.model.norm;E=native.lm_head.weight;rotary=native.model.rotary_emb
    states=[tensors(m) for m in (stack,packed,norm,rotary)];E_before=E.clone()
    result={mode:diagnostic.suffix_logits(packed if mode.startswith('packed') else stack,
        norm,E,rotary,batch,batch['hidden'],mode) for mode in diagnostic.MODES}
    assert all(value.shape==(4,4,16) and not value.requires_grad for value in result.values())
    assert torch.equal(result['packed_inference'],result['packed_torch'])
    assert torch.equal(result['unpacked_eval'],result['unpacked_grad'])
    torch.testing.assert_close(result['packed_torch'],result['unpacked_eval'],atol=2e-6,rtol=2e-5)
    with torch.inference_mode():
        h=stack(batch['hidden'],attention_mask=batch['attention_mask'],position_ids=batch['position_ids'],
            cache_position=batch['position_ids'][0],position_embeddings=rotary(batch['hidden'],batch['position_ids']))[0]
        expected=F.linear(norm(h[torch.arange(4)[:,None],batch['loss_positions']]),E)
    assert torch.equal(result['unpacked_eval'],expected)
    assert packed.enabled and stack.enabled and torch.equal(E,E_before)
    for module,before in zip((stack,packed,norm,rotary),states):
        assert all(torch.equal(v,tensors(module)[k]) for k,v in before.items())
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert batch['hidden'].grad is None and not batch['hidden'].requires_grad


def test_mode_switch_restored_even_if_forward_fails_and_frozen_contract_enforced():
    native,stack,packed,batch=tiny();observed=[]
    def stop(module,args):
        observed.append((module.enabled,torch.is_grad_enabled()))
        raise RuntimeError('probe failure')
    handle=packed.register_forward_pre_hook(stop)
    try:
        with pytest.raises(RuntimeError,match='probe failure'):
            diagnostic.suffix_logits(packed,native.model.norm,native.lm_head.weight,native.model.rotary_emb,
                batch,batch['hidden'],'packed_torch')
    finally:handle.remove()
    assert observed==[(False,False)] and packed.enabled
    with pytest.raises(ValueError,match='packed/unpacked'):
        diagnostic.suffix_logits(stack,native.model.norm,native.lm_head.weight,native.model.rotary_emb,
            batch,batch['hidden'],'packed_torch')
    native.model.norm.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):
        diagnostic.suffix_logits(stack,native.model.norm,native.lm_head.weight,native.model.rotary_emb,
            batch,batch['hidden'],'unpacked_eval')


def test_choice_rows_and_full_vocabulary_kl_use_all_four_positions():
    torch.manual_seed(912);expected=torch.randn(2,4,16);actual=expected.clone()
    actual[0,0,15]+=2;actual[1,-1,3]+=4
    items=[dict(identity=f'case{i}',split='development',family='f',answer=1,
        choices=['a','b'] if i==0 else ['a','b','c','d']) for i in range(2)]
    labels=[0,1,2,3];result=diagnostic.logits_comparison(actual,expected,items,labels)
    reference=(expected/2).log_softmax(-1);student=(actual/2).log_softmax(-1)
    per_case=(reference.exp()*(reference-student)).sum(-1).mean(-1)
    assert result['full_vocab_kl_t2_by_case']==pytest.approx(per_case.tolist())
    assert result['full_vocab_kl_t2']==pytest.approx(float(per_case.mean()))
    assert result['full_vocab_max_abs']==4 and result['count']==2
    for i,row in enumerate(result['records']):
        values=actual[i,-1,labels[:len(items[i]['choices'])]]
        assert row['logits']==values.tolist() and row['prediction']==int(values.argmax())
    assert per_case[0]>0  # Non-answer, non-label logit changes are not discarded.
    items[0]['split']='evaluation'
    with pytest.raises(ValueError,match='development'):
        diagnostic.logits_comparison(actual,expected,items,labels)


def test_run_cannot_initialize_cuda_before_reservation(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    calls=[]
    monkeypatch.setattr(diagnostic,'verify',lambda root:dict(campaign=str(tmp_path)))
    monkeypatch.setattr(diagnostic.source,'verify_stage',lambda root,cut:({},{}))
    def denied(*args,**kwargs):raise RuntimeError('reservation denied')
    monkeypatch.setattr(jspace_full_lease,'wait_ready',denied)
    monkeypatch.setattr(torch.cuda,'init',lambda:calls.append('cuda'))
    monkeypatch.setattr(diagnostic.source,'cap_allocator',lambda *args:calls.append('allocator'))
    with pytest.raises(RuntimeError,match='reservation denied'):diagnostic.run(tmp_path/'new')
    assert calls==[] and not (tmp_path/'new/run').exists()
