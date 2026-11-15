from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net import jspace_3b_hidden_feature_diagnostic as diagnostic
from experiments.structure_net import jspace_qwen_compact_stack as compact
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock


def tiny(dtype=torch.float64, geometry='fp32'):
    torch.manual_seed(8472)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=36,
        num_attention_heads=2, num_key_value_heads=1, attention_dropout=0., max_position_embeddings=16)
    config._attn_implementation = 'eager'
    blocks = []
    for layer in (33,34,35):
        native = Qwen2DecoderLayer(config,layer).to(dtype).eval().requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(8,8,dtype=torch.float64)).Q[:,:4].to(dtype)
        blocks.append(AffineQwenBlock(native,basis,torch.randn(6,8,dtype=dtype)/3,trainable=False))
    stack = compact.CompactQwenStack(blocks,geometry_mode=geometry,pad_to_multiple=0).eval().requires_grad_(False)
    stack.pack(dtype=dtype,buffer_dtype=dtype,strip_unused=True)
    lengths=[3,5]; x=torch.randn(2,5,8,dtype=dtype)
    ids=torch.arange(5)[None].expand(2,-1)
    angles=torch.arange(5,dtype=dtype)[:,None]*torch.tensor([.1,.2],dtype=dtype)[None]
    angles=torch.cat((angles,angles),-1)[None]
    mask=torch.zeros(2,1,5,5,dtype=dtype)
    valid=(torch.arange(5)[None,:]<torch.tensor(lengths)[:,None])
    allowed=(ids[:,None,:]<=ids[:,:,None]) & valid[:,None,:]
    mask.masked_fill_(~allowed[:,None],torch.finfo(dtype).min)
    return stack,x,lengths,dict(position_ids=ids,position_embeddings=(angles.cos(),angles.sin()),attention_mask=mask)


def test_real_stack_trace_preserves_output_and_all_state_with_stripped_geometry():
    stack,x,lengths,kwargs=tiny()
    before=deepcopy(stack.state_dict()); original=compact._run_reduced
    with torch.inference_mode(): expected=stack(x,**kwargs,use_cache=False)[0]
    trace=diagnostic.trace_coordinates(stack,x,**kwargs)
    assert compact._run_reduced is original
    assert set(trace['coordinates'])=={33,34,35}
    torch.testing.assert_close(trace['output'],expected,rtol=0,atol=0)
    actual_exit=diagnostic.reconstruct_final(stack,trace['coordinates'][35],kwargs['position_ids'])
    torch.testing.assert_close(actual_exit,trace['output'],rtol=0,atol=0)
    for name,value in stack.state_dict().items():
        if torch.is_tensor(value):assert torch.equal(value,before[name])
        else:assert value==before[name]
    assert all(p.grad is None and not p.requires_grad for p in stack.parameters())


def test_packed_and_differentiable_floor_does_not_mutate_or_leave_gradients():
    stack,x,lengths,kwargs=tiny()
    with torch.inference_mode():
        floor=diagnostic.packed_arithmetic_floor(stack,x,lengths,**kwargs)
    assert floor['output']['max_abs']==0
    assert all(x['max_abs']==0 for x in floor['by_layer'].values())
    assert all(not x['fuse_qkv'] and not x['fuse_mlp'] for x in floor['differentiable_execution'].values())
    assert all(p.grad is None and not p.requires_grad for p in stack.parameters())
    assert x.grad is None


def test_observer_restores_global_on_executor_failure_and_can_run_again(monkeypatch):
    stack,x,_,kwargs=tiny(); original=compact._run_reduced
    def broken(*args,**kw):raise RuntimeError('deliberate failure')
    monkeypatch.setattr(compact,'_run_reduced',broken)
    with pytest.raises(RuntimeError,match='deliberate failure'):
        diagnostic.trace_coordinates(stack,x,**kwargs)
    assert compact._run_reduced is broken
    monkeypatch.setattr(compact,'_run_reduced',original)
    assert 35 in diagnostic.trace_coordinates(stack,x,**kwargs)['coordinates']


def test_native_projection_uses_source_convention_not_assumed_exact_inverse():
    stack,x,_,kwargs=tiny(torch.float32,geometry='bf16')
    # Match production BF16 activation/parameter arithmetic with FP32 anchors.
    for p in stack.parameters():p.data=p.data.bfloat16()
    x=x.bfloat16()
    z=diagnostic.native_final_coordinates(stack,x)
    expected=torch.nn.functional.linear(x,stack.blocks[-1].basis.T)
    assert torch.equal(z,expected)
    synthetic=torch.randn(2,5,4).bfloat16()
    reconstructed=diagnostic.reconstruct_final(stack,synthetic,kwargs['position_ids'])
    projected=diagnostic.native_final_coordinates(stack,reconstructed)
    assert not torch.equal(projected,synthetic)  # BF16 geometry/rounding is not an exact inverse.


def test_orthonormal_float64_exit_projection_recovers_direct_coordinates():
    stack,_,_,kwargs=tiny();z=torch.randn(2,5,4,dtype=torch.float64)
    h=diagnostic.reconstruct_final(stack,z,kwargs['position_ids'])
    torch.testing.assert_close(diagnostic.native_final_coordinates(stack,h),z,atol=2e-14,rtol=2e-14)


def test_metrics_equal_weight_cases_ignore_padding_and_use_last_valid_answer():
    reference=torch.ones(2,3,2);actual=reference.clone()
    actual[0,0]+=1;actual[0,1:]=float('nan');actual[1]+=2
    result=diagnostic.feature_metrics(actual,reference,[1,3])
    assert result['mse']==2.5  # (one-token case MSE1 + three-token case MSE4)/2.
    assert result['relative_rmse']==pytest.approx(2.5**.5)
    assert result['answer_mse']==2.5 and result['valid_tokens']==4
    with pytest.raises(FloatingPointError):diagnostic.feature_metrics(actual,reference,[2,3])


def test_guard_and_frozen_state_requirements_precede_forward(monkeypatch):
    stack,x,_,kwargs=tiny()
    monkeypatch.setattr(stack,'forward',lambda *_a,**_k:pytest.fail('forward reached'))
    def denied():raise PermissionError('no reservation')
    with pytest.raises(PermissionError):diagnostic.trace_coordinates(stack,x,guard=denied,**kwargs)
    next(stack.parameters()).requires_grad_(True)
    with pytest.raises(ValueError,match='frozen eval'):diagnostic.trace_coordinates(stack,x,**kwargs)


def test_requested_absolute_layers_are_explicit_unique_and_complete():
    stack,x,_,kwargs=tiny()
    assert set(diagnostic.trace_coordinates(stack,x,layers=[34,35],**kwargs)['coordinates'])=={34,35}
    for invalid in ([35,35],[0],[],[True]):
        with pytest.raises(ValueError):diagnostic.trace_coordinates(stack,x,layers=invalid,**kwargs)
