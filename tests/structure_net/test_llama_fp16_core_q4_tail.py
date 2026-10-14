import pytest
import torch
from experiments.structure_net.llama_fp16_core_q4_tail import Config,pack_signed,unpack_signed,quantize,dequantize,Q4Linear,CoreResidualLinear


def test_every_signed_nibble_survives_packing():
    q=torch.arange(-8,8,dtype=torch.int8).repeat(3,1)
    p=pack_signed(q)
    assert p.numel()*2==q.numel()
    assert torch.equal(unpack_signed(p),q)


def test_group_scales_zero_and_range_roundtrip():
    weight=torch.zeros(3,128)
    weight[0,:64]=torch.linspace(-1,1,64)
    weight[0,64:]=torch.linspace(-10,10,64)
    p,s=quantize(weight)
    y=dequantize(p,s)
    assert s.shape==(2,3) and torch.isfinite(s).all() and (s>0).all()
    assert torch.equal(y[1:],weight[1:].half())
    assert (weight[0,:64]-y[0,:64]).abs().max()<=float(s[0,0])/2+.001
    assert (weight[0,64:]-y[0,64:]).abs().max()<=float(s[1,0])/2+.01


def test_invalid_precision_shape_and_nonfinite_weights():
    with pytest.raises(ValueError):Config(rank=1024)
    with pytest.raises(ValueError):quantize(torch.ones(3,65))
    with pytest.raises(ValueError):quantize(torch.full((3,64),float('nan')))
    with pytest.raises(ValueError):pack_signed(torch.full((2,64),9,dtype=torch.int8))
    with pytest.raises(ValueError):Q4Linear(torch.zeros(33,4,dtype=torch.uint8),torch.ones(1,4,dtype=torch.float16))
    p,s=quantize(torch.ones(3,64))
    with pytest.raises(ValueError,match='CUDA FP16'):Q4Linear(p,s)(torch.ones(1,64))


def test_packed_core_state_reload_does_not_retain_dense_tail():
    p,s=quantize(torch.randn(4,64))
    a=CoreResidualLinear(torch.randn(4,2).half(),torch.randn(2,64).half(),p,s)
    b=CoreResidualLinear(torch.zeros(4,2).half(),torch.zeros(2,64).half(),p.clone(),s.clone())
    b.load_state_dict(a.state_dict())
    assert set(a.state_dict())=={'core.left','core.right','tail.packed','tail.scales'}
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_serialized_tied_weight_bytes_survive_reload(tmp_path):
    from experiments.structure_net.run_llama_fp16_core_q4_tail import storage_bytes,cpu_state
    model=torch.nn.Module()
    model.a=torch.nn.Linear(64,4,bias=False,dtype=torch.float16)
    model.b=torch.nn.Linear(64,4,bias=False,dtype=torch.float16)
    model.b.weight=model.a.weight
    state=cpu_state(model)
    torch.save(state,tmp_path/'model.pt')
    restored=torch.load(tmp_path/'model.pt',weights_only=True)
    assert storage_bytes(restored)==4*64*2
    assert restored['a.weight'].data_ptr()==restored['b.weight'].data_ptr()
