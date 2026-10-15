import pytest
import torch
from experiments.structure_net.llama_q4_kernel_v2 import Tile,OptimizedCore
from experiments.structure_net.llama_fp16_core_q4_tail import quantize,CoreResidualLinear


def test_prefill_half_group_tiles_and_invalid_layouts():
    assert Tile(bk=32,split=1).bk==32
    with pytest.raises(ValueError):Tile(bk=16)


def test_revised_wrapper_preserves_packed_state_and_ties():
    p,s=quantize(torch.randn(64,64))
    old=CoreResidualLinear(torch.randn(64,64).half(),torch.randn(64,64).half(),p,s)
    new=OptimizedCore(old,'staged',Tile(),Tile(bk=32,split=1,staged=True))
    assert all(x.data_ptr()==new.state_dict()[n].data_ptr() for n,x in old.state_dict().items())
    assert set(new.state_dict())=={'core.left','core.right','tail.packed','tail.scales'}


def test_third_wrapper_keeps_exactly_the_same_persistent_state():
    from experiments.structure_net.llama_q4_kernel_v3 import OptimizedCore as ThirdCore
    p,s=quantize(torch.randn(64,64))
    old=CoreResidualLinear(torch.randn(64,64).half(),torch.randn(64,64).half(),p,s)
    new=ThirdCore(old,'packed',None,Tile(bk=32,split=1))
    assert all(x.data_ptr()==new.state_dict()[n].data_ptr() for n,x in old.state_dict().items())
