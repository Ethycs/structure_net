from dataclasses import FrozenInstanceError
import pytest
import torch

from experiments.structure_net.llama_q4_kernel import Tile,signed_half,OptimizedCore
from experiments.structure_net.llama_fp16_core_q4_tail import quantize,CoreResidualLinear


def test_bit_conversion_decodes_all_signed_nibbles_exactly():
    v=torch.arange(16,dtype=torch.int16)
    assert torch.equal(signed_half(v),((v^8)-8).half())
    assert torch.equal(signed_half(v)*torch.tensor(2**-24).half(),(((v^8)-8).float()*2**-24).half())


def test_tile_contract_is_immutable_and_rejects_invalid_layouts():
    with pytest.raises(ValueError):Tile(bk=32)
    with pytest.raises(ValueError):Tile(split=3)
    with pytest.raises(FrozenInstanceError):Tile().split=4


def test_installation_preserves_all_parameter_and_buffer_storage():
    p,s=quantize(torch.randn(64,64))
    original=CoreResidualLinear(torch.randn(64,64).half(),torch.randn(64,64).half(),p,s)
    optimized=OptimizedCore(original,"packed",Tile(),Tile(split=1))
    assert list(original.state_dict())==list(optimized.state_dict())
    assert all(v.data_ptr()==optimized.state_dict()[k].data_ptr() for k,v in original.state_dict().items())
    with pytest.raises(ValueError):optimized(torch.ones(1,64).half())
