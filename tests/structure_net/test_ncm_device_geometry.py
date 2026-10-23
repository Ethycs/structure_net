from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net.ncm_llama import vendor_classes
from experiments.structure_net.ncm_prefix_geometry import set_source_frame as reference
from experiments.structure_net.ncm_device_geometry import set_source_frame


class Geometry(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(2,device=device))
        self.config=SimpleNamespace(attn_pattern="block_diagonal",n_embd=32,n_head=4,prefix_len=8)
        self.use_rope=True
        self.use_flex_attention=False
        self.levels=[4,8,16]
        self.rope_base=100000


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_geometry_matches_released_cpu_calculation_and_reuses_frames(device):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA required")
    vendor_classes()
    expected=Geometry("cpu")
    actual=Geometry(device)
    before=actual.weight.detach().clone()
    for frame in [8,12,8]:
        reference(expected,frame,maximum=16)
        set_source_frame(actual,frame,maximum=16)
        assert actual.prefix_len==frame and actual.config.prefix_len==frame
        for name,value in expected._ncm_source_geometry[(frame,"cpu",torch.float32)].items():
            received=getattr(actual,name)
            assert received.device.type==device
            assert torch.equal(value,received.cpu())
    assert torch.equal(before,actual.weight)
