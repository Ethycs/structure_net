"""CUDA parity/causality contracts for actual sparse execution."""
import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net.moba_sparse_kernel import sparse_attention
from experiments.structure_net.tinyllm_moba_distillation import attention


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA kernel required")


def test_training_cannot_silently_use_inference_kernel():
    x = torch.zeros(1, 1, 8, 16, requires_grad=True)
    with pytest.raises(RuntimeError, match="no backward"):
        sparse_attention(x, x, x)


@CUDA
@pytest.mark.parametrize("heads,kvheads,length,dim", [(1,1,17,32), (2,2,63,64), (2,2,97,64), (8,2,137,128), (32,8,256,128)])
def test_fp32_matches_reference_with_tails_and_gqa(heads, kvheads, length, dim):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(851)
    q = torch.randn(2,length,heads,dim,device="cuda").transpose(1,2)
    k,v = [torch.randn(2,length,kvheads,dim,device="cuda").transpose(1,2) for _ in range(2)]
    expected = attention(q,k.repeat_interleave(heads//kvheads,1),v.repeat_interleave(heads//kvheads,1),32,2)[0]
    actual = sparse_attention(q,k,v)
    torch.testing.assert_close(actual,expected,atol=5e-6,rtol=5e-5)


@CUDA
def test_no_future_information_changes_past_outputs():
    torch.manual_seed(151)
    q,k,v = [torch.randn(1,2,137,64,device="cuda") for _ in range(3)]
    before,routes_before = sparse_attention(q,k,v,return_routes=True)
    k[...,75:,:] += 100
    v[...,75:,:] -= 100
    after,routes_after = sparse_attention(q,k,v,return_routes=True)
    torch.testing.assert_close(before[...,:75,:],after[...,:75,:],atol=0,rtol=0)
    assert torch.equal(routes_before[...,:75],routes_after[...,:75])


@CUDA
def test_tied_routing_chooses_first_eligible_remote_block():
    q = torch.zeros(1,2,131,64,device="cuda")
    k,v = [torch.randn_like(q) for _ in range(2)]
    actual,routes = sparse_attention(q,k,v,return_routes=True)
    assert (routes[...,:32] == -1).all()
    assert (routes[...,32:] == 0).all()
    torch.testing.assert_close(actual,attention(q,k,v,32,2)[0],atol=1e-6,rtol=1e-5)


@CUDA
def test_full_coverage_agrees_with_optimized_full_attention():
    q,k,v = [torch.randn(1,2,64,64,device="cuda") for _ in range(3)]
    torch.testing.assert_close(sparse_attention(q,k,v),F.scaled_dot_product_attention(q,k,v,is_causal=True),atol=3e-6,rtol=3e-5)
