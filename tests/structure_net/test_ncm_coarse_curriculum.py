from types import SimpleNamespace

import torch

from experiments.structure_net.ncm_coarse_curriculum import GlobalResidual, enable_coarse_ar


def test_zero_residual_preserves_native_context_but_receives_gradient():
    bridge = GlobalResidual(7, 3)
    native = torch.randn(2, 5, 3)
    last = torch.randn(2, 7)
    pad = torch.tensor([[False, False, False, True, True], [False, False, True, True, True]])
    output = bridge(native, last, pad)
    assert torch.equal(output, native)
    output.sum().backward()
    assert bridge.proj.weight.grad.norm() > 0
    with torch.no_grad():
        bridge.proj.bias.fill_(1.)
    changed = bridge(native, last, pad)
    assert torch.equal(changed[pad], native[pad])
    assert torch.equal(changed[~pad], native[~pad]+1.)


def test_coarse_ar_replaces_loaded_parallel_mask_without_changing_other_levels():
    original = torch.zeros(1, 1, 15, 15)
    net = SimpleNamespace(query_level0=True, use_flex_attention=False, use_rope=True,
        prefix_len=5, levels=[4, 6], attn_mask_for_masking=original.clone(), config=SimpleNamespace())
    enable_coarse_ar(net)
    assert net.l0_ar and net.config.l0_autoregressive
    assert torch.isneginf(net.attn_mask_for_masking[0, 0, 5, 6:9]).all()
    assert (net.attn_mask_for_masking[0, 0, 8, 5:9] == 0).all()
    assert torch.equal(net.attn_mask_for_masking[:, :, :5], original[:, :, :5])
    assert torch.equal(net.attn_mask_for_masking[:, :, 9:], original[:, :, 9:])
