"""Separate channel and head reductions; test the full-rank equivalence control."""
from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net.jspace_first_block_ablation import build_first


def components():
    torch.manual_seed(341)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    native = Qwen2DecoderLayer(cfg, layer_idx=0).bfloat16().float().eval().requires_grad_(False)
    fit = dict(basis=torch.linalg.qr(torch.randn(8, 4)).Q, means=torch.randn(1, 8),
        rope_pairs=torch.tensor([[0]]), value_basis=torch.eye(4)[None, :, :2])
    repaired = {**fit, "basis":torch.linalg.qr(torch.randn(8, 4)).Q, "means":torch.randn(1, 8)}
    return native, fit, repaired, Qwen2RotaryEmbedding(config=cfg)


def test_full_rank_full_head_matches_actual_native_block_with_padding_and_causality():
    native, fit, _, rotary = components()
    control = build_first(deepcopy(native), fit, "full_rank_full_head")
    h = torch.randn(2, 7, 8); positions = torch.arange(7)[None].expand(2, -1)
    lengths = torch.tensor([4, 7])
    allowed = torch.arange(7)[None] < lengths[:, None]
    allowed = allowed[:, None, None] & torch.ones(7, 7, dtype=torch.bool).tril()[None, None]
    mask = torch.zeros(allowed.shape).masked_fill(~allowed, torch.finfo(torch.float32).min)
    kwargs = dict(position_ids=positions, cache_position=positions[0], attention_mask=mask,
        position_embeddings=rotary(h, positions))
    with torch.inference_mode():
        expected = native(h, **kwargs)[0]
        actual = control(h, **kwargs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert control.rank == control.width == 8 and control.head_dim == 4
    assert control.wrapped.metric is None and not any(p.requires_grad for p in control.parameters())


@pytest.mark.parametrize("variant,rank,head", [("full_rank_half_head",8,2), ("half_rank_full_head",4,4), ("mean_rank_full_head",4,4)])
def test_ablation_changes_only_requested_rank_and_head_geometry(variant, rank, head):
    native, fit, repaired, _ = components()
    block = build_first(native, fit, variant, mean_fit=repaired)
    assert block.rank == rank and block.head_dim == head and block.intermediate_size == 12
    assert block.nq == 2 and block.nkv == 1 and block.scale == .5
    assert block.wrapped.mode == "softmax" and block.wrapped.metric is None
    if variant == "mean_rank_full_head":
        torch.testing.assert_close(block.basis, repaired["basis"], atol=0, rtol=0)
        expected_anchor = repaired["means"] - repaired["means"] @ repaired["basis"] @ repaired["basis"].T
        torch.testing.assert_close(block.anchor, expected_anchor, atol=1e-6, rtol=1e-6)
    elif variant == "half_rank_full_head":
        torch.testing.assert_close(block.basis, fit["basis"], atol=0, rtol=0)


def test_repaired_variant_cannot_silently_reuse_original_basis():
    native, fit, _, _ = components()
    with pytest.raises(ValueError, match="requires its fitted"):
        build_first(native, fit, "mean_rank_full_head")
