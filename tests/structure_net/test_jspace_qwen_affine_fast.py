from copy import deepcopy

import pytest
import torch
from transformers import DynamicCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_affine_stack import AffineQwenStack
from experiments.structure_net.jspace_qwen_affine_fast import (
    FastAffineQwenBlock, FastAffineQwenStack, affine_qkv_post,
    affine_qkv_reference, gate_swiglu_features, swiglu_polynomial_reference,
)


def fixture(dtype=torch.float64, polynomial=True, index=0):
    torch.manual_seed(891 + index)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=2, vocab_size=32,
                         attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=index).to(dtype).eval().requires_grad_(False)
    basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :8].to(dtype)
    value = torch.stack([torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q[:, :2]
                         for _ in range(2)]).to(dtype)
    block = AffineQwenBlock(native, basis, torch.randn(8, 16, dtype=dtype) / 4,
        head_projection={"rope_pairs": torch.tensor([[0], [1]]), "value_basis": value})
    if polynomial:
        fit = {"anchors": [0, 2, 5, 8], "mean": torch.randn(28, dtype=dtype) / 5,
               "scale": torch.rand(28, dtype=dtype) + .7,
               "coefficients": {"2": torch.randn(28, 15, dtype=dtype) / 30}}
        block.compile_polynomial(fit, [1, 3, 4, 6, 9, 12])
    return block


def positions(length, dtype=torch.float64):
    angle = torch.arange(length, dtype=dtype)[:, None] * torch.tensor([.1, .3], dtype=dtype)[None]
    angle = torch.cat((angle, angle), -1)[None]
    return angle.cos(), angle.sin()


def test_cpu_qkv_reference_preserves_affine_rounding_and_position_selection():
    block = fixture(torch.float32).pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    raw = torch.randn(2, 4, block.qkv_weight.shape[0], dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])
    denominator = torch.rand(2, 4, 1) + .3
    shift = block.qkv_shift[ids].float()
    expected = (((raw.float() + shift) / denominator) + block.qkv_bias.float()).to(torch.bfloat16)
    actual = affine_qkv_post(raw, denominator, block, ids, use_triton=True)
    assert torch.equal(actual, expected)
    assert torch.equal(affine_qkv_reference(raw, shift, denominator, block.qkv_bias), expected)


@pytest.mark.parametrize("polynomial", [False, True])
def test_cpu_swiglu_reference_preserves_two_bf16_rounding_points_and_padding(polynomial):
    block = fixture(torch.float32, polynomial).pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    raw = torch.randn(1, 4, 2 * block.intermediate_size, dtype=torch.bfloat16) * 3
    ids = torch.arange(4)[None]
    denominator = torch.rand(1, 4, 1) + .3
    shift = block.gate_up_shift[ids].float()
    gate, up = ((raw.float() + shift) / denominator).to(torch.bfloat16).chunk(2, -1)
    hidden = torch.nn.functional.silu(gate) * up
    expected = torch.cat((hidden, block._polynomial_features(hidden).to(hidden.dtype)), -1) if polynomial else hidden
    expected = torch.nn.functional.pad(expected, (0, 128 - expected.shape[-1]))
    actual = gate_swiglu_features(raw, denominator, block, ids, padded_width=128, use_triton=True)
    assert torch.equal(actual, expected)
    assert torch.equal(swiglu_polynomial_reference(raw, shift, denominator, block, padded_width=128), expected)
    logical = block.intermediate_size + block.polynomial_feature_count
    assert torch.count_nonzero(actual[..., logical:]) == 0


def test_unpacked_cpu_fallback_is_exact_and_differentiable():
    base = fixture()
    reference = deepcopy(base)
    fast = FastAffineQwenBlock(base)
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = fast(x, position_embeddings=positions(5))[0]
    expected = reference(xr, position_embeddings=positions(5))[0]
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.equal(x.grad, xr.grad)
    for name, parameter in base.named_parameters():
        assert torch.equal(parameter.grad, dict(reference.named_parameters())[name].grad)
    assert fast.attention_type == base.attention_type and fast.layer_idx == base.layer_idx


@pytest.mark.parametrize("polynomial", [False, True])
@pytest.mark.parametrize("multiple", [16, 128])
def test_padding_only_matches_unpadded_packed_output_and_strict_reload(polynomial, multiple, tmp_path):
    reference = fixture(polynomial=polynomial).pack()
    fast = FastAffineQwenBlock.from_packed(deepcopy(reference), pad_to_multiple=multiple)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    expected = reference(x, position_embeddings=positions(5))[0]
    actual = fast(x, position_embeddings=positions(5))[0]
    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    logical = reference.intermediate_size + reference.polynomial_feature_count
    width = fast.original.mlp_output.shape[1]
    assert width % multiple == 0 and width >= logical
    assert torch.count_nonzero(fast.original.mlp_output[:, logical:]) == 0
    assert not fast.compact_parameters()
    path = tmp_path / "fast.pt"
    torch.save(fast.state_dict(), path)
    fresh = FastAffineQwenBlock.from_packed(deepcopy(reference), pad_to_multiple=multiple)
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(fresh(x, position_embeddings=positions(5))[0], actual)


def test_bf16_geometry_replaces_buffers_supports_joint_training_and_preserves_pack_arithmetic():
    block = fixture(torch.float32)
    manual = deepcopy(block)
    manual.basis = manual.basis.to(torch.bfloat16)
    fast = FastAffineQwenBlock(block, geometry_mode="bf16", pad_to_multiple=128)
    assert fast.original.basis.dtype == torch.bfloat16
    assert fast.original.anchor.dtype == fast.original.anchor_energy.dtype == torch.float32
    assert sum("basis" in name for name, _ in fast.named_buffers()) == 2  # basis and value_basis, no duplicate
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fast(x, position_embeddings=positions(5, torch.bfloat16))[0]
        expected = manual(x, position_embeddings=positions(5, torch.bfloat16))[0]
    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.compact_parameters())
    bias = fast.original.polynomial_bias.detach().clone()
    fast.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    assert fast.original.basis.dtype == torch.bfloat16
    assert fast.original.polynomial_bias.dtype == torch.float32
    assert torch.equal(fast.original.polynomial_bias, bias)
    assert fast.original.qkv_shift.dtype == fast.original.gate_up_shift.dtype == torch.bfloat16
    assert fast.original.mlp_output.shape[1] == 128
    assert torch.isfinite(fast(x, position_embeddings=positions(5, torch.bfloat16))[0]).all()


def test_fast_stack_replaces_bridge_buffers_and_matches_explicit_quantized_structure(tmp_path):
    stack = AffineQwenStack(fixture(torch.float32, index=0), fixture(torch.float32, index=1))
    manual = deepcopy(stack)
    for block in manual.blocks:
        block.basis = block.basis.to(torch.bfloat16)
    manual.bridge_weight = manual.bridge_weight.to(torch.bfloat16)
    fast = FastAffineQwenStack(stack, geometry_mode="bf16", pad_to_multiple=128)
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    args = {"position_embeddings": positions(5, torch.bfloat16)}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = fast(x, **args)[0]
        reference = manual(x, **args)[0]
    assert torch.equal(output, reference)
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.compact_parameters())
    assert fast.original.bridge_weight.dtype == torch.bfloat16
    assert fast.original.bridge_shift.dtype == torch.float32
    assert len([name for name, _ in fast.named_buffers() if "bridge_weight" in name]) == 1
    fresh = deepcopy(fast)
    fast.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    actual = fast(x, **args)[0]
    path = tmp_path / "stack.pt"
    torch.save(fast.state_dict(), path)
    fresh.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(actual, fresh(x, **args)[0])


def test_padded_stack_dynamic_decode_matches_prefill():
    stack = AffineQwenStack(fixture(index=0), fixture(index=1)).pack()
    fast = FastAffineQwenStack.from_packed(stack, pad_to_multiple=128)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    cos, sin = positions(5)
    expected = fast(x, position_embeddings=(cos, sin))[0]
    cache, rows = DynamicCache(), []
    for i in range(5):
        rows.append(fast(x[:, i:i+1], position_embeddings=(cos[:, i:i+1], sin[:, i:i+1]),
                         cache_position=torch.tensor([i]), past_key_value=cache, use_cache=True)[0])
    assert torch.allclose(torch.cat(rows, 1), expected, atol=3e-12, rtol=3e-12)


def test_invalid_configuration_and_capture_are_explicit():
    with pytest.raises(ValueError, match="geometry_mode"):
        FastAffineQwenBlock(fixture(), geometry_mode="tf32")
    with pytest.raises(ValueError, match="requires a packed"):
        FastAffineQwenBlock.from_packed(fixture())
    with pytest.raises(ValueError, match="FP32 anchors"):
        FastAffineQwenBlock(fixture(), geometry_mode="bf16")
    fast = FastAffineQwenBlock.from_packed(fixture().pack(), pad_to_multiple=128)
    fast.original.set_capture_mlp(True)
    with pytest.raises(ValueError, match="Disable diagnostic"):
        fast(torch.ones(1, 2, 16, dtype=torch.float64), position_embeddings=positions(2))
