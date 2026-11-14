from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import DynamicCache, StaticCache, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_affine_stack import AffineQwenStack


def fixture(dtype=torch.float64, projection=True, polynomial=True, position_counts=(8, 8), indices=(34, 35)):
    torch.manual_seed(2951)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=max(indices) + 1,
                         vocab_size=32, attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    blocks, originals = [], []
    for index, rank, count in zip(indices, (8, 10), position_counts):
        original = Qwen2DecoderLayer(config, layer_idx=index).to(dtype).eval().requires_grad_(False)
        originals.append(original)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :rank].to(dtype)
        means = torch.randn(count, 16, dtype=dtype) / 3
        head_projection = None
        if projection:
            values = torch.stack([torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q[:, :2]
                                  for _ in range(2)]).to(dtype)
            head_projection = {"rope_pairs": torch.tensor([[0], [1]]), "value_basis": values}
        block = AffineQwenBlock(original, basis, means, head_projection=head_projection,
                                parameter_dtype=torch.float32 if dtype == torch.bfloat16 else dtype)
        if polynomial:
            fit = {"anchors": [0, 2, 5], "mean": torch.randn(28, dtype=basis.dtype) / 4,
                   "scale": torch.rand(28, dtype=basis.dtype) + .6,
                   "coefficients": {"2": torch.randn(28, 10, dtype=basis.dtype) / 20}}
            block.compile_polynomial(fit, [1, 3, 4, 6, 9, 12])
        with torch.no_grad():
            block.qkv_offset.add_(torch.randn_like(block.qkv_offset) / 20)
            block.gate_up_offset.add_(torch.randn_like(block.gate_up_offset) / 20)
        blocks.append(block)
    return AffineQwenStack(*blocks), originals, config


def positions(length, dtype=torch.float64):
    angles = torch.arange(length, dtype=dtype)[:, None] * torch.tensor([.1, .3], dtype=dtype)[None]
    angles = torch.cat((angles, angles), -1)[None]
    return angles.cos(), angles.sin()


def sequential(blocks, x, **kwargs):
    for block in blocks:
        x = block(x, **kwargs)[0]
    return x


@pytest.mark.parametrize("projection,polynomial", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("position_counts", [(8, 8), (1, 8), (8, 1), (1, 1)])
def test_reduced_stack_matches_sequential_native_boundaries(projection, polynomial, position_counts):
    stack, _, _ = fixture(projection=projection, polynomial=polynomial, position_counts=position_counts)
    reference = deepcopy(stack.blocks)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    kwargs = {"position_embeddings": positions(5), "position_ids": torch.arange(5)[None]}
    actual, attentions = stack(x, output_attentions=True, **kwargs)
    expected = sequential(reference, x, **kwargs)
    assert torch.allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert len(attentions) == 2 and all(p.shape == (2, 4, 5, 5) for p in attentions)
    assert stack.ranks == (8, 10) and stack.layer_indices == (34, 35)
    assert stack.bridge_weight.shape == (10, 8)
    assert stack.bridge_shift.shape == (max(position_counts), 10)
    assert all(not b.requires_grad for b in stack.buffers())


def test_gradients_match_through_both_blocks_and_bridge_remains_fixed():
    stack, originals, _ = fixture()
    reference = deepcopy(stack.blocks)
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    args = {"position_embeddings": positions(5)}
    stack(x, **args)[0].square().mean().backward()
    sequential(reference, xr, **args).square().mean().backward()
    assert torch.allclose(x.grad, xr.grad, atol=2e-12, rtol=2e-11)
    for block, expected in zip(stack.blocks, reference):
        for name, parameter in block.named_parameters():
            comparison = dict(expected.named_parameters())[name]
            assert parameter.grad is not None and parameter.grad.norm() > 0, name
            assert torch.allclose(parameter.grad, comparison.grad, atol=2e-12, rtol=2e-10), name
    assert all(p.grad is None for original in originals for p in original.parameters())
    assert all(not b.requires_grad and b.grad is None for b in stack.buffers())


def test_stack_never_calls_native_boundary_forward_or_second_encode():
    stack, _, _ = fixture()
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    with patch.object(stack.blocks[0], "forward", side_effect=AssertionError("first native boundary")), \
         patch.object(stack.blocks[1], "forward", side_effect=AssertionError("second native boundary")), \
         patch.object(stack.blocks[1], "encode", side_effect=AssertionError("second encode")), \
         patch.object(stack.blocks[0], "encode", wraps=stack.blocks[0].encode) as encode:
        output = stack(x, position_embeddings=positions(5))[0]
    assert output.shape == x.shape and encode.call_count == 1


def test_original_34_35_cache_slots_dynamic_and_static_agree():
    stack, _, config = fixture()
    stack.pack()
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    cos, sin = positions(5)
    full = stack(x, position_embeddings=(cos, sin))[0]
    dynamic = DynamicCache()
    static = StaticCache(config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float64)
    dynamic_rows, static_rows = [], []
    for i in range(5):
        args = {"position_embeddings": (cos[:, i:i+1], sin[:, i:i+1]),
                "cache_position": torch.tensor([i]), "use_cache": True}
        mask = torch.full((1, 1, 1, 8), torch.finfo(torch.float64).min, dtype=torch.float64)
        mask[..., :i+1] = 0
        dynamic_rows.append(stack(x[:, i:i+1], past_key_value=dynamic, **args)[0])
        static_rows.append(stack(x[:, i:i+1], past_key_value=static, attention_mask=mask, **args)[0])
    assert torch.allclose(torch.cat(dynamic_rows, 1), full, atol=2e-12, rtol=2e-12)
    assert torch.allclose(torch.cat(static_rows, 1), full, atol=2e-12, rtol=2e-12)
    for index in (34, 35):
        assert dynamic.key_cache[index].shape[-1] == dynamic.value_cache[index].shape[-1] == 2
        assert static.key_cache[index].shape[-1] == static.value_cache[index].shape[-1] == 4
        assert torch.count_nonzero(static.key_cache[index][..., 2:]) == 0
        assert torch.count_nonzero(static.value_cache[index][..., 2:]) == 0


def test_packing_and_strict_reload_preserve_output_and_capture_cleanup(tmp_path):
    stack, _, _ = fixture()
    fresh = deepcopy(stack)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    for block in stack.blocks:
        block.set_capture_mlp(True)
    expected = stack(x, position_embeddings=positions(5))[0]
    assert all(block.last_mlp_hidden is not None for block in stack.blocks)
    bridge = stack.bridge_weight.clone()
    stack.pack()
    actual = stack(x, position_embeddings=positions(5))[0]
    assert torch.allclose(expected, actual, atol=2e-12, rtol=2e-12)
    assert torch.equal(stack.bridge_weight, bridge)
    assert stack.packed and not stack.compact_parameters()
    assert all(not block.capture_mlp and block.last_mlp_hidden is None for block in stack.blocks)
    path = tmp_path / "stack.pt"
    torch.save(stack.state_dict(), path)
    fresh.pack().load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(fresh(x, position_embeddings=positions(5))[0], actual)


def test_bf16_autocast_training_and_packed_reload(tmp_path):
    # Fixed bases stay FP32, while original projection weights become BF16.
    stack, _, _ = fixture(dtype=torch.float32)
    fresh = deepcopy(stack)
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    args = {"position_embeddings": positions(5, torch.bfloat16)}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        stack(x, **args)[0].float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.compact_parameters())
    stack.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    assert stack.bridge_weight.dtype == stack.bridge_shift.dtype == torch.float32
    expected = stack(x, **args)[0]
    assert expected.dtype == torch.bfloat16 and torch.isfinite(expected).all()
    path = tmp_path / "stack_bf16.pt"
    torch.save(stack.state_dict(), path)
    fresh.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(fresh(x, **args)[0], expected)


def test_bf16_bridge_uses_fixed_arithmetic_geometry_without_native_roundtrip():
    stack, _, _ = fixture(dtype=torch.float32)
    z = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    ids = torch.arange(5)[None].expand(2, -1)
    expected = ((z.float() @ stack.bridge_weight.T)
                + stack.bridge_shift[ids]).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = stack.bridge(z, ids)
    assert torch.equal(actual, expected)
    first, second = stack.blocks
    native = (z.float() @ first.basis.T + first.anchor[ids]).to(torch.bfloat16)
    rounded_roundtrip = second.encode(native)
    # This is an intentional BF16 execution difference, not an equivalence
    # claim: removing a native-width materialization removes its rounding.
    assert not torch.equal(actual, rounded_roundtrip)


def test_hf_two_layers_can_be_replaced_with_stack_preserving_native_config_and_tied_head():
    torch.manual_seed(1281)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=2, vocab_size=32,
                         attention_dropout=0., max_position_embeddings=32, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    basis = torch.eye(16)
    blocks = [AffineQwenBlock(block, basis, torch.zeros(1, 16), trainable=False)
              for block in model.model.layers]
    model.model.layers = nn.ModuleList(blocks)
    reference = deepcopy(model)
    model.model.layers = nn.ModuleList([AffineQwenStack(*blocks)])
    ids = torch.tensor([[3, 5, 7]])
    with torch.no_grad():
        actual = model(ids, use_cache=True)
        expected = reference(ids, use_cache=True)
        actual_next = model(torch.tensor([[8]]), past_key_values=actual.past_key_values, use_cache=True).logits
        expected_next = reference(torch.tensor([[8]]), past_key_values=expected.past_key_values, use_cache=True).logits
    assert torch.equal(actual.logits, expected.logits)
    assert torch.equal(actual_next, expected_next)
    assert model.config.num_hidden_layers == 2 and len(actual.past_key_values.key_cache) == 2
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_invalid_stack_pair_and_uncovered_positions_are_rejected():
    stack, _, _ = fixture()
    with pytest.raises(ValueError, match="distinct"):
        AffineQwenStack(stack.blocks[0], stack.blocks[0])
    wrong = deepcopy(stack.blocks[1])
    wrong.layer_idx = 37
    with pytest.raises(ValueError, match="adjacent"):
        AffineQwenStack(stack.blocks[0], wrong)
    wrong = deepcopy(stack.blocks[1])
    wrong.position_policy = "clamp"
    with pytest.raises(ValueError, match="position policy"):
        AffineQwenStack(stack.blocks[0], wrong)
    x = torch.randn(1, 1, 16, dtype=torch.float64)
    with pytest.raises((IndexError, RuntimeError)):
        stack(x, position_embeddings=positions(1), cache_position=torch.tensor([8]))
