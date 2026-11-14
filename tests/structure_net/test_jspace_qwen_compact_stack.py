from copy import deepcopy

import pytest
import torch
from transformers import DynamicCache, StaticCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack, estimate_qwen3b_stack_bytes


def fixture(dtype=torch.float64, count=3):
    torch.manual_seed(511)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=36, vocab_size=32,
                         attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    result = []
    for i in range(count):
        native = Qwen2DecoderLayer(config, layer_idx=36-count+i).to(dtype).eval().requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :8+i % 3].to(dtype)
        projection = {"rope_pairs": torch.tensor([[0], [1]]),
                      "value_basis": torch.eye(4, dtype=dtype).repeat(2, 1, 1)[:, :, :2]}
        block = AffineQwenBlock(native, basis, torch.randn(8, 16, dtype=dtype) / 4, head_projection=projection)
        fit = {"anchors": [0, 2, 5, 8], "mean": torch.randn(28, dtype=dtype) / 5,
               "scale": torch.rand(28, dtype=dtype) + .7,
               "coefficients": {"2": torch.randn(28, 15, dtype=dtype) / 30}}
        result.append(block.compile_polynomial(fit, [1, 3, 4, 6, 9, 12]))
    return result, config


def positions(length, dtype=torch.float64):
    a = torch.arange(length, dtype=dtype)[:, None] * torch.tensor([.1, .3], dtype=dtype)[None]
    a = torch.cat((a, a), -1)[None]
    return a.cos(), a.sin()


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_arbitrary_stack_matches_sequential_affine_boundaries_and_gradients(count):
    blocks, _ = fixture(count=count)
    reference = deepcopy(blocks)
    stack = CompactQwenStack(blocks, pad_to_multiple=0)
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = stack(x, position_embeddings=positions(5))[0]
    expected = xr
    for block in reference:
        expected = block(expected, position_embeddings=positions(5))[0]
    torch.testing.assert_close(actual, expected, atol=4e-12, rtol=4e-12)
    actual.square().mean().backward(); expected.square().mean().backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=3e-12, rtol=3e-10)
    for block, other in zip(stack.blocks, reference):
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None and parameter.grad.norm() > 0
            torch.testing.assert_close(parameter.grad, dict(other.named_parameters())[name].grad, atol=3e-12, rtol=3e-10)
    assert all(not b.requires_grad for b in stack.buffers())


def test_activation_checkpointing_preserves_gradients_for_all_blocks():
    blocks, _ = fixture()
    control = CompactQwenStack(deepcopy(blocks), pad_to_multiple=0)
    checked = CompactQwenStack(blocks, pad_to_multiple=0, checkpoint_blocks=True)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    for model in (control, checked):
        model(x, position_embeddings=positions(5))[0].square().mean().backward()
    for name, parameter in checked.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, dict(control.named_parameters())[name].grad, atol=1e-13, rtol=1e-12)
    with pytest.raises(ValueError, match="uncached execution"):
        checked(x, position_embeddings=positions(5), use_cache=True)


def test_pack_strips_only_unused_geometry_and_strict_reload_is_exact(tmp_path):
    blocks, _ = fixture()
    stack = CompactQwenStack(blocks, pad_to_multiple=16)
    fresh = deepcopy(stack)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    expected = stack(x, position_embeddings=positions(5))[0]
    stack.pack(strip_unused=False)
    before = stack.storage_breakdown()["total"]
    removed = sum(b.basis.numel() * b.basis.element_size() for b in stack.blocks[1:-1])
    removed += sum(b.anchor.numel() * b.anchor.element_size() for b in stack.blocks[:-1])
    stack.pack(strip_unused=True)
    assert before - stack.storage_breakdown()["total"] == removed
    assert stack.blocks[0].basis is not None and stack.blocks[-1].basis is not None
    assert all(b.basis is None for b in stack.blocks[1:-1])
    assert all(b.anchor is None for b in stack.blocks[:-1]) and stack.blocks[-1].anchor is not None
    actual = stack(x, position_embeddings=positions(5))[0]
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    path = tmp_path / "stack.pt"; torch.save(stack.state_dict(), path)
    fresh.pack().load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(fresh(x, position_embeddings=positions(5))[0], actual)
    with pytest.raises(ValueError, match="cannot be restored"):
        stack.pack(strip_unused=False)


def test_stripped_padded_stack_preserves_native_cache_indices():
    blocks, config = fixture()
    stack = CompactQwenStack(blocks, pad_to_multiple=16).pack()
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    cos, sin = positions(5)
    expected = stack(x, position_embeddings=(cos, sin))[0]
    dynamic = DynamicCache()
    static = StaticCache(config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float64)
    drows, srows = [], []
    for i in range(5):
        kwargs = {"position_embeddings": (cos[:, i:i+1], sin[:, i:i+1]), "cache_position": torch.tensor([i])}
        mask = torch.full((1, 1, 1, 8), torch.finfo(torch.float64).min, dtype=torch.float64); mask[..., :i+1] = 0
        drows.append(stack(x[:, i:i+1], past_key_value=dynamic, **kwargs)[0])
        srows.append(stack(x[:, i:i+1], past_key_value=static, attention_mask=mask, **kwargs)[0])
    torch.testing.assert_close(torch.cat(drows, 1), expected, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(torch.cat(srows, 1), expected, atol=3e-12, rtol=3e-12)
    for index in (33, 34, 35):
        assert dynamic.key_cache[index].shape[-1] == 2
        assert static.key_cache[index].shape[-1] == 4
        assert torch.count_nonzero(static.key_cache[index][..., 2:]) == 0


def test_bf16_geometry_fp16_worker_is_explicit_differentiable_and_no_triton():
    blocks, _ = fixture(torch.float32)
    worker = CompactQwenStack(blocks, geometry_mode="bf16", geometry_compute_dtype=torch.float16,
                               checkpoint_blocks=True, pad_to_multiple=16)
    x = torch.randn(1, 5, 16, dtype=torch.float16)
    with torch.autocast("cpu", dtype=torch.float16):
        output = worker(x, position_embeddings=positions(5, torch.float16))[0]
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in worker.compact_parameters())
    assert all(b.basis.dtype == torch.bfloat16 for b in worker.blocks)
    assert all(b.anchor_energy.dtype == torch.float32 for b in worker.blocks)
    assert not worker._can_fuse(x)
    with pytest.raises(ValueError, match="matching activation dtype"):
        worker(x.bfloat16(), position_embeddings=positions(5, torch.bfloat16))
    worker.pack(dtype=torch.float16, buffer_dtype=torch.float16)
    assert all(b.polynomial_bias.dtype == torch.float32 for b in worker.blocks)
    with torch.no_grad():
        assert torch.isfinite(worker(x, position_embeddings=positions(5, torch.float16))[0]).all()


def test_bf16_trainable_weights_support_checkpointed_recovery():
    blocks, _ = fixture(torch.float32)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", checkpoint_blocks=True)
    for parameter in stack.parameters():
        parameter.data = parameter.data.bfloat16()
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = stack(x, position_embeddings=positions(5, torch.bfloat16))[0]
    output.float().square().mean().backward()
    assert all(p.dtype == p.grad.dtype == torch.bfloat16 and torch.isfinite(p.grad).all()
               for p in stack.compact_parameters())
    assert all(b.anchor.dtype == b.anchor_energy.dtype == torch.float32 for b in stack.blocks)


def test_budget_matches_current_oneblock_and_fullstack_estimates():
    one = estimate_qwen3b_stack_bytes(1536, blocks=1)
    assert one["one_block_bytes"] == 127837456
    assert one["full_model_bytes"] == 6145561104
    rank1024 = estimate_qwen3b_stack_bytes(1024)
    rank896 = estimate_qwen3b_stack_bytes(896)
    assert rank1024["full_model_bytes"] == 3816752960
    assert rank896["full_model_bytes"] == 3490199360
    assert rank1024["margin_below_38_percent_limit_bytes"] == pytest.approx(9811171.84)
    assert rank1024["reduction_fraction"] > .38 and not rank1024["measured"]
    assert estimate_qwen3b_stack_bytes(1024, strip_unused=False)["full_model_bytes"] > rank1024["full_model_bytes"]


def test_invalid_stack_and_reload_configuration_are_rejected():
    blocks, _ = fixture()
    with pytest.raises(ValueError, match="contiguous"):
        CompactQwenStack([blocks[0], blocks[2]])
    with pytest.raises(ValueError, match="distinct"):
        CompactQwenStack([blocks[0], blocks[0]])
    stack = CompactQwenStack(blocks, pad_to_multiple=16)
    state = stack.get_extra_state(); state["geometry_mode"] = "bf16"
    with pytest.raises(ValueError, match="same compact-stack structure"):
        stack.set_extra_state(state)
