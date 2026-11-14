from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import DynamicCache, Qwen2Config, StaticCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net.jspace_quadratic_qwen import (
    FoldedReadout, QuadraticQwenBlock, QuadraticQwenStack, RopePairMetric, normalized_causal_attention)


def fixture(dtype=torch.float64, count=1, global_anchor=False):
    torch.manual_seed(1504)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4, num_key_value_heads=2,
                        num_hidden_layers=4, max_position_embeddings=32, vocab_size=19, attention_dropout=0.)
    config._attn_implementation = "eager"
    blocks = []
    for layer in range(count):
        native = Qwen2DecoderLayer(config, layer_idx=layer).to(dtype).eval()
        u = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :8].to(dtype)
        means = torch.randn(1 if global_anchor else 8, 16, dtype=dtype)/5
        block = AffineQwenBlock(native, u, means, head_projection={"rope_pairs": torch.tensor([[0], [1]]),
            "value_basis": torch.eye(4, dtype=dtype).repeat(2, 1, 1)[:, :, :2]})
        with torch.no_grad():
            block.qkv_bias.add_(torch.randn_like(block.qkv_bias)/10)
            block.qkv_offset.copy_(torch.randn_like(block.qkv_offset)/10)
            block.gate_up_offset.copy_(torch.randn_like(block.gate_up_offset)/10)
        blocks.append(block)
    return blocks, config


def rotary(length, dtype=torch.float64, start=0):
    phase = torch.arange(start, start+length, dtype=dtype)[:, None]*torch.tensor([.13, .41], dtype=dtype)
    phase = torch.cat((phase, phase), -1)[None]
    return phase.cos(), phase.sin()


def test_polynomial_matches_explicit_masked_reference_and_gradients():
    torch.manual_seed(11)
    q = torch.randn(1, 2, 3, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 5, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 5, 3, dtype=torch.float64, requires_grad=True)
    allowed = torch.tensor([[[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 1, 1]]]], dtype=torch.bool)
    positions = torch.tensor([2, 3, 4])
    output, probabilities = normalized_causal_attention(q, k, v, scale=.5, attention_mask=allowed, query_positions=positions)
    scores = (q@k.transpose(-1, -2))*.5
    weights = torch.where(allowed, 1+scores+.5*scores.square(), 0.)
    expected = weights/weights.sum(-1, keepdim=True).clamp_min(1.)
    torch.testing.assert_close(probabilities, expected, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(output, expected@v, atol=2e-15, rtol=2e-15)
    assert torch.count_nonzero(probabilities[:, :, 1]) == 0
    assert torch.autograd.gradcheck(lambda a, b, c: normalized_causal_attention(a, b, c, scale=.5,
        attention_mask=allowed, query_positions=positions)[0], (q, k, v))
    additive = torch.where(allowed, 0., -torch.inf)
    assert torch.equal(normalized_causal_attention(q, k, v, scale=.5, attention_mask=additive,
                                                  query_positions=positions)[0], output)


@pytest.mark.parametrize("mode", ["quadratic", "softmax"])
def test_no_future_leak_all_masked_rows_and_absolute_decode_positions(mode):
    q = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    output, p = normalized_causal_attention(q, k, v, scale=.5, mode=mode)
    changed = v.detach().clone(); changed[:, :, 2] += 1000
    again, _ = normalized_causal_attention(q, k, changed, scale=.5, mode=mode)
    assert torch.equal(output[:, :, :2], again[:, :, :2]) and torch.count_nonzero(p.triu(1)) == 0
    empty, p = normalized_causal_attention(q, k, v, scale=.5, mode=mode, attention_mask=torch.zeros(3, 3, dtype=torch.bool))
    assert torch.count_nonzero(empty) == torch.count_nonzero(p) == 0
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in torch.autograd.grad(empty.sum(), (q, k, v)))
    last, p = normalized_causal_attention(q[:, :, :1], k, v, scale=.5, mode=mode, query_positions=torch.tensor([0]))
    assert torch.equal(last, v[:, :, :1]) and p[0, 0, 0].tolist() == [1., 0., 0.]


def test_polynomial_normalization_avoids_finite_score_square_overflow():
    q = torch.tensor([[[[1e10]]]], dtype=torch.float32)
    k = torch.tensor([[[[1e10], [-1e10], [0.]]]], dtype=torch.float32)
    v = torch.tensor([[[[1.], [3.], [9.]]]], dtype=torch.float32)
    output, p = normalized_causal_attention(q, k, v, scale=1., stable=True)
    assert torch.isfinite(p).all() and torch.isfinite(output).all()
    torch.testing.assert_close(p, torch.tensor([[[[.5, .5, 0.]]]]), atol=1e-6, rtol=0)
    _, direct = normalized_causal_attention(q, k, v, scale=1.)
    assert not torch.isfinite(direct).all()  # Training/evaluation guards reject this unsupported range.


def test_fp32_default_is_literal_polynomial_with_matching_gradients_and_no_amax(monkeypatch):
    torch.manual_seed(52)
    q = torch.randn(1, 2, 4, 3, requires_grad=True)
    k = torch.randn(1, 2, 4, 3, requires_grad=True)
    v = torch.randn(1, 2, 4, 2, requires_grad=True)
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    scores = (q@k.transpose(-1, -2))*.125
    safe = torch.where(mask, scores, 0.)
    weights = torch.where(mask, 1+safe+.5*safe.square(), 0.)
    expected_p = weights/weights.sum(-1, keepdim=True)
    expected = expected_p@v
    monkeypatch.setattr(torch.Tensor, "amax", lambda *a, **kw: pytest.fail("Default path rescales scores"))
    actual, probabilities = normalized_causal_attention(q, k, v, scale=.125)
    assert torch.equal(probabilities, expected_p) and torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_softmax_control_matches_affine_block_with_native_scale():
    blocks, _ = fixture()
    native = blocks[0]
    model = QuadraticQwenBlock(deepcopy(native), mode="softmax")
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    expected = native(x, position_embeddings=rotary(5), output_attentions=True)
    actual = model(x, position_embeddings=rotary(5), output_attentions=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
    assert model.scale == .5 and model.scale != 2**-.5


@pytest.mark.parametrize("packed_first", [False, True])
def test_metric_commutes_with_selected_rope_and_folds_every_affine_qk_term(packed_first):
    blocks, _ = fixture()
    block = blocks[0]
    if packed_first:
        block.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    model = QuadraticQwenBlock(block, learn_metric=True)
    with torch.no_grad():
        model.metric.log_metric.copy_(torch.tensor([[.7], [-.4]], dtype=torch.float64))
    metric = deepcopy(model.metric)
    q, k = torch.randn(1, 4, 4, 2, dtype=torch.float64), torch.randn(1, 2, 4, 2, dtype=torch.float64)
    before = block._rotary(*metric(q, k), *rotary(4))
    after = metric(*block._rotary(q, k, *rotary(4)))
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=2e-15, rtol=2e-15)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    expected = model(x, position_embeddings=rotary(5))[0]
    qk_rows = (block.nq+block.nkv)*block.head_dim
    value_weight = block.qkv_weight[qk_rows:].detach().clone()
    model.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    actual = model(x, position_embeddings=rotary(5))[0]
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
    assert model.metric is None and model.scale == .5
    assert torch.equal(block.qkv_weight[qk_rows:], value_weight)
    assert block.qkv_offset is None and block.gate_up_offset is None


@pytest.mark.parametrize("mode", ["quadratic", "softmax"])
def test_prefill_dynamic_static_cache_and_query_subset_agree(mode):
    blocks, config = fixture()
    model = QuadraticQwenBlock(blocks[0], mode=mode, learn_metric=True)
    with torch.no_grad():
        model.metric.log_metric.copy_(torch.tensor([[.2], [-.15]], dtype=torch.float64))
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    expected = model(x, position_embeddings=rotary(5))[0]
    indices = torch.tensor([1, 4])
    selected, probabilities = model(x, position_embeddings=rotary(5), query_indices=indices, output_attentions=True)
    torch.testing.assert_close(selected, expected[:, indices], atol=3e-13, rtol=3e-13)
    assert probabilities.shape == (1, 4, 2, 5) and torch.count_nonzero(probabilities[:, :, 0, 2:]) == 0
    dynamic = DynamicCache()
    static = StaticCache(config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float64)
    drows, srows = [], []
    for i in range(5):
        kwargs = dict(cache_position=torch.tensor([i]), position_embeddings=rotary(1, start=i))
        drows.append(model(x[:, i:i+1], past_key_value=dynamic, **kwargs)[0])
        srows.append(model(x[:, i:i+1], past_key_value=static, **kwargs)[0])
    for values in (drows, srows):
        torch.testing.assert_close(torch.cat(values, 1), expected, atol=5e-13, rtol=5e-13)
    assert dynamic.key_cache[0].shape[-1] == 2 and static.key_cache[0].shape[-1] == 4
    cache = DynamicCache()
    prefix = model(x[:, :3], past_key_value=cache, cache_position=torch.arange(3), position_embeddings=rotary(3),
                   query_indices=torch.tensor([2]))[0]
    suffix = model(x[:, 3:], past_key_value=cache, cache_position=torch.tensor([3, 4]), position_embeddings=rotary(2, start=3),
                   query_indices=torch.tensor([1]))[0]
    torch.testing.assert_close(prefix, expected[:, 2:3], atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(suffix, expected[:, 4:5], atol=5e-13, rtol=5e-13)
    with pytest.raises(ValueError, match="explicit absolute"):
        model(x[:, :1], past_key_value=static, position_embeddings=rotary(1))


@pytest.mark.parametrize("mode", ["quadratic", "softmax"])
def test_reduced_stack_gradients_checkpoint_pack_and_reload(mode):
    blocks, _ = fixture(count=3)
    native_blocks = deepcopy(blocks)
    original = CompactQwenStack(blocks, pad_to_multiple=16)
    model = QuadraticQwenStack(original, mode=mode, learn_metric=True)
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    expected = x
    for block in native_blocks:
        expected = QuadraticQwenBlock(block, mode=mode)(expected, position_embeddings=rotary(5))[0]
    actual = model(x, position_embeddings=rotary(5))[0]
    torch.testing.assert_close(actual, expected, atol=4e-12, rtol=4e-12)
    z = original._geometry(x, original.blocks[0].basis.T)
    reduced = model.forward_coordinates(z, position_embeddings=rotary(5))
    selected = model.forward_coordinates(z, position_embeddings=rotary(5), query_indices=torch.tensor([4]))
    torch.testing.assert_close(selected, reduced[:, -1:], atol=4e-12, rtol=4e-12)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.compact_parameters())
    assert all(m.log_metric.grad.abs().sum() > 0 for m in model.metrics)
    checked = deepcopy(model); checked.original.checkpoint_blocks = True
    checked.zero_grad(set_to_none=True)
    checked(x.detach(), position_embeddings=rotary(5))[0].square().mean().backward()
    for p, other in zip(model.parameters(), checked.parameters()):
        torch.testing.assert_close(p.grad, other.grad, atol=2e-13, rtol=2e-12)
    fresh = deepcopy(model)
    model.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    fresh.pack(dtype=torch.float64, buffer_dtype=torch.float64).load_state_dict(model.state_dict(), strict=True)
    assert all(m is None for m in model.metrics)
    assert torch.equal(model(x.detach(), position_embeddings=rotary(5))[0], fresh(x.detach(), position_embeddings=rotary(5))[0])
    torch.testing.assert_close(model(x.detach(), position_embeddings=rotary(5))[0], actual.detach(), atol=4e-12, rtol=4e-12)
    assert original.blocks[1].basis is None and original.blocks[0].anchor is None


def test_folded_readout_matches_general_affine_native_formula_and_gradients():
    torch.manual_seed(140)
    e, u, a, gain = (torch.randn(*shape, dtype=torch.float64) for shape in ((13, 9), (9, 5), (9,), (9,)))
    z = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    eps = .17
    head = FoldedReadout(e, u, a, gain, eps, parameter_dtype=torch.float64)
    h = z@u.T+a
    expected = (h/(h.square().mean(-1, keepdim=True)+eps).sqrt()*gain)@e.T
    actual = head(z)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    actual_grad = torch.autograd.grad(actual.square().sum(), z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), z)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-13, rtol=3e-13)
    actual.square().mean().backward()
    assert head.weight.grad.abs().sum() > 0 and head.bias.grad.abs().sum() > 0
    assert all(not b.requires_grad for b in head.buffers())
    assert not hasattr(head, "basis") and not hasattr(head, "anchor")
    with pytest.raises(ValueError, match="global"):
        FoldedReadout(e, u, a.repeat(2, 1), gain, eps)


def test_fp32_static_fullgraph_reduced_stack_and_head_are_traceable():
    blocks, _ = fixture(dtype=torch.float32, count=2, global_anchor=True)
    final_u, final_a = blocks[-1].basis.clone(), blocks[-1].anchor.clone()
    stack = QuadraticQwenStack(CompactQwenStack(blocks, pad_to_multiple=0), learn_metric=True)
    head = FoldedReadout(torch.randn(19, 16), final_u, final_a, torch.randn(16), 1e-6)
    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__(); self.stack, self.head = stack, head
        def forward(self, z, cos, sin, selected):
            return self.head(self.stack.forward_coordinates(z, position_embeddings=(cos, sin), query_indices=selected))
    model = Pipeline().eval()
    z = torch.randn(1, 5, 8)
    args = (z, *rotary(5, torch.float32), torch.tensor([4]))
    # Eager backend checks fullgraph capture without starting an expensive CPU
    # compiler job during the separate reserved campaign. Benchmark uses inductor.
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    expected, actual = model(*args), compiled(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert model.stack.metrics[0].log_metric.grad is not None


def test_polynomial_mlp_is_explicitly_rejected():
    blocks, _ = fixture()
    blocks[0].has_polynomial = True
    with pytest.raises(ValueError, match="complete native SwiGLU"):
        QuadraticQwenBlock(blocks[0])


def test_stack_from_fitted_wrappers_preserves_metric_state_and_aliases():
    blocks, _ = fixture(count=2, global_anchor=True)
    wrappers = [QuadraticQwenBlock(b, mode="softmax", learn_metric=True) for b in blocks]
    with torch.no_grad():
        wrappers[0].metric.log_metric.fill_(.3)
        wrappers[1].metric.log_metric.fill_(-.2)
    x = torch.randn(1, 4, 16, dtype=torch.float64)
    expected = x
    for wrapper in wrappers:
        assert wrapper.block is wrapper.affine is wrapper.original
        expected = wrapper(expected, position_embeddings=rotary(4))[0]
    stack = QuadraticQwenStack(wrappers, pad_to_multiple=0)
    assert stack.mode == "softmax" and stack.metrics[0] is wrappers[0].metric
    torch.testing.assert_close(stack(x, position_embeddings=rotary(4))[0], expected, atol=2e-12, rtol=2e-12)
    stack.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    torch.testing.assert_close(stack(x, position_embeddings=rotary(4))[0], expected, atol=2e-12, rtol=2e-12)
