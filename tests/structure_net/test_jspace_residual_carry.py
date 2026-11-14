from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_compact_model import rope
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import QuadraticQwenBlock, normalized_causal_attention
from experiments.structure_net.jspace_residual_carry import ResidualCarryBlock, dynamic_complement_energy


def fixture(rank=8, dtype=torch.float64, mode="softmax", positions=1):
    torch.manual_seed(615)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                        num_key_value_heads=2, num_hidden_layers=1, vocab_size=23,
                        max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=0).to(dtype).eval()
    with torch.no_grad():
        native.input_layernorm.weight.copy_(torch.rand(16, dtype=dtype)+.5)
        native.post_attention_layernorm.weight.copy_(torch.rand(16, dtype=dtype)+.5)
    u = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :rank].to(dtype)
    means = torch.randn(positions, 16, dtype=dtype)/3
    affine = AffineQwenBlock(native, u, means, parameter_dtype=dtype)
    wrapped = QuadraticQwenBlock(affine, mode=mode, learn_metric=False)
    return native, ResidualCarryBlock(wrapped)


def rotary(length, dtype=torch.float64):
    phase = torch.arange(length, dtype=dtype)[:, None]*torch.tensor([.13, .41], dtype=dtype)
    phase = torch.cat((phase, phase), -1)[None]
    return phase.cos(), phase.sin()


def independent_reference(native, model, x, position_embeddings):
    """Native-width algebra, dynamic full-state RMS, fixed-mean numerators."""
    block, u = model.affine, model.affine.basis
    anchor = block.anchor[:x.shape[1]]
    project = lambda h: (h@u)@u.T + anchor
    # The complement contributes to the denominator, but not to the learned
    # numerator: this is the intended approximation, not a native bypass.
    h = project(x)/(x.square().mean(-1, keepdim=True)+block.input_eps).sqrt()
    h = h*native.input_layernorm.weight
    batch, length = x.shape[:2]
    q = native.self_attn.q_proj(h).reshape(batch, length, 4, 4).transpose(1, 2)
    k = native.self_attn.k_proj(h).reshape(batch, length, 2, 4).transpose(1, 2)
    v = native.self_attn.v_proj(h).reshape(batch, length, 2, 4).transpose(1, 2)
    q, k = rope(q, k, *position_embeddings)
    mixed, _ = normalized_causal_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                                           scale=.5, mode=model.wrapped.mode)
    mixed = mixed.transpose(1, 2).reshape(batch, length, 16)
    h = x+(native.self_attn.o_proj(mixed)@u)@u.T
    numerator = project(h)*native.post_attention_layernorm.weight
    normalized = numerator/(h.square().mean(-1, keepdim=True)+block.post_eps).sqrt()
    mlp = native.mlp.down_proj(F.silu(native.mlp.gate_proj(normalized))*native.mlp.up_proj(normalized))
    return h+(mlp@u)@u.T, h


def test_full_rank_softmax_matches_native_qwen():
    native, model = fixture(rank=16, dtype=torch.float32)
    x = torch.randn(2, 5, 16)
    position_embeddings = rotary(5, torch.float32)
    mask = torch.full((5, 5), torch.finfo(torch.float32).min).triu(1)[None, None]
    expected = native(x, attention_mask=mask, position_embeddings=position_embeddings)[0]
    actual = model(x, attention_mask=mask, position_embeddings=position_embeddings)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode", ["softmax", "quadratic"])
@pytest.mark.parametrize("positions", [1, 7])
def test_dynamic_both_rms_match_native_width_reference_and_carry(mode, positions):
    native, model = fixture(mode=mode, positions=positions)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    # Make the actual complement deliberately unlike its fitted mean.
    u = model.basis
    x = x+5*(x-(x@u)@u.T)
    expected, after_attention = independent_reference(native, model, x, rotary(5))
    actual = model(x, position_embeddings=rotary(5))
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    complement = lambda h: h-(h@u)@u.T
    torch.testing.assert_close(complement(actual), complement(x), atol=2e-13, rtol=2e-13)
    energy = dynamic_complement_energy(x, x@u)
    actual_post_rms = model.rms_denominator(after_attention@u, energy, model.post_eps)
    full_post_rms = (after_attention.square().mean(-1, keepdim=True)+model.post_eps).sqrt()
    torch.testing.assert_close(actual_post_rms, full_post_rms, atol=3e-13, rtol=3e-13)
    # A static fitted-anchor denominator is not accidentally equivalent here.
    wrong = model.rms_denominator(after_attention@u, model.anchor_energy[:5], model.post_eps)
    assert not torch.allclose(wrong, full_post_rms)


def test_input_and_weight_gradients_match_independent_reference():
    native, model = fixture()
    x = torch.randn(1, 4, 16, dtype=torch.float64, requires_grad=True)
    expected, _ = independent_reference(native, model, x, rotary(4))
    output = model(x, position_embeddings=rotary(4))
    actual_grad = torch.autograd.grad(output.square().mean(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-13, rtol=3e-12)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for name in ("qkv_weight", "attention_output", "gate_up_weight", "mlp_output"):
        assert getattr(model, name).grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["softmax", "quadratic"])
def test_causal_mask_future_invariance_and_capture(mode):
    _, model = fixture(mode=mode)
    model.set_capture_mlp(True)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    mask[:, 1] = False
    first, probabilities = model(x, attention_mask=mask, position_embeddings=rotary(5), output_attentions=True)
    changed = x.clone(); changed[:, -1] += 100
    second = model(changed, attention_mask=mask, position_embeddings=rotary(5))
    torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=0, rtol=0)
    assert torch.count_nonzero(probabilities.triu(1)) == 0
    assert torch.count_nonzero(probabilities[..., 1]) == 0
    assert model.last_mlp_hidden.shape == (1, 5, 28)


def test_reload_and_packing_preserve_outputs_without_mutating_anchor_energy():
    _, model = fixture()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    energy_before = model.anchor_energy.clone()
    expected = model(x, position_embeddings=rotary(5))
    state = deepcopy(model.state_dict())
    _, restored = fixture()
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored(x, position_embeddings=rotary(5)), expected)
    assert torch.equal(model.anchor_energy, energy_before)
    model.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    restored.pack(dtype=torch.float64, buffer_dtype=torch.float64)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(model(x, position_embeddings=rotary(5)), restored(x, position_embeddings=rotary(5)))
    torch.testing.assert_close(model(x, position_embeddings=rotary(5)), expected, atol=2e-13, rtol=2e-13)


def test_reject_unsupported_cache_dtype_and_query_subset():
    _, model = fixture(dtype=torch.float32)
    assert not hasattr(model, "forward_coordinates")
    x = torch.randn(1, 2, 16)
    for kwargs in ({"use_cache": True}, {"past_key_value": object()},
                   {"query_indices": torch.tensor([1])}):
        with pytest.raises(ValueError):
            model(x, position_embeddings=rotary(2, torch.float32), **kwargs)
    with pytest.raises(ValueError, match="basis dtype"):
        model(x.to(torch.bfloat16), position_embeddings=rotary(2, torch.float32))
    model.affine.basis = None
    with pytest.raises(ValueError, match="unstripped"):
        ResidualCarryBlock(model.wrapped)
