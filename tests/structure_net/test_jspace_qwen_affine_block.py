from copy import deepcopy
from itertools import combinations_with_replacement

import pytest
import torch
from torch.nn import functional as F
from transformers import DynamicCache, StaticCache, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.structure_net.jspace_compact_model import attend, denominator, rope
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock


def fixture(dtype=torch.float64, positions=8):
    torch.manual_seed(1729)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1, vocab_size=32,
                         attention_dropout=0.0, max_position_embeddings=32)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=0).to(dtype).eval().requires_grad_(False)
    with torch.no_grad():
        native.input_layernorm.weight.copy_(torch.rand(16, dtype=dtype) + .5)
        native.post_attention_layernorm.weight.copy_(torch.rand(16, dtype=dtype) + .5)
    basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q.to(dtype)
    means = torch.randn(positions, 16, dtype=dtype) / 3
    return native, basis, means


def positions(length, dtype=torch.float64):
    angle = torch.arange(length, dtype=dtype)[:, None] * torch.tensor([.1, .3], dtype=dtype)[None]
    angle = torch.cat((angle, angle), -1)[None]
    return angle.cos(), angle.sin()


def head_projection(dtype=torch.float64):
    torch.manual_seed(1741)
    s = torch.stack([torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q[:, :2] for _ in range(2)]).to(dtype)
    return {"rope_pairs": torch.tensor([[0], [1]]), "value_basis": s}


def projected_reference(native, basis, mean, x, pos, projection=None, polynomial=None, removed=()):
    """Independent float64 computation in native coordinates with projections.

    HF's own RMSNorm/softmax force FP32 even when its weights are double. This
    reference uses the same native weights with float64 algebra, allowing the
    actual affine identity to be tested to double precision.
    """
    anchor = mean - (mean @ basis) @ basis.T
    project = lambda h: anchor + (h @ basis) @ basis.T
    norm = lambda h, n: h / denominator(h, n.variance_epsilon) * n.weight
    represented = project(x)
    h = norm(represented, native.input_layernorm)
    batch, length, width = h.shape
    q = native.self_attn.q_proj(h).view(batch, length, 4, 4).transpose(1, 2)
    k = native.self_attn.k_proj(h).view(batch, length, 2, 4).transpose(1, 2)
    v = native.self_attn.v_proj(h).view(batch, length, 2, 4).transpose(1, 2)
    q, k = rope(q, k, *pos)
    if projection is not None:
        pairs = projection["rope_pairs"]
        columns = torch.cat((pairs, pairs + 2), -1)
        q = torch.stack([q[:, i, :, columns[i // 2]] for i in range(4)], 1)
        k = torch.stack([k[:, i, :, columns[i]] for i in range(2)], 1)
        s = projection["value_basis"]
        v = torch.einsum("bhtd,hdf->bhtf", v, s)
    mixed = attend(q, k, v, None, .5)
    if projection is not None:
        mixed = torch.einsum("bhtf,hdf->bhtd", mixed, s.repeat_interleave(2, 0))
    mixed = mixed.transpose(1, 2).reshape(batch, length, width)
    represented = project(represented + native.self_attn.o_proj(mixed))
    h = norm(represented, native.post_attention_layernorm)
    hidden = F.silu(native.mlp.gate_proj(h)) * native.mlp.up_proj(h)
    if polynomial is not None:
        anchors = polynomial["anchors"]
        z = (hidden[..., anchors] - polynomial["mean"][anchors]) / polynomial["scale"][anchors]
        phi = torch.cat((torch.ones_like(z[..., :1]), z,
                         torch.stack([z[..., i] * z[..., j]
                                      for i, j in combinations_with_replacement(range(len(anchors)), 2)], -1)), -1)
        prediction = F.linear(phi, polynomial["coefficients"]["2"][removed])
        hidden = hidden.clone()
        hidden[..., removed] = prediction * polynomial["scale"][removed] + polynomial["mean"][removed]
    return project(represented + native.mlp.down_proj(hidden))


@pytest.mark.parametrize("rank", [8, 16])
@pytest.mark.parametrize("position_count", [1, 8])
def test_affine_block_equals_projected_native_algebra_and_input_gradient(rank, position_count):
    native, q, means = fixture(positions=position_count)
    u = q[:, :rank]
    compiled = AffineQwenBlock(native, u, means, trainable=False)
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    pos = positions(5)
    target = projected_reference(native, u, means[:5], x, pos)
    actual = compiled(x, position_embeddings=pos)[0]
    assert torch.allclose(actual, target, atol=2e-12, rtol=2e-12)
    da = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    dt = torch.autograd.grad(target.square().sum(), x)[0]
    assert torch.allclose(da, dt, atol=1e-11, rtol=1e-11)


def test_full_rank_matches_actual_hf_qwen_decoder():
    native, q, means = fixture(dtype=torch.float32)
    compiled = AffineQwenBlock(native, q, means, trainable=False)
    x = torch.randn(2, 5, 16)
    pos = positions(5, torch.float32)
    mask = torch.full((5, 5), torch.finfo(torch.float32).min).triu(1)[None, None]
    with torch.no_grad():
        expected = native(x, attention_mask=mask, position_embeddings=pos)[0]
        actual = compiled(x, attention_mask=mask, position_embeddings=pos)[0]
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert compiled.position_count == 1
    assert torch.count_nonzero(compiled.anchor) == 0


def test_affine_rms_uses_original_width_and_anchor_energy():
    native, q, means = fixture()
    compiled = AffineQwenBlock(native, q[:, :8], means)
    z = torch.randn(2, 5, 8, dtype=torch.float64)
    h = z @ compiled.basis.T + compiled.anchor[:5]
    actual = compiled.rms_denominator(z, compiled.anchor_energy[:5], native.input_layernorm.variance_epsilon)
    assert torch.allclose(actual, denominator(h, native.input_layernorm.variance_epsilon), atol=1e-12, rtol=1e-12)
    wrong = ((z.square().sum(-1, keepdim=True) + compiled.anchor_energy[:5]) / 8 + 1e-6).sqrt()
    assert not torch.allclose(actual, wrong)


def test_dynamic_cache_decode_matches_causal_full_sequence():
    native, q, means = fixture()
    compiled = AffineQwenBlock(native, q[:, :8], means, trainable=False)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    cos, sin = positions(5)
    expected = compiled(x, position_embeddings=(cos, sin))[0]
    cache = DynamicCache()
    pieces = []
    for i in range(5):
        pieces.append(compiled(x[:, i:i+1], position_embeddings=(cos[:, i:i+1], sin[:, i:i+1]),
                               cache_position=torch.tensor([i]), past_key_value=cache, use_cache=True)[0])
    actual = torch.cat(pieces, 1)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert cache.key_cache[0].shape == (1, 2, 5, 4)
    assert cache.value_cache[0].shape == (1, 2, 5, 4)


@pytest.mark.parametrize("reduce_heads", [False, True])
def test_shared_offsets_train_and_pack_without_learning_position_tables(reduce_heads):
    native, q, means = fixture()
    projection = head_projection() if reduce_heads else None
    compiled = AffineQwenBlock(native, q[:, :8], means, head_projection=projection)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    compiled(x, position_embeddings=positions(5))[0].square().mean().backward()
    parameters = dict(compiled.named_parameters())
    assert set(parameters) == {"qkv_weight", "qkv_bias", "attention_output", "gate_up_weight",
                               "mlp_output", "qkv_offset", "gate_up_offset"}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in parameters.values())
    assert all(p.grad is None for p in native.parameters())
    assert all(not b.requires_grad and b.grad is None for b in compiled.buffers())
    assert compiled.qkv_offset.ndim == compiled.gate_up_offset.ndim == 1
    with torch.no_grad():
        compiled.qkv_offset.add_(torch.randn_like(compiled.qkv_offset) / 20)
        compiled.gate_up_offset.add_(torch.randn_like(compiled.gate_up_offset) / 20)
        before = compiled(x, position_embeddings=positions(5))[0]
        compiled.pack()
        after = compiled(x, position_embeddings=positions(5))[0]
    assert torch.equal(before, after)
    assert not compiled.compact_parameters()
    assert compiled.qkv_offset is compiled.gate_up_offset is None
    assert compiled.pack() is compiled


def test_position_tables_require_valid_absolute_positions():
    native, q, means = fixture(positions=3)
    compiled = AffineQwenBlock(native, q[:, :8], means)
    x = torch.randn(1, 1, 16, dtype=torch.float64)
    with pytest.raises((IndexError, RuntimeError)):
        compiled(x, position_embeddings=positions(1), cache_position=torch.tensor([3]))
    with pytest.raises((IndexError, RuntimeError)):
        compiled(x, position_embeddings=positions(1), cache_position=torch.tensor([-1]))
    with pytest.raises(ValueError, match="explicit absolute"):
        compiled(x, position_embeddings=positions(1), past_key_value=DynamicCache())
    clamped = AffineQwenBlock(native, q[:, :8], means, position_policy="clamp")
    a = clamped(x, position_embeddings=positions(1), cache_position=torch.tensor([20]))[0]
    b = clamped(x, position_embeddings=positions(1), cache_position=torch.tensor([2]))[0]
    assert torch.equal(a, b)


def test_batched_position_ids_select_the_correct_affine_rows():
    native, q, means = fixture()
    compiled = AffineQwenBlock(native, q[:, :8], means)
    x = torch.randn(2, 2, 16, dtype=torch.float64)
    ids = torch.tensor([[0, 2], [3, 5]])
    pos = positions(2)
    together = compiled(x, position_embeddings=pos, position_ids=ids)[0]
    separately = torch.cat([compiled(x[i:i+1], position_embeddings=pos, position_ids=ids[i:i+1])[0]
                            for i in range(2)])
    assert torch.allclose(together, separately, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("reduce_heads", [False, True])
def test_fp32_training_parameters_bf16_activations_and_explicit_pack(tmp_path, reduce_heads):
    native, q, means = fixture(dtype=torch.float32)
    native = native.to(torch.bfloat16)
    projection = head_projection(torch.float32) if reduce_heads else None
    compiled = AffineQwenBlock(native, q[:, :8], means, parameter_dtype=torch.float32, head_projection=projection)
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    pos = tuple(t.to(torch.bfloat16) for t in positions(5, torch.float32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = compiled(x, position_embeddings=pos)[0]
    y.float().square().mean().backward()
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    assert all(p.dtype == torch.float32 and p.grad is not None for p in compiled.compact_parameters())
    compiled.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    assert compiled.qkv_shift.dtype == compiled.gate_up_shift.dtype == torch.bfloat16
    assert compiled.basis.dtype == compiled.anchor.dtype == compiled.anchor_energy.dtype == torch.float32
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in compiled.parameters())
    with torch.no_grad():
        expected = compiled(x, position_embeddings=pos)[0]
        path = tmp_path / "block.pt"
        torch.save(compiled.state_dict(), path)
        fresh = AffineQwenBlock(native, q[:, :8], means, parameter_dtype=torch.float32, head_projection=projection)
        fresh.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
        actual = fresh(x, position_embeddings=pos)[0]
    assert torch.equal(actual, expected)


def test_replaced_final_hf_block_preserves_tied_readout_and_generation_cache():
    torch.manual_seed(917)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=2, vocab_size=32,
                         attention_dropout=0.0, max_position_embeddings=32, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    native = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    modified = deepcopy(native)
    q = torch.linalg.qr(torch.randn(16, 16)).Q
    modified.model.layers[-1] = AffineQwenBlock(modified.model.layers[-1], q, torch.randn(12, 16), trainable=False)
    ids = torch.tensor([[3, 7, 9, 12]])
    with torch.no_grad():
        expected = native(ids, use_cache=True)
        actual = modified(ids, use_cache=True)
        next_token = torch.tensor([[5]])
        expected_next = native(next_token, past_key_values=expected.past_key_values, use_cache=True).logits
        actual_next = modified(next_token, past_key_values=actual.past_key_values, use_cache=True).logits
    assert torch.allclose(expected.logits, actual.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(expected_next, actual_next, atol=1e-6, rtol=1e-5)
    assert modified.lm_head.weight.data_ptr() == modified.model.embed_tokens.weight.data_ptr()


def test_invalid_basis_and_implicit_position_policy_are_rejected():
    native, q, means = fixture()
    with pytest.raises(ValueError, match="orthonormal"):
        AffineQwenBlock(native, q[:, :8] * 2, means)
    with pytest.raises(ValueError, match="Position policy"):
        AffineQwenBlock(native, q[:, :8], means, position_policy="fallback")
    with pytest.raises(ValueError, match="finite"):
        AffineQwenBlock(native, q[:, :8], means * torch.nan)


@pytest.mark.parametrize("rank", [8, 16])
def test_full_head_basis_rotation_preserves_affine_model(rank):
    native, q, means = fixture()
    control = AffineQwenBlock(native, q[:, :rank], means)
    # Reorder complete frequency pairs differently in each GQA group and rotate
    # each complete value space. Both are exact full-width transformations.
    s = torch.stack([torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q for _ in range(2)])
    projection = {"rope_pairs": torch.tensor([[1, 0], [0, 1]]), "value_basis": s}
    changed = AffineQwenBlock(native, q[:, :rank], means, head_projection=projection)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    a = control(x, position_embeddings=positions(5))[0]
    b = changed(x, position_embeddings=positions(5))[0]
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    assert not changed.compact_head and changed.native_head_dim == changed.head_dim == 4


def test_reduced_heads_equal_explicit_native_space_projection_and_gradient():
    native, q, means = fixture()
    projection = head_projection()
    compiled = AffineQwenBlock(native, q[:, :8], means, head_projection=projection)
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    pos = positions(5)
    actual = compiled(x, position_embeddings=pos)[0]
    expected = projected_reference(native, q[:, :8], means[:5], x, pos, projection)
    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    da = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    de = torch.autograd.grad(expected.square().sum(), x)[0]
    assert torch.allclose(da, de, atol=1e-11, rtol=1e-11)
    assert compiled.compact_head and compiled.head_dim == 2 and compiled.native_head_dim == 4
    assert compiled.scale == .5  # Original width, not 1/sqrt(compact width).
    assert compiled.attention_output.shape[1] == 4 * 2
    assert compiled.qkv_weight.shape[0] == (4 + 2 + 2) * 2


def test_reduced_heads_dynamic_and_padded_static_cache_agree():
    native, q, means = fixture()
    compiled = AffineQwenBlock(native, q[:, :8], means, head_projection=head_projection(), trainable=False)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    cos, sin = positions(5)
    full = compiled(x, position_embeddings=(cos, sin))[0]
    dynamic = DynamicCache()
    static = StaticCache(native.self_attn.config, max_batch_size=1, max_cache_len=8,
                         device="cpu", dtype=torch.float64)
    dynamic_rows, static_rows = [], []
    for i in range(5):
        args = dict(position_embeddings=(cos[:, i:i+1], sin[:, i:i+1]), cache_position=torch.tensor([i]), use_cache=True)
        mask = torch.full((8,), torch.finfo(torch.float64).min, dtype=torch.float64)
        mask[:i+1] = 0
        mask = mask[None, None, None]
        dynamic_rows.append(compiled(x[:, i:i+1], past_key_value=dynamic, **args)[0])
        static_rows.append(compiled(x[:, i:i+1], past_key_value=static, attention_mask=mask, **args)[0])
    assert torch.allclose(torch.cat(dynamic_rows, 1), full, atol=1e-12, rtol=1e-12)
    assert torch.allclose(torch.cat(static_rows, 1), full, atol=1e-12, rtol=1e-12)
    assert dynamic.key_cache[0].shape[-1] == dynamic.value_cache[0].shape[-1] == 2
    assert static.key_cache[0].shape[-1] == static.value_cache[0].shape[-1] == 4
    assert torch.count_nonzero(static.key_cache[0][..., 2:]) == 0
    assert torch.count_nonzero(static.value_cache[0][..., 2:]) == 0


def test_invalid_group_head_projections_are_rejected():
    native, q, means = fixture()
    with pytest.raises(ValueError, match="distinct"):
        AffineQwenBlock(native, q[:, :8], means,
                        head_projection={"rope_pairs": torch.tensor([[0, 0], [0, 1]]),
                                         "value_basis": torch.eye(4, dtype=torch.float64).expand(2, -1, -1)})
    wrong = head_projection()
    wrong["value_basis"] = wrong["value_basis"] * 2
    with pytest.raises(ValueError, match="orthonormal"):
        AffineQwenBlock(native, q[:, :8], means, head_projection=wrong)
    wrong = head_projection()
    wrong["rope_pairs"] = torch.tensor([[0]])
    with pytest.raises(ValueError, match="KV heads"):
        AffineQwenBlock(native, q[:, :8], means, head_projection=wrong)


def polynomial_fit(dtype=torch.float64):
    torch.manual_seed(1931)
    anchors = [0, 2, 5, 9]
    fit = {"anchors": anchors, "mean": torch.randn(28, dtype=dtype) / 4,
           "scale": torch.rand(28, dtype=dtype) + .6,
           "coefficients": {"2": torch.randn(28, 15, dtype=dtype) / 20}}
    return fit, [1, 3, 4, 6, 8, 12, 19, 24]


@pytest.mark.parametrize("reduce_heads", [False, True])
def test_polynomial_compilation_matches_explicit_native_activation_substitution(reduce_heads):
    native, q, means = fixture()
    projection = head_projection() if reduce_heads else None
    fit, removed = polynomial_fit()
    compiled = AffineQwenBlock(native, q[:, :8], means, head_projection=projection)
    original_gate_up = compiled.gate_up_weight.detach().clone()
    original_output = compiled.mlp_output.detach().clone()
    original_shift = compiled.gate_up_shift.clone()
    compiled.compile_polynomial(fit, removed)
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    pos = positions(5)
    actual = compiled(x, position_embeddings=pos)[0]
    expected = projected_reference(native, q[:, :8], means[:5], x, pos, projection, fit, removed)
    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    da = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    de = torch.autograd.grad(expected.square().sum(), x)[0]
    assert torch.allclose(da, de, atol=1e-11, rtol=1e-11)
    kept = [i for i in range(28) if i not in removed]
    rows = kept + [i + 28 for i in kept]
    assert compiled.native_intermediate_size == 28 and compiled.intermediate_size == 20
    assert compiled.polynomial_feature_count == 15
    assert compiled.gate_up_weight.shape == (40, 8)
    assert compiled.gate_up_shift.shape == (8, 40)
    assert compiled.gate_up_offset.shape == (40,)
    assert compiled.mlp_output.shape == (8, 20)
    assert compiled.polynomial_output.shape == (8, 15)
    assert torch.equal(compiled.gate_up_weight, original_gate_up[rows])
    assert torch.equal(compiled.gate_up_shift, original_shift[:, rows])
    assert torch.equal(compiled.mlp_output, original_output[:, kept])
    assert compiled.polynomial_removed.tolist() == removed
    assert compiled.polynomial_kept.tolist() == kept
    assert compiled.polynomial_bias.abs().sum() > 0


def test_polynomial_native_and_surrogate_gradients_pack_and_strict_reload(tmp_path):
    native, q, means = fixture()
    fit, removed = polynomial_fit()
    projection = head_projection()
    compiled = AffineQwenBlock(native, q[:, :8], means, head_projection=projection).compile_polynomial(fit, removed)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    compiled(x, position_embeddings=positions(5))[0].square().mean().backward()
    parameters = dict(compiled.named_parameters())
    assert {"gate_up_weight", "mlp_output", "polynomial_output", "polynomial_bias"} <= parameters.keys()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in parameters.values())
    assert all(p.grad is None for p in native.parameters())
    assert all(not b.requires_grad for b in compiled.buffers())
    with torch.no_grad():
        compiled.qkv_offset.add_(torch.randn_like(compiled.qkv_offset) / 30)
        compiled.gate_up_offset.add_(torch.randn_like(compiled.gate_up_offset) / 30)
        expected = compiled(x, position_embeddings=positions(5))[0]
        compiled.pack()
        packed = compiled(x, position_embeddings=positions(5))[0]
        assert torch.allclose(packed, expected, atol=1e-12, rtol=1e-12)
        assert compiled.mlp_output.shape == (8, 20 + 15)
        assert compiled.polynomial_output is None and compiled.polynomial_bias is not None
        assert not compiled.compact_parameters()
        path = tmp_path / "polynomial.pt"
        torch.save(compiled.state_dict(), path)
        fresh = AffineQwenBlock(native, q[:, :8], means, head_projection=projection).compile_polynomial(fit, removed)
        fresh.pack()
        fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
        assert torch.equal(fresh(x, position_embeddings=positions(5))[0], packed)


def test_polynomial_bf16_pack_preserves_bias_in_arithmetic_precision(tmp_path):
    native, q, means = fixture(dtype=torch.float32)
    native = native.to(torch.bfloat16)
    fit, removed = polynomial_fit(torch.float32)
    projection = head_projection(torch.float32)
    compiled = AffineQwenBlock(native, q[:, :8], means, parameter_dtype=torch.float32,
                               head_projection=projection).compile_polynomial(fit, removed)
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    pos = tuple(t.to(torch.bfloat16) for t in positions(5, torch.float32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = compiled(x, position_embeddings=pos)[0]
    y.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in compiled.compact_parameters())
    bias = compiled.polynomial_bias.detach().clone()
    compiled.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    assert compiled.mlp_output.dtype == torch.bfloat16
    assert compiled.polynomial_bias.dtype == torch.float32
    assert torch.equal(compiled.polynomial_bias, bias)
    with torch.no_grad():
        y = compiled(x, position_embeddings=pos)[0]
        assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
        path = tmp_path / "polynomial_bf16.pt"
        torch.save(compiled.state_dict(), path)
        fresh = AffineQwenBlock(native, q[:, :8], means, parameter_dtype=torch.float32,
                                head_projection=projection).compile_polynomial(fit, removed)
        fresh.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
        assert torch.equal(fresh(x, position_embeddings=pos)[0], y)


def test_mlp_capture_is_explicit_diagnostic_and_does_not_enter_state_dict():
    native, q, means = fixture()
    compiled = AffineQwenBlock(native, q[:, :8], means)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    compiled(x, position_embeddings=positions(5))
    assert compiled.last_mlp_hidden is None
    compiled.set_capture_mlp(True)
    output = compiled(x, position_embeddings=positions(5))[0]
    captured = compiled.last_mlp_hidden
    assert captured.shape == (1, 5, 28) and captured.requires_grad
    gradient = torch.autograd.grad(output.square().sum(), captured)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert not any("last_mlp_hidden" in key for key in compiled.state_dict())
    compiled.set_capture_mlp(False)
    assert compiled.last_mlp_hidden is None
    fit, removed = polynomial_fit()
    compiled.compile_polynomial(fit, removed).set_capture_mlp(True)
    compiled(x, position_embeddings=positions(5))
    assert compiled.last_mlp_hidden.shape == (1, 5, 20)
    compiled.pack()
    assert not compiled.capture_mlp and compiled.last_mlp_hidden is None


def test_invalid_or_repeated_polynomial_compilation_is_rejected():
    native, q, means = fixture()
    fit, removed = polynomial_fit()
    compiled = AffineQwenBlock(native, q[:, :8], means)
    with pytest.raises(ValueError, match="anchors cannot"):
        compiled.compile_polynomial(fit, [0, 1])
    wrong = deepcopy(fit)
    wrong["scale"][1] = 0
    with pytest.raises(ValueError, match="positive scales"):
        compiled.compile_polynomial(wrong, removed)
    wrong = deepcopy(fit)
    wrong["coefficients"]["2"] = wrong["coefficients"]["2"][:, :-1]
    with pytest.raises(ValueError, match="fit shapes"):
        compiled.compile_polynomial(wrong, removed)
    compiled.compile_polynomial(fit, removed)
    with pytest.raises(ValueError, match="fresh unpacked"):
        compiled.compile_polynomial(fit, removed)
    other = AffineQwenBlock(native, q[:, :8], means).pack()
    with pytest.raises(ValueError, match="fresh unpacked"):
        other.compile_polynomial(fit, removed)
