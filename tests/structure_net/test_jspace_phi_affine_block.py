from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Phi3Config, Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer, Phi3RotaryEmbedding, apply_rotary_pos_emb,
)

from experiments.structure_net import jspace_phi_affine_block as port


def fixture(layer=0, rank=8, dtype=torch.float32):
    torch.manual_seed(45270 + layer)
    config = Phi3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=32, original_max_position_embeddings=32,
        rope_theta=250000, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=False, resid_pdrop=0., embd_pdrop=0., attention_dropout=0.)
    config._attn_implementation = "eager"
    native = Phi3DecoderLayer(config, layer).eval().requires_grad_(False)
    with torch.no_grad():
        native.input_layernorm.weight.copy_(.3 + torch.rand(16))
        native.post_attention_layernorm.weight.copy_(.3 + torch.rand(16))
    native.to(dtype=dtype)
    basis = torch.linalg.qr(torch.randn(16, 16)).Q[:, :rank]
    means = torch.randn(7, 16) / 3
    projection = {"rope_pairs": torch.tensor([[0], [1]]),
        "value_basis": torch.stack([torch.linalg.qr(torch.randn(4, 4)).Q[:, :2] for _ in range(2)])}
    return native, basis, means, projection


def rotary(native, x):
    positions = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
    return Phi3RotaryEmbedding(native.config)(x, positions)


def causal(x):
    return torch.full((1, 1, x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1)


def relation():
    return {"anchors": [0, 2], "mean": torch.randn(24) / 20,
        "scale": torch.rand(24) + .5, "coefficients": {"2": torch.randn(24, 6) / 30}}


def projected_reference(native, u, means, projection, x, positions):
    """Independent native fused projections, then explicit physical projections."""
    anchor = means[:x.shape[1]] - (means[:x.shape[1]] @ u) @ u.T
    project = lambda value: (value @ u) @ u.T + anchor
    represented = project(x)
    b, length = x.shape[:2]
    qkv = native.self_attn.qkv_proj(native.input_layernorm(represented))
    q, k, v = qkv.split((16, 8, 8), -1)
    q = q.view(b, length, 4, 4).transpose(1, 2)
    k = k.view(b, length, 2, 4).transpose(1, 2)
    v = v.view(b, length, 2, 4).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *positions)
    cols = torch.cat((projection['rope_pairs'], projection['rope_pairs'] + 2), -1)
    q = torch.stack([q[:, i, :, cols[i // 2]] for i in range(4)], 1)
    k = torch.stack([k[:, i, :, cols[i]] for i in range(2)], 1)
    v = torch.einsum('bhld,hdf->bhlf', v, projection['value_basis'])
    probabilities = ((q @ k.repeat_interleave(2, 1).transpose(-1, -2))
                     * native.self_attn.scaling + causal(x)).softmax(-1)
    attended = probabilities @ v.repeat_interleave(2, 1)
    attended = torch.einsum('bhlf,hdf->bhld', attended, projection['value_basis'].repeat_interleave(2, 0))
    represented = project(represented + native.self_attn.o_proj(attended.transpose(1, 2).reshape(b, length, 16)))
    return project(represented + native.mlp(native.post_attention_layernorm(represented)))


def test_fused_views_are_zero_copy_and_compiler_does_not_retain_native():
    native, u, means, _ = fixture()
    before = deepcopy(native.state_dict())
    view = port.compatible_view(native)
    assert view.self_attn.q_proj.weight.untyped_storage().data_ptr() == native.self_attn.qkv_proj.weight.untyped_storage().data_ptr()
    assert torch.equal(view.self_attn.k_proj.weight, native.self_attn.qkv_proj.weight[16:24])
    assert torch.equal(view.self_attn.v_proj.weight, native.self_attn.qkv_proj.weight[24:])
    assert torch.equal(view.mlp.gate_proj.weight, native.mlp.gate_up_proj.weight[:24])
    assert torch.equal(view.mlp.up_proj.weight, native.mlp.gate_up_proj.weight[24:])
    block = port.AffinePhiBlock(native, u, means)
    native_storage = {v.untyped_storage().data_ptr() for v in native.parameters()}
    assert all(v.untyped_storage().data_ptr() not in native_storage for v in block.parameters())
    assert all(torch.equal(v, native.state_dict()[k]) for k, v in before.items())
    assert not hasattr(native, 'attention_type')


@pytest.mark.parametrize('identity', [True, False])
def test_actual_native_full_rank_output_and_input_gradient(identity):
    native, u, means, _ = fixture(rank=16)
    if identity: u = torch.eye(16)
    compiled = port.AffinePhiBlock(native, u, means, trainable=False)
    x = torch.randn(2, 5, 16, requires_grad=True); positions = rotary(native, x)
    expected = native(x, position_embeddings=positions, attention_mask=causal(x))[0]
    actual = compiled(x, position_embeddings=positions, attention_mask=causal(x))[0]
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    ge = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    ga = torch.autograd.grad(actual.square().sum(), x)[0]
    torch.testing.assert_close(ga, ge, atol=2e-5, rtol=2e-5)
    assert compiled.scale == .5 and compiled.native_head_dim == 4


def test_reduced_affine_head_pair_reference_and_gradients():
    native, u, means, projection = fixture()
    block = port.AffinePhiBlock(native, u, means, head_projection=projection)
    x = torch.randn(2, 5, 16, requires_grad=True); positions = rotary(native, x)
    actual = block(x, position_embeddings=positions)[0]
    expected = projected_reference(native, u, means, projection, x, positions)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(ga, ge, atol=3e-5, rtol=3e-5)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in block.parameters())
    assert all(p.grad is None for p in native.parameters())
    assert block.head_dim == 2 and block.scale == .5  # Original score scale, not 1/sqrt(2).


def test_reduced_cache_decode_causality_and_position_capacity():
    native, u, means, projection = fixture()
    block = port.AffinePhiBlock(native, u, means, head_projection=projection, trainable=False)
    x = torch.randn(1, 5, 16); cos, sin = rotary(native, x)
    expected = block(x, position_embeddings=(cos, sin))[0]
    cache = DynamicCache(); parts = []
    for i in range(5):
        parts.append(block(x[:, i:i+1], position_embeddings=(cos[:, i:i+1], sin[:, i:i+1]),
                           past_key_value=cache, cache_position=torch.tensor([i]))[0])
    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=3e-6, rtol=3e-6)
    changed = x.clone(); changed[:, 3:] += 10
    assert torch.equal(block(changed, position_embeddings=(cos, sin))[0][:, :3], expected[:, :3])
    with pytest.raises(IndexError):
        block(x[:, :1], position_embeddings=(cos[:, :1], sin[:, :1]), position_ids=torch.tensor([[7]]))


def test_polynomial_stack_update_packing_and_real_save_reload(tmp_path):
    blocks = []; native_states = []; natives = []
    for layer in range(3):
        native, u, means, projection = fixture(layer=layer)
        natives.append(native); native_states.append(deepcopy(native.state_dict()))
        block = port.AffinePhiBlock(native, u, means, head_projection=projection)
        block.compile_polynomial(relation(), [1, 3, 4])
        blocks.append(block)
    stack = port.CompactPhiStack(blocks, pad_to_multiple=16, checkpoint_blocks=True).eval()
    x = torch.randn(2, 5, 16); positions = rotary(native, x)
    sequential = x
    for block in blocks: sequential = block(sequential, position_embeddings=positions)[0]
    torch.testing.assert_close(stack(x, position_embeddings=positions)[0], sequential, atol=4e-6, rtol=4e-6)
    geometry = {k: v.clone() for k, v in stack.named_buffers()}
    optimizer = torch.optim.AdamW(stack.parameters(), lr=1e-3, weight_decay=0)
    stack(x, position_embeddings=positions)[0].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.parameters())
    prior = [b.qkv_weight.clone() for b in stack.blocks]
    optimizer.step()
    assert all(not torch.equal(b.qkv_weight, old) for b, old in zip(stack.blocks, prior))
    assert all(torch.equal(v, dict(stack.named_buffers())[k]) for k, v in geometry.items())
    assert all(torch.equal(value, native.state_dict()[k]) for native, state in zip(natives, native_states) for k, value in state.items())
    packed = deepcopy(stack).pack(dtype=torch.float32, buffer_dtype=torch.float32)
    reference = stack(x, position_embeddings=positions)[0]
    torch.testing.assert_close(packed(x, position_embeddings=positions)[0], reference, atol=4e-6, rtol=4e-6)
    path = tmp_path / 'phi.pt'; torch.save(packed.state_dict(), path)
    fresh = deepcopy(stack).pack(dtype=torch.float32, buffer_dtype=torch.float32)
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(fresh(x, position_embeddings=positions)[0], packed(x, position_embeddings=positions)[0])
    assert packed.blocks[1].basis is None and packed.blocks[0].anchor is None
    assert packed.get_extra_state()['architecture'] == 'phi4_full_rotary'


@pytest.mark.parametrize('reduced', [False, True])
def test_bf16_native_control_or_compact_polynomial_gradients(reduced):
    native, u, means, projection = fixture(rank=8 if reduced else 16, dtype=torch.bfloat16)
    if not reduced: u = torch.eye(16)
    block = port.AffinePhiBlock(native, u, means, head_projection=projection if reduced else None)
    if reduced: block.compile_polynomial(relation(), [1, 3, 4])
    x = torch.randn(2, 5, 16).bfloat16().requires_grad_(); positions = rotary(native, x)
    actual = block(x, position_embeddings=positions)[0]
    if not reduced:
        expected = native(x, position_embeddings=positions, attention_mask=causal(x))[0]
        torch.testing.assert_close(actual, expected, atol=.03125, rtol=.015625)
    actual.float().square().mean().backward()
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert block.basis.dtype == block.anchor.dtype == torch.float32
    packed = deepcopy(block).pack(dtype=torch.bfloat16, buffer_dtype=torch.float32)
    torch.testing.assert_close(packed(x.detach(), position_embeddings=positions)[0], actual.detach(), atol=.015625, rtol=.015625)


@pytest.mark.parametrize('sharded', [False, True])
def test_selective_local_loaders_keep_untied_head(tmp_path, sharded):
    native, *_ = fixture(); config = native.config
    model = Phi3ForCausalLM(config).eval().bfloat16()
    config.save_pretrained(tmp_path)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if sharded:
        parts = {'a.safetensors': {k: v for k, v in state.items() if k.startswith('model.layers.1.')},
                 'b.safetensors': {k: v for k, v in state.items() if not k.startswith('model.layers.1.')}}
        for name, values in parts.items(): save_file(values, tmp_path / name)
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            k: name for name, values in parts.items() for k in values}}))
    else: save_file(state, tmp_path / 'model.safetensors')
    loaded = port.load_native_block(tmp_path, 1, dtype=torch.float32)
    assert all(torch.equal(v, state['model.layers.1.' + k].float()) for k, v in loaded.state_dict().items())
    embedding = port.load_embedding(tmp_path, dtype=torch.float32)
    output, norm, frequencies = port.load_readout_norm(tmp_path, dtype=torch.float32)
    assert torch.equal(embedding, state['model.embed_tokens.weight'].float())
    assert torch.equal(output, state['lm_head.weight'].float()) and not torch.equal(embedding, output)
    assert torch.equal(norm.weight, state['model.norm.weight'].float())
    assert torch.equal(frequencies.inv_freq, port.load_rotary(tmp_path).inv_freq)
    assert all(not p.requires_grad for p in loaded.parameters())


@pytest.mark.parametrize('key,value', [('partial_rotary_factor', .5), ('rope_scaling', {'rope_type': 'linear', 'factor': 2}),
    ('resid_pdrop', .1), ('attention_dropout', .1), ('sliding_window', 8), ('tie_word_embeddings', True)])
def test_rejects_unsupported_architecture(key, value):
    native, u, means, _ = fixture()
    setattr(native.config, key, value)
    with pytest.raises(ValueError): port.AffinePhiBlock(native, u, means)


def test_strict_phi4_profile_and_checkpoint_geometry_and_quantization():
    native, u, means, _ = fixture()
    config = Phi3Config(hidden_size=5120, intermediate_size=17920, num_hidden_layers=40,
        num_attention_heads=40, num_key_value_heads=10, vocab_size=100352,
        max_position_embeddings=16384, original_max_position_embeddings=16384, rope_theta=250000,
        rms_norm_eps=1e-5, architectures=['Phi3ForCausalLM'], tie_word_embeddings=False)
    assert port.validate_phi4_config(config)['head_dim'] == 128
    with pytest.raises(ValueError, match='14B'): port.validate_phi4_config(native.config)
    config.quantization_config = {'quant_method': 'gptq', 'bits': 4}
    with pytest.raises(ValueError, match='GPTQ'): port.load_config('/not/read', config)
    block = port.AffinePhiBlock(native, u, means)
    state = deepcopy(block.state_dict()); state['_extra_state']['geometry']['rope_theta'] = 10000
    with pytest.raises(ValueError, match='rotary'): block.load_state_dict(state, strict=True)
