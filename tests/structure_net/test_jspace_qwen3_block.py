from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, apply_rotary_pos_emb

from experiments.structure_net import jspace_qwen3_block as port


def fixture(layer=0, head_dim=4, rank=8):
    torch.manual_seed(21401 + layer)
    config = Qwen3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
        vocab_size=32, max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=False)
    config._attn_implementation = "eager"
    native = Qwen3DecoderLayer(config, layer).float().eval().requires_grad_(False)
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if "norm.weight" in name: parameter.copy_(.3 + torch.rand_like(parameter))
    basis = torch.linalg.qr(torch.randn(16, 16)).Q[:, :rank]
    means = torch.randn(1, 16) / 3
    projection = {"rope_pairs": torch.tensor([[0], [head_dim // 2 - 1]]),
        "value_basis": torch.stack([torch.linalg.qr(torch.randn(head_dim, head_dim)).Q[:, :2] for _ in range(2)])}
    return native, basis, means, projection


def rope(native, x):
    ids = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
    return Qwen3RotaryEmbedding(native.self_attn.config)(x, ids)


def mask(length):
    return torch.full((1, 1, length, length), torch.finfo(torch.float32).min).triu(1)


def projected_reference(native, u, mean, projection, x, position_embeddings):
    anchor = mean - (mean @ u) @ u.T
    project = lambda value: (value @ u) @ u.T + anchor
    represented = project(x)
    h = native.input_layernorm(represented)
    b, l = h.shape[:2]; d = native.self_attn.head_dim
    a = native.self_attn
    q = a.q_norm(a.q_proj(h).view(b, l, 4, d)).transpose(1, 2)
    k = a.k_norm(a.k_proj(h).view(b, l, 2, d)).transpose(1, 2)
    v = a.v_proj(h).view(b, l, 2, d).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    cols = torch.cat((projection["rope_pairs"], projection["rope_pairs"] + d//2), -1)
    q = torch.stack([q[:, i, :, cols[i//2]] for i in range(4)], 1)
    k = torch.stack([k[:, i, :, cols[i]] for i in range(2)], 1)
    v = torch.einsum("bhld,hdf->bhlf", v, projection["value_basis"])
    probability = ((q @ k.repeat_interleave(2, 1).transpose(-1, -2)) * a.scaling + mask(l)).softmax(-1)
    mixed = probability @ v.repeat_interleave(2, 1)
    mixed = torch.einsum("bhlf,hdf->bhld", mixed, projection["value_basis"].repeat_interleave(2, 0))
    represented = project(represented + a.o_proj(mixed.transpose(1, 2).reshape(b, l, -1)))
    return project(represented + native.mlp(native.post_attention_layernorm(represented)))


@pytest.mark.parametrize("head_dim", [4, 8])
def test_actual_native_full_rank_output_and_input_gradient(head_dim):
    native, u, means, _ = fixture(head_dim=head_dim, rank=16)
    block = port.Qwen3AffineBlock(native, u, means, trainable=False)
    x = torch.randn(2, 5, 16, requires_grad=True); positions = rope(native, x)
    expected = native(x, position_embeddings=positions, attention_mask=mask(5))[0]
    actual = block(x, position_embeddings=positions, attention_mask=mask(5))[0]
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    ge = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    ga = torch.autograd.grad(actual.square().sum(), x)[0]
    torch.testing.assert_close(ga, ge, atol=2e-5, rtol=2e-5)
    assert torch.equal(block.q_norm_weight, native.self_attn.q_norm.weight)
    assert block.scale == native.self_attn.scaling


def test_reduced_reference_preserves_native_norm_before_pair_selection_and_gradients():
    native, u, means, projection = fixture(head_dim=8)
    block = port.Qwen3AffineBlock(native, u, means, head_projection=projection)
    x = torch.randn(2, 5, 16, requires_grad=True); positions = rope(native, x)
    expected = projected_reference(native, u, means, projection, x, positions)
    actual = block(x, position_embeddings=positions)[0]
    torch.testing.assert_close(actual, expected, atol=8e-6, rtol=8e-6)
    ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(ga, ge, atol=4e-5, rtol=4e-5)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in block.parameters())
    assert all(p.grad is None for p in native.parameters())
    assert block.qkv_weight.shape[0] == (4+2)*8 + 2*2
    # Discarded Q channels affect the true normalization denominator.
    q, k = torch.randn(1, 4, 2, 8), torch.randn(1, 2, 2, 8)
    before = block._normalized_selected_qk(q, k)[0]
    changed = q.clone(); changed[..., 1] += 50
    after = block._normalized_selected_qk(changed, k)[0]
    assert not torch.allclose(before[:, :2], after[:, :2])


@pytest.mark.parametrize("static", [False, True])
def test_compact_causal_prefill_decode_cache(static):
    native, u, means, projection = fixture()
    block = port.Qwen3AffineBlock(native, u, means, head_projection=projection, trainable=False)
    x = torch.randn(1, 5, 16); cos, sin = rope(native, x)
    expected = block(x, position_embeddings=(cos, sin))[0]
    cache = StaticCache(native.self_attn.config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float32) if static else DynamicCache()
    pieces = []
    for i in range(5):
        attention_mask = None
        if static:
            attention_mask = torch.zeros(1, 1, 1, 8); attention_mask[..., i+1:] = torch.finfo(torch.float32).min
        pieces.append(block(x[:, i:i+1], position_embeddings=(cos[:, i:i+1], sin[:, i:i+1]),
            cache_position=torch.tensor([i]), past_key_value=cache, attention_mask=attention_mask)[0])
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=3e-6, rtol=3e-6)
    altered = x.clone(); altered[:, 3:] += 10
    torch.testing.assert_close(block(altered, position_embeddings=(cos, sin))[0][:, :3], expected[:, :3])


def polynomial():
    return {"anchors": [0, 2], "mean": torch.randn(24)/10, "scale": torch.rand(24)+.5,
        "coefficients": {"2": torch.randn(24, 6)/30}}


def test_polynomial_packing_padding_strict_reload_and_stack_bridges():
    blocks = []
    for layer in range(3):
        native, u, means, projection = fixture(layer=layer)
        blocks.append(port.Qwen3AffineBlock(native, u, means, head_projection=projection))
    blocks[1].compile_polynomial(polynomial(), [1, 3, 4])
    stack = port.Qwen3CompactStack(deepcopy(blocks), pad_to_multiple=16).eval()
    x = torch.randn(1, 5, 16); positions = rope(native, x)
    expected = x
    for block in blocks: expected = block(expected, position_embeddings=positions)[0]
    actual = stack(x, position_embeddings=positions)[0]
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    packed = deepcopy(stack).pack()
    torch.testing.assert_close(packed(x, position_embeddings=positions)[0], actual, atol=4e-6, rtol=4e-6)
    reloaded = deepcopy(stack).pack(); reloaded.load_state_dict(packed.state_dict(), strict=True)
    assert torch.equal(reloaded(x, position_embeddings=positions)[0], packed(x, position_embeddings=positions)[0])
    assert packed.blocks[1].basis is None and packed.blocks[0].anchor is None
    assert all(v.dtype == torch.float32 for v in packed.state_dict().values() if isinstance(v, torch.Tensor) and v.is_floating_point())
    assert packed.rank == 8 and packed.layer_indices == (0, 1, 2)


def test_checkpointed_stack_matches_gradients_through_multiple_blocks():
    blocks = []
    for layer in range(2):
        n, u, m, p = fixture(layer=layer)
        blocks.append(port.Qwen3AffineBlock(n, u, m, head_projection=p))
    ordinary = port.Qwen3CompactStack(blocks)
    checked = deepcopy(ordinary); checked.checkpoint_blocks = True
    x = torch.randn(1, 4, 16); positions = rope(n, x)
    for module in (ordinary, checked): module(x, position_embeddings=positions)[0].square().mean().backward()
    for first, second in zip(ordinary.parameters(), checked.parameters()):
        torch.testing.assert_close(first.grad, second.grad)


@pytest.mark.parametrize("sharded", [False, True])
def test_fp32_loaders_preserve_untied_embedding_and_head(tmp_path, sharded):
    native, *_ = fixture()
    config = native.self_attn.config
    model = Qwen3ForCausalLM(config).eval().bfloat16()
    config.save_pretrained(tmp_path)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if sharded:
        parts = {"a.safetensors": {k: v for k,v in state.items() if k.startswith("model.layers.1.")},
                 "b.safetensors": {k: v for k,v in state.items() if not k.startswith("model.layers.1.")}}
        for name, values in parts.items(): save_file(values, tmp_path/name)
        (tmp_path/'model.safetensors.index.json').write_text(json.dumps({"weight_map": {k:name for name, values in parts.items() for k in values}}))
    else: save_file(state, tmp_path/'model.safetensors')
    block = port.load_native_block(tmp_path, 1, dtype=torch.float32)
    assert all(torch.equal(v, state['model.layers.1.'+k].float()) for k,v in block.state_dict().items())
    embedding = port.load_embedding(tmp_path, dtype=torch.float32)
    readout, norm, rotary = port.load_readout_norm(tmp_path, dtype=torch.float32)
    assert torch.equal(embedding, state['model.embed_tokens.weight'].float())
    assert torch.equal(readout, state['lm_head.weight'].float())
    assert not torch.equal(embedding, readout)
    assert torch.equal(norm.weight, state['model.norm.weight'].float())
    assert torch.equal(port.load_rotary(tmp_path).inv_freq, rotary.inv_freq)
    assert all(not p.requires_grad for p in block.parameters())


def test_rejects_precision_changes_and_wrong_architecture():
    n, u, m, p = fixture()
    with pytest.raises(ValueError, match="FP32"): port.Qwen3AffineBlock(n, u, m, parameter_dtype=torch.float16)
    block = port.Qwen3AffineBlock(n, u, m, head_projection=p)
    x = torch.randn(1, 2, 16); positions = rope(n, x)
    with pytest.raises(ValueError, match="FP32"): block(x.half(), position_embeddings=positions)
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match="Autocast"):
        block(x, position_embeddings=positions)
    with pytest.raises(ValueError, match="FP32"): block.pack(dtype=torch.float16)
    n.self_attn.config.model_type = 'qwen2'
    with pytest.raises(ValueError, match="Qwen3"): port.Qwen3AffineBlock(n, u, m)


@pytest.mark.parametrize("rank,heads", [(16, False), (8, True)])
def test_bf16_native_control_and_compact_gradient_packing(rank, heads):
    native, u, means, projection = fixture(rank=rank)
    native = native.bfloat16()
    block = port.Qwen3AffineBlock(native, u, means, head_projection=projection if heads else None)
    x = torch.randn(2, 5, 16).bfloat16().requires_grad_(True); positions = rope(native, x)
    output = block(x, position_embeddings=positions)[0]
    if rank == 16:
        expected = native(x, position_embeddings=positions, attention_mask=mask(5).bfloat16())[0]
        torch.testing.assert_close(output, expected, atol=.03125, rtol=.03125)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.dtype == torch.bfloat16 and p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert block.basis.dtype == block.anchor.dtype == block.qkv_shift.dtype == torch.float32
    packed = deepcopy(block).pack()
    reloaded = deepcopy(block).pack(); reloaded.load_state_dict(packed.state_dict(), strict=True)
    actual = packed(x.detach(), position_embeddings=positions)[0]
    assert torch.equal(actual, reloaded(x.detach(), position_embeddings=positions)[0])
    torch.testing.assert_close(actual, output.detach(), atol=.015625, rtol=.015625)


def test_bf16_polynomial_stack_keeps_fp32_geometry_and_strict_reload():
    blocks = []
    for layer in range(2):
        native, u, mean, projection = fixture(layer=layer)
        block = port.Qwen3AffineBlock(native.bfloat16(), u, mean, head_projection=projection)
        block.compile_polynomial(polynomial(), [1,3,4])
        blocks.append(block)
    stack = port.Qwen3CompactStack(blocks, checkpoint_blocks=True).eval()
    x = torch.randn(1,4,16).bfloat16(); positions = rope(native, x)
    before = stack(x, position_embeddings=positions)[0]
    before.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.parameters())
    assert all(b.weight.dtype == b.shift.dtype == torch.float32 for b in stack.bridges)
    packed = deepcopy(stack).pack()
    after = packed(x, position_embeddings=positions)[0]
    torch.testing.assert_close(after, before.detach(), atol=.03125, rtol=.03125)
    other = deepcopy(stack).pack(); other.load_state_dict(packed.state_dict(), strict=True)
    assert torch.equal(after, other(x, position_embeddings=positions)[0])
    assert packed.blocks[0].basis.dtype == torch.float32
    assert packed.blocks[-1].anchor.dtype == torch.float32
    assert packed.blocks[0].qkv_weight.dtype == torch.bfloat16
