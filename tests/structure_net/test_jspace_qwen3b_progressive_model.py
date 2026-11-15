from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_qwen3b_progressive_model as model
from experiments.structure_net.jspace_qwen_parallel_model import state_dict_cpu


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(9873)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=9,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=16,
        max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = "eager"
    native = Qwen2ForCausalLM(config).bfloat16().eval()
    path = tmp_path / "native"; path.mkdir(); config.save_pretrained(path)
    save_file({k: v.detach().clone() for k, v in native.state_dict().items()
               if k != "lm_head.weight"}, path / "model.safetensors")
    fits = {}
    for layer in range(9):
        directory = tmp_path / f"fit_{layer}"; directory.mkdir(); fits[layer] = directory
        fit = dict(basis=torch.linalg.qr(torch.randn(8, 8)).Q, means=torch.randn(6, 8)/10,
            rope_pairs=torch.tensor([[0]]), value_basis=torch.eye(4)[None, :, :2])
        relation = dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
            coefficients={"2": torch.randn(12, 6)/100}, removed=torch.tensor([10, 11]))
        torch.save(fit, directory / "fit.pt"); torch.save(relation, directory / "relation.pt")
    return path, fits, config, native


def build(source, cut, previous=None, **kw):
    path, fits, config, _ = source
    return model.build_stack(path, fits, cut, previous, rank=4, config=config, **kw)


def recovered(source):
    stack, norm, *_ = build(source, 2)
    with torch.no_grad():
        for parameter in stack.parameters(): parameter.add_(.015625)
        norm.weight.fill_(1.5)
        # Detect accidental bridge recomputation even when old FP32 fits exist.
        stack.bridges[2].shift.add_(.004)
    return dict(stack=state_dict_cpu(stack), norm=state_dict_cpu(norm))


def forward(stack, rotary, x):
    positions = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
    return stack(x, position_ids=positions, position_embeddings=rotary(x, positions))[0]


def test_eight_block_expansion_preserves_seven_parent_blocks_norm_and_bridges(source):
    parent = recovered(source); parent_copy = deepcopy(parent)
    calls = []
    def loader(path, layer, **kw):
        calls.append(layer)
        return model.native.load_native_block(path, layer, **kw)
    stack, norm, embedding, rotary, proof = build(source, 1, parent, native_loader=loader)
    assert calls == list(range(1, 9))
    assert proof == dict(passed=True, unchanged_layers=list(range(2, 9)),
        interior_bridges_preserved=6, new_layer=1, new_bridge=[1, 2], norm_exact=True)
    assert torch.equal(norm.weight, parent["norm"]["weight"])
    assert model.state_equal(parent["stack"], parent_copy["stack"])
    for index, block in enumerate(stack.blocks[1:]):
        prefix = f"blocks.{index}."
        expected = {k[len(prefix):]: v for k, v in parent["stack"].items() if k.startswith(prefix)}
        assert model.state_equal(block.state_dict(), expected)
    first = torch.load(source[1][1]/"fit.pt", weights_only=True)
    second = torch.load(source[1][2]/"fit.pt", weights_only=True)
    u, v = first["basis"][:, :4], second["basis"][:, :4]
    assert torch.equal(stack.bridges[0].weight, (v.T @ u).bfloat16())
    delta = stack.blocks[0].anchor - stack.blocks[1].anchor
    assert torch.equal(stack.bridges[0].shift, delta @ v)
    assert all(b.basis.dtype == torch.bfloat16 and b.anchor.dtype == torch.float32 for b in stack.blocks)
    assert all(b.has_polynomial and b.intermediate_size == 10 and b.head_dim == 2 for b in stack.blocks)
    assert all(not b.requires_grad for b in stack.buffers()) and not embedding.requires_grad


def test_same_cut_checkpoint_path_is_exact_and_frozen_teacher_retains_input_gradient(source, tmp_path):
    parent = recovered(source); checkpoint = tmp_path / "selected.pt"; torch.save(parent, checkpoint)
    a, norm, _, rotary, proof = build(source, 2, checkpoint, trainable=False)
    assert proof["new_layer"] is None and proof["interior_bridges_preserved"] == 6
    assert model.state_equal(a.state_dict(), parent["stack"])
    assert model.state_equal(norm.state_dict(), parent["norm"])
    x = torch.randn(1, 4, 8).bfloat16().requires_grad_()
    forward(a, rotary, x).float().square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in a.parameters())


def test_packed_clone_preserves_live_state_reload_and_cached_decode(source):
    stack, norm, _, rotary, _ = build(source, 1, recovered(source))
    state = state_dict_cpu(stack)
    x = torch.randn(1, 5, 8).bfloat16()
    expected = forward(stack, rotary, x)
    packed = model.packed_clone(stack)
    actual = forward(packed, rotary, x)
    torch.testing.assert_close(actual, expected, atol=.0625, rtol=.0625)
    assert model.state_equal(stack.state_dict(), state) and not stack.packed
    assert packed.packed and packed.stripped and all(not p.requires_grad for p in packed.parameters())
    copy = model.packed_clone(stack); copy.load_state_dict(packed.state_dict(), strict=True)
    assert torch.equal(forward(copy, rotary, x), actual)
    cache = DynamicCache(); pieces = []
    for i in range(x.shape[1]):
        ids = torch.tensor([[i]])
        pieces.append(packed(x[:, i:i+1], position_ids=ids, cache_position=ids[0],
            position_embeddings=rotary(x[:, i:i+1], ids), past_key_value=cache, use_cache=True)[0])
    torch.testing.assert_close(torch.cat(pieces, 1), actual, atol=.0625, rtol=.0625)
    assert cache.get_seq_length(1) == 5 and cache.get_seq_length(8) == 5


def test_all_eight_blocks_and_norm_get_gradients_but_geometry_stays_fixed(source):
    stack, norm, embedding, rotary, _ = build(source, 1, recovered(source))
    before = {k: v.clone() for k, v in stack.named_buffers()}
    x = torch.randn(1, 4, 8).bfloat16()
    logits = torch.nn.functional.linear(norm(forward(stack, rotary, x)), embedding)
    logits.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.parameters())
    assert all(any(p.grad.abs().sum() > 0 for p in block.parameters()) for block in stack.blocks)
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0
    assert all(torch.equal(value, before[name]) for name, value in stack.named_buffers())
    assert embedding.grad is None


@pytest.mark.parametrize("cut", [0, 1, 8])
def test_inventory_actual_arbitrary_suffix_and_zero_prefix(source, cut):
    stack, norm, embedding, rotary, _ = build(source, cut)
    result = model.inventory(source[0], stack, norm, rotary, config=source[2])
    native = source[3]
    prefix = sum(model.tensor_bytes(b) for b in native.model.layers[:cut])
    # Use the actual deployment rotary, not the fixture's extra BF16 rotary.
    baseline = model.tensor_bytes(native.model.layers, native.model.embed_tokens,
                                 native.model.norm, rotary)
    assert result["frozen_native_prefix_bytes"] == prefix
    assert (prefix == 0) == (cut == 0)
    assert result["tied_embedding_readout_bytes"] == embedding.numel() * 2
    assert result["native_full_tensor_bytes"] == baseline
    assert result["full_model_tensor_bytes"] == prefix + model.tensor_bytes(embedding, norm, rotary, model.packed_clone(stack))
    assert result["native_treated_suffix_and_norm_bytes"] == baseline-prefix-model.tensor_bytes(embedding, rotary)


def test_wrong_basis_missing_fit_packed_parent_and_skipped_layer_are_rejected(source):
    parent = recovered(source)
    with pytest.raises(ValueError, match="exactly one"):
        build(source, 0, parent)
    paths = dict(source[1]); del paths[3]
    with pytest.raises(ValueError, match="Missing explicit"):
        model.build_stack(source[0], paths, 1, rank=4, config=source[2])
    paths = dict(source[1]); paths[2] = paths[3]
    with pytest.raises(ValueError, match="basis mismatch"):
        model.build_stack(source[0], paths, 1, parent, rank=4, config=source[2])
    bad = deepcopy(parent); bad["stack"]["_extra_state"]["packed"] = True
    with pytest.raises(ValueError, match="unpacked"):
        build(source, 1, bad)
    bad = deepcopy(parent); del bad["stack"]["blocks.0.qkv_weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        build(source, 1, bad)
