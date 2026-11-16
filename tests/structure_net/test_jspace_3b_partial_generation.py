from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_partial_generation as partial
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def fixture():
    torch.manual_seed(109)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=True,
        eos_token_id=31, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    blocks = []
    for index in (2, 3):
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        block = AffineQwenBlock(model.model.layers[index], basis, torch.randn(1, 8)/20)
        for p in block.parameters():
            p.data = p.data.bfloat16()
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=16)
    stack.eval().requires_grad_(False).pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16, strip_unused=True)
    norm = deepcopy(model.model.norm)
    norm.weight.copy_(torch.linspace(.5, 2., 8).bfloat16())
    return model, stack, norm


@torch.inference_mode()
def test_partial_hf_forward_preserves_prefix_and_uses_trained_suffix_norm():
    model, stack, norm = fixture()
    original = deepcopy(model)
    prefix = list(model.model.layers[:2])
    tied = model.lm_head.weight
    partial.attach_suffix(model, stack, norm, cut=2)
    ids = torch.tensor([[2, 5, 9, 1]])
    saved = []
    hook = original.model.layers[2].register_forward_pre_hook(lambda _m, a: saved.append(a[0]))
    original.model(input_ids=ids, use_cache=False)
    hook.remove()
    hidden = saved[0]
    positions = torch.arange(ids.shape[1])[None]
    mask = torch.full((4, 4), torch.finfo(hidden.dtype).min, dtype=hidden.dtype).triu(1)[None, None]
    expected = norm(stack(hidden, attention_mask=mask, position_ids=positions,
        cache_position=positions[0], position_embeddings=original.model.rotary_emb(hidden, positions), use_cache=False)[0])
    actual = model.model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(actual, expected)
    assert all(a is b for a, b in zip(prefix, model.model.layers[:2]))
    assert model.lm_head.weight is tied and model.model.embed_tokens.weight is tied
    assert model.config.num_hidden_layers == 4 and len(model.model.layers) == 3
    assert model.model.norm is norm and not model.generation_config.use_cache


class Tokenizer:
    def decode(self, values, **kwargs):
        return ",".join(map(str, values))


@torch.inference_mode()
def test_generation_uses_no_cache_and_checks_guard_each_forward():
    model, stack, norm = fixture()
    partial.attach_suffix(model, stack, norm, cut=2)
    calls = []
    def guard():
        calls.append(True)
        return {"active_lease": "sentinel-must-not-replace-forward-arguments"}
    output = partial.generate(model, Tokenizer(), [2, 5, 9], max_new_tokens=3,
        position_capacity=8, guard=guard)
    assert output["input_ids"] == [2, 5, 9]
    assert 1 <= len(output["output_ids"]) <= 3
    assert len(calls) >= len(output["output_ids"])+2
    assert output["generation_config"]["use_cache"] is False
    assert output["generation_config"]["do_sample"] is False
    with pytest.raises(ValueError, match="capacity"):
        partial.generate(model, Tokenizer(), [1, 2], max_new_tokens=7, position_capacity=8)


def test_incomplete_or_wrong_cut_suffix_rejected():
    model, stack, norm = fixture()
    with pytest.raises(ValueError, match="partial suffix"):
        partial.attach_suffix(model, stack, norm, cut=1)


@torch.inference_mode()
def test_suffix_has_its_own_rotary_without_changing_native_prefix():
    model, stack, norm = fixture()
    native = deepcopy(model)
    suffix_rotary = partial.training_rotary(model.config, "cpu")
    # Simulate a construction-device difference, magnified so this CPU test
    # detects an accidental reuse of the native-prefix rotary.
    suffix_rotary.inv_freq.mul_(11.)
    native_rotary = model.model.rotary_emb
    original_bits = native_rotary.inv_freq.clone()
    ids = torch.tensor([[2, 5, 9, 1, 7, 3]])
    captured = []
    hook = native.model.layers[2].register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    native.model(input_ids=ids, use_cache=False)
    hook.remove()
    hidden = captured[0]
    positions = torch.arange(ids.shape[1])[None]
    mask = torch.full((6, 6), torch.finfo(hidden.dtype).min, dtype=hidden.dtype).triu(1)[None, None]
    kwargs = dict(attention_mask=mask, position_ids=positions, cache_position=positions[0], use_cache=False)
    expected = norm(stack(hidden, position_embeddings=suffix_rotary(hidden, positions), **kwargs)[0])
    wrong = norm(stack(hidden, position_embeddings=native_rotary(hidden, positions), **kwargs)[0])
    partial.attach_suffix(model, stack, norm, cut=2, suffix_rotary=suffix_rotary)
    observed = []
    hook = model.model.layers[2].register_forward_pre_hook(lambda _m, args: observed.append(args[0]))
    actual = model.model(input_ids=ids, use_cache=False).last_hidden_state
    hook.remove()
    assert torch.equal(observed[0], hidden)  # Original native prefix, bit for bit.
    assert torch.equal(actual, expected) and not torch.equal(actual, wrong)
    assert model.model.rotary_emb is native_rotary
    assert torch.equal(native_rotary.inv_freq, original_bits)
    assert model.model.layers[2].rotary is suffix_rotary
    with pytest.raises(ValueError, match="cache-free"):
        model.model.layers[2](hidden, position_ids=positions, use_cache=True)


def test_whole_fixed_state_includes_rotary_and_geometry():
    model, stack, _ = fixture()
    rotary = partial.training_rotary(model.config, "cpu")
    expected = partial.fixed_state(stack, model.lm_head.weight, rotary)
    assert partial.assert_fixed_state(stack, model.lm_head.weight, rotary, expected)["passed"]
    rotary.inv_freq[0] += .01
    with pytest.raises(ValueError, match="Complete fixed"):
        partial.assert_fixed_state(stack, model.lm_head.weight, rotary, expected)
    rotary = partial.training_rotary(model.config, "cpu")
    stack.blocks[0].basis.flatten()[0] += .01
    with pytest.raises(ValueError, match="Complete fixed"):
        partial.assert_fixed_state(stack, model.lm_head.weight, rotary, expected)


def test_training_rotary_denies_unreserved_cuda_before_constructor(monkeypatch):
    monkeypatch.setattr(partial,"Qwen2RotaryEmbedding",lambda **kw:pytest.fail("Constructor before reservation"))
    with pytest.raises(ValueError,match="reservation guard"):
        partial.training_rotary(None,"cuda:0")


def test_build_checks_original_whole_state_before_packing_and_attaches_recreated_rotary(monkeypatch):
    model, stack, norm = fixture()
    reference = partial.training_rotary(model.config,"cpu")
    expected = partial.fixed_state(stack,model.lm_head.weight,reference)
    # A CPU reconstruction's nonpersistent buffers must never be used merely
    # because every checkpoint tensor loaded strictly.
    stale = deepcopy(reference);stale.inv_freq.mul_(2)
    proof = dict(new_layer=None,unchanged_layers=[2,3])
    monkeypatch.setattr(partial.execution,"load_native_model",lambda *a,**kw:model)
    monkeypatch.setattr(partial.builder,"build_stack",lambda *a,**kw:(stack,norm,model.lm_head.weight,stale,proof))
    calls=[]
    monkeypatch.setattr(partial.builder,"packed_clone",lambda block,**kw:calls.append("pack") or deepcopy(block))
    wrong=dict(expected,sha256="changed")
    with pytest.raises(ValueError,match="Complete fixed"):
        partial.build_partial("unused",{},"checkpoint",cut=2,expected_fixed_state=wrong)
    assert calls==[]
    built=partial.build_partial("unused",{},"checkpoint",cut=2,expected_fixed_state=expected)
    assert calls==["pack"] and built.partial_fixed_state_receipt["actual"]==expected
    assert torch.equal(built.model.layers[2].rotary.inv_freq,reference.inv_freq)
    assert not torch.equal(built.model.layers[2].rotary.inv_freq,stale.inv_freq)
