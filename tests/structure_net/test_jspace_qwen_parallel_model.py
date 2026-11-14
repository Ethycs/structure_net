from copy import deepcopy
import json

import pytest
from safetensors.torch import save_file
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_qwen_parallel_model as builder


def model_fixture(tmp_path, sharded=False):
    torch.manual_seed(551)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=2, vocab_size=32,
                         attention_dropout=0., max_position_embeddings=32, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    path = tmp_path / "native"; path.mkdir(parents=True); config.save_pretrained(path)
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    if sharded:
        first = {k: v for k, v in state.items() if k.startswith("model.layers.1.")}
        second = {k: v for k, v in state.items() if k not in first}
        save_file(first, path / "one.safetensors"); save_file(second, path / "two.safetensors")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            **{k: "one.safetensors" for k in first}, **{k: "two.safetensors" for k in second}}}))
    else:
        save_file(state, path / "model.safetensors")
    root = tmp_path / "campaign"; root.mkdir()
    (root / "plan.json").write_text(json.dumps({"model": {"path": str(path)}, "config": {"rank": 8}}))
    for layer in (0, 1):
        directory = builder.group_fit_path(root, layer); directory.mkdir(parents=True)
        basis = torch.linalg.qr(torch.randn(16, 16)).Q
        fit = {"basis": basis, "means": torch.randn(8, 16) / 4,
               "rope_pairs": torch.tensor([[0], [1]]),
               "value_basis": torch.eye(4).repeat(2, 1, 1)[:, :, :2]}
        relation = {"anchors": [0, 2, 5, 8], "mean": torch.randn(28) / 5, "scale": torch.rand(28) + .7,
                    "coefficients": {"2": torch.randn(28, 15) / 30}, "removed": [1, 3, 4, 6, 9, 12]}
        torch.save(fit, directory / "fit.pt"); torch.save(relation, directory / "relation.pt")
    return root, path, model, config


@pytest.mark.parametrize("sharded", [False, True])
def test_selective_native_loading_uses_meta_and_only_requested_keys(tmp_path, monkeypatch, sharded):
    _, path, model, config = model_fixture(tmp_path, sharded)
    original_class = builder.Qwen2DecoderLayer
    seen = []
    def construct(*args, **kwargs):
        block = original_class(*args, **kwargs)
        seen.append(all(p.is_meta for p in block.parameters()))
        return block
    monkeypatch.setattr(builder, "Qwen2DecoderLayer", construct)
    monkeypatch.setattr(builder, "load_small_qwen", lambda *a, **k: pytest.fail("Full model loaded"))
    block = builder.load_native_block(path, 1, dtype=torch.float32, config=config)
    assert seen == [True] and block.self_attn.layer_idx == 1
    assert all(torch.equal(v, model.model.layers[1].state_dict()[k]) for k, v in block.state_dict().items())
    assert all(not p.is_meta and not p.requires_grad for p in block.parameters())
    readout, norm, rotary = builder.load_readout_norm(path, dtype=torch.float32)
    assert torch.equal(readout, model.lm_head.weight)
    assert torch.equal(norm.weight, model.model.norm.weight)
    x, ids = torch.randn(1, 4, 16), torch.arange(4)[None]
    actual, expected = rotary(x, ids), model.model.rotary_emb(x, ids)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def save_recovery(root, stack, norm):
    path = builder.recovery_checkpoint_path(root, 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stack": builder.state_dict_cpu(stack), "norm": builder.state_dict_cpu(norm)}, path)
    return path


def test_warmstart_maps_original_layers_and_restores_fp32_basis_before_new_bridge(tmp_path):
    root, _, model, _ = model_fixture(tmp_path)
    previous = builder.build_stack(root, 1, True)
    with torch.no_grad():
        previous.blocks[0].qkv_weight.add_(.125)
        previous.blocks[0].qkv_offset.add_(.25)
        model.model.norm.weight.add_(.2)
    save_recovery(root, previous, model.model.norm)
    recovered = builder.load_recovered_checkpoint(root, 5)
    assert isinstance(recovered["stack"]["_extra_state"], dict)
    stack = builder.build_stack(root, 0, True, 5)
    assert stack.layer_indices == (0, 1) and not stack.packed
    for name, parameter in previous.blocks[0].named_parameters():
        assert torch.equal(parameter, dict(stack.blocks[1].named_parameters())[name]), name
    fit0 = torch.load(builder.group_fit_path(root, 0) / "fit.pt", weights_only=True)
    fit1 = torch.load(builder.group_fit_path(root, 1) / "fit.pt", weights_only=True)
    expected_bridge = (fit1["basis"][:, :8].T @ fit0["basis"][:, :8]).bfloat16()
    assert torch.equal(stack.bridges[0].weight, expected_bridge)
    assert all(b.basis.dtype == torch.bfloat16 and b.qkv_shift.dtype == torch.float32 for b in stack.blocks)
    assert all(p.requires_grad for p in stack.parameters())
    assert all(p.dtype == (torch.float32 if name.endswith("polynomial_bias") else torch.bfloat16)
               for name, p in stack.named_parameters())
    cloned = builder.state_dict_cpu(stack)
    cloned["_extra_state"]["geometry_mode"] = "changed"
    assert stack.get_extra_state()["geometry_mode"] == "bf16"


def test_teacher_loads_model_once_replaces_only_suffix_and_preserves_norm_and_embedding_tie(tmp_path):
    root, _, native, config = model_fixture(tmp_path)
    previous = builder.build_stack(root, 1, True)
    with torch.no_grad():
        native.model.norm.weight.add_(.3)
    save_recovery(root, previous, native.model.norm)
    calls = []
    def loader(path, **kwargs):
        calls.append(path)
        return deepcopy(native).to(dtype=kwargs["dtype"]), "local-tokenizer"
    teacher, tokenizer = builder.build_teacher(root, 5, device="cpu", dtype=torch.bfloat16,
                                               config=config, model_loader=loader)
    assert len(calls) == 1 and tokenizer == "local-tokenizer"
    assert teacher.config.num_hidden_layers == 2 and len(teacher.model.layers) == 2
    stack = teacher.model.layers[-1]
    assert stack.packed and stack.stripped and stack.layer_indices == (1,)
    assert teacher.lm_head.weight.data_ptr() == teacher.model.embed_tokens.weight.data_ptr()
    assert torch.equal(teacher.model.norm.weight, native.model.norm.weight.bfloat16())
    assert torch.equal(teacher.model.layers[0].self_attn.q_proj.weight, native.model.layers[0].self_attn.q_proj.weight.bfloat16())
    with torch.no_grad():
        result = teacher(torch.tensor([[3, 5, 7]]), use_cache=True)
    assert torch.isfinite(result.logits).all() and len(result.past_key_values.key_cache) == 2


def test_inference_builder_strips_internal_geometry_and_worker_override_is_preserved(tmp_path):
    root, _, _, _ = model_fixture(tmp_path)
    stack = builder.build_stack(root, 0, False)
    assert stack.packed and stack.stripped and stack.blocks[0].anchor is None
    assert all(not p.requires_grad for p in stack.parameters())
    assert all(p.dtype == (torch.float32 if name.endswith("polynomial_bias") else torch.bfloat16)
               for name, p in stack.named_parameters())
    worker = builder.build_stack(root, 0, True, dtype=torch.float16,
                                  geometry_compute_dtype=torch.float16, checkpoint_blocks=True)
    assert worker.geometry_compute_dtype == torch.float16 and worker.checkpoint_blocks


def test_group_mapping_and_incompatible_recovery_are_rejected(tmp_path):
    assert builder.group_fit_path(tmp_path, 35) == tmp_path / "groups/group_00/block_35"
    assert builder.group_fit_path(tmp_path, 30) == tmp_path / "groups/group_00/block_30"
    assert builder.group_fit_path(tmp_path, 29) == tmp_path / "groups/group_01/block_29"
    assert builder.group_fit_path(tmp_path, 0) == tmp_path / "groups/group_05/block_00"
    with pytest.raises(ValueError):
        builder.group_fit_path(tmp_path, 36)
    root, _, model, _ = model_fixture(tmp_path / "nested")
    previous = builder.build_stack(root, 0, True)
    save_recovery(root, previous, model.model.norm)
    with pytest.raises(ValueError, match="contained contiguous suffix"):
        builder.build_stack(root, 1, True, 5)
    previous.pack()
    save_recovery(root, previous, model.model.norm)
    with pytest.raises(ValueError, match="unpacked, unstripped"):
        builder.load_recovered_checkpoint(root, 5)


def test_full_rank_warmstart_does_not_overwrite_original_fit_geometry(tmp_path):
    root, _, model, _ = model_fixture(tmp_path)
    previous = builder.build_stack(root, 1, True, rank=16)
    save_recovery(root, previous, model.model.norm)
    stack = builder.build_stack(root, 0, True, 5, rank=16)
    first = torch.load(builder.group_fit_path(root, 0) / "fit.pt", weights_only=True)["basis"]
    second = torch.load(builder.group_fit_path(root, 1) / "fit.pt", weights_only=True)["basis"]
    assert torch.equal(stack.bridges[0].weight, (second.T @ first).bfloat16())


@pytest.mark.parametrize("stabilized", [False, True])
def test_systems_stack_compiles_one_native_block_and_labels_repeated_initialization(tmp_path, monkeypatch, stabilized):
    root, path, _, config = model_fixture(tmp_path)
    original = builder.load_native_block
    calls = []
    def load(*args, **kwargs):
        calls.append(args[1]); return original(*args, **kwargs)
    monkeypatch.setattr(builder, "load_native_block", load)
    fit = builder.group_fit_path(root, 1)
    stack = builder.systems_stack(path, fit / "fit.pt", fit / "relation.pt", blocks=2, rank=8,
                                  config=config, stabilized=stabilized)
    assert calls == [1] and stack.layer_indices == (0, 1)
    assert stack.systems_only and stack.systems_metadata["stabilized"] == stabilized
    assert stack.systems_metadata["residual_scale"] == (.5 if stabilized else 1.)
    assert stack.blocks[0].qkv_weight.data_ptr() != stack.blocks[1].qkv_weight.data_ptr()
    x, ids = torch.randn(1, 3, 16, dtype=torch.bfloat16), torch.arange(3)[None]
    _, _, rotary = builder.load_readout_norm(path)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = stack(x, position_embeddings=rotary(x, ids))[0]
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.compact_parameters())
