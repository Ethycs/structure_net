"""Actual tiny HF native prefixes and same-checkpoint full-vocabulary paths."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_response_model as study
from experiments.structure_net.jspace_3b_response_train import ResponseModel
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def make_parts(tmp_path, cut):
    torch.manual_seed(8536)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        max_position_embeddings=32, tie_word_embeddings=True, attention_dropout=0.,
        pad_token_id=0, eos_token_id=31)
    cfg._attn_implementation = "eager"
    native = Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    path = tmp_path / "native"
    native.save_pretrained(path)
    blocks = []
    for layer in range(cut, cfg.num_hidden_layers):
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        block = AffineQwenBlock(native.model.layers[layer], basis, torch.randn(12, 8)/20,
                               trainable=True, parameter_dtype=torch.bfloat16)
        for p in block.parameters(): p.data = p.data.bfloat16()
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=16)
    norm = deepcopy(native.model.norm).requires_grad_(True)
    norm.weight.data.mul_(1.25)
    model = ResponseModel(stack, norm, native.lm_head.weight.detach(),
        study.partial.training_rotary(cfg, "cpu"), max_length=12, expected_layers=4)
    return path, native, model


def test_prefix_loads_only_required_native_layers_and_preserves_terminal_residual(tmp_path, monkeypatch):
    path, native, model = make_parts(tmp_path, 2)
    calls = []
    original = study.builder.native.load_native_block
    def loader(_path, index, **kwargs):
        calls.append(index)
        return original(_path, index, **kwargs)
    monkeypatch.setattr(study.builder.native, "load_native_block", loader)
    ids = torch.tensor([[2, 8, 11, 4]])
    captured = []
    hook = native.model.layers[2].register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    with torch.no_grad(): native(input_ids=ids, use_cache=False)
    hook.remove()
    embedding_pointer = model.embedding.data_ptr()
    with study.attach_native_prefix(model, path) as receipt:
        actual = model.native_prefix(ids, torch.ones_like(ids, dtype=torch.bool))
        assert torch.equal(actual, captured[0])
        assert receipt["loaded_layers"] == 2 and receipt["shared_embedding"]
        assert receipt["final_norm"] == "Identity" and not receipt["cache_lookup"]
        assert calls == [0, 1]
    assert model.native_prefix is None and model.embedding.data_ptr() == embedding_pointer


@pytest.mark.parametrize("cut", [0, 2])
@pytest.mark.parametrize("reference_mode", ["packed", "unpacked"])
def test_full_vocabulary_parity_preserves_exact_student_and_parameter_identities(tmp_path, cut, reference_mode):
    path, native, model = make_parts(tmp_path, cut)
    before = study.offload.fingerprint(model.named_parameters())
    parameters = tuple(map(id, model.parameters()))
    embedding_pointer = model.embedding.data_ptr()
    check = study.check_unpacked_execution_parity if reference_mode == "unpacked" else study.check_full_vocabulary_parity
    result = check(model, path, [2, 8, 11], steps=3)
    assert result["passed"] and result["same_checkpoint"]
    assert result["reference_mode"] == reference_mode
    assert len(result["rows"]) == 3
    assert all(row["vocabulary"] == 32 and row["greedy_equal"] for row in result["rows"])
    assert result["rows"][1]["input_ids"][-1] == result["rows"][0]["adapter_argmax"]
    assert model.native_prefix is None
    assert model.embedding.data_ptr() == embedding_pointer
    assert tuple(map(id, model.parameters())) == parameters
    assert study.offload.fingerprint(model.named_parameters()) == before


def test_parity_refuses_live_optimizer_hooks_and_missing_reservation_before_allocation(tmp_path, monkeypatch):
    path, native, model = make_parts(tmp_path, 0)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    handle = parameter.register_post_accumulate_grad_hook(lambda p: None)
    try:
        with pytest.raises(ValueError, match="before optimizer registration"):
            study.check_full_vocabulary_parity(model, path, [2, 8])
    finally:
        handle.remove()
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA allocated before guard"))
    with pytest.raises(ValueError, match="reservation guard"):
        study.build_student("progressive", {}, "absent", "cuda:0", None)


def test_prefix_context_cleans_up_after_generation_failure(tmp_path):
    path, native, model = make_parts(tmp_path, 2)
    with pytest.raises(RuntimeError, match="injected"):
        with study.attach_native_prefix(model, path):
            assert model.native_prefix is not None
            raise RuntimeError("injected")
    assert model.native_prefix is None


def test_mc_retention_helper_reuses_dev_and_cleans_up_without_changing_mode(tmp_path, monkeypatch):
    _, _, model = make_parts(tmp_path, 0)
    model.train()
    cases = [dict(identity=str(i), split="development") for i in range(256)]
    closed = []
    dataset = SimpleNamespace(items=cases+[dict(identity="train", split="train")],
                              close=lambda: closed.append(True))
    p = dict(schema=study.joint.SCHEMA, native_correct=195, native_records=["native"], label_ids=[1, 2])
    p["fingerprint"] = study.progressive.base.digest(p)
    monkeypatch.setattr(study.joint, "dataset", lambda plan: dataset)
    monkeypatch.setattr(study.joint.training, "score_unpacked", lambda *args: ({"count": 256}, ["candidate"]))
    received = []
    def eligibility(records, native, selected_cases):
        received.append((records, native, selected_cases))
        return dict(passed=False)
    monkeypatch.setattr(study.joint.broad, "eligibility", eligibility)
    result = study.old_mc_evaluate("all36", p, model)
    assert received == [(["candidate"], ["native"], cases)]
    assert not result["eligibility"]["passed"] and result["native_count"] == 256
    assert closed == [True] and model.training
