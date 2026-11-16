from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_generation_path_probe as probe
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def partial_model():
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
        block = AffineQwenBlock(model.model.layers[index], basis, torch.randn(1, 8) / 20)
        for parameter in block.parameters():
            parameter.data = parameter.data.bfloat16()
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=16)
    stack.eval().requires_grad_(False).pack(dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16, strip_unused=True)
    norm = deepcopy(model.model.norm)
    norm.weight.copy_(torch.linspace(.5, 2., 8).bfloat16())
    rotary = probe.generation.training_rotary(config, "cpu")
    rotary.inv_freq.mul_(11.)  # Detect accidental reuse of native-prefix RoPE.
    return probe.generation.attach_suffix(model, stack, norm, cut=2, suffix_rotary=rotary)


def archive():
    canary = dict(id="concept_heat_transfer", input_ids=[2, 5, 9, 1])
    record = dict(canary, output_ids=[7, 3, 6, 4, 2, 8, 11, 12, 13])
    return canary, record


def test_real_partial_full_vocabulary_matches_at_all_archived_prefixes_and_is_pure():
    model = partial_model()
    before = {name: value.clone() for name, value in
              (*model.named_parameters(), *model.named_buffers())}
    calls = []
    for prefix in probe.prefixes(*archive()):
        wrapper = probe.wrapper_logits(model, prefix["input_ids"], lambda: calls.append("wrapper"))
        direct = probe.direct_logits(model, prefix["input_ids"], lambda: calls.append("direct"))
        assert wrapper.dtype == torch.bfloat16 and wrapper.shape == (1, 32)
        assert torch.equal(wrapper, direct)
        comparison = probe.compare_logits(wrapper, direct)
        assert comparison["exact"] and comparison["differing_logits"] == 0
        assert comparison["vocabulary_size"] == 32
    assert calls == ["wrapper", "direct"] * 3
    assert all(torch.equal(value, before[name]) for name, value in
               (*model.named_parameters(), *model.named_buffers()))


def test_actual_trained_norm_and_tied_readout_are_used_and_wrapper_corruption_is_visible(monkeypatch):
    model = partial_model()
    ids = archive()[0]["input_ids"]
    initial = probe.direct_logits(model, ids)
    model.model.norm.weight.mul_(2.)
    changed = probe.direct_logits(model, ids)
    assert not torch.equal(initial, changed)
    assert torch.equal(changed, probe.wrapper_logits(model, ids))
    model.model.embed_tokens.weight[17].add_(.5)
    changed_readout = probe.direct_logits(model, ids)
    assert changed_readout[0, 17] != changed[0, 17]
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(changed_readout, probe.wrapper_logits(model, ids))
    original_forward = model.lm_head.forward
    monkeypatch.setattr(model.lm_head, "forward", lambda hidden: original_forward(hidden) + .125)
    assert not probe.compare_logits(probe.wrapper_logits(model, ids),
                                    probe.direct_logits(model, ids))["exact"]


def test_archived_prefixes_are_strict_and_tolerance_never_becomes_exact():
    canary, record = archive()
    rows = probe.prefixes(canary, record)
    assert [row["continuation_tokens"] for row in rows] == [0, 1, 8]
    assert rows[-1]["input_ids"] == canary["input_ids"] + record["output_ids"][:8]
    for altered in (dict(record, id="another"), dict(record, input_ids=[1]),
                    dict(record, output_ids=[1] * 7), dict(record, output_ids=[True] * 8)):
        with pytest.raises(ValueError, match="archived"):
            probe.prefixes(canary, altered)
    reference = torch.tensor([[0., 1., 2.]])
    result = probe.compare_logits(reference + .001, reference)
    assert not result["exact"] and result["tolerance_pass"] and result["argmax_exact"]
    with pytest.raises(ValueError, match="finite"):
        probe.compare_logits(torch.tensor([[float("nan"), 1., 2.]]), reference)


def test_archived_repetition_penalty_and_unsupported_processors():
    config = dict(do_sample=False, num_beams=1, use_cache=False,
                  max_new_tokens=64, repetition_penalty=1.05)
    scores = torch.tensor([[2.05, 2., -1., -1.04]])
    assert int(scores.argmax()) == 0
    assert probe.processed_argmax(scores, [0, 2, 0], config) == 1
    assert torch.equal(scores, torch.tensor([[2.05, 2., -1., -1.04]]))
    negative = torch.tensor([[-4., -3., -1., -1.04]])
    assert probe.processed_argmax(negative, [2], config) == 3
    for extra in (dict(no_repeat_ngram_size=2), dict(do_sample=True),
                  dict(forced_eos_token_id=3), dict(min_new_tokens=1)):
        with pytest.raises(ValueError, match="processor"):
            probe.processed_argmax(scores, [0, 2], dict(config, **extra))


def test_denied_guard_precedes_forward_and_cuda(monkeypatch, tmp_path):
    def denied(*_args, **_kwargs):
        raise RuntimeError("lease denied")
    model = partial_model()
    monkeypatch.setattr(model, "forward", lambda **kw: pytest.fail("Forward without lease"))
    for fn in (probe.direct_logits, probe.wrapper_logits):
        with pytest.raises(RuntimeError, match="lease denied"):
            fn(model, [1, 2], denied)
    monkeypatch.setattr(probe, "verify", lambda root: {"fingerprint": "frozen"})
    monkeypatch.setattr(probe, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA before lease"))
    monkeypatch.setattr(probe.assessment.cache.source.inherited, "cap_allocator",
                        lambda *args: pytest.fail("Allocator before lease"))
    with pytest.raises(RuntimeError, match="lease denied"):
        probe.run(tmp_path)
    assert probe.read(tmp_path / "run/failure.json")["lease_id"] is None


def test_run_converts_json_fit_keys_before_building(monkeypatch, tmp_path):
    plan = dict(fingerprint="frozen", model={"path": "model"}, fits={"22": "fit22", "35": "fit35"},
                checkpoint="checkpoint", cut=22, expected_fixed_state={"sha256": "fixed"})
    monkeypatch.setattr(probe, "verify", lambda root: plan)
    monkeypatch.setattr(probe, "wait_ready", lambda *_: {"id": "lease", "allocation": {"3060": 11264}})
    monkeypatch.setattr(probe.assessment.cache.source.inherited, "cap_allocator",
                        lambda *args: {"allocator_limit_bytes": 1_000_000})
    monkeypatch.setattr(probe.assessment.sanity.execution, "arithmetic_policy", nullcontext)
    for name in ("init", "set_device", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    def build(model_path, fits, checkpoint, **kwargs):
        assert fits == {22: "fit22", 35: "fit35"}
        assert kwargs["expected_fixed_state"] == plan["expected_fixed_state"]
        raise RuntimeError("inspected build")
    monkeypatch.setattr(probe.generation, "build_partial", build)
    with pytest.raises(RuntimeError, match="inspected build"):
        probe.run(tmp_path)
