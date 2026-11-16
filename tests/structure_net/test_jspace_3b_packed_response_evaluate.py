from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_3b_packed_response_evaluate as packed_eval


def source_plan():
    return dict(fingerprint="campaign", model={"revision":"native"}, checkpoint_sha256="seed",
                corpus_fingerprint="corpus", cache_fingerprint="cache", configuration={"temperature":1.},
                checkpoints=[dict(update=1, supervised_tokens=6)], training={"windows":[[0, 1]]})


def saved_checkpoint(path, plan, *, update=1):
    payload = dict(schema="jspace.complete-response-checkpoint.v1", identities=packed_eval.campaign._identities(plan),
                   config=plan["configuration"], stack={"weight":torch.ones(2, 2), "_extra_state":{"packed":False}},
                   norm={"weight":torch.ones(2)}, updates=update, supervised_tokens=6 if update else 0,
                   input_tokens=12 if update else 0, visited=[0, 1] if update else [], fixed_state={"hash":"fixed"})
    torch.save(payload, path)
    return payload


def test_checkpoint_admission_binds_actual_payload_config_and_training_boundary(tmp_path):
    plan = source_plan(); path = tmp_path/"saved.pt"
    payload = saved_checkpoint(path, plan)
    result = packed_eval.checkpoint_metadata(path, plan)
    assert result["update"] == 1 and result["supervised_tokens"] == 6
    assert result["identities"] == payload["identities"]
    payload["identities"]["study"] = "another-campaign"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identities"):
        packed_eval.checkpoint_metadata(path, plan)


def test_partial_or_wrong_boundary_checkpoint_is_never_admitted(tmp_path):
    plan = source_plan(); path = tmp_path/"saved.pt"
    payload = saved_checkpoint(path, plan)
    payload["visited"] = [2, 3]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="membership"):
        packed_eval.checkpoint_metadata(path, plan)
    path.write_bytes(b"PK partial model write")
    with pytest.raises(Exception):
        packed_eval.checkpoint_metadata(path, plan)


def test_reload_equality_applies_only_to_the_same_packed_export():
    original = [dict(input_ids=[1, 2]+[3]*i, logits=torch.tensor([1., 2., 3.])) for i in range(3)]
    assert packed_eval.reload_parity(original, deepcopy(original))["passed"]
    changed = deepcopy(original)
    changed[2]["logits"][0] += .001
    with pytest.raises(ValueError, match="reload changed"):
        packed_eval.reload_parity(original, changed)


def test_actual_task_quality_can_pass_without_unpacked_logit_equality():
    good = {"passed":True}
    mc = dict(native_correct=195, native_count=256, metrics={"correct":190, "accuracy":190/256},
              eligibility=good)
    result = packed_eval.combine_quality({"nll":2.1}, {"nll":2.}, good, good, mc, good,
                                         {"maximum_native_nll_ratio":1.1, "maximum_drop_pp":5.})
    assert result["passed"] and result["student_mc_correct"] == 190
    assert not result["full_logit_parity_required"]
    failed = packed_eval.combine_quality({"nll":2.1}, {"nll":2.}, good, {"passed":False}, mc, good,
                                         {"maximum_native_nll_ratio":1.1, "maximum_drop_pp":5.})
    assert not failed["passed"] and not failed["checks"]["semantic"]


def test_warm_latency_includes_full_extended_prefix_and_has_no_cached_decode_claim():
    calls = []
    model = SimpleNamespace(embedding=torch.ones(4, 2))
    def next_logits(ids):
        calls.append(ids.tolist()[0])
        return torch.tensor([0., 1., 2., 3.])
    model.next_logits = next_logits
    timing = packed_eval.warm_timing(model, [1, 2], warmup_calls=1, repeats=3)
    assert timing["full_prompt_next_logit"]["input_tokens"] == 2
    assert timing["extended_prefix_next_logit"]["input_tokens"] == 3
    assert [1, 2, 3] in calls and all(len(ids) >= 2 for ids in calls)
    assert timing["use_cache"] is False and "KV-cached" in timing["scope"]
    assert "speedup" not in timing and timing["extended_prefix_next_logit"]["median_ms"] >= 0


def test_packed_probe_uses_saved_generated_prefixes_on_fresh_reload():
    model = SimpleNamespace(embedding=torch.ones(4, 2), next_logits=lambda ids:torch.arange(4.))
    reference = packed_eval.packed_probe(model, [1])
    assert [row["input_ids"] for row in reference] == [[1], [1, 3], [1, 3, 3]]
    actual = packed_eval.packed_probe(model, [9], prefixes=[row["input_ids"] for row in reference])
    assert packed_eval.reload_parity(reference, actual)["passed"]


def test_reloaded_packed_model_scores_every_response_token_with_cached_native_teacher(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from torch.nn import functional as F
    source = packed_eval.packed_source
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        max_position_embeddings=32, tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = "eager"
    torch.manual_seed(375)
    native = Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    block = source.AffineQwenBlock(native.model.layers[0], torch.eye(8)[:, :4], torch.zeros(12, 8),
                                   trainable=False, parameter_dtype=torch.bfloat16)
    stack = source.CompactQwenStack([block], geometry_mode="bf16", pad_to_multiple=16)
    packed = source.source.builder.packed_clone(stack, device="cpu")
    model = source.PackedResponseModel(packed, deepcopy(native.model.norm), native.lm_head.weight.detach(),
        source.source.partial.training_rotary(config, "cpu"), config=config, max_length=12)
    source.export_packed_snapshot(model, tmp_path/"export", identities={"fixture":"response-loss"})
    del model
    reloaded, receipt = source.load_packed_snapshot(tmp_path/"export")
    assert receipt["packed_state_exact"]
    data = packed_eval.cache.data
    record = data.ResponseRecord(example_id="heldout-fixture", group_id="group", split="development",
        task="fixture", prompt_ids=(1, 2), response_ids=(3, 4, 31), finish_reason="eos", eos_token_id=31,
        provenance=dict(source_uri="synthetic://test", source_revision="1", tokenizer_sha256="a"*64,
            chat_template_sha256="b"*64, prompt_provenance="test", response_provenance="test",
            assistant_boundary_verified=True))
    batch = data.response_batch([record], pad_token_id=0, max_length=12)
    with torch.no_grad():
        hidden = native.model(input_ids=batch["input_ids"], use_cache=False).last_hidden_state
    def prepare(original):
        assert original["example_ids"] == [record.example_id]
        return dict(original, teacher_hidden=hidden, teacher_input_ids=original["input_ids"].clone(),
                    teacher_record_ids=[record.example_id])
    provider = packed_eval.training.CachedHiddenTeacher(reloaded.embedding, teacher_id="native-fixture")
    scored = packed_eval.evaluation.response_losses(reloaded, [record], pad_token_id=0, max_length=12,
        teacher_provider=provider, prepare_batch=prepare, token_chunk_size=1)
    with torch.no_grad():
        logits = reloaded.readout(reloaded.hidden_forward(batch)[batch["loss_mask"]]).float()
        expected = F.cross_entropy(logits, batch["targets"][batch["loss_mask"]])
    assert scored["response_tokens"] == 3 and scored["records"][0]["eos_supervised"]
    assert scored["nll"] == pytest.approx(float(expected), rel=1e-6)
    assert scored["kl"] is not None and scored["teacher_nll"] is not None
