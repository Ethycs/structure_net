import copy
import json

import pytest
import torch

from experiments.structure_net.ncm_frozen_body import (
    Config, bridge, fake_prefix, module_hash, parent_inputs, select, supplied_context)


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt):
        return [1] * int(messages[-1]["content"])


def test_selection_filters_lengths_without_truncation_and_is_order_independent():
    records = [{"id": str(i), "source": str(n)} for i, n in enumerate([600, 511, 512, 200])]
    a, _ = select(records, Tokenizer(), "train", 3)
    b, _ = select(list(reversed(records)), Tokenizer(), "train", 3)
    assert a == b and {len(r["parent_ids"]) for r in a} == {511, 512, 200}
    with pytest.raises(ValueError, match="not enough"):
        select(records, Tokenizer(), "train", 4)


def test_parent_padding_preserves_positions_and_all_tokens():
    data = parent_inputs([{"parent_ids": [3, 4, 5]}, {"parent_ids": [7]}], Tokenizer(), "cpu")
    assert data["input_ids"][0, -3:].tolist() == [3, 4, 5]
    assert data["attention_mask"].sum(1).tolist() == [3, 1]
    assert data["position_ids"][0, -3:].tolist() == [0, 1, 2]
    assert data["position_ids"][1, -1].item() == 0
    assert fake_prefix(data["attention_mask"].eq(0)).eq(50257).equal(data["attention_mask"].eq(0))


def test_external_context_restores_method_and_keeps_adapter_gradient():
    class Net(torch.nn.Module):
        use_ctx_encoder = False

        def _encode_context(self, ids):
            return ids + 1

    net = Net()
    adapter = bridge()
    features = torch.randn(1, 2, Config().parent_width)
    context = adapter(features)
    with supplied_context(net, context):
        assert net.use_ctx_encoder and net._encode_context(None) is context
        net._encode_context(None).square().mean().backward()
    assert not net.use_ctx_encoder and "_encode_context" not in net.__dict__
    assert net._encode_context(torch.tensor(1)).item() == 2
    assert adapter[1].weight.grad.abs().sum() > 0 and features.grad is None
    with pytest.raises(RuntimeError):
        with supplied_context(net, context):
            raise RuntimeError("simulate failure")
    assert not net.use_ctx_encoder and "_encode_context" not in net.__dict__


def test_frozen_digest_covers_bfloat16_weights_and_detects_mutation():
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    original = module_hash(model)
    assert module_hash(copy.deepcopy(model)) == original
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert module_hash(model) != original


def test_registry_keeps_exploratory_result_conservative_and_rejects_changed_evidence(tmp_path):
    from experiments.neural_architecture_lab.finalize_ncm_frozen_body import Evidence
    from experiments.structure_net.ncm_memory_data import file_hash
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "artifacts": {str(raw): file_hash(raw)}, "models": {"hybrid": {"passes": 61, "n": 64}},
        "classification": "frozen_body_training_succeeded_quality_point_target_met", "body_unchanged": True}))
    report = tmp_path / "report.md"
    report.write_text("An exploratory point estimate is not production validation.")
    record = Evidence(campaign, report).record()
    assert record["tested"] and not record["confirmed"] and record["underpowered"]
    assert not record["human_rated"] and not record["speed_tested"]
    assert record["task_passes"] == 61 and record["task_cases"] == 64
    raw.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="evidence changed"):
        Evidence(campaign, report).record()


def test_public_provenance_rejects_nonpublic_or_modified_passages():
    from experiments.structure_net.ncm_frozen_body_public_data import verify_cases
    from experiments.structure_net.ncm_memory_data import text_hash
    text = "The public meeting was cancelled."
    public = [{"haystack_session_ids": ["public-1"],
               "haystack_sessions": [[{"role": "assistant", "content": text}]]}]
    case = {"id": "a", "source": text, "spans": [{"source_id": "public-1", "role": "assistant",
        "turn_text_sha256": text_hash(text), "span_text_sha256": text_hash(text),
        "start_char": 0, "end_char": len(text)}]}
    assert verify_cases(public, [case]) == {"a": text_hash(text)}
    changed = copy.deepcopy(case)
    changed["source"] = "A different private passage."
    with pytest.raises(ValueError, match="differs"):
        verify_cases(public, [changed])
    changed = copy.deepcopy(case)
    changed["spans"][0]["source_id"] = "private-1"
    with pytest.raises(ValueError, match="absent"):
        verify_cases(public, [changed])
