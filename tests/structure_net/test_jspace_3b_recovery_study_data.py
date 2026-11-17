import json
from dataclasses import asdict

import pytest

from experiments.structure_net import jspace_3b_recovery_study_data as m


def row(**changes):
    value = dict(example_id="example", source_group="source-a", instruction="Solve this task.",
                 context="A distinct source context.", response="A complete response.",
                 family="dolly", category="instruction", task="instruction:open_qa",
                 source_uri="https://example.org/train", source_revision="a"*40,
                 source_row=1, reference_quality="reference")
    value.update(changes)
    return value


def record(**changes):
    value = dict(example_id="example", group_id="group-a", split="train", task="instruction",
                 prompt_ids=(1, 2), response_ids=(3, 4), finish_reason="eos", eos_token_id=4,
                 provenance=dict(source_uri="https://example.org/train", source_revision="a"*40,
                     tokenizer_sha256="tokenizer", chat_template_sha256="template",
                     prompt_provenance="source", response_provenance="complete_dataset_reference",
                     assistant_boundary_verified=True))
    value.update(changes)
    return m.data.ResponseRecord(**value)


def test_historical_train_ancestry_cannot_move_to_confirmation(tmp_path):
    index = m.LineageIndex(tmp_path/"lineage.sqlite", 123)
    historical = row(split="train", group_id="inherited-group")
    index.historical(historical)
    split, group, exposure = index.assign(row(example_id="derivative", source_group="derived-source"))
    assert (split, group, exposure) == ("train", "inherited-group", True)
    index.close()


def test_historical_development_is_exposed_not_sealed(tmp_path):
    index = m.LineageIndex(tmp_path/"lineage.sqlite", 123)
    index.historical(row(split="development", group_id="old-dev"))
    assert index.assign(row()) == ("development", "old-dev", True)
    index.close()


def test_conflicting_ancestry_fails_closed(tmp_path):
    index = m.LineageIndex(tmp_path/"lineage.sqlite", 123)
    index.historical(row(split="train", group_id="old-train"))
    with pytest.raises(ValueError, match="Conflicting historical"):
        index.historical(row(split="development", group_id="old-dev"))
    index.close()


def test_unique_prompt_budget_does_not_count_derivatives_twice(tmp_path):
    index = m.LineageIndex(tmp_path/"lineage.sqlite", 123)
    assert index.unique(record())
    assert not index.unique(record(example_id="different-id", response_ids=(6, 4)))
    assert not index.unique(record(example_id="different-id-2"))
    assert index.unique(record(example_id="different-id-3", prompt_ids=(8, 9)))
    index.close()


def test_confirmation_remains_test_and_is_inaccessible_to_training_loader(tmp_path):
    path = tmp_path/"confirmation.jsonl"
    path.write_text(json.dumps(asdict(record(split="test")))+"\n")
    with pytest.raises(ValueError, match="Unadmitted split"):
        list(m.data.stream_records(path, allowed_splits=("train",)))
    assert list(m.data.stream_records(path, allowed_splits=("test",)))[0].split == "test"


def test_public_code_reference_selection_is_fixed_not_shortest(tmp_path):
    path = tmp_path/"apps.jsonl"
    first = "# complete reference\n" + "print('long')\n"*3000
    original = dict(id=17, question="Print something", solutions=json.dumps([first, "print(1)"]),
                    input_output=json.dumps(dict(inputs=[""], outputs=["1\n"])), difficulty="introductory")
    path.write_text(json.dumps(original)+"\n")
    source = dict(family="apps", category="code", path=str(path), revision=m.APPS_REVISION, uri="https://huggingface.co/datasets/codeparrot/apps")
    value, = m.source_rows(source)
    assert value["response"] == first
    assert value["source_split"] == "train"
    assert value["apps_tests"]["outputs"] == ["1\n"]
    assert value["example_id"].endswith(":train:17")


def test_invalid_apps_tests_preserve_reference_and_explicit_unavailability(tmp_path):
    path = tmp_path/"apps.jsonl"
    original = dict(id=17, question="Print something", solutions='["print(1)"]', input_output="not-json")
    path.write_text(json.dumps(original)+"\n")
    value, = m.source_rows(dict(family="apps", category="code", path=str(path), revision=m.APPS_REVISION, uri="https://public.example"))
    assert value["response"] == "print(1)"
    assert value["apps_tests"] is None
    assert value["apps_tests_error"] == "JSONDecodeError"
    assert value["apps_tests_raw_sha256"] == m.data.digest("not-json")


def test_xsum_requires_complete_original_train(tmp_path):
    path = tmp_path/"xsum.jsonl"
    path.write_text(json.dumps(dict(split="TEST", source_truncated=False,
        public_provenance=dict(split="test", revision=m.old.XSUM_REVISION)))+"\n")
    with pytest.raises(ValueError, match="Non-TRAIN"):
        list(m.source_rows(dict(family="xsum", category="summary", path=str(path), revision=m.old.XSUM_REVISION, uri="https://public.example")))


def test_sample_size_is_not_tiny_diagnostic_certificate():
    assert m.sample_size() > 3000
    assert m.sample_size(discordance=.5) < m.sample_size()
    assert sum(m.TRAIN_QUOTAS.values()) >= 1000000
    assert min(m.TRAIN_QUOTAS.values()) >= 250000
    with pytest.raises(ValueError):
        m.sample_size(discordance=0)


def test_real_pinned_template_complete_reference_and_total_length():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(m.old.MODEL, local_files_only=True, trust_remote_code=False)
    tokenization = dict(tokenizer_sha256="fixture-local-tokenizer", chat_template_sha256="fixture-pinned-template")
    value = m.old.tokenize_reference(tokenizer, row(), split="train", group_id="g", tokenization=tokenization,
                                    max_prompt=2048, max_response=2048, max_total=2048)
    assert value.finish_reason == "eos" and value.response_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(value.response_ids, skip_special_tokens=True) == "A complete response."
    assert len(value.prediction_positions) == len(value.response_ids)
    with pytest.raises(ValueError, match="total_over_limit"):
        m.old.tokenize_reference(tokenizer, row(), split="train", group_id="g", tokenization=tokenization,
                                 max_prompt=2048, max_response=2048, max_total=len(value.input_ids)-1)


def test_shared_guard_is_exact_not_near_duplicate_certificate():
    guard = m.data.SplitGuard()
    guard.add(record())
    with pytest.raises(ValueError, match="Cross-split exact prompt"):
        guard.add(record(example_id="heldout", group_id="another", split="test"))
    # Different tokenization of a related text is not evidence this guard detects
    # paraphrases: the manifest explicitly retains that contamination uncertainty.
    guard.add(record(example_id="other", group_id="other-group", split="test", prompt_ids=(8, 9), response_ids=(10, 4)))
