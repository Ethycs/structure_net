import copy

import pytest

from experiments.structure_net.ncm_memory_data import hydrate, split_rows, text_hash


def test_hydration_preserves_exact_unicode_slice_and_checks_source():
    text = "user: café — corrected to €50"
    section = {"spans": [{"turn_id": "t1", "source_id": "s1", "role": "user",
                           "start_char": 6, "end_char": len(text),
                           "turn_text_sha256": text_hash(text),
                           "span_text_sha256": text_hash(text[6:])}]}
    turns = {"t1": {"text": text, "role": "user", "source_id": "s1"}}
    assert hydrate(section, turns) == text[6:]
    changed = copy.deepcopy(turns)
    changed["t1"]["text"] = text.replace("50", "60")
    with pytest.raises(ValueError, match="provenance"):
        hydrate(section, changed)
    changed_section = copy.deepcopy(section)
    changed_section["spans"][0]["start_char"] = 7
    with pytest.raises(ValueError, match="span hash"):
        hydrate(changed_section, turns)


def test_split_groups_shared_inputs_without_connecting_generic_bypass_rows():
    rows = []
    for i in range(20):
        for text, tokens in [(f"long dialogue {i}", 200), ("Thanks", 1)]:
            rows.append({"id": f"{i}:{tokens}", "source_id": f"s{i}",
                         "source_sha256": text_hash(text), "input_tokens": tokens})
    rows += [{"id": "duplicate", "source_id": "other-session",
              "source_sha256": text_hash("long dialogue 0"), "input_tokens": 200}]
    result = split_rows(rows)
    reverse = split_rows(list(reversed(rows)))
    assert {r["id"]: r["split"] for r in result} == {r["id"]: r["split"] for r in reverse}
    assert len({r["group_id"] for r in result}) == 20
    assert len({r["group_id"] for r in result if r["split"] == "test"}) == 2
    train = {r["source_id"] for r in result if r["split"] == "train"}
    test = {r["source_id"] for r in result if r["split"] == "test"}
    assert not train & test
    shared = [r for r in result if r["source_sha256"] == text_hash("long dialogue 0")]
    assert len({r["split"] for r in shared}) == 1


def test_no_independent_test_group_is_rejected():
    rows = [{"source_id": f"s{i}", "source_sha256": "same", "input_tokens": 200}
            for i in range(3)]
    with pytest.raises(ValueError, match="two source groups"):
        split_rows(rows)
