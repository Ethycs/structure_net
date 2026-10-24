import json

import pytest

from experiments.structure_net import ncm_lm1b_capacity_labels as run


def test_teacher_schema_does_not_retry_capacity_or_semantic_outcomes():
    text = "possibly wrong " * 500
    assert run.parse(json.dumps({"id": "a", "summary": text}), "a") == text.strip()
    with pytest.raises(ValueError): run.parse(json.dumps({"id": "other", "summary": "x"}), "a")
    with pytest.raises(ValueError): run.parse(json.dumps({"id": "a", "summary": "x", "extra": 1}), "a")


def test_merge_preserves_all_sources_and655_texts_and_drops_stale_targets():
    original = [{"id": str(i), "source": "source"+str(i), "target_text": "old"+str(i),
        "word_ids": [1], "concept_ids": [2], "context_ids": [3]} for i in range(690)]
    updates = {str(i): "new"+str(i) for i in range(35)}
    merged = run.merge(original, updates)
    assert len(merged) == 690 and sum(a["target_text"] != b["target_text"] for a,b in zip(original,merged)) == 35
    assert all(a["source"] == b["source"] and b["context_ids"] == [3] for a,b in zip(original,merged))
    assert all("word_ids" not in r and "concept_ids" not in r for r in merged)
    assert all(r["word_ids"] == [1] for r in original)
    with pytest.raises(ValueError): run.merge(original, {"a": "new"})


def test_current_capacity_rule_selects_exactly35_before_teacher_execution():
    run.configure()
    import tiktoken
    t, storage = run.native.tokenizer(), tiktoken.get_encoding("cl100k_base")
    rows = run.q.read(run.native.CASES)["train"]
    selected = [r for r in rows if not run.capacity(r["target_text"], t, storage)["fits"]]
    assert len(rows) == 690 and len(selected) == 35
    assert len(run.base_cache()) == 2284
