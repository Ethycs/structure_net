import json

import pytest

from experiments.structure_net import ncm_additional_teacher_pilot as run


def test_schema_valid_bad_text_is_preserved_without_semantic_repair():
    assert run.parse('{"summaries":[{"slot":0,"summary":""}]}', 1) == [""]
    text = "unsupported repetition " * 300
    assert run.parse(json.dumps({"summaries": [{"slot": 0, "summary": text}]}), 1) == [text]


@pytest.mark.parametrize("rows", (
    [{"slot": False, "summary": "x"}],
    [{"slot": 0, "summary": "x"}, {"slot": 0, "summary": "y"}],
    [{"slot": 1, "summary": "x"}],
))
def test_ambiguous_or_incomplete_teacher_slots_are_invalid(rows):
    with pytest.raises(ValueError):
        run.parse(json.dumps({"summaries": rows}), 1)


def test_prompt_packing_keeps_all_full_inputs_and_order_within_budget():
    encoding = run.data.storage_encoding()
    rows = [{"id": str(i), "source": "word " * 2000} for i in range(10)]
    actual = run.batches(rows, encoding)
    assert [r for batch in actual for r in batch] == rows
    assert all(len(batch) <= 4 and len(encoding.encode(json.dumps(run.payload(batch)["messages"], ensure_ascii=False),
        disallowed_special=())) + 16 <= 6300 for batch in actual)
    with pytest.raises(ValueError, match="clipping prohibited"):
        run.batches([{"id": "too-long", "source": "word " * 7000}], encoding)


def test_fixed64_training_label_population_is_not_outcome_selected():
    rows = [{"id": f"{b}-{i}", "bin": b, "source_component": f"component-{b}-{i}"} for b in range(5) for i in range(40)]
    selected = run.select(rows)
    assert selected == run.select(list(reversed(rows)))
    assert len(selected) == 64 and sum(r["bin"] == 0 for r in selected) == 32
    for r in rows:
        r["source_component"] = "same-component"
    with pytest.raises(ValueError, match="distinct"):
        run.select(rows)
