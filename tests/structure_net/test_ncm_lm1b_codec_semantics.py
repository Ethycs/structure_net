import pytest

from experiments.structure_net import ncm_lm1b_codec_semantics as run


def test_only_tail_padding_changes_and_genuine_endpoint_stays_explicit():
    a = {"id": "case", "target_ids": [101, 1000, 101], "packed_ids": [[101, 1000, 101] + [0] * 125]}
    b = run.replace_tail_padding(a)
    assert b["target_ids"] == a["target_ids"]
    assert b["packed_ids"] == [[101, 1000] + [101] * 126]
    assert b["frame_ends"] == [2]
    assert a["packed_ids"][0][-1] == 0
    with pytest.raises(ValueError):
        run.replace_tail_padding({**a, "packed_ids": [[101, 1000, 101, 0, 1001] + [0] * 123]})
    with pytest.raises(ValueError):
        run.replace_tail_padding({**a, "packed_ids": [[]]})


def test_semantic_sample_is_balanced_and_independent_of_output_quality():
    rows = [{"id": str(i)} for i in range(690)]
    selected = run.select(rows)
    assert len(selected) == len({r["id"] for r in selected}) == 64
    assert sum(int(r["id"]) < 252 for r in selected) == 32
    assert [r["id"] for r in run.select([{**r, "passed": False} for r in rows])] == [r["id"] for r in selected]
    with pytest.raises(ValueError):
        run.select(rows[:64])


def test_semantic_gate_requires_all64_valid_and_keeps_negative_firsts():
    assert run.oracle.qualifies({"cases": 64, "valid": 64, "passed": 61})
    assert not run.oracle.qualifies({"cases": 64, "valid": 63, "passed": 61})
    cache = {}
    run.reference.first.remember(cache, "key", {"passed": False})
    with pytest.raises(ValueError):
        run.reference.first.remember(cache, "key", {"passed": True})
