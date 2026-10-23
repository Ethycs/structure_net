import pytest

from experiments.structure_net.ncm_source_readout import candidates, normalized, select_index, selection
from experiments.structure_net.ncm_memory_data import text_hash


def test_candidates_are_contiguous_complete_spans_and_never_truncated():
    text = "First sentence.\nSecond sentence!  Third sentence? Fourth sentence."
    units = ["First sentence.\n", "Second sentence!  ", "Third sentence? ", "Fourth sentence."]
    spans, start = [], 0
    for unit in units:
        spans.append({"start": start, "end": start+len(unit)})
        start += len(unit)
    result = candidates(text, spans, lambda s: len(s.split()))
    assert len(result) == 9  # 4 singles + 3 pairs + 2 triples; never four.
    for row in result:
        assert text[row["start"]:row["end"]] == row["raw"]
        assert row["text"] == normalized(row["raw"])
        assert len(row["sentence_ids"]) == row["sentence_count"] <= 3
    assert candidates(text, spans, lambda _: 129) == []
    assert len(candidates(text, spans, lambda _: 128)) == 9


def test_cosine_selection_maximizes_score_with_fixed_tie_rules():
    rows = [{"tokens": 4, "start": 0, "end": 9}, {"tokens": 3, "start": 10, "end": 15},
            {"tokens": 3, "start": 5, "end": 8}]
    assert select_index(rows, [.5, .7, .7]) == 2
    assert select_index(rows, [.8, .7, .7]) == 0
    assert select_index([], []) is None
    with pytest.raises(ValueError):
        select_index(rows, [float("nan")]*3)


def test_empty_case_is_retained_and_reference_query_is_target_aware():
    import torch
    base = {"id": "a", "source_id": "s", "source": "text", "source_sha256": "h", "group_id": "g", "spans": []}
    case = {**base, "cohort": "development", "candidates": [], "sentences": [], "queries": {}}
    output = selection([case], {})
    assert all(len(output[k]) == 1 and output[k][0]["text"] == "" for k in ("lead", "source_centroid", "ncm_query", "parent_query"))
    option = {"text": "text", "tokens": 1, "start": 0, "end": 4, "sentence_count": 1}
    reference = {**case, "cohort": "reference", "candidates": [option], "queries": {"reference_query": "query"}}
    picked = selection([reference], {text_hash("text"): torch.tensor([1.,0.]), text_hash("query"): torch.tensor([1.,0.])})
    assert picked["reference_query"][0]["target_aware"] and picked["reference_query"][0]["text"] == "text"
