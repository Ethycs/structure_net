import json

import pytest

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_expanded_label_recovery import checked_teacher_batch


def attempt(text, valid=True, source="Public source."):
    return {"valid": valid, "http_status": 200,
        "request": {"messages": [{"content": "system"}, {"content": json.dumps({"passages": [{"id": "a", "text": source}]})}]},
        "response": {"model": q.JUDGES["primary"], "choices": [{"message": {"content": text}}]}}


def response(text):
    return json.dumps({"summaries": [{"id": "a", "summary": text}]})


def test_recovery_keeps_first_valid_text_without_quality_filtering():
    data = {"0": attempt(response("A potentially poor first summary."))}
    row = {"status": "valid", "attempts": ["0"], "summaries": {"a": "A potentially poor first summary."}}
    assert checked_teacher_batch(row, [{"id": "a", "source": "Public source."}], data.__getitem__) == row


def test_recovery_rejects_retry_after_a_valid_response():
    data = {"0": attempt(response("First.")), "1": attempt(response("Later."))}
    row = {"status": "valid", "attempts": ["0", "1"], "summaries": {"a": "Later."}}
    with pytest.raises(ValueError, match="first-valid"):
        checked_teacher_batch(row, [{"id": "a", "source": "Public source."}], data.__getitem__)


def test_recovery_rejects_a_parseable_response_marked_invalid():
    data = {"0": attempt(response("First."), valid=False)}
    row = {"status": "invalid", "attempts": ["0"], "summaries": {}}
    with pytest.raises(ValueError, match="parser replay"):
        checked_teacher_batch(row, [{"id": "a", "source": "Public source."}], data.__getitem__)


def test_invalid_batch_requires_all_three_genuine_format_failures():
    data = {str(i): attempt("not JSON", valid=False) for i in range(3)}
    row = {"status": "invalid", "attempts": list(data), "summaries": {}}
    assert checked_teacher_batch(row, [{"id": "a", "source": "Public source."}], data.__getitem__) == row


def test_recovery_rejects_changed_request_source():
    data = {"0": attempt(response("Summary."), source="Different source.")}
    row = {"status": "valid", "attempts": ["0"], "summaries": {"a": "Summary."}}
    with pytest.raises(ValueError, match="source membership"):
        checked_teacher_batch(row, [{"id": "a", "source": "Public source."}], data.__getitem__)
