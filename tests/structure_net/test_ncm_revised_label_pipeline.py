import json
import pytest
from experiments.structure_net.ncm_revised_label_pipeline import parse


def test_teacher_keeps_semantic_failures_but_rejects_unknown_evidence_ids():
    cases = [{"id": "one", "source": "first line\nsecond line"}]
    row = {"id": "one", "summary": "An unsupported statement", "claims": [
        {"claim": "An unsupported statement", "evidence_ids": ["L1"]}]}
    assert parse(json.dumps({"summaries": [row]}), cases)["one"]["summary"] == row["summary"]
    row["claims"][0]["evidence_ids"] = ["L2"]
    with pytest.raises(ValueError): parse(json.dumps({"summaries": [row]}), cases)
    row["summary"] = ""
    row["claims"] = []
    assert parse(json.dumps({"summaries": [row]}), cases)["one"]["summary"] == ""
