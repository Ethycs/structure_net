import json
import pytest

from experiments.structure_net import ncm_local_judge as local
from experiments.structure_net import ncm_local_judge_gpu2 as gpu2
from experiments.structure_net import ncm_local_judge_gpu2_json as structured
from experiments.structure_net import ncm_memory_accuracy_v2 as q


@pytest.mark.parametrize("local", [local, gpu2, structured])
def test_local_judge_preserves_first_valid_failure_and_never_sends_credentials(tmp_path, monkeypatch, local):
    calls = []
    packet = q.packet("case", "The meeting is on Monday.", [{"candidate_id": "x", "text": "The meeting is on Friday."}])
    response_content = {"evaluations": [{"candidate_id": "x", "faithful": False, "salient": True,
        "readable": True, "claims": [{"claim": "The meeting is on Friday.", "verdict": "contradicted",
        "evidence_ids": ["L0"]}], "reason": "The source says Monday."}]}

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"model": local.MODEL, "choices": [{"message": {"content": json.dumps(response_content)}}]}

    class Session:
        trust_env = True
        def post(self, url, **kwargs):
            assert url.startswith("http://127.0.0.1:")
            assert self.trust_env is False
            assert "headers" not in kwargs and "auth" not in kwargs
            calls.append(kwargs)
            return Response()

    import requests
    monkeypatch.setattr(requests, "Session", Session)
    (tmp_path / "attempts").mkdir()
    row = local.request(packet, tmp_path)
    assert row["status"] == "valid"
    assert row["evaluations"][0]["faithful"] is False
    assert len(calls) == len(row["attempts"]) == 1


def test_schema_binds_ids_without_fixing_semantic_decisions():
    packet = q.packet("case", "Monday.\nTwo people.", [{"candidate_id": "x", "text": "Friday."}])
    schema = structured.response_format(packet)["json_schema"]["schema"]
    grade = schema["properties"]["evaluations"]["items"]
    assert grade["properties"]["candidate_id"]["enum"] == ["x"]
    assert grade["properties"]["faithful"] == {"type": "boolean"}
    claim = grade["properties"]["claims"]["items"]
    assert set(claim["properties"]["verdict"]["enum"]) == {"supported", "contradicted", "insufficient evidence"}
    assert claim["properties"]["evidence_ids"]["items"]["enum"] == ["L0", "L1"]
