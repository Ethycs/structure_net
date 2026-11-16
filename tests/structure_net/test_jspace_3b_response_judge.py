import json

import pytest

from experiments.structure_net import jspace_3b_response_judge as judge


def outputs():
    common = dict(example_id="case-one", input_ids=[1, 2], output_ids=[3, 6], max_new_tokens=128,
                  eos_token_ids=[6], stop_reason="eos", task="summarization",
                  prompt="Summarize the shipment report.", reference="12 crates arrived on Friday.")
    return dict(common, response="On Friday, 12 crates arrived."), dict(common, response="15 crates arrived on Friday.")


def verdict(candidate="A", passed=True):
    return dict(candidate_id=candidate, task_success=passed, generation_success=True,
                faithfulness="supported" if passed else "contradicted",
                numerical_errors=[] if passed else ["Wrong crate count."], critical_omissions=[],
                rationale="The source specifies twelve crates.")


def test_blinding_is_deterministic_and_does_not_reveal_arm_names():
    native, student = outputs()
    packet, binding = judge.pair_packet(native, student, context={"source": "12 crates arrived on Friday."})
    assert {row["candidate_id"] for row in packet["candidates"]} == {"A", "B"}
    assert set(binding["assignment"].values()) == {"native", "student"}
    assert "native" not in json.dumps(packet) and "student" not in json.dumps(packet)
    changed = dict(student, response="Something else")
    _, again = judge.pair_packet(native, changed)
    assert binding["assignment"] == again["assignment"]
    assert binding["student"] != again["student"]
    with pytest.raises(ValueError, match="Paired"):
        judge.pair_packet(native, dict(student, input_ids=[9]))


def test_strict_semantic_verdict_rejects_missing_or_inconsistent_evidence():
    packet, _ = judge.pair_packet(*outputs())
    valid = {"evaluations": [verdict(), verdict("B", False)]}
    assert judge.parse_judgment(json.dumps(valid), packet) == valid
    bad = {"evaluations": [verdict(), verdict()]}
    with pytest.raises(ValueError, match="Exactly"):
        judge.parse_judgment(json.dumps(bad), packet)
    wrong = verdict()
    wrong["numerical_errors"] = ["Count is wrong."]
    with pytest.raises(ValueError, match="conflicts"):
        judge.parse_judgment(json.dumps({"evaluations": [wrong, verdict("B")]}), packet)


class Response:
    status_code = 200
    def __init__(self, body):
        self.text = json.dumps(body)
    def raise_for_status(self):
        pass


def test_request_lifecycle_persists_raw_blinded_evidence_and_omits_credentials(tmp_path):
    native, student = outputs()
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        packet = json.loads(kwargs["json"]["messages"][1]["content"])
        grades = [verdict(row["candidate_id"], "12" in row["response"]) for row in packet["candidates"]]
        return Response(dict(model="fixture-judge-v1", id="request1", usage={"completion_tokens": 40},
            choices=[dict(finish_reason="stop", message={"content": json.dumps({"evaluations": grades})})]))
    config = judge.JudgeConfig(model="fixture-route", workers=1, expected_returned_model="fixture-judge-v1")
    result = judge.judge_pairs([native], [student], tmp_path/"judge", config=config, post=post, key="secret-test-key")
    assert result["status"] == "completed"
    row = result["judgments"][0]
    assert row["native"]["task_success"] and not row["student"]["task_success"]
    assert not result["semantic_gate"]["passed"]
    assert result["judge_provenance"]["judge_version"] == "fixture-judge-v1"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-test-key"
    for path in (tmp_path/"judge").rglob("*.json"):
        assert "secret-test-key" not in path.read_text()
    with pytest.raises(FileExistsError):
        judge.judge_pairs([native], [student], tmp_path/"judge", config=config, post=post, key="secret-test-key")


def test_transport_failure_is_preserved_and_cannot_promote(tmp_path):
    def post(*args, **kwargs):
        raise TimeoutError("includes credentials that must not be logged")
    result = judge.judge_pairs(*([row] for row in outputs()), tmp_path/"failed",
                               post=post, key="not-saved")
    assert result["status"] == "judge_unavailable"
    assert not result["semantic_gate"]["passed"]
    assert result["judgments"][0]["error_type"] == "TimeoutError"
    attempts = list((tmp_path/"failed"/"attempts").glob("*.json"))
    assert len(attempts) == 1 and "includes credentials" not in attempts[0].read_text()


def test_model_identity_change_and_length_truncation_fail_closed(tmp_path):
    def post(*args, **kwargs):
        return Response(dict(model="other-model", choices=[dict(finish_reason="length",
            message={"content": json.dumps({"evaluations": [verdict(), verdict("B")]})})]))
    native, student = outputs()
    result = judge.judge_pairs([native], [student], tmp_path/"wrong",
        config=judge.JudgeConfig(expected_returned_model="pinned-model"), post=post, key="hidden")
    assert result["status"] == "judge_unavailable" and not result["semantic_gate"]["passed"]
