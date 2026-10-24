from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_clean_word_train_semantic as run
from experiments.neural_architecture_lab import finalize_ncm_clean_word_train_semantic as audit


def fixture():
    selected, measured = [], {model: [] for model in run.MODELS}
    for i in range(32):
        source = {"id": str(i), "source_id": f"source{i}", "group_id": f"group{i}",
            "source_sha256": f"sha{i}", "source": "The passage gives useful advice.", "spans": [],
            "diagnostic_group": "original_16" if i < 16 else "additional_16", "target_text": "HIDDEN TARGET"}
        selected.append(source)
        for model in run.MODELS:
            value = {"text": "Useful advice.", "raw_word_ids": [7, 50256] + [8] * 190,
                "global_EOS_position": 1}
            measured[model].append({"id": source["id"], "source_sha256": source["source_sha256"],
                "diagnostic_group": source["diagnostic_group"],
                "arms": {arm: deepcopy(value) for arm in run.ARMS}})
    return selected, measured


def entry(packet, role, root, passed=True, valid=True):
    key = run.first.packet_key(packet, role)
    payload = {"model": run.q.JUDGES[role], "temperature": 0, "max_tokens": 4096,
        "messages": [{"role": "system", "content": run.q.RUBRIC},
                     {"role": "user", "content": json.dumps(packet)}]}
    content = json.dumps({"evaluations": [{"candidate_id": "summary", "faithful": True,
        "salient": passed, "readable": True, "reason": "The candidate is assessed against the source.",
        "claims": [{"claim": "The passage gives useful advice.", "verdict": "supported", "evidence_ids": ["L0"]}]}]})
    paths = []
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1 if valid else 3):
        path = root / (key + f"_{i}.json")
        path.write_text(json.dumps({"request": payload, "valid": valid, "http_status": 200 if valid else 503,
            "response": {"model": run.q.JUDGES[role], "choices": [{"message": {"content": content}}]}}))
        paths.append(str(path))
    row = {"case_id": packet["case_id"], "packet_sha256": run.q.digest(packet),
        "judge": run.q.JUDGES[role], "status": "valid" if valid else "invalid", "attempts": paths}
    if valid:
        row["evaluations"] = run.q.parse(content, packet)
    return {"packet": packet, "role": role, "row": row, "origin": str(root)}


ENCODING = SimpleNamespace(encode=lambda text, **kwargs: text.split())


def test_pairing_anonymity_and_complete_frames():
    selected, measured = fixture()
    observations = run.build_observations(selected, measured)
    assert tuple(observations) == run.CONDITIONS
    for name, rows in observations.items():
        packet = run.packets_for(rows)[0]
        assert set(packet) == {"case_id", "source_lines", "candidates"}
        assert packet["candidates"] == [{"candidate_id": "summary", "text": "Useful advice."}]
        assert "HIDDEN TARGET" not in json.dumps(packet)
        assert name not in json.dumps(packet)
        assert rows[0]["oracle_target_codes"] == name.endswith("correct_history")
    measured[run.MODELS[1]][0]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source"):
        run.build_observations(selected, measured)


def test_wrong_order_and_incomplete_raw_frames_are_rejected():
    selected, measured = fixture()
    measured[run.MODELS[0]].reverse()
    with pytest.raises(ValueError, match="ordered"):
        run.build_observations(selected, measured)
    selected, measured = fixture()
    measured[run.MODELS[0]][0]["arms"][run.ARMS[0]]["raw_word_ids"].pop()
    with pytest.raises(ValueError, match="raw192"):
        run.build_observations(selected, measured)


def test_both_valid_token_gate_outcomes_admit_diagnosis_but_pending_does_not(monkeypatch):
    for upstream, expected in ((None, False), (False, True), (True, True)):
        monkeypatch.setattr(run.forecaster, "admission", lambda: upstream)
        assert run.admitted() is expected
    assert run.qualifies({"cases": 32, "valid": 32, "passed": 31})
    assert not run.qualifies({"cases": 32, "valid": 31, "passed": 31})
    assert not run.qualifies({"cases": 32, "valid": 32, "passed": 30})


def test_raw_negative_judgments_reuse_first_decision_without_new_request(tmp_path, monkeypatch):
    packet = run.q.packet("case", "The passage gives useful advice.", [{"candidate_id": "summary", "text": "Advice."}])
    calls = []
    def request(packet, role, root):
        calls.append(packet)
        return entry(packet, role, root, passed=False)["row"]
    monkeypatch.setattr(run.q, "request", request)
    cache = {}
    assert run.request_packets([packet, packet], "secondary", tmp_path, cache) == 1
    assert len(calls) == 1 and len(cache) == 1
    prior = deepcopy(cache)
    assert run.request_packets([packet], "secondary", tmp_path, cache) == 0
    assert cache == prior
    key = run.first.packet_key(packet, "secondary")
    changed = deepcopy(cache[key]); changed["row"]["evaluations"][0]["salient"] = True
    with pytest.raises(ValueError, match="overwritten"):
        run.first.remember(cache, key, changed)


def test_invalids_keep_fixed_denominators_and_oversize_or_empty_text_fails(tmp_path):
    selected, measured = fixture()
    rows = run.build_observations(selected, measured)[run.CONDITIONS[0]]
    rows[0]["text"] = "word " * 129
    rows[1]["text"] = ""
    packets = run.packets_for(rows)
    cache = {run.first.packet_key(packet, "secondary"): entry(packet, "secondary", tmp_path,
        valid=i != 2) for i, packet in enumerate(packets)}
    scores = run.scores_for(rows, packets, "secondary", cache, ENCODING)
    metric = run.metric(scores)
    assert metric["cases"] == 32 and metric["valid"] == 31 and metric["invalid"] == 1
    assert metric["passed"] == 29 and metric["semantic_pass_rate"] is None
    assert metric["stored_overflow"] == 1 and metric["empty"] == 1
    assert metric["all_case_pass_lower_bound"] == 29 / 32
    assert metric["all_case_pass_upper_bound"] == 30 / 32
    assert metric["groups"]["original_16"]["cases"] == 16
    assert not run.qualifies(metric)


def test_joint_counts_require_same_cases_and_same_case_passes():
    left = [{"id": str(i), "valid": True, "passed": i < 16} for i in range(32)]
    right = [{"id": str(i), "valid": True, "passed": i >= 16} for i in range(32)]
    assert run.joint_count(left, right) == {"cases": 32, "both_valid": 32, "both_passed": 0}
    with pytest.raises(ValueError, match="ordered"):
        run.joint_count(left, list(reversed(right)))


def test_complete_audit_replays_raw_scores_and_rejects_changed_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "GRADE", tmp_path)
    selected, measured = fixture()
    observations = run.build_observations(selected, measured)
    packets = {name: run.packets_for(rows) for name, rows in observations.items()}
    cache = {}
    for role in ("secondary", "primary"):
        for i, packet in enumerate(packets[run.CONDITIONS[0]]):
            cache[run.first.packet_key(packet, role)] = entry(packet, role, tmp_path / "raw", passed=i != 0)
    saved = {"results": {}}
    for condition in run.CONDITIONS:
        scores = {role: run.scores_for(observations[condition], packets[condition], role, cache, ENCODING)
                  for role in ("secondary", "primary")}
        for role, values in scores.items():
            (tmp_path / (condition + "_" + role + "_scores.json")).write_text(json.dumps(values))
        saved["results"][condition] = {"secondary": run.metric(scores["secondary"]),
            "primary": run.metric(scores["primary"]), "joint": run.joint_count(scores["secondary"], scores["primary"])}
    result = audit.replay(observations, packets, cache, {}, {}, saved, ENCODING)
    assert all(row["joint"]["both_passed"] == 31 for row in result.values())
    broken = deepcopy(saved)
    broken["results"][run.CONDITIONS[0]]["secondary"]["passed"] = 32
    with pytest.raises(ValueError, match="denominators"):
        audit.replay(observations, packets, cache, {}, {}, broken, ENCODING)
    first_key = next(iter(cache))
    with pytest.raises(ValueError, match="first judgments"):
        audit.replay(observations, packets, cache, {first_key: {"altered": True}}, {}, saved, ENCODING)


def test_cache_audit_rejects_role_and_rubric_identity_changes(tmp_path):
    packet = run.q.packet("case", "The passage gives useful advice.", [{"candidate_id": "summary", "text": "Advice."}])
    value = entry(packet, "secondary", tmp_path)
    key = run.first.packet_key(packet, "secondary")
    run.audit_cache({key: value})
    wrong = deepcopy(value); wrong["role"] = "primary"
    with pytest.raises(ValueError, match="key"):
        run.audit_cache({key: wrong})
