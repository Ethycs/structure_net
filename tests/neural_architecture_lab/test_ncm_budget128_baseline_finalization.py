from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import finalize_ncm_budget128_baselines as f


def test_baseline_replay_rejects_token_metadata_tampering_and_replaced_first_negative(tmp_path, monkeypatch):
    f.b.dependencies()
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    monkeypatch.setattr(f.b, "GRADING", tmp_path)
    monkeypatch.setattr(f, "verify", lambda producer: None)
    monkeypatch.setattr(f.first, "audit_row", lambda packet, role, row: None)
    root = tmp_path / "secondary_native_3b"
    root.mkdir()
    outputs = [{"id": str(i), "source": "The value is five.", "text": "The value is five."} for i in range(64)]
    packets = [f.q.packet(o["id"], o["source"], [{"candidate_id": "summary", "text": o["text"]}]) for o in outputs]
    scores, cache = [], {}
    for i, (output, packet) in enumerate(zip(outputs, packets)):
        key = f.first.packet_key(packet, "secondary")
        grade = {"faithful": i != 0, "salient": True, "readable": True}
        cache[key] = {"packet": packet, "role": "secondary", "origin": "first",
                      "row": {"status": "valid", "evaluations": [grade]}}
        scores.append({"id": output["id"], "text": output["text"], "tokens": len(encoding.encode(output["text"])),
                       "valid": True, "passed": i != 0, "packet_key": key, "first_origin": "first"})
    for name, value in (("producer.json", {}), ("case_scores.json", scores), ("packets.json", packets)):
        f.write_json(root / name, value)
    endpoint = {"valid": 64, "passed": 63}
    assert f.replay("native_3b", "secondary", outputs, endpoint, cache, {})["passed"] == 63
    scores[0]["tokens"] += 1
    f.write_json(root / "case_scores.json", scores)
    with pytest.raises(ValueError, match="stored token count"):
        f.replay("native_3b", "secondary", outputs, endpoint, cache, {})
    scores[0]["tokens"] -= 1
    f.write_json(root / "case_scores.json", scores)
    pilot_cache = deepcopy(cache)
    first_key = scores[0]["packet_key"]
    cache[first_key]["row"]["evaluations"][0]["faithful"] = True
    with pytest.raises(ValueError, match="first pilot judgment replaced"):
        f.replay("native_3b", "secondary", outputs, endpoint, cache, pilot_cache)
