import json

import pytest

from experiments.neural_architecture_lab import audit_ncm_additional_teacher_candidates as audit


def raw_files(root, count=1):
    (root / "attempts").mkdir()
    (root / "batches").mkdir()
    rows = [{"id": "case", "source": "Original complete source"}]
    paths = []
    for i in range(count):
        path = root / "attempts" / f"batch_0000_{i}.json"
        raw = {"request": audit.run.payload(rows), "valid": True, "http_status": 200,
            "response": {"model": audit.run.MODEL, "choices": [{"finish_reason": "stop", "message": {
                "content": '{"summaries":[{"slot":0,"summary":"First paragraph"}]}'}}]}}
        path.write_text(json.dumps(raw))
        paths.append(str(path))
    (root / "batches/batch_0000.json").write_text(json.dumps({"index": 0, "status": "valid",
        "case_ids": ["case"], "texts": ["First paragraph"], "attempts": paths}))
    return rows


def test_complete_first_raw_candidate_is_replayed(tmp_path):
    rows = raw_files(tmp_path)
    assert audit.replay_batch(tmp_path, 0, rows) == {"case": "First paragraph"}
    rows[0]["source"] = "Changed source"
    with pytest.raises(ValueError, match="full-source"):
        audit.replay_batch(tmp_path, 0, rows)


def test_any_call_after_a_valid_first_is_rejected(tmp_path):
    rows = raw_files(tmp_path, count=2)
    with pytest.raises(ValueError, match="later rerolls"):
        audit.replay_batch(tmp_path, 0, rows)
