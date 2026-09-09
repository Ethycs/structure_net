"""Evidence audit must reject tampering and unsupported proxy claims."""
import hashlib
import json

import pytest

from experiments.neural_architecture_lab import audit_qwen_moba_feasibility as a


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"score": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        a.strict_read(path)


def test_readout_gate_is_inclusive_and_rejects_inactive_or_failed_cells():
    cell = {"status": "completed", "sparse_active": True,
            "metrics": {"heads_cosine_mean": .95, "ov_cosine_mean": .95,
                        "change_pearson": .95, "top_quarter_change_f1": .9}}
    assert a.quality_pass(cell)
    assert not a.quality_pass({**cell, "sparse_active": False})
    assert not a.quality_pass({"status": "baseline_out_of_memory"})
    assert not a.quality_pass({**cell, "metrics": {**cell["metrics"], "change_pearson": None}})


def test_audit_checks_hash_before_parsing(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="hash mismatch"):
        a.audit(path, "0" * 64)


def test_audit_rejects_mutated_producer(tmp_path):
    producer = {"stage": "screen", "fingerprint": hashlib.sha256(b'{}').hexdigest()}
    (tmp_path / "producer.json").write_text(json.dumps(producer))
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"producer_fingerprint": producer["fingerprint"]}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        a.audit(campaign, a.sha256(campaign))
