import hashlib
import json
import pytest
from neural_architecture_lab.llama_cotenancy_hypothesis import LlamaCotenancyEvidence
from experiments.neural_architecture_lab.finalize_llama_cotenancy import audited, digest


def test_exploratory_tensorization_failure_stays_unconfirmed():
    evidence = LlamaCotenancyEvidence("result.json", "a"*64, "report.md", "b"*64,
                                      "tensorization_quality_failed", 4, 0)
    assert evidence.record()["tested"]
    assert not evidence.record()["confirmed"]
    assert not evidence.record()["application_accuracy_tested"]
    with pytest.raises(ValueError):
        LlamaCotenancyEvidence("x", "a"*64, "y", "b"*64, "practical_cotenancy_supported", 4, 0)
    with pytest.raises(ValueError):
        LlamaCotenancyEvidence("x", "a"*64, "y", "b"*64, "tensorization_quality_failed", 2, 0)


def test_audit_rejects_source_and_fingerprint_tampering(tmp_path):
    snapshots = tmp_path/"producer_source"
    snapshots.mkdir()
    source = snapshots/"00_dependency.py"
    source.write_text("frozen producer")
    producer = {"sources": {"dependency.py": hashlib.sha256(source.read_bytes()).hexdigest()}, "config": {"rank": 64}}
    fingerprint = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
    producer["fingerprint"] = fingerprint
    (tmp_path/"producer.json").write_text(json.dumps(producer))
    result = tmp_path/"result.json"
    result.write_text(json.dumps({"status": "completed", "producer_fingerprint": fingerprint}))
    audited(result)
    source.write_text("modified producer")
    digest.cache_clear()
    with pytest.raises(ValueError, match="snapshot changed"):
        audited(result)
    source.write_text("frozen producer")
    producer["config"]["rank"] = 256
    (tmp_path/"producer.json").write_text(json.dumps(producer))
    digest.cache_clear()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        audited(result)
