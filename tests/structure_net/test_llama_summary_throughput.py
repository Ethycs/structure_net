"""Artifact identity and native prompt lookup contracts before CUDA work."""
import subprocess

import pytest

from experiments.structure_net.llama_summary_throughput import BINARY, digest, prepare


def test_fingerprint_is_order_invariant_and_sensitive():
    assert digest({"cap": 768, "draft": 4}) == digest({"draft": 4, "cap": 768})
    assert digest({"cap": 768}) != digest({"cap": 512})
    with pytest.raises(ValueError):
        digest({"score": float("nan")})


def test_completed_fixtures_cannot_be_overwritten(tmp_path):
    path = tmp_path / "fixtures.json"
    path.write_text("immutable")
    with pytest.raises(ValueError, match="immutable"):
        prepare(path, "pilot")
    assert path.read_text() == "immutable"


def test_native_lookup_contracts():
    if not BINARY.exists():
        pytest.skip("build the native runtime before its CPU contract checks")
    result = subprocess.run([str(BINARY), "self-test"], capture_output=True, text=True, check=True,
                            env={"LD_LIBRARY_PATH": ""})
    assert "5 prompt lookup contracts passed" in result.stdout


def test_fact_presence_is_explicitly_not_a_support_verdict():
    from experiments.neural_architecture_lab.analyze_llama_summary_throughput import fact_diagnostic
    example = {"fragments": [{"id": "F01", "required": {"owner": "Maya"}, "superseded": {"owner": "Omar"}}]}
    result = fact_diagnostic("Maya is not the owner.", example)
    assert result["all_required_present"]
    assert "cannot establish" in result["boundary"]
    assert "quality_pass" not in result


def test_review_gate_rejects_unreviewed_output_and_changed_source(tmp_path, monkeypatch):
    import json
    from experiments.neural_architecture_lab import finalize_llama_summary_throughput as final
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    source = [{"source": "Maya is the approved owner."}]
    key = final.review_key("one", "Maya is the approved owner.")
    inventory = {key: {"id": "one", "fragments": source}}
    monkeypatch.setattr(final, "collect", lambda _: ({}, inventory, {}, []))
    review = tmp_path / "review.json"
    body = {"manifest_sha256": final.sha256(manifest), "items": []}
    review.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="every distinct output"):
        final.finalize(manifest, review, tmp_path / "out.json")
    body["items"] = [{"key": key, "id": "one", "passed": True,
                      "rationale": "Compared to source.", "source_sha256": "0" * 64}]
    review.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="review source changed"):
        final.finalize(manifest, review, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_review_identity_includes_request_and_exact_output():
    from experiments.neural_architecture_lab.finalize_llama_summary_throughput import review_key
    assert review_key("a", "bc") != review_key("ab", "c")
    assert review_key("a", "Owner: Maya") != review_key("a", "Owner: Omar")
