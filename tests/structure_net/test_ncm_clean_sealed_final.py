from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_clean_sealed_final as run


def qualified_development():
    endpoint = {"cases": 64, "valid": 64, "passed": 61}
    return {"classification": "clean_full690_semantic_development_passed_final_unproven",
        "semantic_development_gate": True, "same_case_joint_gate": True, "per_judge_development_gate": True,
        "semantic_results": {s: {"secondary": deepcopy(endpoint), "primary": deepcopy(endpoint)}
            for s in ("short_development", "long_diagnostic")},
        "same_case_joint": {s: deepcopy(endpoint) for s in ("short_development", "long_diagnostic")}}


def test_final_requires_actual_same_case_development_not_marginals():
    campaign = qualified_development()
    assert run.development_gate(campaign) is True
    campaign["same_case_joint"]["short_development"]["passed"] = 58
    with pytest.raises(ValueError, match="same61of64"):
        run.development_gate(campaign)
    campaign = qualified_development()
    campaign["semantic_results"]["long_diagnostic"]["primary"] = None
    with pytest.raises(ValueError, match="same61of64"):
        run.development_gate(campaign)
    assert run.development_gate({"semantic_development_gate": False}) is False


def test_no_final_source_read_or_hash_before_development_qualification(monkeypatch):
    monkeypatch.setattr(run, "admission_state", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified final source access")

    monkeypatch.setattr(run, "file_hash", forbidden)
    monkeypatch.setattr(run.q, "read", forbidden)
    for function in (run.require_admission, run.final_rows, run.prepare, run.sources,
            run.inputs, run.features, run.body_replay, run.head_replay, run.generate):
        with pytest.raises(ValueError, match="before final source access"):
            function()


def test_reserved_metadata_and_pending_producer_never_open_sealed_contents(tmp_path, monkeypatch):
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if path.name in ("final_short_sources.json", "final_long_sources.json", "sealed_test_sources.json", "sealed_final_sources.json"):
            raise AssertionError("sealed content must remain closed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    values, _ = run.sealed_metadata()
    assert {s: len(v) for s, v in values.items()} == run.COUNTS
    first = run.producing(tmp_path / "first", "systems_pending_seal_contract", pending=True)
    original_minimum = run.minimum
    monkeypatch.setattr(run, "minimum", lambda count: original_minimum(count) + 1)
    changed = run.producing(tmp_path / "changed", "systems_pending_seal_contract", pending=True)
    assert changed["fingerprint"] != first["fingerprint"]
    monkeypatch.setattr(run, "COUNTS", {"short_final": 256, "long_final": 65})
    with pytest.raises(ValueError, match="metadata"):
        run.sealed_metadata()
    assert first["config"]["final_test_opened"] is False
    assert {"experiments/neural_architecture_lab/finalize_ncm_sealed_final_quality.py",
        "experiments/neural_architecture_lab/finalize_ncm_clean_development_features.py"} <= first["source_hashes"].keys()
    assert not any(Path(p).name in ("final_short_sources.json", "final_long_sources.json", "sealed_test_sources.json", "sealed_final_sources.json")
        for p in first["input_hashes"])
    run.base.verify(first)


def test_final_consensus_uses_244_and61_same_case_thresholds_and_all_valid():
    assert run.minimum(256) == 244 and run.minimum(64) == 61
    secondary = [{"id": str(i), "valid": True, "passed": i < 244} for i in range(256)]
    primary = deepcopy(secondary)
    result = run.consensus(secondary, primary)
    assert result["qualified"] and result["joint_passed"] == 244
    primary[0]["passed"] = False
    primary[244]["passed"] = True
    assert sum(r["passed"] for r in primary) == 244
    assert run.consensus(secondary, primary)["qualified"] is False
    primary[1].update(valid=False, passed=None)
    assert run.consensus(secondary, primary)["task_accuracy"] is None
    assert run.consensus(secondary, None)["qualified"] is False


def test_final_source_content_needs_access_intent_even_after_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORKFLOW", tmp_path)
    monkeypatch.setattr(run, "require_admission", lambda: {})
    monkeypatch.setattr(run, "sealed_metadata", lambda: ({}, {}))
    with pytest.raises(ValueError, match="durable final access intent"):
        run.final_rows()


def test_pilot_selection_retains_first_and_complete_largest_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DATA", tmp_path)
    run.write_json(tmp_path / "selection.json", {"inference_pilot_ids": ["first", "longest"]})
    cases = {"short_final": [{"id": "first"}], "long_final": [{"id": "longest"}, {"id": "other"}]}
    assert [r["id"] for r in run.selected_rows(cases)] == ["first", "longest"]


def test_fresh_body_integration_checks_actual_identity_precision_and_values():
    row = {"id": "source", "source_sha256": "sha", "alignment_sha256": "alignment",
        "native_prompt_sha256": "prompt", "source_ids": [1, 2]}
    tensor = torch.zeros(2, 3072)
    cached = {k: v for k, v in row.items() if k != "source_ids"}
    cached["features"] = tensor.clone()
    run.require_body_match(row, tensor, cached)
    for value in (torch.ones(2, 3072), torch.zeros(1, 3072), torch.zeros(2, 3072, dtype=torch.float16),
            torch.full((2, 3072), float("nan")), torch.zeros(2, 3072, requires_grad=True)):
        with pytest.raises(ValueError, match="exact source identity"):
            run.require_body_match(row, value, cached)
    with pytest.raises(ValueError, match="exact source identity"):
        run.require_body_match(row, tensor, {**cached, "source_sha256": "another source"})
