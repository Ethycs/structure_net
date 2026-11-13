"""Storage readback must preserve endpoint scopes and the one-study evidence count."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.neural_architecture_lab import store_jspace_active_cav_heads as registry


def fake_registry(monkeypatch):
    result = {"scientific_fingerprint": "abc", "classification": "country_CAV_causality_not_established"}
    experiment_id = "jspace-active-cav-heads-abc"
    experiment = {"hypothesis_id": registry.STUDY_ID, "accuracy": .96875, "primary_metric": -.015625,
                  "accuracy_scope": registry.SCOPE, "primary_scope": registry.PRIMARY_SCOPE,
                  "confirmed": False, "confirmation_status": result["classification"], "experiment_id": experiment_id}
    hypothesis = {"confirmed": False, "confirmation_status": result["classification"],
                  "evidence_count": 1, "direct_experiment_count": 1,
                  "direct_experiment_ids_json": json.dumps([experiment_id]), "direct_result_hashes_json": '["result-hash"]'}
    logger = SimpleNamespace(
        experiments_collection=SimpleNamespace(get=lambda **kwargs: {"ids": ["result-hash"], "metadatas": [experiment]}),
        hypotheses_collection=SimpleNamespace(get=lambda **kwargs: {"ids": [registry.STUDY_ID], "metadatas": [hypothesis]}))
    monkeypatch.setattr(registry, "_make_logger", lambda path: logger)
    monkeypatch.setattr(registry, "checked_result", lambda root: ({}, {}, result, .96875, -.015625))
    return experiment, hypothesis


def test_readback_preserves_accuracy_primary_scope_and_direct_ids(monkeypatch):
    fake_registry(monkeypatch)
    result = registry.readback(Path("root"), Path("record"), Path("db"), result_hash="result-hash")
    assert result["readback_verified"] and result["confirmed"] is False
    assert result["accuracy_scope"] == registry.SCOPE
    assert result["primary_metric"] == -.015625
    assert result["direct_experiment_count"] == result["evidence_count"] == 1


@pytest.mark.parametrize("target,key,value", [
    ("experiment", "accuracy_scope", "general_language_model_accuracy"),
    ("experiment", "confirmed", True),
    ("hypothesis", "evidence_count", 3),
    ("hypothesis", "direct_result_hashes_json", '["wrong-result"]'),
])
def test_readback_rejects_scope_upgrades_and_broken_provenance(monkeypatch, target, key, value):
    experiment, hypothesis = fake_registry(monkeypatch)
    {"experiment": experiment, "hypothesis": hypothesis}[target][key] = value
    with pytest.raises(RuntimeError, match="mismatch"):
        registry.readback(Path("root"), Path("record"), Path("db"), result_hash="result-hash")
