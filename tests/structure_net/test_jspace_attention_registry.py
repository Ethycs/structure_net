import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.neural_architecture_lab import store_jspace_attention_transport as registry


@pytest.mark.parametrize("corruption", [None, "accuracy_scope", "primary_scope", "confirmed", "aggregate_sha256", "evidence_count"])
def test_readback_keeps_teacher_accuracy_and_causal_rate_separate(monkeypatch, corruption):
    plan = {"scientific_fingerprint": "abcdef"}
    result = {"classification": "pilot_only", "teacher": {"accuracy": .75}}
    monkeypatch.setattr(registry, "checked_result", lambda root:(plan, result, .125))
    monkeypatch.setattr(registry, "file_hash", lambda path:"aggregate")
    metadata = {"hypothesis_id": registry.STUDY_ID, "accuracy": .75, "primary_metric": .125,
                "accuracy_scope": registry.ACCURACY_SCOPE, "primary_scope": registry.PRIMARY_SCOPE,
                "confirmed": False, "confirmation_status": "pilot_only", "aggregate_sha256": "aggregate"}
    hypothesis = {"confirmed": False, "confirmation_status": "pilot_only", "evidence_count": 1, "direct_experiment_count": 1,
                  "direct_experiment_ids_json": json.dumps(["jspace-attention-transport-abcdef"]), "direct_result_hashes_json": '["record"]'}
    if corruption == "evidence_count":
        hypothesis[corruption] = 4
    elif corruption:
        metadata[corruption] = True if corruption == "confirmed" else "wrong scope or hash"
    logger = SimpleNamespace(experiments_collection=SimpleNamespace(get=lambda **kwargs:{"ids":["record"], "metadatas":[metadata]}),
                             hypotheses_collection=SimpleNamespace(get=lambda **kwargs:{"ids":[registry.STUDY_ID], "metadatas":[hypothesis]}))
    monkeypatch.setattr(registry, "_make_logger", lambda path:logger)
    if corruption:
        with pytest.raises(RuntimeError, match="mismatch"):
            registry.readback(Path("root"), Path("output"), Path("db"), "record")
    else:
        out = registry.readback(Path("root"), Path("output"), Path("db"), "record")
        assert out["accuracy"] == .75 and out["primary_metric"] == .125 and out["confirmed"] is False
