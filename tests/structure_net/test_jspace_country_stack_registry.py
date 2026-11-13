import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from experiments.neural_architecture_lab import store_jspace_country_stack as registry


@pytest.mark.parametrize("corruption",[None,"accuracy_scope","confirmed","evidence_count"])
def test_registry_does_not_turn_readout_fidelity_into_model_accuracy(monkeypatch,corruption):
    monkeypatch.setattr(registry,"result_for",lambda root:({"scientific_fingerprint":"abc"},{"classification":"pilot"},.98))
    monkeypatch.setattr(registry,"file_hash",lambda path:"aggregate")
    metadata={"hypothesis_id":registry.STUDY_ID,"accuracy":.98,"primary_metric":.98,"accuracy_scope":registry.SCOPE,
              "confirmed":False,"confirmation_status":"pilot","aggregate_sha256":"aggregate"}
    hypothesis={"confirmed":False,"confirmation_status":"pilot","evidence_count":1,"direct_experiment_count":1,
                "direct_result_hashes_json":'["record"]',"direct_experiment_ids_json":'["jspace-country-stack-abc"]'}
    if corruption=="evidence_count":hypothesis[corruption]=3
    elif corruption:metadata[corruption]=True if corruption=="confirmed" else "general_model_accuracy"
    logger=SimpleNamespace(experiments_collection=SimpleNamespace(get=lambda **kw:{"ids":["record"],"metadatas":[metadata]}),
        hypotheses_collection=SimpleNamespace(get=lambda **kw:{"ids":[registry.STUDY_ID],"metadatas":[hypothesis]}))
    monkeypatch.setattr(registry,"_make_logger",lambda path:logger)
    if corruption:
        with pytest.raises(RuntimeError,match="mismatch"):
            registry.readback(Path("root"),Path("record"),Path("db"),"record")
    else:
        assert registry.readback(Path("root"),Path("record"),Path("db"),"record")["accuracy_scope"]==registry.SCOPE
