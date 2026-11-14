import json
from types import SimpleNamespace
import pytest
from experiments.neural_architecture_lab import store_jspace_bench_tensor as store


@pytest.mark.parametrize("corruption",[None,"accuracy_scope","confirmed"])
def test_registry_keeps_combined_task_accuracy_separate_from_basis_counts(monkeypatch,tmp_path,corruption):
    monkeypatch.setattr(store,"expected",lambda root:({}, {"classification":"quality_preserved_no_speed_gate"},.75))
    monkeypatch.setattr(store,"sha",lambda path:"aggregate")
    meta={"hypothesis_id":store.STUDY_ID,"accuracy":.75,"primary_metric":.75,"accuracy_scope":store.SCOPE,
          "confirmed":False,"confirmation_status":"quality_preserved_no_speed_gate","aggregate_sha256":"aggregate"}
    h={"confirmed":False,"confirmation_status":"quality_preserved_no_speed_gate","evidence_count":1,
       "direct_experiment_count":1,"direct_result_hashes_json":json.dumps(["record"])}
    if corruption:meta[corruption]=True if corruption=="confirmed" else "basis_redundant_fraction"
    logger=SimpleNamespace(experiments_collection=SimpleNamespace(get=lambda **kwargs:{"ids":["record"],"metadatas":[meta]}),
        hypotheses_collection=SimpleNamespace(get=lambda **kwargs:{"ids":[store.STUDY_ID],"metadatas":[h]}))
    monkeypatch.setattr(store,"_make_logger",lambda db:logger)
    if corruption:
        with pytest.raises(RuntimeError,match="mismatch"):
            store.readback(tmp_path,tmp_path/"record.json",tmp_path/"db","record")
    else:
        result=store.readback(tmp_path,tmp_path/"record.json",tmp_path/"db","record")
        assert result["accuracy"]==.75 and result["confirmed"] is False


@pytest.mark.parametrize("corrupt_followup",[False,True])
@pytest.mark.parametrize("with_engineering",[False,True])
def test_readback_distinguishes_primary_failure_from_followup(monkeypatch,tmp_path,corrupt_followup,with_engineering):
    monkeypatch.setattr(store,"expected",lambda root:({}, {"classification":"combined_quality_gate_failed"},.75))
    monkeypatch.setattr(store,"sha",lambda path:"aggregate")
    supplement={"followup_accuracy":.7,"followup_confirmation_status":"covered_quality_preserved_no_speed_gate",
                "followup_quality_gate":True,"followup_aggregate_sha256":"followup_hash"}
    engineering={"stacked_accuracy":.72,"stacked_classification":"dense_stacked_core_engineering_measurement",
                 "stacked_aggregate_sha256":"stacked_hash"}
    monkeypatch.setattr(store,"expected_followup",lambda root:({},supplement))
    monkeypatch.setattr(store,"expected_engineering",lambda root:({},engineering))
    if with_engineering:supplement.update(engineering)
    meta={"hypothesis_id":store.STUDY_ID,"accuracy":.75,"primary_metric":.75,"accuracy_scope":store.SCOPE,
          "confirmed":False,"confirmation_status":"combined_quality_gate_failed","aggregate_sha256":"aggregate",**supplement}
    h={"confirmed":False,"confirmation_status":"combined_quality_gate_failed","evidence_count":1,
       "direct_experiment_count":1,"direct_result_hashes_json":json.dumps(["record"]),**supplement}
    if corrupt_followup:meta["followup_accuracy"]=.75
    logger=SimpleNamespace(experiments_collection=SimpleNamespace(get=lambda **kwargs:{"ids":["record"],"metadatas":[meta]}),
        hypotheses_collection=SimpleNamespace(get=lambda **kwargs:{"ids":[store.STUDY_ID],"metadatas":[h]}))
    monkeypatch.setattr(store,"_make_logger",lambda db:logger)
    if corrupt_followup:
        with pytest.raises(RuntimeError,match="mismatch"):
            store.readback(tmp_path,tmp_path/"record.json",tmp_path/"db","record",tmp_path/"followup",
                           tmp_path/"stacked" if with_engineering else None)
    else:
        result=store.readback(tmp_path,tmp_path/"record.json",tmp_path/"db","record",tmp_path/"followup",
                              tmp_path/"stacked" if with_engineering else None)
        assert result["accuracy"]==.75 and result["followup_accuracy"]==.7
        assert result["confirmation_status"]=="combined_quality_gate_failed"
        assert result["followup_quality_gate"] is True and result["confirmed"] is False
        if with_engineering:assert result["stacked_accuracy"]==.72
