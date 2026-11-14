"""Store and independently read back the scoped all-layer compression result."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_full_compact_record import JSpaceFullCompactRecord
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net import jspace_full_capture as base
from experiments.structure_net.jspace_full_affine_v2 import DEFAULT_ROOT
from experiments.structure_net.jspace_bench_data import sha,write

REPORT=base.ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-full-compact.md"
REGISTRY=base.ROOT/"data/hypothesis_registry/jspace_full_compact_20261113.json"
CLASSIFICATION="completed_exploratory_full_model_compression_campaign"


def expected(root):
    plan=base.verify(root);result=json.loads((root/"result.json").read_text())
    aggregate=json.loads((root/"campaign_results.json").read_text())
    if aggregate["result_sha256"]!=sha(root/"result.json") or aggregate["result"]!=result:raise ValueError("Aggregate mismatch")
    if result["fingerprint"]!=plan["scientific_fingerprint"]:raise ValueError("Result fingerprint mismatch")
    record=JSpaceFullCompactRecord.from_result(result)
    reload=aggregate["checkpoint_reload"]
    if reload["all_256_records_exact"] is not True or reload["source_checkpoint_sha256"]!=result["checkpoint_sha256"]:
        raise ValueError("Checkpoint readback not verified")
    return plan,result,record


def wanted(root,record):
    return {"hypothesis_id":record.study_id,"accuracy":record.accuracy,"primary_metric":record.accuracy,
            "accuracy_scope":record.scope,"confirmed":False,"confirmation_status":CLASSIFICATION,
            "aggregate_sha256":sha(root/"campaign_results.json"),"matched_graph_speedup":record.speedup,
            "trained_layers":record.trained_layers,"correct":record.correct,"count":record.count}


def readback(root,output,db,key=None):
    plan,result,record=expected(root)
    if key is None:key=json.loads(output.read_text())["storage"]["result_hash"]
    if output.exists():
        stored=json.loads(output.read_text())
        for path,value in stored["source_sha256"].items():
            if sha(path)!=value:raise ValueError("Registry provenance changed")
    logger=_make_logger(db);rows=logger.experiments_collection.get(ids=[key],include=["metadatas"])
    hypotheses=logger.hypotheses_collection.get(ids=[record.study_id],include=["metadatas"])
    metadata=wanted(root,record)
    if rows["ids"]!=[key] or any(rows["metadatas"][0].get(k)!=v for k,v in metadata.items()):
        raise ValueError("Experiment readback mismatch")
    if hypotheses["ids"]!=[record.study_id]:raise ValueError("Missing hypothesis")
    h=hypotheses["metadatas"][0]
    if (h.get("tested") is not True or h.get("confirmed") is not False or h.get("confirmation_status")!=CLASSIFICATION
        or h.get("evidence_count")!=1 or json.loads(h["direct_result_hashes_json"])!=[key]):
        raise ValueError("Hypothesis readback mismatch")
    return {"readback_verified":True,"result_hash":key,**metadata}


def store(root,report,output,db):
    if output.exists():raise FileExistsError(output)
    plan,result,record=expected(root)
    aggregate=json.loads((root/"campaign_results.json").read_text())
    hypothesis={"id":record.study_id,"name":"Full-model functional coordinates and polynomial core with retained residual",
        "category":"compression","tested":True,"confirmed":False,"evidence_count":1,"direct_experiment_count":1,
        "confirmation_status":CLASSIFICATION,"created_at":datetime.now(timezone.utc).isoformat(),"tested_scope":record.scope,
        "description":"All 36 Qwen2.5-3B blocks: common task-Jacobian coordinates, affine RMS/value folding, reduced RoPE-compatible attention group, Groebner-normal-form MLP core, progressive distillation. Includes the preserved v1 mean-omission failure and a corrected implementation; one exploratory campaign."}
    experiment=ExperimentResult(experiment_id="jspace-full-compact-"+plan["scientific_fingerprint"][:16],
        hypothesis_id=record.study_id,metrics={"accuracy":record.accuracy,"matched_graph_speedup":record.speedup,
            "model_tensor_byte_reduction":1-record.tensor_bytes/record.original_tensor_bytes,
            "original_accuracy":record.original_correct/record.count},primary_metric=record.accuracy,
        model_architecture=[2048]*36,model_parameters=record.parameter_count,model_checkpoint=aggregate["checkpoint_reload"]["checkpoint"],
        training_time=sum(v["seconds"] for v in result["layers"]),observations=[record.scope,str(report),CLASSIFICATION,
            "One seed; reused evaluation pool. Actual answer accuracy, not a reconstruction proxy.",
            "Same graph runtime on teacher and student. KV padding means no KV-cache byte reduction claim.",
            "Materialized model reloaded in a fresh process with all 256 evaluation logits identical."])
    logger=_make_logger(db);logger.log_hypothesis(hypothesis);key=logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[key],metadatas=[wanted(root,record)])
    logger.hypotheses_collection.update(ids=[record.study_id],metadatas=[{"direct_result_hashes_json":json.dumps([key]),
        "direct_experiment_ids_json":json.dumps([experiment.experiment_id]),"report_path":str(report),
        "aggregate_sha256":sha(root/"campaign_results.json")}])
    storage=readback(root,output,db,key)
    paths=[Path(__file__),base.ROOT/"src/neural_architecture_lab/jspace_full_compact_record.py",report,
           root/"plan.json",root/"result.json",root/"campaign_results.json",Path(aggregate["export_root"])/"reload/verification.json"]
    write(output,{"schema_version":"nal.meta-hypothesis.v1","hypothesis":hypothesis,"record":record.to_dict(),
                  "storage":storage,"source_sha256":{str(p):sha(p) for p in paths}})
    return storage


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("store","readback"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--output",type=Path,default=REGISTRY);p.add_argument("--db",type=Path,default=base.ROOT/"data/chroma_db")
    a=p.parse_args();print(json.dumps(store(a.root,a.report,a.output,a.db) if a.stage=="store" else readback(a.root,a.output,a.db),allow_nan=False))
