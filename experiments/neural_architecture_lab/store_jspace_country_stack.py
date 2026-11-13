"""Store scoped country-stack readout compression, never full-model accuracy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_country_stack import DEFAULT_ROOT, STUDY_ID, checked, verify
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


SCOPE="fraction_of_reused_dense_swap_worlds_preserving_all_nine_country_probe_labels"


def result_for(root):
    plan=verify(root); checked(root,"fit"); checked(root,"capture")
    result=json.loads((root/"result.json").read_text()); aggregate=json.loads((root/"campaign_results.json").read_text())
    if (result["study_id"]!=STUDY_ID or result["status"]!="completed" or result["confirmed"] is not False
            or result["scientific_fingerprint"]!=plan["scientific_fingerprint"]
            or aggregate["result_sha256"]!=file_hash(root/"result.json")
            or {k:v for k,v in aggregate.items() if k not in ("result_path","result_sha256")}!=result):
        raise ValueError("Aggregate scope/provenance mismatch")
    for name,expected in result["artifacts"].items():
        if file_hash(root/name)!=expected:raise ValueError("Result artifact changed")
    selected=str(result["selected_degree"])
    accuracy=result["metrics"]["dense"]["predictors"][selected]["world_agreement"]
    return plan,result,accuracy


def readback(root, output, chromadb, key=None):
    plan,result,accuracy=result_for(root)
    if key is None:key=json.loads(output.read_text())["storage"]["result_hash"]
    if output.exists():
        record=json.loads(output.read_text())
        for path,expected_hash in record["source_sha256"].items():
            if file_hash(path)!=expected_hash:raise RuntimeError("Stored source provenance mismatch")
    logger=_make_logger(chromadb)
    rows=logger.experiments_collection.get(ids=[key],include=["metadatas"])
    h=logger.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    if rows["ids"]!=[key] or h["ids"]!=[STUDY_ID]:raise RuntimeError("Registry IDs missing")
    expected={"hypothesis_id":STUDY_ID,"accuracy":accuracy,"primary_metric":accuracy,"accuracy_scope":SCOPE,
              "confirmed":False,"confirmation_status":result["classification"],"aggregate_sha256":file_hash(root/"campaign_results.json")}
    if any(rows["metadatas"][0].get(k)!=v for k,v in expected.items()):raise RuntimeError("Experiment readback mismatch")
    hm=h["metadatas"][0]
    if (hm.get("confirmed") is not False or hm.get("confirmation_status")!=result["classification"]
            or hm.get("evidence_count")!=1 or hm.get("direct_experiment_count")!=1
            or json.loads(hm.get("direct_result_hashes_json","[]"))!=[key]
            or json.loads(hm.get("direct_experiment_ids_json","[]"))!=["jspace-country-stack-"+plan["scientific_fingerprint"][:16]]):
        raise RuntimeError("Hypothesis readback mismatch")
    return {"readback_verified":True,"result_hash":key,**expected}


def store(root, report, output, chromadb):
    if output.exists():raise FileExistsError(output)
    plan,result,accuracy=result_for(root)
    diagnostic=json.loads((root/"identifiability_audit.json").read_text())
    pullback=json.loads((root/"attention_pullback_audit.json").read_text())
    exported=json.loads((root/"export_manifest.json").read_text())
    if diagnostic["main_result_sha256"]!=file_hash(root/"result.json") or diagnostic["confirmed"] is not False:
        raise ValueError("Posthoc diagnostic provenance mismatch")
    if pullback["confirmed"] is not False or pullback["control"]["passed"] is not True:
        raise ValueError("Pullback control scope mismatch")
    if (exported["report_sha256"]!=file_hash(report) or exported["readback_verified"] is not True
            or exported["archive_sha256"]!=file_hash(root/"country_stack_bundle.zip")):
        raise ValueError("Export provenance mismatch")
    paths=[root/"plan.json",root/"result.json",root/"campaign_results.json",report,Path(__file__),
           root/"identifiability_audit.json",root/"presentation_metrics.json",root/"linear_readout_reference.pt",
           root/"attention_pullback_audit.json",root/"export_manifest.json",root/"country_stack_bundle.zip"]
    hypothesis={"id":STUDY_ID,"name":"Stacked country readouts with exact surrogate Groebner compression",
                "category":"representation","tested":True,"confirmed":False,"evidence_count":1,"direct_experiment_count":1,
                "confirmation_status":result["classification"],"created_at":datetime.now(timezone.utc).isoformat(),
                "tested_scope":"posthoc_reused_worlds_country_readout_stack_not_transformer_replacement",
                "description":"Exact finite country algebra and exact canonicalization of a fitted continuous stack surrogate; neural fidelity and readout payload measured separately."}
    model_parameters=json.loads((Path(plan["parent_root"])/"training_manifest.json").read_text())["model_parameters"]
    experiment=ExperimentResult(experiment_id="jspace-country-stack-"+plan["scientific_fingerprint"][:16],hypothesis_id=STUDY_ID,
        metrics={"accuracy":accuracy,"readout_numerical_payload_ratio":result["gates"]["readout_payload_ratio"]},primary_metric=accuracy,
        model_architecture=[2048]*36,model_parameters=model_parameters,model_checkpoint=plan["model"]["path"],
        training_time=result["fit"]["elapsed_seconds"],observations=[SCOPE,str(report),result["classification"],
        "The parameter count describes the unchanged Qwen checkpoint. Fitting time includes CPU symbolic certification. No full-model compression, general language accuracy or speedup was measured."])
    logger=_make_logger(chromadb);logger.log_hypothesis(hypothesis);key=logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[key],metadatas=[{"accuracy_scope":SCOPE,"confirmed":False,
        "confirmation_status":result["classification"],"aggregate_sha256":file_hash(root/"campaign_results.json"),
        "full_model_compression_tested":False,"evaluation_reused":True}])
    logger.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[{"direct_result_hashes_json":json.dumps([key]),
        "direct_experiment_ids_json":json.dumps([experiment.experiment_id]),"report_path":str(report),
        "aggregate_sha256":file_hash(root/"campaign_results.json")}])
    storage=readback(root,output,chromadb,key)
    write_json(output,{"schema_version":"nal.meta-hypothesis.v1","hypothesis":hypothesis,"result":result,"storage":storage,
                       "posthoc_identifiability_audit":diagnostic,
                       "posthoc_attention_pullback_control":pullback,
                       "source_sha256":{str(p):file_hash(p) for p in paths}})
    return storage


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("store","readback"));parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--report",type=Path,default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-country-stack.md")
    parser.add_argument("--output",type=Path,default=ROOT/"data/hypothesis_registry/jspace_country_stack_20261113.json")
    parser.add_argument("--chromadb",type=Path,default=ROOT/"data/chroma_db")
    args=parser.parse_args()
    print(json.dumps(store(args.root,args.report,args.output,args.chromadb) if args.stage=="store" else readback(args.root,args.output,args.chromadb),allow_nan=False))
