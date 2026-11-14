"""Register actual combined-model benchmark accuracy and scoped basis reuse."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_bench_tensor import ROOT, STUDY_ID, CONFIG, verify, checked
from experiments.structure_net.jspace_bench_tensor_v2 import DEFAULT_ROOT
from experiments.structure_net.jspace_bench_covered import DEFAULT_ROOT as FOLLOWUP_ROOT
from experiments.structure_net.jspace_bench_stacked import DEFAULT_ROOT as STACKED_ROOT
from experiments.structure_net.jspace_stacked_graph_decode_v3 import DEFAULT_ROOT as GRAPH_ROOT
from experiments.structure_net.jspace_bench_data import sha, write


SCOPE="four_family_length_filtered_multiple_choice_accuracy_of_combined_residual_model"


def expected(root):
    plan=verify(root)
    for stage in ("capture","fit","evaluation"):checked(root,stage)
    result=json.loads((root/"result.json").read_text())
    aggregate=json.loads((root/"campaign_results.json").read_text())
    if aggregate.pop("result_sha256")!=sha(root/"result.json") or aggregate!=result or result["confirmed"] is not False:
        raise ValueError("Aggregate mismatch")
    for name,value in result["artifacts"].items():
        if sha(root/name)!=value:raise ValueError("Artifact changed")
    accuracy=result["result"]["metrics"]["compressed"]["accuracy"]["overall"]
    return plan,result,accuracy


def expected_followup(root):
    plan=verify(root)
    checked(root,"covered")
    result=json.loads((root/"result.json").read_text())
    aggregate=json.loads((root/"campaign_results.json").read_text())
    if (aggregate.pop("result_sha256")!=sha(root/"result.json") or aggregate!=result
            or result["confirmed"] is not False
            or result["scientific_fingerprint"]!=plan["scientific_fingerprint"]
            or plan["followup_config"]["new_independent_confirmation"] is not False):
        raise ValueError("Follow-up aggregate mismatch")
    accuracy=result["result"]["metrics"]["factorized"]["accuracy"]["overall"]
    metadata={"followup_accuracy":accuracy,"followup_confirmation_status":result["classification"],
              "followup_quality_gate":result["quality_gate"],
              "followup_aggregate_sha256":sha(root/"campaign_results.json")}
    return result,metadata


def expected_engineering(root):
    plan=verify(root)
    checked(root,"stacked")
    result=json.loads((root/"result.json").read_text())
    aggregate=json.loads((root/"campaign_results.json").read_text())
    if (aggregate.pop("result_sha256")!=sha(root/"result.json") or aggregate!=result
            or result["confirmed"] is not False
            or result["scientific_fingerprint"]!=plan["scientific_fingerprint"]):
        raise ValueError("Engineering aggregate mismatch")
    accuracy=result["result"]["metrics"]["stacked_polynomial_core"]["accuracy"]["overall"]
    return result,{"stacked_accuracy":accuracy,"stacked_classification":result["classification"],
                   "stacked_aggregate_sha256":sha(root/"campaign_results.json"),
                   "stacked_speedup_vs_previous_factorized":result["speedup_vs_previous_factorized"],
                   "stacked_speedup_vs_matched_stacked_original":result["speedup_vs_matched_stacked_original"]}


def expected_runtime(root):
    plan=verify(root);checked(root,"graph")
    result=json.loads((root/"result.json").read_text())
    aggregate=json.loads((root/"campaign_results.json").read_text())
    if (aggregate.pop("result_sha256")!=sha(root/"result.json") or aggregate!=result
            or result["confirmed"] is not False or result["scientific_fingerprint"]!=plan["scientific_fingerprint"]):
        raise ValueError("Runtime aggregate mismatch")
    return result,{"graph_accuracy":result["accuracy"]["overall"],
                   "graph_tokens_per_second":result["timings"]["cuda_graph"]["tokens_per_second"],
                   "graph_speedup":result["graph_speedup"],"graph_aggregate_sha256":sha(root/"campaign_results.json")}


def readback(root,output,db,key=None,followup_root=None,engineering_root=None,runtime_root=None):
    plan,result,accuracy=expected(root)
    if key is None:key=json.loads(output.read_text())["storage"]["result_hash"]
    if output.exists():
        record=json.loads(output.read_text())
        followup_root=Path(record["supplemental_evidence"]["root"])
        engineering_root=Path(record["engineering_evidence"]["root"])
        runtime_root=Path(record["runtime_evidence"]["root"])
        for path,value in record["source_sha256"].items():
            if sha(path)!=value:raise ValueError("Registry provenance changed")
    logger=_make_logger(db)
    rows=logger.experiments_collection.get(ids=[key],include=["metadatas"])
    hypotheses=logger.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    wanted={"hypothesis_id":STUDY_ID,"accuracy":accuracy,"primary_metric":accuracy,"accuracy_scope":SCOPE,
            "confirmed":False,"confirmation_status":result["classification"],"aggregate_sha256":sha(root/"campaign_results.json")}
    supplement={} if followup_root is None else expected_followup(followup_root)[1]
    if engineering_root is not None:supplement.update(expected_engineering(engineering_root)[1])
    if runtime_root is not None:supplement.update(expected_runtime(runtime_root)[1])
    wanted.update(supplement)
    if rows["ids"]!=[key] or any(rows["metadatas"][0].get(k)!=v for k,v in wanted.items()):
        raise RuntimeError("Experiment readback mismatch")
    if hypotheses["ids"]!=[STUDY_ID]:raise RuntimeError("Hypothesis missing")
    h=hypotheses["metadatas"][0]
    if (h.get("confirmed") is not False or h.get("confirmation_status")!=result["classification"]
            or h.get("evidence_count")!=1 or h.get("direct_experiment_count")!=1
            or json.loads(h["direct_result_hashes_json"])!=[key]
            or any(h.get(k)!=v for k,v in supplement.items())):
        raise RuntimeError("Hypothesis readback mismatch")
    return {"readback_verified":True,"result_hash":key,**wanted}


def store(root,report,output,db,followup_root=FOLLOWUP_ROOT,engineering_root=STACKED_ROOT,runtime_root=GRAPH_ROOT):
    if output.exists():raise FileExistsError(output)
    plan,result,accuracy=expected(root)
    followup,supplement=expected_followup(followup_root)
    engineering,engineering_metadata=expected_engineering(engineering_root)
    supplement.update(engineering_metadata)
    runtime,runtime_metadata=expected_runtime(runtime_root)
    supplement.update(runtime_metadata)
    storage_audit=json.loads((root/"storage_audit.json").read_text())
    hypothesis={"id":STUDY_ID,"name":"Shared task-Jacobian basis and Groebner tensor compilation with residual",
                "category":"compression","tested":True,"confirmed":False,"evidence_count":1,"direct_experiment_count":1,
                "confirmation_status":result["classification"],"created_at":datetime.now(timezone.utc).isoformat(),
                "tested_scope":SCOPE,"description":"Common chart across four tasks, polynomial elimination of selected MLP channels, retained nonlinear residual, and cross-task generator reuse."}
    experiment=ExperimentResult(experiment_id="jspace-bench-tensor-"+plan["scientific_fingerprint"][:16],hypothesis_id=STUDY_ID,
        metrics={"accuracy":accuracy,"speedup":result["speedup"],"model_tensor_byte_reduction":result["model_tensor_byte_reduction"],
                 "followup_accuracy":supplement["followup_accuracy"],"followup_speedup":followup["speedup"],
                 "stacked_accuracy":supplement["stacked_accuracy"],
                 "graph_accuracy":supplement["graph_accuracy"],"graph_speedup":supplement["graph_speedup"],
                 "stacked_speedup_vs_previous_factorized":engineering["speedup_vs_previous_factorized"]},
        primary_metric=accuracy,model_architecture=[2048]*36,
        model_parameters=storage_audit["combined_learned_weight_values_including_tensor_buffers"],
        model_checkpoint=str(root/"compressed_modules.pt"),training_time=checked(root,"fit")["elapsed_seconds"],
        observations=[SCOPE,str(report),result["classification"],
            "Checkpoint is a replacement-module delta requiring the pinned original Qwen checkpoint and fit metadata.",
            "Parameter count includes polynomial coefficient buffers; timing is a small batch-one generation pilot; not a semantic CAV causality result.",
            "One campaign record includes strict-coverage and stacked-kernel extensions using the same evaluation pool; these are not independent evidence."])
    logger=_make_logger(db);logger.log_hypothesis(hypothesis);key=logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[key],metadatas=[{"accuracy_scope":SCOPE,"confirmed":False,
        "confirmation_status":result["classification"],"aggregate_sha256":sha(root/"campaign_results.json"),**supplement}])
    logger.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[{"direct_result_hashes_json":json.dumps([key]),
        "direct_experiment_ids_json":json.dumps([experiment.experiment_id]),"report_path":str(report),
        "aggregate_sha256":sha(root/"campaign_results.json"),**supplement}])
    storage=readback(root,output,db,key,followup_root,engineering_root,runtime_root)
    paths=[Path(__file__),report,root/"plan.json",root/"result.json",root/"campaign_results.json",root/"storage_audit.json",
           root/"basis_redundancy_summary.json",followup_root/"plan.json",followup_root/"result.json",
           followup_root/"campaign_results.json",followup_root/"storage_audit.json",engineering_root/"plan.json",
           engineering_root/"result.json",engineering_root/"campaign_results.json",engineering_root/"storage_audit.json",
           runtime_root/"plan.json",runtime_root/"result.json",runtime_root/"campaign_results.json"]
    write(output,{"schema_version":"nal.meta-hypothesis.v1","hypothesis":hypothesis,"result":result,"storage":storage,
                  "supplemental_evidence":{"root":str(followup_root),"result":followup,"new_independent_confirmation":False},
                  "engineering_evidence":{"root":str(engineering_root),"result":engineering,"new_independent_confirmation":False},
                  "runtime_evidence":{"root":str(runtime_root),"result":runtime,"new_independent_confirmation":False},
                  "source_sha256":{str(p):sha(p) for p in paths}})
    return storage


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("store","readback"));parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--report",type=Path,default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-bench-tensor.md")
    parser.add_argument("--output",type=Path,default=ROOT/"data/hypothesis_registry/jspace_bench_tensor_20261113.json")
    parser.add_argument("--db",type=Path,default=ROOT/"data/chroma_db")
    args=parser.parse_args()
    print(json.dumps(store(args.root,args.report,args.output,args.db) if args.stage=="store" else readback(args.root,args.output,args.db),allow_nan=False))
