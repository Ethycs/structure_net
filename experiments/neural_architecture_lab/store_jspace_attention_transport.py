"""Index the attention pilot with teacher accuracy separate from intervention success."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_attention_study import DEFAULT_ROOT, STUDY_ID, verify, checked
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


ACCURACY_SCOPE = "unmodified_Qwen_country_attribute_task_accuracy_128_fresh_worlds"
PRIMARY_SCOPE = "selected_attention_J_edit_target_answer_rate_all_128_worlds_not_model_accuracy"


def checked_result(root):
    plan = verify(root)
    for stage in ("discovery", "training", "evaluation"):
        checked(root, stage)
    result = json.loads((root/"result.json").read_text())
    aggregate = json.loads((root/"campaign_results.json").read_text())
    if (result["study_id"] != STUDY_ID or result["status"] != "completed" or result["confirmed"] is not False
            or result["scientific_fingerprint"] != plan["scientific_fingerprint"]
            or aggregate["result_sha256"] != file_hash(root/"result.json")
            or {k:v for k,v in aggregate.items() if k not in ("result_path", "result_sha256")} != result):
        raise ValueError("Result/aggregate mismatch")
    for name in ("trace", "fit", "evaluation_manifest"):
        if file_hash(root/f"{name}.json") != result[f"{name}_sha256"]:
            raise ValueError("Result artifact changed")
    arm = result["arms"][result["selected_origin"]["method"]]
    return plan, result, arm["all_case_target_correct"]/arm["all_case_total"]


def readback(root, output, chromadb, result_hash=None):
    plan, result, primary = checked_result(root)
    if result_hash is None:
        result_hash = json.loads(output.read_text())["storage"]["result_hash"]
    logger = _make_logger(chromadb)
    back = logger.experiments_collection.get(ids=[result_hash], include=["metadatas"])
    h = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    if back["ids"] != [result_hash] or h["ids"] != [STUDY_ID]:
        raise RuntimeError("Exact registry IDs absent")
    m, hm = back["metadatas"][0], h["metadatas"][0]
    expected = {"hypothesis_id": STUDY_ID, "accuracy": result["teacher"]["accuracy"], "primary_metric": primary,
                "accuracy_scope": ACCURACY_SCOPE, "primary_scope": PRIMARY_SCOPE, "confirmed": False,
                "confirmation_status": result["classification"], "aggregate_sha256": file_hash(root/"campaign_results.json")}
    if any(m.get(k) != v for k,v in expected.items()):
        raise RuntimeError("Experiment readback scope/provenance mismatch")
    experiment_id = "jspace-attention-transport-"+plan["scientific_fingerprint"][:16]
    if (hm.get("confirmed") is not False or hm.get("confirmation_status") != result["classification"]
            or hm.get("evidence_count") != 1 or hm.get("direct_experiment_count") != 1
            or json.loads(hm.get("direct_experiment_ids_json", "[]")) != [experiment_id]
            or json.loads(hm.get("direct_result_hashes_json", "[]")) != [result_hash]):
        raise RuntimeError("Hypothesis readback mismatch")
    return {"readback_verified": True, "result_hash": result_hash, "experiment_id": experiment_id, **expected}


def store(root, report, output, chromadb):
    if output.exists():
        raise FileExistsError(output)
    plan, result, primary = checked_result(root)
    training = checked(root, "training")
    fit = json.loads((root/"fit.json").read_text())
    hypothesis = {"id": STUDY_ID, "name": "Compact attention concepts and contextual observable transport",
                  "category": "representation", "tested": True, "confirmed": False,
                  "evidence_count": 1, "direct_experiment_count": 1,
                  "confirmation_status": result["classification"], "created_at": datetime.now(timezone.utc).isoformat(),
                  "tested_scope": "single_checkpoint_attention_edit_and_composed_linear_quadratic_observables",
                  "description": "Eight country J directions at actual attention writes; fresh paired task worlds; consecutive attention/MLP traces. No weight training, compression or exact Groebner claim."}
    experiment = ExperimentResult(
        experiment_id="jspace-attention-transport-"+plan["scientific_fingerprint"][:16], hypothesis_id=STUDY_ID,
        metrics={"accuracy": result["teacher"]["accuracy"], "J_target_rate_all_worlds": primary}, primary_metric=primary,
        model_architecture=[2048]*36, model_parameters=training["model_parameters"], model_checkpoint=plan["model"]["path"],
        training_time=fit["elapsed_seconds"], observations=[ACCURACY_SCOPE, PRIMARY_SCOPE, str(report), result["classification"],
        "Accuracy and parameter fields describe the unchanged teacher. Training time is CPU probe/map fitting. No compressed model or general LM quality measured."])
    logger = _make_logger(chromadb)
    logger.log_hypothesis(hypothesis)
    result_hash = logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[result_hash], metadatas=[{
        "accuracy_scope": ACCURACY_SCOPE, "primary_scope": PRIMARY_SCOPE, "confirmed": False,
        "confirmation_status": result["classification"], "aggregate_sha256": file_hash(root/"campaign_results.json"),
        "compression_tested": False, "parameter_scope": "frozen_teacher_checkpoint"}])
    logger.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[{
        "direct_experiment_ids_json": json.dumps([experiment.experiment_id]), "direct_result_hashes_json": json.dumps([result_hash]),
        "report_path": str(report), "aggregate_sha256": file_hash(root/"campaign_results.json")}])
    storage = readback(root, output, chromadb, result_hash)
    paths = [root/"plan.json", root/"result.json", root/"campaign_results.json", report, Path(__file__)]
    audit = json.loads((root/"identity_audit.json").read_text())
    if (audit["plan_sha256"] != file_hash(root/"identity_audit_plan.json")
            or audit["evaluation_manifest_sha256"] != file_hash(root/"evaluation_manifest.json") or audit["confirmed"] is not False):
        raise ValueError("Secondary country-identity audit provenance changed")
    paths.extend([root/"identity_audit.json", root/"identity_audit_plan.json"])
    checked(root, "full_trace")
    full_trace = json.loads((root/"full_trace/result.json").read_text())
    if (full_trace["plan_sha256"] != file_hash(root/"full_trace_plan.json")
            or full_trace["confirmed"] is not False or full_trace["generation_replay_exact"] is not True):
        raise ValueError("Full-state follow-up provenance changed")
    paths.extend([root/"full_trace_plan.json", root/"full_trace/result.json", root/"full_trace_manifest.json"])
    write_json(output, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "result": result,
                        "secondary_country_identity_audit": audit, "posthoc_reused_full_trace": full_trace, "storage": storage,
                        "source_sha256": {str(p):file_hash(p) for p in paths}})
    return storage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("store", "readback"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-attention-transport.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_attention_transport_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    print(json.dumps(store(args.root, args.report, args.output, args.chromadb) if args.stage == "store"
                     else readback(args.root, args.output, args.chromadb), allow_nan=False))
