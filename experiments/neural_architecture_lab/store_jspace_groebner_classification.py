"""Index scoped algebra evidence without inventing a neural accuracy metric."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_groebner_classification import DEFAULT_OUTPUT, STUDY_ID, verify
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


def checked_result(root):
    plan = verify(root)
    result = json.loads((root/"result.json").read_text())
    aggregate = json.loads((root/"campaign_results.json").read_text())
    if (result["study_id"] != STUDY_ID or result["status"] != "completed"
            or result["scientific_fingerprint"] != plan["scientific_fingerprint"]
            or result["confirmed"] is not False or result["neural_task_accuracy"] is not None
            or result["neural_speedup"] is not None or result["neural_compression_tested"] is not False
            or aggregate["result_sha256"] != file_hash(root/"result.json")
            or {k: v for k, v in aggregate.items() if k not in ("result_path", "result_sha256")} != result):
        raise ValueError("Result/aggregate integrity or scope mismatch")
    for name, expected in result["artifacts"].items():
        if file_hash(root/name) != expected:
            raise ValueError(f"Analysis artifact changed: {name}")
    return result


def readback(root, chromadb):
    result = checked_result(root)
    logger = _make_logger(chromadb)
    back = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas", "documents"])
    if back["ids"] != [STUDY_ID]:
        raise RuntimeError("Exact hypothesis ID not found")
    metadata = back["metadatas"][0]
    if (metadata.get("confirmed") is not False
            or metadata.get("confirmation_status") != result["classification"]
            or metadata.get("direct_experiment_count") != 0
            or metadata.get("tested_scope") != "constructed_algebra_and_reused_neural_relations_no_new_model_run"
            or metadata.get("aggregate_sha256") != file_hash(root/"campaign_results.json")
            or "accuracy" in metadata):
        raise RuntimeError("Registry scope or provenance readback mismatch")
    return {"readback_verified": True, "hypothesis_id": STUDY_ID,
            "classification": result["classification"], "confirmed": False,
            "registration_scope": "hypothesis_and_artifact_aggregate_no_accuracy_bearing_experiment_record",
            "new_model_runs": 0, "aggregate_sha256": metadata["aggregate_sha256"]}


def store(root, report, output, chromadb):
    if output.exists():
        raise FileExistsError(output)
    result = checked_result(root)
    hypothesis = {
        "id": STUDY_ID, "name": "Classify Gröbner methods for interpretation and compression",
        "category": "representation", "tested": True, "confirmed": False, "evidence_count": 1,
        "direct_experiment_count": 0, "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmation_status": result["classification"],
        "tested_scope": "constructed_algebra_and_reused_neural_relations_no_new_model_run",
        "description": "Eleven constructed algebra controls passed, including a finite lossless payload codec. Frozen Qwen graph audit found no validated destination relation; six symbolic cells unresolved. No neural accuracy, compression or speedup measured.",
        "question": "Which algebraic conditions enable interpretation, coordinate decoding or reduced dynamics, and do frozen neural graph surrogates create destination constraints?",
    }
    logger = _make_logger(chromadb)
    if logger.hypotheses_collection is None:
        raise RuntimeError("Hypothesis registry unavailable")
    logger.log_hypothesis(hypothesis)
    logger.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[{
        "aggregate_sha256": file_hash(root/"campaign_results.json"),
        "neural_accuracy_measured": False, "neural_compression_tested": False,
        "new_model_runs": 0, "report_path": str(report)}])
    storage = readback(root, chromadb)
    paths = [root/"plan.json", root/"result.json", root/"campaign_results.json", report, Path(__file__)]
    write_json(output, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis,
                        "result": result, "storage": storage,
                        "source_sha256": {str(p): file_hash(p) for p in paths}})
    return storage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("store", "readback"))
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-groebner-classification.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_groebner_classification_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    value = (store(args.root, args.report, args.output, args.chromadb) if args.stage == "store"
             else readback(args.root, args.chromadb))
    print(json.dumps(value, allow_nan=False))
