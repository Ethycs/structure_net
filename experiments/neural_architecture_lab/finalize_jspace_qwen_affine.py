"""Append-only analysis, registration and independent readback; no GPU work."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from neural_architecture_lab.jspace_qwen_affine_record import JSpaceQwenAffineRecord

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/experiments/jspace_qwen_affine/20261113_v2"
REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-13_jspace-qwen-affine-v2.md"
REGISTRY = ROOT / "data/hypothesis_registry/jspace_qwen_affine_20261113_v2.json"
LOG = ROOT / "docs/07 - Status Reports/jspace/research-log.jsonl"
TYPED = ROOT / "src/neural_architecture_lab/jspace_qwen_affine_record.py"
ARMS = ("r1536_h128", "r1536_h64", "r1024_h64", "r1536_h64_gb")
CLASSIFICATION = "partial_quality_recovery_no_speedup"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        stream.write(text)


def paired(native, student):
    if [x["identity"] for x in native] != [x["identity"] for x in student]:
        raise ValueError("Unpaired answers")
    gains = losses = changes = 0
    for a, b in zip(native, student):
        if a["answer"] != b["answer"]:
            raise ValueError("Labels changed")
        ac, bc = a["prediction"] == a["answer"], b["prediction"] == b["answer"]
        gains += not ac and bc
        losses += ac and not bc
        changes += a["prediction"] != b["prediction"]
    n = gains + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n) if n else 1.0
    return {"gains": gains, "losses": losses, "changed_answers": changes,
            "net_correct_change": gains - losses, "exact_mcnemar_two_sided_p": p,
            "scope": "descriptive paired test; no equivalence or noninferiority claim"}


def block_ms(runtime):
    rows = runtime["isolated_block_prefill"]
    return 1000 * sum(x["seconds_per_call"] for x in rows) / len(rows)


def analyze():
    plan, result = read(RUN / "plan.json"), read(RUN / "result.json")
    stages = ("calibration", "relations", "controls", *("train_" + a for a in ARMS), "evaluation", "reload")
    manifests, expected = {}, {}
    for stage in stages:
        manifest = read(RUN / f"{stage}_manifest.json")
        if manifest["status"] != "completed" or manifest["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Incomplete or mismatched stage: " + stage)
        manifests[stage] = manifest
        for path, digest in manifest["artifact_sha256"].items():
            if path in expected and expected[path] != digest:
                raise ValueError("Conflicting artifact hashes")
            expected[path] = digest
    for path, digest in expected.items():
        if sha(RUN / path) != digest:
            raise ValueError("Artifact changed: " + path)
    for path, digest in plan["source_sha256"].items():
        for candidate in (Path(path), RUN / "producer" / Path(path).relative_to(ROOT)):
            if sha(candidate) != digest:
                raise ValueError("Producer changed: " + str(candidate))
    if result["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Result fingerprint changed")
    native = read(RUN / "evaluation/native_fresh.json")
    answers = {arm: read(RUN / f"evaluation/{arm}_fresh.json") for arm in ARMS}
    training = {arm: read(RUN / f"training_{arm}.json") for arm in ARMS}
    reload = read(RUN / "reload/verification.json")
    results = result["results"]
    matched = paired(answers["r1536_h64"], answers["r1536_h64_gb"])
    matched.update(tensor_bytes_saved=results["r1536_h64"]["tensor_bytes"] - results["r1536_h64_gb"]["tensor_bytes"],
                   block_prefill_time_ratio=block_ms(results["r1536_h64_gb"]["runtime"]) / block_ms(results["r1536_h64"]["runtime"]))
    aggregate = {
        "schema_version": "jspace.qwen-affine.analysis.v1", "study_id": result["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed", "confirmed": False,
        "classification": CLASSIFICATION,
        "classification_scope": "descriptive exploratory classification, not a retrospective confirmatory threshold",
        "result_sha256": sha(RUN / "result.json"), "result": result, "config": plan["config"],
        "model": plan["model"], "training": training, "reload": reload,
        "controls": read(RUN / "controls.json"), "fit": read(RUN / "fit.json"),
        "relations": read(RUN / "relations.json"),
        "paired_to_native": {arm: paired(native, rows) for arm, rows in answers.items()},
        "matched_groebner_comparison": matched,
        "manifests": manifests,
        "integrity": {"completed_stages": len(stages), "trained_arms": 4, "training_seed": 861,
                      "independent_confirmations": 0, "artifact_hashes_verified": len(expected),
                      "producer_files_verified_live_and_archived": len(plan["source_sha256"]),
                      "failed_v1_shakedown_preserved": "CUDA int64 addmm unsupported; Boolean overlap reduction fixed before v2",
                      "v1_main": "prepared but unexecuted; superseded, not omitted quality evidence"},
        "artifact_sha256": expected,
        "analysis_source_sha256": {str(Path(__file__)): sha(__file__), str(TYPED): sha(TYPED)},
        "scope": "one trained seed; last block only; 512 fresh and 256 reused multiple-choice tasks; no free-generation accuracy claim",
    }
    JSpaceQwenAffineRecord.from_aggregate(aggregate)
    write(RUN / "campaign_results.json", aggregate)
    return {"aggregate": str(RUN / "campaign_results.json"), "sha256": sha(RUN / "campaign_results.json")}


def expected():
    aggregate = read(RUN / "campaign_results.json")
    if aggregate["result_sha256"] != sha(RUN / "result.json") or aggregate["result"] != read(RUN / "result.json"):
        raise ValueError("Aggregate/result mismatch")
    for path, digest in aggregate["analysis_source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Analysis source changed")
    return aggregate, JSpaceQwenAffineRecord.from_aggregate(aggregate)


def logger():
    import numpy as np
    if "NaN" not in vars(np):
        np.NaN = np.nan  # Process-local compatibility for locked Chroma/NumPy versions.
    from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
    return _make_logger(ROOT / "data/chroma_db")


def wanted(record):
    return {"hypothesis_id": record.study_id, "confirmed": False, "confirmation_status": CLASSIFICATION,
            "accuracy": record.accuracy, "primary_metric": record.accuracy, "correct": record.correct,
            "count": record.count, "accuracy_scope": record.scope,
            "aggregate_sha256": sha(RUN / "campaign_results.json"), "primary_arm": record.primary_arm}


def readback(key=None):
    aggregate, record = expected()
    if key is None:
        stored = read(REGISTRY)
        key = stored["storage"]["result_hash"]
        for path, digest in stored["source_sha256"].items():
            if sha(path) != digest:
                raise ValueError("Registry provenance changed")
    db = logger()
    experiments = db.experiments_collection.get(ids=[key], include=["metadatas"])
    hypotheses = db.hypotheses_collection.get(ids=[record.study_id], include=["metadatas"])
    values = wanted(record)
    if experiments["ids"] != [key] or any(experiments["metadatas"][0].get(k) != v for k, v in values.items()):
        raise ValueError("Experiment readback mismatch")
    if hypotheses["ids"] != [record.study_id]:
        raise ValueError("Hypothesis absent")
    h = hypotheses["metadatas"][0]
    if h.get("tested") is not True or h.get("confirmed") is not False or h.get("confirmation_status") != CLASSIFICATION:
        raise ValueError("Hypothesis status mismatch")
    if h.get("evidence_count") != 1 or json.loads(h["direct_result_hashes_json"]) != [key]:
        raise ValueError("Wrong evidence count/link")
    return {"readback_verified": True, "result_hash": key, "reader_source_sha256": sha(__file__), **values}


def store():
    if REGISTRY.exists():
        raise FileExistsError(REGISTRY)
    aggregate, record = expected()
    from neural_architecture_lab.core import ExperimentResult
    import torch
    state = torch.load(RUN / "r1536_h64_gb.pt", map_location="cpu", weights_only=True)
    parameter_names = ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "polynomial_bias")
    compact_parameters = sum(state["block"][name].numel() for name in parameter_names)
    # Pinned Qwen: tied vocabulary counted once; native last block includes both RMS gains.
    native_parameters, native_block_parameters = 3085938688, 77076992
    model_parameters = native_parameters - native_block_parameters + compact_parameters
    hypothesis = {"id": record.study_id, "name": "Qwen final-block functional coordinates and benchmark polynomial compilation",
                  "category": "compression", "tested": True, "confirmed": False, "evidence_count": 1,
                  "direct_experiment_count": 1, "confirmation_status": CLASSIFICATION,
                  "created_at": datetime.now(timezone.utc).isoformat(), "tested_scope": record.scope,
                  "description": "Frozen native prefix; four matched KL+CE arms; fresh held-out choices, full-vocabulary sensitivity, matched runtime and exact fresh-process reload. GB is a fitted surrogate, not exact neural relations."}
    experiment = ExperimentResult(
        experiment_id="jspace-qwen-affine-" + record.scientific_fingerprint[:16], hypothesis_id=record.study_id,
        metrics={"accuracy": record.accuracy, "original_accuracy": record.original_correct / 512,
                 "tensor_byte_reduction": 1 - record.tensor_bytes / record.original_tensor_bytes,
                 "graph_tps": record.graph_tokens_per_second,
                 "graph_speed_ratio_to_native_repeat": record.graph_tokens_per_second / record.native_repeat_graph_tokens_per_second},
        primary_metric=record.accuracy, model_architecture=[2048] * 35 + [1536], model_parameters=model_parameters,
        training_time=sum(aggregate["manifests"]["train_" + a]["elapsed_seconds"] for a in ARMS),
        model_checkpoint=str(RUN / "r1536_h64_gb.pt"),
        observations=[record.scope, CLASSIFICATION, str(REPORT),
                      "Primary registry arm is the declared GB arm; all four arms remain in the aggregate.",
                      "Checkpoint is a block-plus-final-norm delta requiring the pinned native model.",
                      "One seed; descriptive McNemar p=0.125 does not establish accuracy equivalence."])
    db = logger()
    prior = db.hypotheses_collection.get(ids=[record.study_id], include=["metadatas"])
    if prior["ids"] and prior["metadatas"][0].get("tested") is True:
        raise ValueError("Refusing to replace tested hypothesis")
    db.log_hypothesis(hypothesis)
    key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key], metadatas=[wanted(record)])
    db.hypotheses_collection.update(ids=[record.study_id], metadatas=[{
        "direct_result_hashes_json": json.dumps([key]), "direct_experiment_ids_json": json.dumps([experiment.experiment_id]),
        "report_path": str(REPORT), "aggregate_sha256": sha(RUN / "campaign_results.json")}])
    receipt = readback(key)
    paths = (Path(__file__), TYPED, REPORT, RUN / "plan.json", RUN / "result.json",
             RUN / "campaign_results.json", RUN / "reload/verification.json", RUN / "reservation_closeout.json")
    write(REGISTRY, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis,
                    "record": record.to_dict(), "model_parameters": model_parameters, "storage": receipt,
                    "source_sha256": {str(p): sha(p) for p in paths}})
    return receipt


def closeout():
    aggregate, record = expected()
    receipt = read(RUN / "registry_readback.json")
    if receipt.get("readback_verified") is not True or receipt["result_hash"] != read(REGISTRY)["storage"]["result_hash"]:
        raise ValueError("Missing independent readback")
    paths = (Path(__file__), TYPED, REPORT, REGISTRY, RUN / "campaign_results.json",
             RUN / "registry_readback.json", RUN / "reservation_closeout.json")
    event = {"schema_version": "jspace.research-log.v1", "id": "2026-11-13-jspace-qwen-affine-v2-completed",
             "event": "qwen_final_block_affine_campaign_completed", "study_id": record.study_id,
             "recorded_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), "timezone": "America/Los_Angeles",
             "confirmed": False, "classification": CLASSIFICATION, "independent_confirmations": 0,
             "scientific_fingerprint": record.scientific_fingerprint, "new_experimental_evidence": True,
             "accuracy": {"correct": record.correct, "count": 512, "overall": record.accuracy},
             "original_accuracy": {"correct": record.original_correct, "count": 512},
             "paired_comparison": aggregate["paired_to_native"][record.primary_arm],
             "matched_groebner_comparison": aggregate["matched_groebner_comparison"],
             "graph_tokens_per_second": record.graph_tokens_per_second,
             "native_graph_tokens_per_second": [record.native_graph_tokens_per_second, record.native_repeat_graph_tokens_per_second],
             "registry_readback_verified": True, "registry_result_id": receipt["result_hash"],
             "report_path": str(REPORT.relative_to(ROOT)), "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
             "research_direction": "Keep the near-native final-block quality result and matched polynomial storage saving; investigate staged joint recovery and runtime cost separately. No efficiency win or exact neural identity claim."}
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()):
            raise ValueError("Research event already logged")
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    archive = RUN / "postprocessing_source"
    archive.mkdir(exist_ok=False)
    for path in (*paths[:3], LOG):
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    final = {"study_id": record.study_id, "status": "completed", "confirmed": False,
             "classification": CLASSIFICATION, "registry_readback": receipt,
             "aggregate_sha256": sha(RUN / "campaign_results.json"), "research_event": event,
             "source_sha256": {str(p): sha(p) for p in paths},
             "research_log_snapshot_sha256": sha(archive / LOG.relative_to(ROOT))}
    write(RUN / "finalization.json", final)
    return {"status": "finalized", "report_sha256": sha(REPORT), "registry_sha256": sha(REGISTRY),
            "aggregate_sha256": sha(RUN / "campaign_results.json"), "finalization_sha256": sha(RUN / "finalization.json")}


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' before Python starts")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "store", "readback", "closeout"))
    args = parser.parse_args()
    value = {"analyze": analyze, "store": store, "readback": readback, "closeout": closeout}[args.stage]()
    if args.stage == "readback":
        write(RUN / "registry_readback.json", value)
    print(json.dumps(value, allow_nan=False))
