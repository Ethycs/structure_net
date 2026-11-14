"""Append-only layerKD closeout. Run AFTER GPU performance timing is finished.

``analyze`` hashes multiple GB and checks CPU checkpoint tensors. ``store`` and
``readback`` access the existing local registry. None of these stages trains,
touches GPUs, changes reservations or edits the frozen producer.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from neural_architecture_lab.jspace_qwen_layerkd_record import ARMS, CLASSIFICATION, JSpaceQwenLayerKDRecord
from . import finalize_jspace_qwen_affine as common

ROOT = common.ROOT
RUN = ROOT / "data/experiments/jspace_qwen_layerkd/20261113_v1"
REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-13_jspace-qwen-layerkd.md"
REGISTRY = ROOT / "data/hypothesis_registry/jspace_qwen_layerkd_20261113.json"
TYPED = ROOT / "src/neural_architecture_lab/jspace_qwen_layerkd_record.py"
LOG = ROOT / "docs/07 - Status Reports/jspace/research-log.jsonl"
POSTPROCESSING = (Path(__file__), TYPED, Path(common.__file__), common.TYPED)
sha, read, write = common.sha, common.read, common.write


def equal_state(first, second):
    import torch
    if first.keys() != second.keys():
        raise ValueError("Checkpoint keys differ")
    tensors = 0
    for name, value in first.items():
        other = second[name]
        if isinstance(value, dict):
            tensors += equal_state(value, other)
        elif isinstance(value, torch.Tensor):
            if value.dtype != other.dtype or not torch.equal(value, other):
                raise ValueError("Checkpoint tensor differs: " + name)
            tensors += 1
        elif value != other:
            raise ValueError("Checkpoint metadata differs: " + name)
    return tensors


def analyze():
    import torch
    plan, result = read(RUN / "plan.json"), read(RUN / "result.json")
    stages = ("setup", *("train_" + arm for arm in ARMS), "evaluation", "reload")
    artifacts, manifests = {}, {}
    for stage in stages:
        m = read(RUN / f"{stage}_manifest.json")
        if m["status"] != "completed" or m["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Incomplete or mismatched stage: " + stage)
        manifests[stage] = m
        for path, digest in m["artifact_sha256"].items():
            if path in artifacts and artifacts[path] != digest:
                raise ValueError("Conflicting artifact digest")
            artifacts[path] = digest
    for path, digest in artifacts.items():
        if sha(RUN / path) != digest:
            raise ValueError("Artifact changed: " + path)
    for path, digest in plan["source_sha256"].items():
        for candidate in (Path(path), RUN / "producer" / Path(path).relative_to(ROOT)):
            if sha(candidate) != digest:
                raise ValueError("Producer changed: " + str(candidate))
    if result["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Result fingerprint mismatch")
    training = {arm: read(RUN / f"training_{arm}.json") for arm in ARMS}
    initial = torch.load(RUN / "initialized.pt", map_location="cpu", weights_only=True, mmap=True)
    tensor_checks = {}
    for arm in ARMS:
        if training[arm]["initialized_checkpoint_sha256"] != sha(RUN / "initialized.pt"):
            raise ValueError("Wrong initializer provenance")
        state = torch.load(RUN / f"train_{arm}/epoch_00.pt", map_location="cpu", weights_only=True, mmap=True)
        tensor_checks[arm] = {"initial_tensors_exact": equal_state(initial, state)}
        del state
    del initial
    parent = Path(plan["parent_root"])
    selected = torch.load(RUN / "endpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    previous = torch.load(parent / "r1536_h64_gb.pt", map_location="cpu", weights_only=True, mmap=True)
    tensor_checks["endpoint_parent_reproduction"] = equal_state(previous, selected)
    parameter_names = ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "polynomial_bias")
    compact_parameters = sum(selected["block"][name].numel() for name in parameter_names)
    model_parameters = 3085938688 - 77076992 + compact_parameters
    del selected, previous
    records = {arm: read(RUN / f"evaluation/{arm}_fresh.json") for arm in ("native", *ARMS)}
    aggregate = {
        "schema_version": "jspace.qwen-layerkd.analysis.v1", "study_id": result["study_id"],
        "status": "completed", "confirmed": False, "classification": CLASSIFICATION,
        "classification_scope": "descriptive exploratory result; no demonstrated auxiliary-loss or performance improvement",
        "created_at": datetime.now(timezone.utc).isoformat(), "result": result,
        "result_sha256": sha(RUN / "result.json"), "config": plan["config"], "model": plan["model"],
        "training": training, "calibration": read(RUN / "calibration.json"),
        "reload": read(RUN / "reload/verification.json"), "model_parameters": model_parameters,
        "tensor_identity_checks": tensor_checks,
        "paired_to_native": {arm: common.paired(records["native"], records[arm]) for arm in ARMS},
        "paired_to_endpoint": {arm: common.paired(records["endpoint"], records[arm]) for arm in ARMS[1:]},
        "manifests": manifests, "artifact_sha256": artifacts,
        "analysis_source_sha256": {str(p): sha(p) for p in POSTPROCESSING},
        "scope": "one seed and one final block; three objectives, not three independent replications; staged joint recovery unexecuted by this study",
        "integrity": {"completed_stages": len(stages), "artifact_hashes_verified": len(artifacts),
                      "producer_files_verified_live_and_archived": len(plan["source_sha256"]),
                      "trained_arms": 3, "independent_confirmations": 0},
    }
    JSpaceQwenLayerKDRecord.from_aggregate(aggregate)
    write(RUN / "campaign_results.json", aggregate)
    return {"aggregate": str(RUN / "campaign_results.json"), "sha256": sha(RUN / "campaign_results.json")}


def expected():
    aggregate = read(RUN / "campaign_results.json")
    if aggregate["result_sha256"] != sha(RUN / "result.json") or aggregate["result"] != read(RUN / "result.json"):
        raise ValueError("Aggregate/result mismatch")
    for path, digest in aggregate["analysis_source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Analysis source changed")
    return aggregate, JSpaceQwenLayerKDRecord.from_aggregate(aggregate)


def wanted(record, arm):
    return {"hypothesis_id": record.study_id, "arm": arm, "confirmed": False,
            "confirmation_status": CLASSIFICATION, "accuracy": record.correct_by_arm[arm] / record.count,
            "primary_metric": record.correct_by_arm[arm] / record.count, "correct": record.correct_by_arm[arm],
            "count": record.count, "accuracy_scope": record.scope,
            "selected_epoch": record.selected_epoch_by_arm[arm], "optimizer_updates": record.updates_per_arm,
            "aggregate_sha256": sha(RUN / "campaign_results.json")}


def readback(keys=None):
    aggregate, record = expected()
    if keys is None:
        stored = read(REGISTRY)
        keys = stored["storage"]["result_hashes"]
        for path, digest in stored["source_sha256"].items():
            if sha(path) != digest:
                raise ValueError("Registry provenance changed")
    if set(keys) != set(ARMS) or len(set(keys.values())) != 3:
        raise ValueError("Wrong registry arm population")
    db = common.logger()
    for arm in ARMS:
        rows = db.experiments_collection.get(ids=[keys[arm]], include=["metadatas"])
        if rows["ids"] != [keys[arm]] or any(rows["metadatas"][0].get(k) != v for k, v in wanted(record, arm).items()):
            raise ValueError("Experiment readback mismatch: " + arm)
    rows = db.hypotheses_collection.get(ids=[record.study_id], include=["metadatas"])
    if rows["ids"] != [record.study_id]:
        raise ValueError("Missing hypothesis")
    h = rows["metadatas"][0]
    if h.get("tested") is not True or h.get("confirmed") is not False or h.get("confirmation_status") != CLASSIFICATION:
        raise ValueError("Hypothesis status mismatch")
    if h.get("evidence_count") != 1 or h.get("direct_experiment_count") != 3:
        raise ValueError("Arms confused with independent confirmations")
    if json.loads(h["direct_result_hashes_json"]) != [keys[a] for a in ARMS]:
        raise ValueError("Incorrect direct evidence links")
    return {"readback_verified": True, "hypothesis_id": record.study_id, "result_hashes": keys,
            "confirmed": False, "classification": CLASSIFICATION, "reader_source_sha256": sha(__file__),
            "aggregate_sha256": sha(RUN / "campaign_results.json"), "correct_by_arm": record.correct_by_arm,
            "count": record.count, "independent_confirmations": 0}


def store():
    if REGISTRY.exists():
        raise FileExistsError(REGISTRY)
    aggregate, record = expected()
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = {"id": record.study_id, "name": "Auxiliary distillation of one compressed Qwen final block",
                  "category": "compression", "tested": True, "confirmed": False, "evidence_count": 1,
                  "direct_experiment_count": 3, "confirmation_status": CLASSIFICATION,
                  "created_at": datetime.now(timezone.utc).isoformat(), "tested_scope": record.scope,
                  "description": "One matched campaign: output-only, projected Euclidean and analytic full-logit-Jacobian objectives. Auxiliary differences are inconclusive; no measured performance improvement; staged joint recovery is outside this study."}
    db = common.logger()
    prior = db.hypotheses_collection.get(ids=[record.study_id], include=["metadatas"])
    if prior["ids"] and prior["metadatas"][0].get("tested") is True:
        raise ValueError("Refusing to replace tested hypothesis")
    db.log_hypothesis(hypothesis)
    keys, experiment_ids = {}, []
    for arm in ARMS:
        experiment_id = "jspace-qwen-layerkd-" + record.scientific_fingerprint[:16] + "-" + arm
        experiment_ids.append(experiment_id)
        metric = aggregate["result"]["results"][arm]
        experiment = ExperimentResult(
            experiment_id=experiment_id, hypothesis_id=record.study_id,
            metrics={"accuracy": record.correct_by_arm[arm] / record.count,
                     "native_accuracy": record.native_correct / record.count,
                     "development_kl": metric["development"]["kl"],
                     "graph_tokens_per_second": record.graph_tokens_per_second_by_arm[arm]},
            primary_metric=record.correct_by_arm[arm] / record.count,
            model_architecture=[2048] * 35 + [1536], model_parameters=aggregate["model_parameters"],
            training_time=aggregate["manifests"]["train_" + arm]["elapsed_seconds"],
            model_checkpoint=str(RUN / (arm + ".pt")),
            observations=[record.scope, CLASSIFICATION, str(REPORT), "Arm: " + arm,
                          "Same initializer, 12 epochs and 1,536 updates; same deployed tensor size.",
                          "A block-plus-final-norm delta; native pinned prefix/readout required.",
                          "Three arms constitute one exploratory campaign, not independent confirmations."])
        keys[arm] = db.log_experiment_result(experiment)
        db.experiments_collection.update(ids=[keys[arm]], metadatas=[wanted(record, arm)])
    db.hypotheses_collection.update(ids=[record.study_id], metadatas=[{
        "direct_result_hashes_json": json.dumps([keys[a] for a in ARMS]),
        "direct_experiment_ids_json": json.dumps(experiment_ids), "report_path": str(REPORT),
        "aggregate_sha256": sha(RUN / "campaign_results.json")}])
    receipt = readback(keys)
    paths = (*POSTPROCESSING, REPORT, RUN / "plan.json", RUN / "result.json",
             RUN / "campaign_results.json", RUN / "reload/verification.json")
    write(REGISTRY, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis,
                    "record": record.to_dict(), "storage": receipt,
                    "source_sha256": {str(p): sha(p) for p in paths}})
    return receipt


def closeout():
    aggregate, record = expected()
    receipt = read(RUN / "registry_readback.json")
    if receipt.get("readback_verified") is not True or receipt["result_hashes"] != read(REGISTRY)["storage"]["result_hashes"]:
        raise ValueError("Independent registry readback missing")
    paths = (*POSTPROCESSING, REPORT, REGISTRY, RUN / "campaign_results.json", RUN / "registry_readback.json")
    event = {"schema_version": "jspace.research-log.v1", "id": "2026-11-13-jspace-qwen-layerkd-v1-completed",
             "event": "single_final_block_auxiliary_distillation_completed", "study_id": record.study_id,
             "recorded_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), "timezone": "America/Los_Angeles",
             "confirmed": False, "classification": CLASSIFICATION, "independent_confirmations": 0,
             "scientific_fingerprint": record.scientific_fingerprint, "new_experimental_evidence": True,
             "native_correct": record.native_correct, "correct_by_arm": record.correct_by_arm, "count": record.count,
             "paired_to_endpoint": aggregate["paired_to_endpoint"], "tensor_bytes": record.tensor_bytes,
             "registry_readback_verified": True, "registry_result_ids": receipt["result_hashes"],
             "report_path": str(REPORT.relative_to(ROOT)), "scope": record.scope,
             "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}}
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
    for path in (*POSTPROCESSING, REPORT, LOG):
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    final = {"study_id": record.study_id, "status": "completed", "confirmed": False,
             "classification": CLASSIFICATION, "registry_readback": receipt, "research_event": event,
             "source_sha256": {str(p): sha(p) for p in paths},
             "research_log_snapshot_sha256": sha(archive / LOG.relative_to(ROOT))}
    write(RUN / "finalization.json", final)
    return {"status": "finalized", "report_sha256": sha(REPORT), "registry_sha256": sha(REGISTRY),
            "aggregate_sha256": sha(RUN / "campaign_results.json"), "finalization_sha256": sha(RUN / "finalization.json")}


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before Python starts")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "store", "readback", "closeout"))
    args = parser.parse_args()
    value = {"analyze": analyze, "store": store, "readback": readback, "closeout": closeout}[args.stage]()
    if args.stage == "readback":
        write(RUN / "registry_readback.json", value)
    print(json.dumps(value, allow_nan=False))
