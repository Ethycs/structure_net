"""Append-only CPU registration after BOTH Qwen recipe arms have completed.

Run analyze, store, readback, closeout in separate locked default-Pixi processes.
Analysis hashes retained artifacts only after all GPU/CPU timing has finished.
The producer's campaign results and Markdown report are never rewritten.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_quadratic_hypothesis import (
    ARMS, JSpaceQuadraticRecord, SCOPE, STUDY_ID, required_stages, require,
)
from . import finalize_jspace_qwen_affine as common

ROOT, LOG = common.ROOT, common.LOG
DEFAULT_ROOT = ROOT / "data/experiments/jspace_quadratic_recipe/20261114_v1"
DEFAULT_REGISTRY = ROOT / "data/hypothesis_registry/jspace_quadratic_20261114.json"
TYPED = ROOT / "src/neural_architecture_lab/jspace_quadratic_hypothesis.py"
TEST = ROOT / "tests/neural_architecture_lab/test_jspace_quadratic_hypothesis.py"
POSTPROCESSING = (Path(__file__), TYPED, TEST, Path(common.__file__), common.TYPED)
read, write, sha = common.read, common.write, common.sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def analyze(run):
    run = Path(run)
    plan, result, receipt = (read(run / name) for name in ("plan.json", "campaign_results.json", "report_receipt.json"))
    require(plan["fingerprint"] == digest({k: v for k, v in plan.items() if k != "fingerprint"}), "Changed scientific plan")
    require(receipt["campaign_sha256"] == sha(run / "campaign_results.json") and receipt["sha256"] == sha(receipt["report"]), "Producer report receipt changed")
    tasks, items = read(run / "tasks.json"), read(run / "items.json")
    require(digest(tasks) == plan["task_digest"] and digest(items) == plan["items_digest"], "Changed task/teacher-cache identities")
    splits = tasks["splits"]
    require({k: len(v) for k, v in splits.items()} == dict(train=512, development=64, evaluation=512), "Changed sample populations")
    identities = {k: {r["identity"] for r in rows} for k, rows in splits.items()}
    require(all(len(identities[k]) == len(rows) for k, rows in splits.items()) and
            not identities["train"] & identities["development"] and not identities["evaluation"] & (identities["train"] | identities["development"]), "Split leakage")
    manifests, checksums, verification = {}, {}, {}
    for name in required_stages():
        path = run / name / "manifest.json"; manifest = read(path)
        require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"], "Incomplete stage: " + name)
        manifests[name] = manifest; verification[str(path)] = sha(path)
        for relative, expected in manifest["artifacts"].items():
            target = run / relative
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Artifact outside campaign root")
            require(str(target) not in checksums or checksums[str(target)] == expected, "Conflicting artifact hashes")
            checksums[str(target)] = expected
    for path, expected in plan["source_sha256"].items():
        checksums[path] = expected
        checksums[str(run / "producer" / Path(path).relative_to(ROOT))] = expected
    checksums.update(plan["input_sha256"])
    checksums.update({str(Path(plan["model"]["path"]) / name): expected for name, expected in plan["model"]["files"].items()})
    for path, expected in checksums.items():
        require(sha(path) == expected, "Changed artifact/input/source: " + path)
    records, timing, exports, training = {}, {}, {}, {}
    for arm in ("native", *ARMS):
        path = run / "evaluation" / arm / "result.json"; value = read(path)
        require(value["fingerprint"] == plan["fingerprint"] and value["arm"] == arm and value["dtype"] == "float32", "Wrong evaluated model provenance")
        records[arm] = value["records"]
        require(len(records[arm]) == 512 and all(all(row[k] == case[k] for k in ("identity", "family", "answer"))
            for row, case in zip(records[arm], splits["evaluation"])), "Scored population differs from frozen tasks")
        verification[str(path)] = sha(path)
    for arm in ARMS:
        path = run / "cpu_benchmark" / arm / "result.json"; value = read(path)
        require(value["fingerprint"] == plan["fingerprint"] and value["arm"] == arm, "CPU benchmark provenance mismatch")
        timing[arm] = value; verification[str(path)] = sha(path)
        path = run / "export" / arm / "metadata.json"; value = read(path)
        require(value["fingerprint"] == plan["fingerprint"] and value["arm"] == arm and value["dtype"] == "float32", "Export provenance mismatch")
        exports[arm] = value; verification[str(path)] = sha(path)
        for layer in range(36):
            path = run / "arms" / arm / f"layer_{layer:02d}" / "training.json"; value = read(path)
            training[f"{arm}/{layer}"] = {k: v for k, v in value.items() if k != "history"}
            training[f"{arm}/{layer}"]["history"] = [{"epoch": row["epoch"], "development": row["development"]} for row in value["history"]]
            verification[str(path)] = sha(path)
            parity_path = path.with_name("fold_parity.json"); parity = read(parity_path)
            require(parity["passed"] is True and parity["finite"] is True and parity["relative_rmse"] < 1e-4, "Selected block folding failed")
            verification[str(parity_path)] = sha(parity_path)
    verification.update({str(run / name): sha(run / name) for name in ("plan.json", "tasks.json", "items.json", "campaign_results.json", "report_receipt.json")})
    verification[receipt["report"]] = receipt["sha256"]
    evidence = dict(schema="jspace.quadratic-closeout.v1", created_at=datetime.now(timezone.utc).isoformat(),
        plan=plan, result=result, manifests=manifests, records=records, timing=timing, exports=exports, training=training,
        report_receipt=receipt, artifact_sha256=checksums, verification_sha256=verification,
        analysis_source_sha256={str(path): sha(path) for path in POSTPROCESSING},
        integrity=dict(stage_manifests=len(manifests), verified_artifacts_inputs_sources=len(checksums),
            split_disjoint=True, confirmed=False, independent_confirmations=0,
            note="The shared prior-study holdout is not an independent confirmation; CPU parity is limited to benchmark prefixes"))
    record = JSpaceQuadraticRecord.from_evidence(evidence)
    evidence["record"] = record.to_dict()
    archive = run / "quadratic_closeout_source"; archive.mkdir(exist_ok=False)
    for path in POSTPROCESSING:
        target = archive / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(run / "quadratic_closeout_analysis.json", evidence)
    return {"analysis": str(run / "quadratic_closeout_analysis.json"), "sha256": sha(run / "quadratic_closeout_analysis.json"), "classification": record.classification}


def expected(run):
    value = read(run / "quadratic_closeout_analysis.json")
    for path, expected_sha in {**value["verification_sha256"], **value["analysis_source_sha256"]}.items():
        require(sha(path) == expected_sha, "Closeout provenance changed: " + path)
    record = JSpaceQuadraticRecord.from_evidence(value)
    require(record.to_dict() == value["record"], "Typed record changed")
    return value, record


def metadata(run, record):
    return dict(hypothesis_id=STUDY_ID, tested=True, confirmed=False, confirmation_status=record.classification,
        accuracy=record.quadratic_correct / 512, primary_metric=record.quadratic_correct / 512,
        correct=record.quadratic_correct, count=512, native_correct=record.native_correct, softmax_correct=record.softmax_correct,
        targets_met=record.targets_met, minimum_cpu_compiled_prefill_speedup=min(record.cpu_compiled_prefill_speedups["quadratic"].values()),
        tensor_reduction=record.tensor_reductions["quadratic"], independent_confirmations=0,
        scientific_fingerprint=record.scientific_fingerprint, analysis_sha256=sha(run / "quadratic_closeout_analysis.json"),
        scope=SCOPE)


def readback(run, registry, key=None):
    _, record = expected(run)
    if key is None:
        stored = read(registry); key = stored["storage"]["result_hash"]
        require(stored["analysis_sha256"] == sha(run / "quadratic_closeout_analysis.json") and
            stored["report_sha256"] == sha(stored["report_path"]), "Registry/report changed")
    db = common.logger()
    result = db.experiments_collection.get(ids=[key], include=["metadatas"])
    require(result["ids"] == [key] and all(result["metadatas"][0].get(k) == v for k, v in metadata(run, record).items()), "Experiment readback differs")
    result = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(result["ids"] == [STUDY_ID], "Hypothesis missing")
    hypothesis = result["metadatas"][0]
    require(hypothesis.get("tested") is True and hypothesis.get("confirmed") is False and
        hypothesis.get("confirmation_status") == record.classification and hypothesis.get("evidence_count") == 1 and
        json.loads(hypothesis["direct_result_hashes_json"]) == [key], "Hypothesis evidence/readback differs")
    return {"readback_verified": True, "result_hash": key, **metadata(run, record)}


def store(run, registry):
    require(not registry.exists(), "Refusing to overwrite registry evidence")
    value, record = expected(run)
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID, name="Sequential Qwen compact quadratic attention against a matched softmax control",
        category="compression", tested=True, confirmed=False, evidence_count=1, direct_experiment_count=1,
        confirmation_status=record.classification, tested_scope=SCOPE,
        description="Fixed original teacher; 36 sequential frozen-prefix stages per arm; aggregate5pp quality allowance and matched compiled CPU prefill/size targets.")
    experiment = ExperimentResult(experiment_id="jspace-quadratic-" + record.scientific_fingerprint[:16], hypothesis_id=STUDY_ID,
        metrics={"accuracy": record.quadratic_correct / 512,
            "minimum_cpu_compiled_prefill_speedup": min(record.cpu_compiled_prefill_speedups["quadratic"].values()),
            "tensor_reduction": record.tensor_reductions["quadratic"]}, primary_metric=record.quadratic_correct / 512,
        model_architecture=[1024] * 36, model_parameters=value["exports"]["quadratic"]["model_parameters"],
        training_time=sum(value["training"][f"quadratic/{layer}"]["elapsed_seconds"] for layer in range(36)),
        model_checkpoint=str(run / "export/quadratic/model.pt"), observations=[SCOPE, record.classification,
            value["report_receipt"]["report"], "No inference Gröbner/MLP polynomial replacement; direct causal normalized quadratic attention."])
    db = common.logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key], metadatas=[metadata(run, record)])
    db.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[{"direct_result_hashes_json": json.dumps([key]),
        "analysis_sha256": sha(run / "quadratic_closeout_analysis.json"), "report_path": value["report_receipt"]["report"]}])
    receipt = readback(run, registry, key)
    write(registry, dict(schema="nal.meta-hypothesis.v1", hypothesis=hypothesis, record=record.to_dict(), storage=receipt,
        analysis_sha256=sha(run / "quadratic_closeout_analysis.json"), report_path=value["report_receipt"]["report"],
        report_sha256=value["report_receipt"]["sha256"]))
    return receipt


def closeout(run, registry):
    value, record = expected(run)
    receipt = read(run / "quadratic_registry_readback.json")
    require(receipt == readback(run, registry), "Fresh-process registry readback is missing")
    event = dict(schema_version="jspace.research-log.v1", id="jspace-quadratic-" + record.scientific_fingerprint,
        event="quadratic_recipe_finalized", study_id=STUDY_ID, recorded_at=datetime.now(timezone.utc).isoformat(),
        confirmed=False, classification=record.classification, targets_met=record.targets_met,
        record=record.to_dict(), new_experimental_evidence=True, independent_confirmations=0,
        registry_result_id=receipt["result_hash"], registry_sha256=sha(registry),
        analysis_sha256=sha(run / "quadratic_closeout_analysis.json"), report_receipt=value["report_receipt"])
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()), "Research event already logged")
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    write(run / "quadratic_finalization.json", dict(status="completed", confirmed=False, record=record.to_dict(),
        registry_readback=receipt, research_event=event, analysis_source_sha256=value["analysis_source_sha256"]))
    return {"finalized": True, "registry": str(registry), "sha256": sha(run / "quadratic_finalization.json")}


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU closeout requires CUDA_VISIBLE_DEVICES=''")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "store", "readback", "closeout"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(); root, registry = args.root.resolve(), args.registry.resolve()
    if args.stage == "analyze":
        output = analyze(root)
    elif args.stage == "readback":
        output = readback(root, registry); write(root / "quadratic_registry_readback.json", output)
    else:
        output = globals()[args.stage](root, registry)
    print(json.dumps(output, allow_nan=False))
