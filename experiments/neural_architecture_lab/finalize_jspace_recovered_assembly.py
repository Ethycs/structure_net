"""Append-only CPU closeout of the four-arm recovered Qwen assembly evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_recovered_assembly_record import (
    JSpaceRecoveredAssemblyRecord, ARMS, STUDY_ID, SCOPE, require)
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import require_committed

ROOT, read, sha, write, logger = common.ROOT, common.read, common.sha, common.write, common.logger
DEFAULT = ROOT/"data/experiments/jspace_recovered_assembly/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-recovered-assembly-results.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_recovered_assembly_20261114.json"
TYPED = ROOT/"src/neural_architecture_lab/jspace_recovered_assembly_record.py"
TEST = ROOT/"tests/neural_architecture_lab/test_jspace_recovered_assembly.py"
ANALYSIS = "assembly_analysis.json"


def sources():
    return [Path(__file__).resolve(), TYPED, TEST, Path(common.__file__).resolve(),
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py"]


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def analyze(root):
    from experiments.structure_net import jspace_recovered_assembly as producer
    root = Path(root).resolve(); plan = producer.verify(root)
    require(not (root/ANALYSIS).exists() and not (root/"assembly_analysis_source").exists(), "Closeout already attempted")
    aggregate = read(root/"paired_result.json")
    runs, artifacts, large = {}, {}, {}
    def pin(path, expected=None):
        path = Path(path).resolve(); checksum = sha(path)
        require(expected is None or checksum == expected, "Changed closeout input: "+str(path))
        artifacts[str(path)] = checksum
    pin(root/"plan.json"); pin(root/"paired_result.json")
    for arm in ARMS:
        runs[arm] = producer.completed_arm(root, plan, arm)
        pin(root/arm/"manifest.json"); pin(root/arm/"result.json", aggregate["source_result_sha256"][arm])
        manifest = read(root/arm/"manifest.json"); run = runs[arm]
        require(manifest["lease_id"] == run["lease_id"] and manifest["allocation"] == run["allocation"]
                and manifest["peak_reserved_bytes"] == run["peak_reserved_bytes"]
                and 0 <= run["peak_reserved_bytes"] <= run["allocator"]["allocator_limit_bytes"]
                    <= sum(run["allocation"].values())*1024**2, "Arm reservation or allocator receipt disagrees")
    for path, expected in plan["source_sha256"].items():
        pin(path, expected); pin(root/"producer"/Path(path).relative_to(ROOT), expected)
    for path, expected in plan["input_sha256"].items():
        if Path(path).stat().st_size < 64*1024**2:
            pin(path, expected)
        else:
            large[path] = dict(sha256=expected, identity=identity(path))
    references = {arm:read(plan[arm+"_reference"]) for arm in ("native", "accepted_two")}
    for arm in references: pin(plan[arm+"_reference"], plan["input_sha256"][plan[arm+"_reference"]])
    config_path = Path(plan["model"]["path"])/"config.json"
    config = read(config_path); pin(config_path, plan["model"]["files"]["config.json"])
    require(config["tie_word_embeddings"] is True and config["hidden_size"] == 2048
            and config["num_hidden_layers"] == 36, "Original tied vocabulary and36-layer model required")
    launch = None
    if (root/"launch_receipt.json").exists():
        pin(root/"launch_receipt.json"); launch = read(root/"launch_receipt.json")
        require(launch["fingerprint"] == plan["fingerprint"] and launch["arms_in_order"] == list(ARMS), "Launch receipt differs")
    value = dict(status="completed", confirmed=False, plan=plan, aggregate=aggregate, arms=runs, references=references,
        endpoint_parameters=config["vocab_size"]*config["hidden_size"]+config["hidden_size"], launch_receipt=launch,
        artifact_sha256=artifacts, authenticated_large_artifacts=large,
        analysis_source_sha256={str(p):sha(p) for p in sources()}, created_at=datetime.now(timezone.utc).isoformat(),
        input_authentication="Plan, aggregate, four arm manifests/results, reference rows, live/archived producers and small inputs rehashed. Large selected states retain frozen producer SHA256 plus current immutable file-identity checks; assembly rehashed each selected state immediately before loading it.")
    record = JSpaceRecoveredAssemblyRecord.from_evidence(value); value["record"] = record.to_dict()
    archive = root/"assembly_analysis_source"; archive.mkdir(exist_ok=False)
    for path in sources():
        target = archive/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(root/ANALYSIS, value)
    return dict(analysis=str(root/ANALYSIS), sha256=sha(root/ANALYSIS), classification=record.classification,
        counts={arm:record.arms[arm]["correct"] for arm in ARMS})


def expected(root):
    root = Path(root); value = read(root/ANALYSIS)
    record = JSpaceRecoveredAssemblyRecord.from_evidence(value)
    require(value["record"] == record.to_dict(), "Typed record differs from observed evidence")
    for path, checksum in {**value["artifact_sha256"], **value["analysis_source_sha256"]}.items():
        require(sha(path) == checksum, "Closeout artifact/source changed: "+path)
    for path, pin in value["authenticated_large_artifacts"].items():
        require(identity(path) == pin["identity"], "Frozen selected state changed: "+path)
    for path, checksum in value["analysis_source_sha256"].items():
        require(sha(root/"assembly_analysis_source"/Path(path).relative_to(ROOT)) == checksum, "Archived analysis source changed")
    return value, record


def report(root, report_path):
    root, report_path = Path(root), Path(report_path); value, r = expected(root)
    counts = {arm:r.arms[arm]["correct"] for arm in ARMS}
    outcome = "passed" if r.primary_quality_pass else "failed"
    lines = ["# Recovered Qwen assembly results", "",
        f"Study: **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**", "",
        f"The primary hybrid scored **{counts['hybrid']}/64** against **{counts['native']}/64** native and **{outcome}** the fixed47/64 development gate. "
        f"All36 compact blocks scored **{counts['all_compact']}/64**. The accepted two-block control reproduced **{counts['accepted_two']}/64**. "
        "These are actual complete model paths on the same64 repeatedly inspected development questions.", "",
        "| Arm | Correct /64 | Accuracy | MatchedFP32 tensor bytes | Reduction vs nativeFP32 |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row, state = r.arms[arm], r.storage[arm]
        reduction = "—" if arm == "native" else f"{100*state['matched_fp32_reduction']:.2f}%"
        lines.append(f"| {arm} | {row['correct']} | {100*row['accuracy']:.2f}% | {state['bytes']:,} | {reduction} |")
    lines += ["", "The hybrid uses the selected compact blocks except12 and15, which remain native because their independent tests missed the isolated gate. "
        "The all-compact result is a separate diagnostic and cannot replace the declared primary outcome. Every arm began with original token embeddings, executed all36 blocks, and used the original tied vocabulary readout. No cached internal states bypassed composition.", "",
        "| Arm | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gains vs native | Losses vs native |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = r.arms[arm]; families = row["by_family"]; pair = row["paired_to_native"]
        lines.append(f"| {arm} | "+" | ".join(str(families[f]["correct"]) for f in ("arc", "boolq", "hellaswag", "winogrande"))
            +f" | {pair['gains']} | {pair['losses']} |")
    controls = value["arms"]
    lines += ["", f"Both controls passed exact prediction replay and logit tolerance5e-4+1e-5×|reference|. Maximum logit differences were "
        f"{controls['native']['control']['logit_max_abs_delta']:.9g} (native) and {controls['accepted_two']['control']['logit_max_abs_delta']:.9g} (accepted two). "
        "Selected checkpoints, all epoch-report hashes, their fresh-verification receipts, per-layer fits and the frozen protocol were pinned before assembly. Block0 retained its separate first-block enrichment fit.", "",
        "Storage uses the actual arm inventories, including fixed bases/geometry, packed parameters, final norm and rotary buffers, and tied embedding/readout counted once. "
        "All compared tensors are FP32. **These reductions are not comparisons against the original BF16 stored model.** Activations, KV cache, temporary label-row copies and allocator reserve are excluded.", "",
        "The evaluator streamed one block at a time. Its elapsed time includes loading and disk work and is not inference throughput. No speed benchmark or untouched confirmation evaluation occurred. "
        "A development pass still requires separately frozen confirmation and deployment measurements; a miss shows that isolated recovery did not preserve the tested assembled behavior.", "",
        f"Evidence: `{root.relative_to(ROOT)}/{ANALYSIS}`, SHA-256 `{sha(root/ANALYSIS)}`. {value['input_authentication']}", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream: stream.write("\n".join(lines))
    return dict(report=str(report_path), sha256=sha(report_path))


def metadata(root, record):
    return dict(hypothesis_id=STUDY_ID, confirmed=False, confirmation_status=record.classification,
        primary_arm="hybrid", primary_metric=record.primary_metric, primary_metric_name="hybrid_development_accuracy",
        primary_metric_unit="fraction", accuracy=record.primary_metric, count=64,
        native_correct=record.arms["native"]["correct"], accepted_two_correct=record.arms["accepted_two"]["correct"],
        all_compact_correct=record.arms["all_compact"]["correct"], hybrid_correct=record.arms["hybrid"]["correct"],
        quality_pass=record.primary_quality_pass, minimum_correct=47, independent_confirmations=0,
        assembly_evaluated=True, speed_measured=False, storage_comparison="matched_fp32_only",
        hybrid_tensor_storage_bytes=record.storage["hybrid"]["bytes"],
        hybrid_matched_fp32_reduction=record.storage["hybrid"]["matched_fp32_reduction"],
        aggregate_sha256=sha(Path(root)/ANALYSIS), scope=SCOPE)


def readback(root, registry, report_path, key=None):
    _, record = expected(root)
    if key is None:
        saved = read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path), "Registry/report provenance mismatch")
        key = saved["storage"]["result_hash"]
    db, wanted = logger(), metadata(root, record)
    row = db.experiments_collection.get(ids=[key], include=["metadatas"])
    require(row["ids"] == [key] and all(row["metadatas"][0].get(k) == v for k,v in wanted.items()), "Exact experiment readback mismatch")
    hypothesis = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(hypothesis["ids"] == [STUDY_ID], "Exact hypothesis missing")
    h = hypothesis["metadatas"][0]
    require(h.get("tested") is True and h.get("confirmed") is False and h.get("evidence_count") == 1
            and h.get("confirmation_status") == record.classification
            and json.loads(h["direct_result_hashes_json"]) == [key], "Hypothesis readback mismatch")
    return dict(readback_verified=True, result_hash=key, **wanted)


def store(root, registry, report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(), "Registry exists or report missing")
    _, r = expected(root); require_committed([*sources(), Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID, name="Recovered Qwen full-path assembly", category="compression", tested=True,
        confirmed=False, evidence_count=1, direct_experiment_count=1, confirmation_status=r.classification,
        tested_scope=SCOPE, description="Four frozen assembly arms; the hybrid47/64 gate is primary and every observed outcome is retained.")
    experiment = ExperimentResult(experiment_id="jspace-assembly-"+sha(Path(root)/ANALYSIS)[:16], hypothesis_id=STUDY_ID,
        metrics={**{arm+"_correct":r.arms[arm]["correct"] for arm in ARMS}, "hybrid_accuracy":r.primary_metric,
            "hybrid_tensor_storage_bytes":r.storage["hybrid"]["bytes"]}, primary_metric=r.primary_metric,
        model_architecture=[2048]*36, model_parameters=r.model_parameters["hybrid"], training_time=0.,
        observations=[SCOPE, r.classification, "No training occurred in this assembly evaluation. Elapsed streaming time is not generation speed."])
    db = logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key], metadatas=[metadata(root, r)])
    db.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/ANALYSIS), report_path=str(report_path))])
    receipt = readback(root, registry, report_path, key)
    write(registry, dict(schema_version="nal.meta-hypothesis.v1", hypothesis=hypothesis, storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS), report_path=str(report_path), report_sha256=sha(report_path)))
    return receipt


def closeout(root, registry, report_path):
    root = Path(root); receipt = read(root/"assembly_registry_readback.json")
    require(receipt == readback(root, registry, report_path) and receipt["readback_verified"] is True, "Readback changed")
    value = dict(status="completed", study_id=STUDY_ID, confirmed=False, classification=receipt["confirmation_status"],
        registry_readback=receipt, report_path=str(report_path), report_sha256=sha(report_path), aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(), research_log_modified=False)
    write(root/"assembly_finalization.json", value)
    return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for CPU finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "report", "store", "readback", "closeout"))
    parser.add_argument("--root", type=Path, default=DEFAULT); parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--registry", type=Path, default=REGISTRY); args = parser.parse_args()
    if args.stage == "analyze": output = analyze(args.root)
    elif args.stage == "report": output = report(args.root, args.report)
    else:
        output = {"store":store, "readback":readback, "closeout":closeout}[args.stage](args.root, args.registry, args.report)
        if args.stage == "readback": write(args.root/"assembly_registry_readback.json", output)
    print(json.dumps(output, allow_nan=False))
