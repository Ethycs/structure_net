"""CPU-only append-only finalization of the first-block development ablations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_residual_diagnostics_record import (
    ARMS, CLASSIFICATION, SCOPE, STUDY_ID, VARIANTS, JSpaceResidualDiagnosticsRecord, require)
from .finalize_jspace_qwen_affine import ROOT, LOG, logger, read, sha, write

DEFAULT = ROOT/"data/experiments/jspace_accuracy_recovery_diagnostics/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-accuracy-recovery-diagnostics.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_residual_diagnostics_20261114.json"
EXPECTED = "30733a8ad6a6fdc0befd195062a9f567a26c568358cd8f6c0eb7f3d9579b548f"
TYPED = ROOT/"src/neural_architecture_lab/jspace_residual_diagnostics_record.py"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def audit(root):
    root = Path(root)
    require(sha(root/"result.json") == EXPECTED, "Diagnostic aggregate hash mismatch")
    aggregate = read(root/"result.json")
    sources, plans, hashes = {}, {}, {str(root/"result.json"):EXPECTED}
    for relative,expected in aggregate["sources"].items():
        path = ROOT/relative
        require(sha(path) == expected, "Changed diagnostic source result")
        value, plan = read(path), read(path.parent/"plan.json")
        name = next((k for k,v in VARIANTS.items() if v == value.get("variant")), "native")
        require(name not in sources, "Duplicate source arm")
        if name == "native":
            config = plan["config"]
            require(plan["fingerprint"] == digest(config) and config["cut"] == 0
                    and config["selected"] is False and config["split"] == "development"
                    and config["dtype"] == "float32", "Native control configuration changed")
            ids = config["case_identities"]
            source_hashes = {**config["source_sha256"], **config["original_frozen_sources"]}
            input_hashes = value["consumed_sha256"]
        else:
            config = plan
            require(plan["fingerprint"] == digest({k:v for k,v in plan.items() if k != "fingerprint"})
                    and plan["variant"] == VARIANTS[name] and plan["replaced_layers"] == [0]
                    and plan["original_suffix_layers"] == list(range(1,36))
                    and plan["split"] == "development" and plan["dtype"] == "float32"
                    and plan["native_embedding_and_readout"] is True
                    and plan["attention"] == "softmax; no learned metric", "Ablation configuration changed")
            ids, source_hashes, input_hashes = plan["identities"], plan["source_sha256"], plan["input_sha256"]
            manifest = read(path.parent/"manifest.json")
            require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"], "Incomplete source manifest")
            for file,expected_hash in manifest["artifacts"].items():
                require(sha(path.parent/file) == expected_hash, "Source manifest artifact changed")
            hashes[str(path.parent/"manifest.json")] = sha(path.parent/"manifest.json")
        require(value["fingerprint"] == plan["fingerprint"] and ids == [r["identity"] for r in value["records"]], "Source plan/result mismatch")
        for file,expected_hash in {**source_hashes,**input_hashes}.items():
            require(sha(file) == expected_hash, "Changed producer or consumed input: "+file)
            hashes[file] = expected_hash
            snapshot = path.parent/"producer"/Path(file).relative_to(ROOT) if file in source_hashes else None
            if name != "native" and snapshot is not None:
                require(sha(snapshot) == expected_hash, "Archived producer changed")
                hashes[str(snapshot)] = expected_hash
        sources[name], plans[name] = value, plan
        hashes[str(path)], hashes[str(path.parent/"plan.json")] = expected, sha(path.parent/"plan.json")
    record = JSpaceResidualDiagnosticsRecord.from_evidence(aggregate,sources,EXPECTED)
    fingerprints = {p["config"]["source_fingerprint"] if name == "native" else p["source_fingerprint"] for name,p in plans.items()}
    require(len(fingerprints) == 1, "Arms derive from different original models")
    return aggregate,record,plans,hashes


def analyze(root):
    root = Path(root)
    aggregate,record,plans,hashes = audit(root)
    paths = [Path(__file__).resolve(),TYPED,ROOT/"tests/neural_architecture_lab/test_jspace_residual_diagnostics.py",
             Path(__file__).with_name("finalize_jspace_qwen_affine.py")]
    archive = root/"postprocessing_source"
    require(not (root/"campaign_results.json").exists() and not archive.exists(), "Analysis already exists")
    for path in paths:
        target = archive/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    value = dict(schema="jspace.residual-diagnostics.analysis.v1",study_id=STUDY_ID,status="completed",
        confirmed=False,classification=CLASSIFICATION,scope=SCOPE,created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),result=aggregate,result_sha256=EXPECTED,source_plans=plans,
        artifact_sha256=hashes,analysis_source_sha256={str(p):sha(p) for p in paths},
        integrity=dict(matched_cases=64,matched_arms=5,families=4,cases_per_family=16,
            independent_confirmations=0,training_updates=0,heldout_used=False,
            native_probe_source_archived=False,ablation_sources_archived=True,
            native_weight_hash_scope="Original model pinned by source plan; large weight shards not rehashed during this diagnostic closeout"))
    write(root/"campaign_results.json",value)
    return dict(status="analyzed",study_id=STUDY_ID,aggregate_sha256=sha(root/"campaign_results.json"))


def expected(root):
    root = Path(root)
    value = read(root/"campaign_results.json")
    require(value["study_id"] == STUDY_ID and value["confirmed"] is False and value["classification"] == CLASSIFICATION
            and value["result_sha256"] == EXPECTED == sha(root/"result.json"), "Changed diagnostic analysis")
    for path,expected_hash in {**value["analysis_source_sha256"],**value["artifact_sha256"]}.items():
        require(sha(path) == expected_hash, "Changed analysis evidence/source")
    return value


def report(root,report_path):
    root,report_path = Path(root),Path(report_path)
    value = expected(root)
    rows = value["record"]["arms"]
    labels = dict(native="Native",full_control="Rank2048 / head128 control",half_head="Rank2048 / head64",
                  half_rank="Rank1024 / head128",mean_rank="Mean-preserving rank1024 / head128")
    lines = ["# Qwen first-block residual diagnostics", "", f"Study: **{STUDY_ID}**. Status: completed exploratory diagnostic. **Confirmed: false.**",
        f"Classification: `{CLASSIFICATION}`.", "", "Reducing the first block's residual-update rank damaged development accuracy substantially more than reducing its head width. The full-rank control preserved all64 native predictions. This isolates an architecture problem before recovery training; it does not establish that recovery cannot work.", "",
        "| Condition | Correct /64 | Accuracy | Wins / losses vs native | Changed predictions |",
        "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {labels[r['name']]} | {r['correct']} | {100*r['correct']/64:.4f}% | {r['gains']} / {r['losses']} | {r['changed_predictions']} |")
    lines += ["", "Only block0 was changed; blocks1–35, embedding and readout remained native. All arms use softmax, native BF16-rounded weights promoted to FP32, and the same64 development questions:16 each from ARC, BoolQ, HellaSwag and WinoGrande. No recovery checkpoint or learned attention metric was used.", "",
        "The full-rank control's maximum absolute choice-logit difference was0.00359344, with no changed predictions. Halving rank lost30 previously correct answers and gained7; halving head width lost7 and gained2. Preserving the training mean in one replacement basis direction raised rank1024 accuracy from27/64 to30/64, but did not repair the failure.", "",
        "The result concerns this task-Jacobian-derived residual-update subspace at the first block. It does not show that all low-rank subspaces fail, identify one unique cause, or invalidate compression elsewhere in the model. These repeatedly inspected development cases provide no independent held-out guarantee. Runtime from one-block-at-a-time evaluation is not a deployment speed measurement. The ongoing all-block joint recovery remains a separate, unfinished study.", "",
        f"Raw aggregate: `{root.relative_to(ROOT)}/result.json`; SHA-256 `{EXPECTED}`.",
        f"Audited analysis: `{root.relative_to(ROOT)}/campaign_results.json`; SHA-256 `{sha(root/'campaign_results.json')}`.", "",
        "All320 per-case records, matched identities/labels, paired counts, family totals, choice-logit deltas, producing plans and recorded source hashes were checked. The four ablation producers have archived source snapshots; the native probe has hashed source but no archived probe snapshot. Large native weight shards remain pinned by the original model plan and were not rehashed by this closeout.", "",
        "Reproduction after committing the source/report (CPU only, default Pixi):", "", "```sh",
        "CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics store",
        "CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics readback",
        "CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics closeout", "```", ""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:
        stream.write("\n".join(lines))
    return dict(report=str(report_path),sha256=sha(report_path))


def metadata(root,value):
    result = dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=CLASSIFICATION,
        accuracy_scope=SCOPE,count=64,matched_arms=5,independent_confirmations=0,
        aggregate_sha256=sha(Path(root)/"campaign_results.json"),source_aggregate_sha256=EXPECTED)
    result.update({r["name"]+"_accuracy":r["correct"]/64 for r in value["record"]["arms"]})
    result.update(primary_metric=result["half_rank_accuracy"],primary_arm="half_rank")
    return result


def readback(root,registry,report_path,key=None):
    value = expected(root)
    if key is None:
        stored = read(registry)
        require(stored["aggregate_sha256"] == sha(Path(root)/"campaign_results.json")
                and stored["report_sha256"] == sha(report_path), "Registry/report provenance mismatch")
        key = stored["storage"]["result_hash"]
    db = logger()
    result = db.experiments_collection.get(ids=[key],include=["metadatas"])
    wanted = metadata(root,value)
    require(result["ids"] == [key] and all(result["metadatas"][0].get(k) == v for k,v in wanted.items()), "Diagnostic experiment readback mismatch")
    hypothesis = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(hypothesis["ids"] == [STUDY_ID], "Diagnostic hypothesis absent")
    row = hypothesis["metadatas"][0]
    require(row.get("tested") is True and row.get("confirmed") is False
            and row.get("confirmation_status") == CLASSIFICATION and row.get("evidence_count") == 1
            and json.loads(row["direct_result_hashes_json"]) == [key], "Diagnostic hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(), "Registry exists or report is missing")
    value = expected(root)
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="First-block Qwen residual-rank versus head-width diagnosis",category="compression",
        tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=CLASSIFICATION,
        tested_scope=SCOPE,description="Five matched development-only native-initialized controls isolate residual-update rank damage from head-width reduction.")
    values = metadata(root,value)
    metrics = {k:v for k,v in values.items() if k.endswith("_accuracy")}
    experiment = ExperimentResult(experiment_id="jspace-residual-diagnostics-"+EXPECTED[:16],hypothesis_id=STUDY_ID,
        metrics=metrics,primary_metric=values["primary_metric"],model_architecture=[2048]*36,
        model_parameters=0,training_time=0.,observations=[SCOPE,CLASSIFICATION,
            "model_parameters=0 is an unmeasured sentinel for this multi-arm diagnostic, not a parameter count."])
    db = logger()
    prior = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis)
    key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[values])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/"campaign_results.json"),report_path=str(report_path))])
    receipt = readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/"campaign_results.json"),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root,registry,report_path):
    root = Path(root)
    value,receipt = expected(root),read(root/"registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True, "Independent readback changed")
    event = dict(schema_version="jspace.research-log.v1",id=STUDY_ID+"-"+EXPECTED,event="residual_diagnostics_completed",
        study_id=STUDY_ID,recorded_at=datetime.now(timezone.utc).isoformat(),confirmed=False,
        classification=CLASSIFICATION,scope=SCOPE,record=value["record"],new_experimental_evidence=True,
        registry_result_id=receipt["result_hash"],aggregate_sha256=sha(root/"campaign_results.json"),
        registry_sha256=sha(registry),report_path=str(report_path),report_sha256=sha(report_path))
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.seek(0)
        require(not any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()), "Diagnostic already logged")
        stream.write(json.dumps(event,sort_keys=True,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())
    write(root/"finalization.json",dict(status="completed",study_id=STUDY_ID,confirmed=False,
        classification=CLASSIFICATION,registry_readback=receipt,research_event=event))
    return dict(finalized=True,study_id=STUDY_ID)


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for CPU diagnostic finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--registry",type=Path,default=REGISTRY)
    args = parser.parse_args()
    if args.stage == "analyze": output = analyze(args.root)
    elif args.stage == "report": output = report(args.root,args.report)
    else:
        output = {"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback": write(args.root/"registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
