"""CPU-only closeout for independent layer fits; assembly remains a separate study."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_parallel_single_layer_record import (
    JSpaceParallelSingleLayerRecord, STUDY_ID, SCOPE, require, scores)
from .finalize_jspace_qwen_affine import ROOT, logger, read, sha, write
from .finalize_jspace_single_layer import storage_accounting, require_committed

DEFAULT = ROOT/"data/experiments/jspace_parallel_single_layer/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-parallel-single-layer-results.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_parallel_single_layer_20261114.json"
TYPED = ROOT/"src/neural_architecture_lab/jspace_parallel_single_layer_record.py"
TEST = ROOT/"tests/neural_architecture_lab/test_jspace_parallel_single_layer.py"
ANALYSIS = "parallel_layer_analysis.json"


def identity(path):
    s = Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def summarize_storage(rows):
    first = rows[0][1]
    require(len(rows) == 34 and [layer for layer,_ in rows] == list(range(2,36)), "34 inventories required")
    for _, row in rows:
        require(row["packed_block"] == first["packed_block"] and row["selected_unpacked"] == first["selected_unpacked"]
                and row["native_matched_fp32_bytes"] == first["native_matched_fp32_bytes"],
                "Selected tensor shapes, dtypes or native storage differ")
    packed = first["packed_block"]["tensor_bytes"]
    native32, native_source = first["native_matched_fp32_bytes"], first["native_source_tensor_bytes"]
    block32, block_source = first["native_block_fp32_bytes"], first["native_block_source_bytes"]
    def projection(count, native, block):
        total = native-count*block+count*packed
        return dict(bytes=total,reduction_fraction=1-total/native,changed_blocks=count,
                    scope="Arithmetic projection only; no assembled model was evaluated in this study")
    return dict(same_fp32_tensor_signature=True,tied_embedding_readout_counted_once=True,projection_only=True,
        packed_block_tensor_bytes=packed,packed_block_parameters=first["packed_block"]["parameter_count"],
        fixed_geometry_and_buffer_bytes=first["packed_block"]["fixed_geometry_and_buffer_bytes"],
        selected_unpacked_tensor_bytes=first["selected_unpacked"]["tensor_bytes"],
        native_block_fp32_bytes=block32,native_block_bf16_bytes=block_source,
        native_matched_fp32_bytes=native32,native_source_bf16_bytes=native_source,
        native_parameter_count=first["native_parameters"],one_replacement_model_parameters=first["selected_whole_parameters"],
        block_matched_fp32_reduction=1-packed/block32,independent_packed_state_bank_bytes=34*packed,
        independent34_matched_fp32_projection=projection(34,native32,block32),
        all36_matched_fp32_projection=projection(36,native32,block32),
        all36_source_bf16_plus_fp32_compact_projection=projection(36,native_source,block_source),
        shared_tensor_inventory=first["packed_block"],native_headers=first["native_headers"],
        per_layer=[dict(layer=layer,packed_tensor_bytes=row["packed_block"]["tensor_bytes"],
            checkpoint_file_bytes=row["selected_checkpoint_file_bytes"]) for layer,row in rows],
        scope="Persistent tensors including fixed geometry; tied E once. Excludes optimizer, activations, KV cache, transient rotary tables and serialization headers. FP32 compact states must not be compared to native BF16 as a compression gain.")


def audit(root):
    from experiments.structure_net import jspace_parallel_single_layer as coordinator
    from experiments.structure_net import jspace_independent_layer_distill as worker
    from experiments.structure_net import jspace_next_layer_distill as sequential
    root = Path(root).resolve(); plan = coordinator.verify_state(root)
    result = read(root/"result.json")
    require(result["fingerprint"] == plan["fingerprint"] and result["status"] == "completed", "Coordinator incomplete or mismatched")
    artifacts, large = {}, {}
    def pin(path, expected=None, large_file=False):
        path = Path(path).resolve(); actual = sha(path)
        require(expected is None or actual == expected, "Changed artifact: "+str(path))
        if large_file: large[str(path)] = dict(sha256=actual,identity=identity(path))
        else: artifacts[str(path)] = actual
        return actual
    for name in ("plan.json","result.json","pilots_gate.json","run_started.json"):
        pin(root/name)
    gate = read(root/"pilots_gate.json")
    require(gate["status"] == "completed" and gate["passed"] is True and gate["fingerprint"] == plan["fingerprint"], "Concurrent pilot gate incomplete")
    for path,digest in plan["input_sha256"].items(): pin(path,digest)
    for path,digest in plan["source_sha256"].items():
        pin(path,digest);pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    native_path = Path(plan["quality_prefix_root"])/"cache/result.json"; pin(native_path)
    native = read(native_path)
    original = read(Path(plan["source"])/"plan.json")
    cards = {card[0]:card for card in coordinator.CARDS}; assignment = {}
    for summary in result["workers"]:
        saved = read(root/"workers"/(summary["worker"]+".json"))
        require(saved == summary and summary["status"] == "completed", "Worker completion changed")
        pin(root/"workers"/(summary["worker"]+".json"))
        for row in summary["completed"]:
            require(row["layer"] not in assignment, "Layer assigned twice")
            assignment[row["layer"]] = (cards[summary["worker"]],row)
    require(sorted(assignment) == list(range(2,36)), "Incomplete worker coverage")
    trials, inventories = [], []
    for layer in range(2,36):
        child = root/"blocks"/f"block_{layer:02d}"; child_plan = worker.verify(child)
        card, worker_row = assignment[layer]
        require(child_plan["source"] == plan["source"] and child_plan["basis_root"] == plan["basis_root"]
                and child_plan["quality_prefix_root"] == plan["quality_prefix_root"], "Worker input provenance changed")
        for name in ("plan.json","items.json"): pin(child/name)
        for path,digest in child_plan["source_sha256"].items():
            if path not in artifacts: pin(path,digest)
            require(artifacts[path] == digest, "Inconsistent live source pin")
            pin(child/"producer"/Path(path).relative_to(ROOT),digest)
        # The frozen worker verifier checks original input identities. Rehash
        # consumed small fits/metadata, not the shared multi-GB teacher banks.
        for path,digest in child_plan["input_sha256"].items():
            if Path(path).stat().st_size < 64*1024**2 and path not in artifacts: pin(path,digest)
        for stage in ("prepare","pilot","train","verify"):
            receipt = coordinator.check_output(stage,child,layer,card)
            stage_path = root/"stages"/f"block_{layer:02d}_{stage}.json"
            require(read(stage_path) == dict(status="completed",returncode=0,**receipt), "Coordinator stage receipt changed")
            pin(stage_path)
            if stage == "train":
                require(worker_row == dict(layer=layer,**receipt), "Worker training summary changed")
        train = sequential.completed_training(child,child_plan)  # Hashes selected checkpoint before loading.
        checkpoint = child/"train/selected.pt"
        large[str(checkpoint)] = dict(sha256=train["selected_checkpoint_sha256"],identity=identity(checkpoint))
        for stage in ("pilot","train","verification"):
            pin(child/stage/"manifest.json"); pin(child/stage/"result.json")
        fresh = read(child/"verification/result.json"); fresh_manifest = read(child/"verification/manifest.json")
        require(fresh_manifest["checkpoint_sha256"] == fresh["checkpoint_sha256"] == train["selected_checkpoint_sha256"], "Verification checkpoint differs")
        pilot = read(child/"pilot/result.json")
        require(pilot["native_control"]["reference_sha256"] == child_plan["quality_reference_sha256"] == sha(native_path), "Native control provenance changed")
        reports = {}
        for row in train["history"]:
            path = child/"train"/f"epoch_{row['epoch']:02d}_development.json"; pin(path,row["development_sha256"])
            data = read(path); score = scores(data["records"],native["records"])
            require(data["metrics"] == row["metrics"] and score["correct"] == row["metrics"]["correct"], "Epoch metrics disagree with predictions")
            require(all(score["by_family"][f]["correct"] == row["metrics"]["by_family"][f]["correct"] for f in score["by_family"]), "Epoch family metrics disagree")
            if row["epoch"] in (0,train["selected_epoch"]): reports[row["epoch"]] = data
        inventories.append((layer,storage_accounting(checkpoint,original["model"]["path"])))
        trials.append(dict(plan=child_plan,training=train,verification=fresh,pilot=pilot,
            train_manifest=read(child/"train/manifest.json"),initial_records=reports[0]["records"],
            selected_records=reports[train["selected_epoch"]]["records"]))
        print(json.dumps(dict(stage="audit",layer=layer,correct=train["selected"]["correct"],verified=True)),flush=True)
    value = dict(status="completed",confirmed=False,coordinator_plan=plan,coordinator_result=result,trials=trials,
        native_records=native["records"],storage=summarize_storage(inventories),artifact_sha256=artifacts,
        authenticated_large_artifacts=large,native_weight_pins=original["model"],
        input_authentication="All selected checkpoint bytes, stage/development records and live/archived producers rehashed at analysis. Original teacher/cache payloads retain pinned producer hashes plus current immutable file-identity verification; small consumed inputs rehashed. Later closeout stages verify hashes of small artifacts and unchanged device/inode/size/mtime/ctime of already-hashed selected checkpoints.",
        total_worker_training_seconds=sum(r["train_manifest"]["elapsed_seconds"] for r in trials))
    return value,JSpaceParallelSingleLayerRecord.from_evidence(value)


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(__file__).with_name("finalize_jspace_single_layer.py"),
            ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py",Path(__file__).with_name("finalize_jspace_qwen_affine.py")]


def analyze(root):
    root = Path(root); require(not (root/ANALYSIS).exists(),"Analysis already exists")
    value,record = audit(root); archive = root/"parallel_layer_analysis_source"
    require(not archive.exists(),"Source archive already exists")
    for path in sources():
        target = archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    value.update(schema="jspace.parallel-layer-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",study_id=STUDY_ID,classification=record.classification,analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root = Path(root);value = read(root/ANALYSIS);record = JSpaceParallelSingleLayerRecord.from_evidence(value)
    require(value["record"] == record.to_dict(),"Typed record differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path) == digest,"Changed closeout artifact/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items():
        require(identity(path) == pin["identity"],"Previously hashed checkpoint changed: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/"parallel_layer_analysis_source"/Path(path).relative_to(ROOT)) == digest,"Changed archived closeout source")
    return value,record


def report(root,report_path):
    root,report_path = Path(root),Path(report_path);value,r = expected(root);s = r.storage
    lines = ["# Independent Qwen layer recovery results","",f"Study: **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**",
        "",f"All **34 independent trials (layers 2–35)** completed 12 epochs and fresh-process verification. **32 layers passed** the fixed isolated gate of at least 47/64; layer 12 scored 44/64 and layer 15 scored 46/64. The native control scored 50/64. These are separate one-layer substitutions on the same 64 development questions. **32/34 is not a model accuracy.**",
        "",f"Training improved the number correct for {r.summary['layers_improved_correct_count']} layers; {r.summary['layers_unchanged_correct_count']} retained their initial count. Initializers won selection for layers {', '.join(map(str,r.summary['initializer_selected_layers']))}; their final scores must not be attributed to training. All trials, including misses, completed **{r.summary['optimizer_updates']:,} optimizer updates** in aggregate without a retry.",
        "","| Layer | Initial /64 | Selected /64 | Epoch | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in r.layers:
        f = row["selected"]["by_family"]
        lines.append(f"| {row['layer']} | {row['initial']['correct']} | {row['selected']['correct']} | {row['selected_epoch']} | "
            +" | ".join(str(f[k]["correct"]) for k in ("arc","boolq","hellaswag","winogrande"))
            +f" | {'pass' if row['quality_pass'] else 'miss'} |")
    lines += ["","Each worker trained only one compact block on the same original 512 training examples and native BF16 input/output pairs. Task checks instead used authenticated native FP32 development boundaries, the selected packed block, the original downstream layers and tied vocabulary readout. Every worker's native substitution control reproduced 50/64 before training. Accuracy-first selection used relative feature error to break ties; initialization was eligible. The independent reload reproduced each selected prediction exactly, with the recorded feature/logit tolerances.",
        "","The residual stream remains 2048 wide; computed updates have rank 1024. Native 128-dimensional attention heads, softmax and the 11008-wide MLP remain. Training uses FP32 parameters, computation, gradients and AdamW. The objective is equal-case/equal-token output error divided by original teacher-update energy, with a fixed floor fitted on training data. Full epoch histories, normalized features, paired changes against initialization/native and family counts are retained in the analysis artifact.",
        "","| Tensor accounting | Bytes |","|---|---:|",
        f"| Native block, FP32 | {s['native_block_fp32_bytes']:,} |",
        f"| Selected packed block, FP32 including basis/geometry | {s['packed_block_tensor_bytes']:,} |",
        f"| Native full model, FP32 | {s['native_matched_fp32_bytes']:,} |",
        f"| Projection: 34 replacements, rest native FP32 | {s['independent34_matched_fp32_projection']['bytes']:,} |",
        f"| Projection: 36 replacements, all FP32 | {s['all36_matched_fp32_projection']['bytes']:,} |",
        f"| Original full model, BF16 | {s['native_source_bf16_bytes']:,} |",
        f"| Projection: 36 FP32 replacements, remaining tensors BF16 | {s['all36_source_bf16_plus_fp32_compact_projection']['bytes']:,} |",
        "",f"All 34 selected checkpoint tensor shapes and dtypes match. Each packed block saves **{100*s['block_matched_fp32_reduction']:.2f}%** against the same native block in FP32. The all-36 FP32 projection saves **{100*s['all36_matched_fp32_projection']['reduction_fraction']:.2f}%**, but replacing native BF16 blocks with these FP32 compact blocks increases total tensor bytes by **{-100*s['all36_source_bf16_plus_fp32_compact_projection']['reduction_fraction']:.2f}%**. These whole-model figures are arithmetic projections, not a packaged/evaluated assembly. Tied embedding/readout is counted once; fixed geometry is included. Optimizer, activations, KV cache and serialization overhead are excluded. No speed measurement belongs to these checkpoints.",
        "","Assembly is pending in this study and has a separate [frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-14_qwen-recovered-assembly.md). Passing isolated replacements does not establish that their errors compose. The accepted block0/block1 results are separate prior experiments; they are not included in this campaign's 34 trials. No untouched confirmation examples were accessed for this closeout.",
        "",f"Evidence: `{root.relative_to(ROOT)}/{ANALYSIS}`, SHA-256 `{sha(root/ANALYSIS)}`. {value['input_authentication']}",
        "","Registry registration requires the exact source, tests and this report to be committed, followed by exact-ID readback. The registry primary metric is **32 isolated layers passing (unit: count)**; no aggregate model-accuracy metric is stored.",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(report_path),sha256=sha(report_path))


def metadata(root,r):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=r.classification,
        primary_metric=r.primary_metric,primary_metric_name="isolated_layers_passing_gate",primary_metric_unit="count",
        isolated_trials=34,isolated_layers_passing_gate=32,isolated_quality_misses_json=json.dumps([12,15]),
        optimizer_updates=13056,unique_development_cases=64,assembly_evaluated=False,
        assembly_status=r.assembly_status,independent_confirmations=0,scope=SCOPE,aggregate_sha256=sha(Path(root)/ANALYSIS))


def readback(root,registry,report_path,key=None):
    _,record = expected(root)
    if key is None:
        saved = read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path),"Registry/report provenance mismatch")
        key = saved["storage"]["result_hash"]
    db = logger();wanted = metadata(root,record)
    actual = db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(actual["ids"] == [key] and all(actual["metadatas"][0].get(k) == v for k,v in wanted.items()),"Exact experiment readback mismatch")
    h = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(h["ids"] == [STUDY_ID],"Hypothesis missing")
    row = h["metadatas"][0]
    require(row.get("tested") is True and row.get("confirmed") is False and row.get("evidence_count") == 1
            and row.get("confirmation_status") == record.classification
            and json.loads(row["direct_result_hashes_json"]) == [key],"Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    value,r = expected(root);require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="Independent Qwen layer recovery before assembly",category="compression",
        tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=r.classification,
        tested_scope=SCOPE,description="34 isolated native-context replacements; completed quality misses retained and every selected checkpoint freshly verified.")
    experiment = ExperimentResult(experiment_id="jspace-parallel-layers-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(isolated_layers_passing_gate=32,isolated_trials=34,optimizer_updates=13056,
            packed_block_tensor_bytes=r.storage["packed_block_tensor_bytes"]),primary_metric=32.,
        model_architecture=[2048]*36,model_parameters=r.storage["one_replacement_model_parameters"],
        training_time=value["total_worker_training_seconds"],observations=[SCOPE,r.classification,
            "model_parameters describes each separate one-replacement model; training_time is summed worker training seconds, not wall time or inference latency."])
    db = logger();prior = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis);key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,r)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt = readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root,registry,report_path):
    root = Path(root);receipt = read(root/"parallel_layer_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value = dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],
        registry_readback=receipt,report_path=str(report_path),report_sha256=sha(report_path),
        aggregate_sha256=sha(root/ANALYSIS),recorded_at=datetime.now(timezone.utc).isoformat(),
        research_log_append="Delegated to root to avoid concurrent research-log edits",assembly_evaluated=False)
    write(root/"parallel_layer_finalization.json",value)
    return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Hide CUDA for CPU finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT);parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--registry",type=Path,default=REGISTRY);args = parser.parse_args()
    if args.stage == "analyze": output = analyze(args.root)
    elif args.stage == "report": output = report(args.root,args.report)
    else:
        output = {"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback":write(args.root/"parallel_layer_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
