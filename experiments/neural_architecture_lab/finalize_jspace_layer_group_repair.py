"""CPU-only closeout of actual all-compact recovery by a selected block triple."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

import torch

from neural_architecture_lab.jspace_layer_group_repair_record import JSpaceLayerGroupRepairRecord,STUDY_ID,SCOPE,require
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import native_inventory,tensor_inventory,require_committed
from .finalize_jspace_recovered_assembly import identity

ROOT,read,sha,write,logger = common.ROOT,common.read,common.sha,common.write,common.logger
DEFAULT = ROOT/"data/experiments/jspace_layer_group_repair/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-layer-group-repair-results.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_layer_group_repair_20261114.json"
TYPED = ROOT/"src/neural_architecture_lab/jspace_layer_group_repair_record.py"
TEST = ROOT/"tests/neural_architecture_lab/test_jspace_layer_group_repair.py"
ANALYSIS = "layer_group_repair_analysis.json"
ARCHIVE = "layer_group_repair_analysis_source"


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(common.__file__).resolve(),
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py",
        ROOT/"src/neural_architecture_lab/jspace_recovered_joint_record.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_recovered_assembly.py"]


def storage_inventory(plan,directory):
    parent = read(Path(plan["assembly_root"])/"plan.json"); native = native_inventory(plan["model"]["path"])
    names = {"wrapped.original."+k for k in ("qkv_weight","qkv_bias","attention_output","gate_up_weight","mlp_output","qkv_offset","gate_up_offset")}
    rows = []; trainable = full_parameters = delta_bytes = 0
    for layer in range(36):
        selected = layer in plan["selected_layers"]
        path = Path(directory)/f"block_{layer:02d}.pt" if selected else Path(parent["blocks"][str(layer)]["checkpoint"])
        loaded = torch.load(path,map_location="cpu",weights_only=True,mmap=True); state = loaded if selected else loaded["block"]
        require(state["wrapped._extra_state"] == dict(version=1,mode="softmax",metric_active=False)
            and names <= set(state) and state["wrapped.original.basis"].shape == (2048,1024)
            and all(t.dtype == torch.float32 for t in state.values() if isinstance(t,torch.Tensor)),"Original full-head/no-polynomial FP32 compact architecture required")
        inventory = tensor_inventory(state); parameters = sum(state[name].numel() for name in names)
        full_parameters += parameters
        if selected: trainable += parameters; delta_bytes += inventory["tensor_bytes"]
        rows.append(dict(layer=layer,trained=selected,parameter_count=parameters,tensor_bytes=inventory["tensor_bytes"]))
    endpoints = native["tensors"]["model.embed_tokens.weight"]["elements"]+native["tensors"]["model.norm.weight"]["elements"]
    return dict(all_fp32=True,trainable_layers=plan["selected_layers"],parameter_tensor_count=21,
        native_norm_frozen=True,tied_embedding_readout_counted_once=True,model_parameters=full_parameters+endpoints,
        trainable_parameters=trainable,delta_tensor_bytes=delta_bytes,
        full_unpacked_tensor_bytes=sum(r["tensor_bytes"] for r in rows)+4*endpoints,blocks=rows,
        scope="Actual three-file delta plus all33 frozen original compact states and tied nativeE/norm once, allFP32. Unpacked saved tensors include geometry; exclude transient rotary tables, activations, KV cache and optimizer. Delta is not a standalone model or packed export.")


def audit(root):
    from experiments.structure_net import jspace_layer_group_repair as producer
    from experiments.structure_net import jspace_recovered_teacher as teacher
    from experiments.structure_net.jspace_layer_combination_search import verified_selection
    root = Path(root).resolve(); plan = producer.verify(root); artifacts,large,stages,manifests = {},{},{},{}
    def pin(path,expected=None):
        path = Path(path).resolve(); checksum = sha(path)
        require(expected is None or checksum == expected,"Changed repair closeout artifact: "+str(path))
        if path.stat().st_size > 64*1024**2: large[str(path)] = dict(sha256=checksum,identity=identity(path))
        else: artifacts[str(path)] = checksum
    for name in ("plan.json","items.json","events.jsonl"): pin(root/name)
    for stage in ("capture","pilot","train","verification"):
        stages[stage] = producer.checked(root,stage,plan); manifest = read(root/stage/"manifest.json"); manifests[stage] = manifest
        pin(root/stage/"manifest.json"); pin(root/stage/"result.json",manifest["result_sha256"])
        require(manifest["allocator"]["allocation"] == manifest["allocation"] and len(manifest["allocator"]["settings"]) == 2
            and sorted(manifest["allocation"].values()) == [7168,11264],"Matching frozen two-card allocation required")
        for row in manifest["allocator"]["settings"]:
            peak = manifest["device_peaks"][row["physical_uuid"]]
            require(0 <= peak["peak_reserved_bytes"] <= row["allocator_limit_bytes"] < row["lease_bytes"],"GPU peak exceeded reservation cap")
        for name,digest in manifest["artifacts"].items():
            path = root/stage/name
            # checked(capture) already authenticated every complete feature bank.
            if stage == "capture": large[str(path)] = dict(sha256=digest,identity=identity(path))
            else: pin(path,digest)
    require(manifests["verification"]["passed"] is True,"Completed independent fresh verification required")
    train = stages["train"]; directory = root/"train/selected"; delta = read(directory/"manifest.json")
    pin(directory/"manifest.json",train["selected_manifest_sha256"])
    require(set(delta["artifacts"]) == {f"block_{i:02d}.pt" for i in plan["selected_layers"]},"Exactly three selected delta files required")
    for name,digest in delta["artifacts"].items(): pin(directory/name,digest)
    epochs = []
    for row in train["history"]:
        path = root/"train"/f"epoch_{row['epoch']:02d}_development.json"; pin(path,row["development_sha256"])
        epochs.append(dict(epoch=row["epoch"],**read(path)))
    control = read(root/"train/initial_control.json"); pin(root/"train/initial_control.json")
    require(control["passed"] is True and control["correct"] == 27,"Scientific initial model must reproduce27/64")
    teacher_root = Path(plan["teacher_root"]); teacher_plan = teacher.verify(teacher_root,completed=True)
    require(teacher_plan["fingerprint"] == plan["teacher_fingerprint"] and teacher_plan["items_digest"] == plan["items_digest"],"Endpoint teacher identity changed")
    for name in ("plan.json","cache/manifest.json"): pin(teacher_root/name)
    for name,digest in read(teacher_root/"cache/manifest.json")["artifact_sha256"].items():
        path = teacher_root/name
        if path.stat().st_size > 64*1024**2: large[str(path)] = dict(sha256=digest,identity=identity(path))
        else: pin(path,digest)
    proof = verified_selection(plan["search_root"])
    require(proof == plan["selection"],"Frozen search target changed")
    search_root = Path(plan["search_root"]); search = read(search_root/"search/result.json"); chosen = search["summary"]["selected"]
    for name in ("plan.json","search/manifest.json","search/result.json","search/selection.json"): pin(search_root/name)
    for path,digest in read(search_root/"plan.json")["source_sha256"].items():
        pin(path,digest); pin(search_root/"producer"/Path(path).relative_to(ROOT),digest)
    context = dict(selection=chosen,records=search["results"][chosen["restored_mask_id"]]["records"],
        candidates=search["summary"]["candidate_record"],search_fingerprint=proof["search_fingerprint"])
    for path,digest in plan["source_sha256"].items():
        pin(path,digest); pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    for path,digest in plan["input_sha256"].items():
        if path in artifacts or path in large: continue
        if Path(path).stat().st_size > 64*1024**2: large[path] = dict(sha256=digest,identity=identity(path))
        else: pin(path,digest)
    storage = storage_inventory(plan,directory)
    updates = [r for r in map(json.loads,(root/"events.jsonl").read_text().splitlines()) if r.get("kind") == "update" and r.get("stage") == "train"]
    require([r["update"] for r in updates] == list(range(1,193)) and [r["epoch"] for r in updates] == [e for e in range(1,7) for _ in range(32)]
        and all(r["cases"] == 16 and r["gradient_parameter_count"] == 21 and r["gradient_scalar_count"] == storage["trainable_parameters"] for r in updates),
        "All192 updates must cover exactly the selected21 parameter tensors")
    value = dict(status="completed",confirmed=False,plan=plan,training=train,verification=stages["verification"],pilot=stages["pilot"],capture=stages["capture"],
        teacher=read(teacher_root/"cache/result.json"),epochs=epochs,delta_manifest=delta,frozen_state=read(root/"train/frozen_state.json"),
        selection_proof=proof,search_context=context,storage=storage,stage_manifests=manifests,
        update_audit=dict(count=192,gradient_parameter_tensors=21,gradient_scalars=storage["trainable_parameters"]),
        artifact_sha256=artifacts,authenticated_large_artifacts=large,
        input_authentication="Search selection, all stages/epoch rows, selected3-file delta, originalFP32logit cache and new3-boundary target banks authenticated. Frozen other33 states retain inherited hashes and immutable identities. Subsequent closeout rechecks small hashes and identities of authenticated large inputs.")
    return value,JSpaceLayerGroupRepairRecord.from_evidence(value)


def analyze(root):
    root = Path(root).resolve(); require(not (root/ANALYSIS).exists() and not (root/ARCHIVE).exists(),"Closeout already attempted")
    value,record = audit(root)
    for path in sources():
        target = root/ARCHIVE/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    value.update(schema="jspace.layer-group-repair-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),record=record.to_dict(),
        analysis_source_sha256={str(path):sha(path) for path in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",classification=record.classification,correct=record.selected["correct"],analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root = Path(root); value = read(root/ANALYSIS); record = JSpaceLayerGroupRepairRecord.from_evidence(value)
    require(value["record"] == record.to_dict(),"Typed repair record differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items(): require(sha(path) == digest,"Changed closeout artifact/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items(): require(identity(path) == pin["identity"],"Changed previously authenticated model/teacher input: "+path)
    for path,digest in value["analysis_source_sha256"].items(): require(sha(root/ARCHIVE/Path(path).relative_to(ROOT)) == digest,"Changed archived closeout source")
    return value,record


def report(root,report_path):
    root,report_path = Path(root),Path(report_path); value,r = expected(root); target = r.search_context["selection"]
    lines = ["# Targeted joint repair of the recovered Qwen assembly","",f"Study **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**","",
        f"After jointly training only compact blocks **{r.selected_layers}**, the selected **all36-compact** model scored **{r.selected['correct']}/64 ({100*r.primary_metric:.2f}%)**, versus27/64 before repair and50/64 native. The fixed47/64 development gate {'passed' if r.quality_pass else 'failed'}. Epoch{r.selected_epoch} was selected by actual task accuracy, then lower conditional-choice CE; the unchanged initializer remained eligible.",
        "",f"The preceding search found that replacing these three blocks with **native** blocks scored **{target['restored_correct']}/64**, a conditional rescue of{target['rescue_correct']:+d}. That is a different model. Its accuracy is not the learned compact repair score and does not establish that the rescue is learnable.",
        "","| Epoch | Actual correct /64 | Full-vocabulary KL | Choice KL | Choice CE | Feature loss | Native agreement |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in r.epochs:
        m = row["metrics"]; lines.append(f"| {row['epoch']} | {row['correct']} | {m['full_kl']:.6f} | {m['choice_kl']:.6f} | {m['choice_ce']:.6f} | {m['feature_loss']:.6f} | {100*m['teacher_agreement']:.2f}% |")
    lines += ["","| Selected layer | Initial feature loss | Selected feature loss |","|---|---:|---:|"]
    for layer in r.selected_layers:
        lines.append(f"| {layer} | {r.initial['feature_loss_by_layer'][str(layer)]:.6f} | {r.selected['feature_loss_by_layer'][str(layer)]:.6f} |")
    lines += ["","Feature loss averages equal cases and the three selected layers: valid-token squared error in the fixed basis divided by the detached original-teacher coordinate energy. It retains gradients through the student output and earlier selected layers. A lower feature loss is not successful task recovery.",
        "","| Model | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 |","|---|---:|---:|---:|---:|"]
    for name,row in (("Native",r.native),("Initial all compact",r.initial),("Selected learned repair",r.selected),("Search: triple native",r.search_context["native_restored_score"])):
        lines.append(f"| {name} | "+" | ".join(str(row["by_family"][f]["correct"]) for f in ("arc","boolq","hellaswag","winogrande"))+" |")
    pair = r.selected["paired_to_initial"]
    lines += ["",f"The selected learned model gained{pair['gains']} and lost{pair['losses']} correct answers relative to the27/64 initializer. All192 updates across6 epochs completed on the original512 training cases; each update reported gradients for exactly21 tensors in the chosen three blocks. The other33 blocks, native final norm, tied embedding/readout, bases and all buffers remained fixed. The one-update pilot was discarded.",
        "","The endpoint objective remained .2T² full-vocabulary KL +.5T² conditional-choice KL +.3 gold-choice CE atT=2, using midpoint and answer only. A unit-weight feature auxiliary was added. This longer schedule plus auxiliary is exploratory; benefit cannot be attributed solely to selecting this group rather than the earlier all-layer continuation.",
        "",f"The selected checkpoint stores a **three-block delta** ({r.storage['delta_tensor_bytes']:,} tensor bytes), requiring the33 frozen blocks and original tiedE/norm. The complete reconstructed unpackedFP32 model contains{r.storage['full_unpacked_tensor_bytes']:,} saved tensor bytes. The delta is not a standalone compressed model. No packing-parity or deployment-speed result is claimed by this closeout.",
        "","OriginalFP32 teacher targets passed native decision and selected-boundary parity checks. Strict reload and independent fresh-process verification reproduced the selected actual full-model predictions/logits and feature metrics. The same64 development cases were reused for search and checkpoint selection; no untouched confirmation occurred.",
        "",f"Evidence: `{root/ANALYSIS}`, SHA256 `{sha(root/ANALYSIS)}`. {value['input_authentication']}",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as handle: handle.write("\n".join(lines))
    return dict(report=str(report_path),report_sha256=sha(report_path))


def metadata(root,record):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=record.classification,
        primary_metric=record.primary_metric,primary_metric_name="selected_all_compact_group_repair_development_accuracy",primary_metric_unit="fraction",
        accuracy=record.primary_metric,correct=record.selected["correct"],count=64,initial_correct=27,native_correct=50,
        quality_pass=record.quality_pass,minimum_correct=47,selected_epoch=record.selected_epoch,updates=192,
        trained_layers_json=json.dumps(record.selected_layers),trained_block_count=3,native_norm_frozen=True,
        native_restored_search_correct=record.search_context["selection"]["restored_correct"],
        feature_loss=record.selected["metrics"]["feature_loss"],independent_confirmations=0,packing_validated=False,speed_measured=False,
        aggregate_sha256=sha(Path(root)/ANALYSIS),scope=SCOPE)


def readback(root,registry,report_path,key=None):
    _,record = expected(root)
    if key is None:
        saved = read(registry); require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path),"Registry/report provenance mismatch")
        key = saved["storage"]["result_hash"]
    db,wanted = logger(),metadata(root,record); row = db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(row["ids"] == [key] and all(row["metadatas"][0].get(k) == v for k,v in wanted.items()),"Exact repair experiment readback mismatch")
    h = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"]); require(h["ids"] == [STUDY_ID],"Repair hypothesis missing")
    v = h["metadatas"][0]
    require(v.get("tested") is True and v.get("confirmed") is False and v.get("evidence_count") == 1
        and v.get("confirmation_status") == record.classification and json.loads(v["direct_result_hashes_json"]) == [key],"Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    value,record = expected(root); require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="Joint repair of a search-selected compact Qwen triple",category="compression",tested=True,
        confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=record.classification,tested_scope=SCOPE,
        description="Actual all-compact model after six-epoch training of exactly three selected blocks with feature and output distillation.")
    experiment = ExperimentResult(experiment_id="jspace-layer-group-repair-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(accuracy=record.primary_metric,initial_accuracy=27/64,native_accuracy=50/64,feature_loss=record.selected["metrics"]["feature_loss"],
            choice_ce=record.selected["metrics"]["choice_ce"],native_restored_search_correct=record.search_context["selection"]["restored_correct"]),
        primary_metric=record.primary_metric,model_architecture=[2048]*36,model_parameters=record.storage["model_parameters"],
        training_time=value["stage_manifests"]["train"]["elapsed_seconds"],model_checkpoint=value["training"]["selected_directory"],
        training_history=[dict(epoch=row["epoch"],accuracy=row["accuracy"],feature_loss=row["metrics"]["feature_loss"]) for row in record.epochs],
        observations=[SCOPE,record.classification,"Selected checkpoint is a three-block delta requiring the original frozen assembly."])
    db = logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested repair hypothesis")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,record)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt = readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,aggregate_sha256=sha(Path(root)/ANALYSIS),
        report_path=str(report_path),report_sha256=sha(report_path))); return receipt


def closeout(root,registry,report_path):
    root = Path(root); receipt = read(root/"layer_group_repair_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value = dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],registry_readback=receipt,
        report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_modified=False)
    write(root/"layer_group_repair_finalization.json",value); return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Hide CUDA for CPU finalization")
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT); parser.add_argument("--report",type=Path,default=REPORT); parser.add_argument("--registry",type=Path,default=REGISTRY)
    args = parser.parse_args()
    if args.stage == "analyze": output = analyze(args.root)
    elif args.stage == "report": output = report(args.root,args.report)
    else:
        output = {"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback": write(args.root/"layer_group_repair_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
