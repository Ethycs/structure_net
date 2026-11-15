"""CPU-only progress snapshots and terminal closeout of progressive Qwen3B."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil

import torch

from neural_architecture_lab.jspace_qwen3b_progressive_record import (
    JSpaceQwen3BProgressiveRecord, STUDY_ID, PRIMARY_METRIC_NAME, SCOPE, require,
)
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import require_committed, native_inventory
from .finalize_jspace_recovered_assembly import identity

ROOT,read,sha,write,logger=common.ROOT,common.read,common.sha,common.write,common.logger
ANALYSIS="qwen3b_progressive_analysis.json"
ARCHIVE="qwen3b_progressive_analysis_source"
TYPED=ROOT/"src/neural_architecture_lab/jspace_qwen3b_progressive_record.py"
TEST=ROOT/"tests/neural_architecture_lab/test_jspace_qwen3b_progressive.py"
PROTOCOL=ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-progressive-completion.md"


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,PROTOCOL,Path(common.__file__).resolve(),
        ROOT/"src/neural_architecture_lab/jspace_incremental_suffix_record.py",
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_recovered_assembly.py"]


def parameter_count(checkpoint,native,cut):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    names={"qkv_weight","qkv_bias","attention_output","gate_up_weight","mlp_output",
           "qkv_offset","gate_up_offset","polynomial_output","polynomial_bias"}
    trainable=sum(v.numel() for k,v in state["stack"].items() if isinstance(v,torch.Tensor) and k.rsplit(".",1)[-1] in names)
    trainable+=sum(v.numel() for v in state["norm"].values() if isinstance(v,torch.Tensor))
    prefix=sum(v["elements"] for k,v in native["tensors"].items() if k=="model.embed_tokens.weight" or
               any(k.startswith(f"model.layers.{i}.") for i in range(cut)))
    return trainable,prefix+trainable


def audit(root):
    from experiments.structure_net import jspace_qwen3b_progressive as producer
    root=Path(root).resolve();plan=producer.verify_root(root);artifacts={};large={}
    def pin(path,digest=None,authenticated=False):
        path=Path(path).resolve();actual=digest if authenticated and digest is not None else sha(path)
        require(digest is None or actual==digest,"Changed closeout input: "+str(path))
        if path.stat().st_size>64*1024**2:large[str(path)]=dict(sha256=actual,identity=identity(path))
        else:artifacts[str(path)]=actual
    pin(root/"plan.json")
    for path,digest in plan["input_sha256"].items():pin(path,digest,authenticated=Path(path).stat().st_size>64*1024**2)
    for path,digest in plan["source_sha256"].items():
        pin(path,digest);pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    data=producer.data_for(plan,29)
    try:native_rows=producer.old.rows_from_cache(data,plan["label_ids"])
    finally:data.close()
    seed_rows=producer.records(plan["seed_records"])
    seed_path=Path(plan["seed_root"])/"verification/result.json";pin(seed_path)
    seed_storage=read(seed_path)["storage"]
    native=native_inventory(plan["model"]["path"])
    _,seed_parameters=parameter_count(plan["seed_checkpoint"],native,29)
    seed_storage=dict(seed_storage,cut=29,compressed_layers=7,model_parameters=seed_parameters,
        full_model_reduction_fraction=1-seed_storage["full_model_tensor_bytes"]/seed_storage["native_full_tensor_bytes"])
    stages=[]
    for cut in range(28,-1,-1):
        directory=producer.stage_dir(root,cut)
        if not directory.exists():break
        _,stage=producer.verify_stage(root,cut);pin(directory/"plan.json")
        for path,digest in stage["input_sha256"].items():pin(path,digest,authenticated=Path(path).stat().st_size>64*1024**2)
        raw={};manifests={};failure=None;pending=None
        for phase in producer.PHASES:
            location=directory/phase;manifest_path=location/"manifest.json"
            if not manifest_path.exists():
                pending=phase
                if (location/"failure.json").exists():
                    pin(location/"failure.json");failure=read(location/"failure.json")
                    require(failure.get("failure_kind") in ("resource_failure","execution_failure"),"Typed failure classification required")
                break
            result=producer.checked(directory,phase,stage);manifest=read(manifest_path)
            pin(manifest_path)
            for name,digest in manifest["artifacts"].items():pin(location/name,digest,authenticated=True)
            allocation=manifest["allocation"];allocator=manifest["allocator"]
            require(allocation=={allocator["physical_uuid"]:plan["allocation_mib"]} and
                allocator["physical_uuid"].startswith("GPU-0370fdb0") and
                allocator["lease_bytes"]==plan["allocation_mib"]*1024**2 and
                allocator["allocator_limit_bytes"]==plan["allocator_mib"]*1024**2 and
                0<=manifest["peak_reserved_bytes"]<=allocator["allocator_limit_bytes"],"Reserved allocator/peak evidence failed")
            raw[phase]=result;manifests[phase]=manifest
        value=dict(plan=stage,raw_stages=raw,manifests=manifests,status="completed")
        if pending is not None:
            value.update(status="in_progress" if failure is None else failure["failure_kind"],phase=pending,
                         error=None if failure is None else failure["error"],failure=failure)
            stages.append(value);break
        train=raw["train"];checkpoint=directory/"train/selected.pt"
        count,total_parameters=parameter_count(checkpoint,native,cut)
        require(sha(checkpoint)==train["selected_checkpoint_sha256"],"Selected checkpoint hash changed")
        epochs=[];updates=[]
        for row in train["history"]:
            report=directory/"train"/f"epoch_{row['epoch']:02d}_development.json"
            pin(report,row["development_sha256"]);epochs.append(read(report));updates.extend(row["training"])
        require(len(updates)==384 and [r["update"] for r in updates]==list(range(1,385)) and
            all(r["cases"]==16 and r["gradients_present_by_block"]==[True]*(36-cut) and
                r["optimizer"]["updated_scalars"]==count and
                all(math.isfinite(r[k]) for k in ("loss","kl","ce")) and
                math.isfinite(r["optimizer"]["unclipped_gradient_norm"]) for r in updates),
            "384 finite all-block/norm updates required")
        require(read(directory/"train/initializer_replay.json")["passed"] is True,"Initializer replay failed")
        raw["verify"]=dict(raw["verify"],storage=dict(raw["verify"]["storage"],model_parameters=total_parameters))
        value.update(epochs=epochs,update_audit=dict(count=384,all_blocks_and_norm=True),
            selected_checkpoint=str(checkpoint),training_seconds=manifests["train"]["elapsed_seconds"])
        stages.append(value)
    outcome=None
    if (root/"outcome.json").exists():pin(root/"outcome.json");outcome=read(root/"outcome.json")
    value=dict(confirmed=False,plan=plan,native_records=native_rows,seed_records=seed_rows,seed_storage=seed_storage,
        stages=stages,outcome=outcome,artifact_sha256=artifacts,authenticated_large_artifacts=large)
    record=JSpaceQwen3BProgressiveRecord.from_evidence(value)
    if record.terminal and (root/"progress.jsonl").exists():pin(root/"progress.jsonl")
    return value,record


def analyze(root,*,snapshot=False):
    root=Path(root).resolve();value,record=audit(root)
    require(snapshot or record.terminal,"An active campaign permits explicit snapshots only")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name=f"qwen3b_progressive_snapshot_{stamp}.json" if snapshot else ANALYSIS
    archive=f"qwen3b_progressive_snapshot_source_{stamp}" if snapshot else ARCHIVE
    require(not (root/name).exists() and not (root/archive).exists(),"Closeout already exists")
    for path in sources():
        target=root/archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    value.update(schema="jspace.qwen3b-progressive-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),snapshot=bool(snapshot),analysis_archive=archive,
        analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/name,value)
    return dict(analysis=str(root/name),analysis_sha256=sha(root/name),classification=record.classification,terminal=record.terminal)


def expected(root,analysis=None):
    root=Path(root);path=Path(analysis) if analysis is not None else root/ANALYSIS
    value=read(path);record=JSpaceQwen3BProgressiveRecord.from_evidence(value)
    require(value["record"]==record.to_dict(),"Typed record differs from evidence")
    for name,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(name)==digest,"Changed evidence/source: "+name)
    for name,pin in value["authenticated_large_artifacts"].items():
        require(identity(name)==pin["identity"],"Changed authenticated large artifact: "+name)
    for name,digest in value["analysis_source_sha256"].items():
        require(sha(root/value["analysis_archive"]/Path(name).relative_to(ROOT))==digest,"Changed archived analysis source")
    return value,record


def report(root,path,analysis=None):
    value,r=expected(root,analysis);path=Path(path)
    control_scope=("Executed parent/native/seam control rows are retained and independently compared to their authenticated references."
        if r.stages and all(s["control_rows_independently_recomputed"] for s in r.stages) else
        "Some parent/native/seam controls are authenticated aggregate checks from the frozen producer; absent executed raw rows cannot be independently recomputed.")
    lines=["# Progressive Qwen2.5-3B suffix recovery","",
        f"**{r.accepted_compressed_layers}/36 blocks accepted; {r.accepted_correct}/64 reused development answers correct.** "
        f"Classification: `{r.classification}`. Confirmed: false.","",SCOPE,"",
        "The primary metric counts verified accepted compressed blocks. Accuracy, the immediate-parent and original-native five-percentage-point gates, and storage are reported separately. A resource or execution stop is not a completed quality failure.","",
        "| Cut | Blocks | Parent | Initial | Selected | Epoch | Minimum | Gate | Whole tensor bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    notes=[]
    for s in r.stages:
        lines.append(f"| {s['cut']} | {s['compressed_layers']} | {s['parent_correct']}/64 | {s['initial']['correct']}/64 | {s['selected']['correct']}/64 | {s['selected_epoch']} | {s['quality']['minimum_correct']}/64 | {'pass' if s['quality']['quality_pass'] else 'fail'} | {s['storage']['full_model_tensor_bytes']:,} |")
        p=s["selected"]["paired_to_parent"]
        notes += ["",f"Cut {s['cut']}: parent comparison gains {p['gains']}, loses {p['losses']} (net {p['net_correct']:+d}); epoch correct counts: "+", ".join(str(e["correct"]) for e in s["epochs"])+"."]
        notes += [f"Recovery targets: `{s['recovery_target_execution']}`. The separate differentiable calibration teacher scores {s['differentiable_teacher_correct']}/64; the accepted packed parent scores {s['parent_correct']}/64."]
    lines += notes+["",f"Last accepted model: cut {r.accepted_cut}, {r.accepted_storage['full_model_tensor_bytes']:,} persistent tensor bytes versus {r.accepted_storage['native_full_tensor_bytes']:,} native bytes ({100*r.accepted_storage['full_model_reduction_fraction']:.2f}% reduction). Includes fixed geometry and tied E once; excludes activations, KV cache and optimizer.","",
        "All36 recovery is reported only after verified passing cut0. The initial seven-block seed is51/64; the matched original BF16 teacher is48/64. Parent-relative budgets are local, while every stage also retains the original-native budget. A passing stage does not establish untouched-test accuracy or speed.","",
        "The accepted teacher uses the unchanged two-group/eight-probe full-vocabulary Jacobian estimator. Each twelve-epoch stage recomputes the entire student suffix, keeps geometry/readout fixed and adapts all suffix parameters plus final norm. Epoch0 remains eligible; accuracy selects first, full-vocabulary CE breaks ties.","",
        "When packed inference targets are present, they are stored separately from the original differentiable teacher cache and use the parent's development batch grouping. The calibration Jacobians remain unchanged; a rounding-dependent difference between the two execution paths is retained as a diagnostic.","",
        control_scope+" All selected-epoch and fresh-verification rows are retained and independently checked.","",
        f"Stop/progress: `{json.dumps(r.stop,sort_keys=True)}`.","",
        f"Evidence: `{analysis or Path(root)/ANALYSIS}`.",""]
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(path),report_sha256=sha(path))


def metadata(root,r):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=r.classification,
        primary_metric=r.primary_metric,primary_metric_name=PRIMARY_METRIC_NAME,primary_metric_unit="layers",
        accepted_cut=r.accepted_cut,compressed_layer_count=r.accepted_compressed_layers,
        correct=r.accepted_correct,count=64,accuracy=r.accepted_correct/64,terminal=r.terminal,
        full36_recovered=r.accepted_cut==0,completed_stages=len(r.stages),speed_measured=False,
        independent_confirmation_count=0,scientific_stage_completed_count=len(r.stages),
        aggregate_sha256=sha(Path(root)/ANALYSIS),scope=SCOPE)


def readback(root,registry,report_path,key=None):
    _,r=expected(root)
    if key is None:
        saved=read(registry);require(saved["aggregate_sha256"]==sha(Path(root)/ANALYSIS) and saved["report_sha256"]==sha(report_path),"Registry/report mismatch")
        key=saved["storage"]["result_hash"]
    db=logger();wanted=metadata(root,r);row=db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(row["ids"]==[key] and all(row["metadatas"][0].get(k)==v for k,v in wanted.items()),"Exact experiment readback failed")
    h=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(h["ids"]==[STUDY_ID],"Hypothesis missing");m=h["metadatas"][0]
    require(m.get("tested") is bool(r.stages) and m.get("confirmed") is False and m.get("evidence_count")==1 and
        m.get("confirmation_status")==r.classification and json.loads(m["direct_result_hashes_json"])==[key],"Exact hypothesis readback failed")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    value,r=expected(root)
    require(r.terminal and value["snapshot"] is False,"Snapshots/live campaigns cannot be registered")
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis=dict(id=STUDY_ID,name="Progressive one-block Qwen3B suffix recovery",category="compression",tested=bool(r.stages),
        confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=r.classification,tested_scope=SCOPE,
        description="Grow accepted seven-block suffix to all36 via fixed-geometry joint output-only recovery.")
    accepted=next((s for s in reversed(r.stages) if s["quality"]["quality_pass"]),None)
    checkpoint=value["plan"]["seed_checkpoint"] if accepted is None else accepted["selected_checkpoint"]
    experiment=ExperimentResult(experiment_id="jspace-qwen3b-progressive-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(accepted_compressed_layers=r.primary_metric,development_accuracy=r.accepted_correct/64),
        primary_metric=r.primary_metric,model_architecture=[2048]*r.accepted_cut+[1024]*r.accepted_compressed_layers,
        model_parameters=r.accepted_storage["model_parameters"],training_time=sum(s["training_seconds"] for s in r.stages),
        model_checkpoint=checkpoint,training_history=[dict(cut=s["cut"],accuracy=s["selected"]["accuracy"],accepted=s["quality"]["quality_pass"]) for s in r.stages],
        observations=[SCOPE,r.classification,"Full model includes remaining native prefix, fixed geometry and tied E once."])
    db=logger();prior=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested campaign")
    db.log_hypothesis(hypothesis);key=db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,r)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt=readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root,registry,report_path):
    root=Path(root);receipt=read(root/"qwen3b_progressive_registry_readback.json")
    require(receipt==readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    result=dict(status="completed",study_id=STUDY_ID,confirmed=False,registry_readback=receipt,
        report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_modified=False)
    write(root/"qwen3b_progressive_finalization.json",result);return result


if __name__=="__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Hide CUDA for CPU finalization")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--report",type=Path)
    parser.add_argument("--registry",type=Path);parser.add_argument("--snapshot",action="store_true")
    parser.add_argument("--analysis",type=Path);args=parser.parse_args()
    if args.stage=="analyze":result=analyze(args.root,snapshot=args.snapshot)
    elif args.stage=="report":result=report(args.root,args.report,args.analysis)
    else:
        result={"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage=="readback":write(args.root/"qwen3b_progressive_registry_readback.json",result)
    print(json.dumps(result,allow_nan=False))
