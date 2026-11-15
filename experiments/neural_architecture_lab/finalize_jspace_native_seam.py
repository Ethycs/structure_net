"""Register the completed eleven-block stop and separate seam diagnosis; CPU only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_native_seam_record import record_for, require
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import require_committed, native_inventory
from .finalize_jspace_recovered_assembly import identity
from .finalize_jspace_qwen3b_progressive import parameter_count

ROOT,read,sha,write,logger=common.ROOT,common.read,common.sha,common.write,common.logger
ANALYSIS="native_seam_analysis.json"
ARCHIVE="native_seam_analysis_source"
TYPED=ROOT/"src/neural_architecture_lab/jspace_native_seam_record.py"
TEST=ROOT/"tests/neural_architecture_lab/test_jspace_native_seam.py"


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(common.__file__).resolve(),
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_recovered_assembly.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_qwen3b_progressive.py"]


def check_resources(manifest,peak):
    allocator=manifest["allocator"];uuid=allocator["physical_uuid"]
    require(bool(manifest["lease_id"]) and uuid.startswith("GPU-0370fdb0")
            and manifest["allocation"]=={uuid:11264} and allocator["lease_bytes"]==11264*1024**2
            and allocator["allocator_limit_bytes"]==11008*1024**2
            and 0<=peak<=allocator["allocator_limit_bytes"],"Reserved allocation/peak evidence changed")


def audit(kind,root):
    from experiments.structure_net import jspace_qwen3b_native_budget as producer
    root=Path(root).resolve(); small={};large={}
    def pin(path,digest=None,authenticated=False):
        path=Path(path).resolve()
        actual=digest if authenticated and digest is not None else sha(path)
        require(digest is None or actual==digest,"Changed evidence: "+str(path))
        if path.stat().st_size>64*1024**2:large[str(path)]=dict(sha256=actual,identity=identity(path))
        else:small[str(path)]=actual
    def pin_plan(plan,directory):
        pin(directory/"plan.json")
        for p,d in plan.get("input_sha256",{}).items():pin(p,d,authenticated=Path(p).stat().st_size>64*1024**2)
        for p,d in plan.get("source_sha256",{}).items():
            pin(p,d);pin(directory/"producer"/Path(p).relative_to(ROOT),d)
    def completed_phase(directory,phase,stage):
        result=producer.checked(directory,phase,stage)
        manifest=read(directory/phase/"manifest.json");pin(directory/phase/"manifest.json")
        check_resources(manifest,manifest["peak_reserved_bytes"])
        for p,d in manifest["artifacts"].items():pin(directory/phase/p,d,authenticated=True)
        return result,manifest
    require(kind in ("native","seam"),"Unknown closeout kind")
    if kind=="native":
        plan,stage=producer.verify_stage(root,25);pin_plan(plan,root);pin_plan(stage,root/"stages/block_25")
        phases={};manifests={};directory=root/"stages/block_25"
        for phase in producer.PHASES:phases[phase],manifests[phase]=completed_phase(directory,phase,stage)
        _,failed=producer.verify_stage(root,24);pin_plan(failed,root/"stages/block_24")
        failure_path=root/"stages/block_24/capture/failure.json";pin(failure_path)
        require(not (root/"stages/block_24/capture/manifest.json").exists(),"Next capture later completed; this stop record is stale")
        epochs=[]
        for h in phases["train"]["history"]:
            path=directory/"train"/f"epoch_{h['epoch']:02d}_development.json"
            pin(path,h["development_sha256"]);epochs.append(read(path))
        value=dict(plan=plan,stage_plan=stage,phases=phases,manifests=manifests,epochs=epochs,
            failure=read(failure_path),failed_stage_plan=failed,failed_cut=24,failed_phase="capture",failed_phase_completed=False)
        checkpoint=directory/"train/selected.pt";seconds=manifests["train"]["elapsed_seconds"]
        model_path=plan["model"]["path"]
    else:
        from experiments.structure_net import jspace_3b_seam_diagnostic as diagnostic
        plan=diagnostic.verify(root);pin_plan(plan,root)
        manifest=read(root/"run/manifest.json");pin(root/"run/manifest.json")
        require(manifest["status"]=="completed" and manifest["fingerprint"]==plan["fingerprint"]
                and "result.json" in manifest["artifacts"],"Completed matching diagnostic receipt required")
        for p,d in manifest["artifacts"].items():
            path=root/"run"/p
            require(path.resolve().is_relative_to((root/"run").resolve()),"Invalid artifact path")
            pin(path,d)
        campaign=Path(plan["campaign"]);source,stage=producer.verify_stage(campaign,25)
        require(source["fingerprint"]==plan["campaign_fingerprint"],"Diagnostic source campaign changed")
        pin_plan(source,campaign);pin_plan(stage,campaign/"stages/block_25")
        train,_=completed_phase(campaign/"stages/block_25","train",stage)
        fresh,_=completed_phase(campaign/"stages/block_25","verify",stage)
        require(plan["parent_checkpoint_sha256"]==train["selected_checkpoint_sha256"]==fresh["selected_checkpoint_sha256"],"Diagnostic selected parent changed")
        value=dict(plan=plan,evaluation=read(root/"run/result.json"),manifest=manifest,reference=fresh)
        check_resources(manifest,value["evaluation"]["peak_reserved_bytes"])
        checkpoint=campaign/"stages/block_25/train/selected.pt";seconds=0.;model_path=source["model"]["path"]
    _,parameters=parameter_count(checkpoint,native_inventory(model_path),25)
    value.update(artifact_sha256=small,authenticated_large_artifacts=large,
        model_parameters=parameters,model_checkpoint=str(checkpoint),training_seconds=seconds)
    return value,record_for(kind,value)


def analyze(kind,root,report):
    root=Path(root).resolve();report=Path(report).resolve()
    require(report.is_file() and not (root/ANALYSIS).exists() and not (root/ARCHIVE).exists(),"Missing report or existing closeout")
    value,record=audit(kind,root)
    for path in sources():
        target=root/ARCHIVE/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    value.update(schema="jspace.native-seam-analysis.v1",kind=kind,confirmed=False,record=record.to_dict(),
        created_at=datetime.now(timezone.utc).isoformat(),report_path=str(report),report_sha256=sha(report),
        analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/ANALYSIS,value)
    return dict(analysis=str(root/ANALYSIS),analysis_sha256=sha(root/ANALYSIS),study_id=record.study_id,
                classification=record.classification,primary_metric=record.primary_metric)


def expected(root):
    root=Path(root);value=read(root/ANALYSIS);record=record_for(value["kind"],value)
    require(value["schema"]=="jspace.native-seam-analysis.v1" and value["confirmed"] is False
            and value["record"]==record.to_dict(),"Typed record differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path)==digest,"Changed evidence/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items():
        require(identity(path)==pin["identity"],"Changed authenticated large artifact: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/ARCHIVE/Path(path).relative_to(ROOT))==digest,"Archived analysis source changed")
    require(sha(value["report_path"])==value["report_sha256"],"Committed report changed")
    return value,record


def metadata(root,record,kind):
    return dict(hypothesis_id=record.study_id,confirmed=False,confirmation_status=record.classification,
        primary_metric=record.primary_metric,primary_metric_name=("verified_compressed_layer_count" if kind=="native" else "singleton_boundary_exact_case_fraction"),
        primary_metric_unit="layers" if kind=="native" else "fraction",compressed_layer_count=11,
        correct=record.selected["correct"] if kind=="native" else record.control["correct"],count=64,
        full36_recovered=False,speed_measured=False,independent_confirmation_count=0,
        scientific_quality_rejection=False,aggregate_sha256=sha(Path(root)/ANALYSIS),scope=record.scope)


def readback(root,registry,key=None):
    value,r=expected(root);wanted=metadata(root,r,value["kind"])
    if key is None:
        saved=read(registry)
        require(saved["aggregate_sha256"]==sha(Path(root)/ANALYSIS) and saved["report_sha256"]==value["report_sha256"],"Registry/report mismatch")
        key=saved["storage"]["result_hash"]
    db=logger();row=db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(row["ids"]==[key] and all(row["metadatas"][0].get(k)==v for k,v in wanted.items()),"Exact experiment readback failed")
    h=db.hypotheses_collection.get(ids=[r.study_id],include=["metadatas"])
    require(h["ids"]==[r.study_id],"Hypothesis missing");m=h["metadatas"][0]
    require(m.get("tested") is True and m.get("confirmed") is False and m.get("evidence_count")==1
            and m.get("confirmation_status")==r.classification and json.loads(m["direct_result_hashes_json"])==[key],"Exact hypothesis readback failed")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry):
    value,r=expected(root);registry=Path(registry)
    require(not registry.exists(),"Registry already exists")
    require_committed([*sources(),Path(value["report_path"])])
    from neural_architecture_lab.core import ExperimentResult
    kind=value["kind"]
    hypothesis=dict(id=r.study_id,name="Qwen3B native-budget execution stop" if kind=="native" else "Qwen3B singleton seam and packed-kernel diagnosis",
        category="compression",tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,
        confirmation_status=r.classification,tested_scope=r.scope,description=r.scope)
    experiment=ExperimentResult(experiment_id="jspace-native-seam-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=r.study_id,
        metrics=dict(primary=r.primary_metric,development_accuracy=(r.selected["accuracy"] if kind=="native" else r.control["accuracy"])),
        primary_metric=r.primary_metric,model_architecture=[2048]*25+[1024]*11,
        model_parameters=value["model_parameters"],training_time=value["training_seconds"],
        model_checkpoint=value["model_checkpoint"],observations=[r.scope,r.classification,
            "Execution failure is not scientific rejection; diagnostic timing is not model throughput."])
    db=logger();prior=db.hypotheses_collection.get(ids=[r.study_id],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested evidence")
    db.log_hypothesis(hypothesis);key=db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,r,kind)])
    db.hypotheses_collection.update(ids=[r.study_id],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=value["report_path"])])
    receipt=readback(root,registry,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=value["report_path"],report_sha256=value["report_sha256"]))
    return receipt


def closeout(root,registry):
    root=Path(root);receipt=read(root/"native_seam_registry_readback.json")
    require(receipt==readback(root,registry) and receipt["readback_verified"] is True,"Independent readback changed")
    value,r=expected(root)
    result=dict(status="completed",study_id=r.study_id,confirmed=False,registry_path=str(registry),
        registry_readback=receipt,registry_sha256=sha(registry),aggregate_sha256=sha(root/ANALYSIS),
        report_path=value["report_path"],report_sha256=value["report_sha256"],
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_modified=False)
    write(root/"native_seam_finalization.json",result);return result


if __name__=="__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Hide CUDA for CPU finalization")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","store","readback","closeout"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--kind",choices=("native","seam"))
    parser.add_argument("--report",type=Path);parser.add_argument("--registry",type=Path)
    args=parser.parse_args()
    if args.stage=="analyze":result=analyze(args.kind,args.root,args.report)
    else:
        result={"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry)
        if args.stage=="readback":write(args.root/"native_seam_registry_readback.json",result)
    print(json.dumps(result,allow_nan=False))
