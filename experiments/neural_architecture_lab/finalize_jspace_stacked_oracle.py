"""CPU closeout for the completed teacher-guided projection diagnostic."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_stacked_oracle_record import JSpaceStackedOracleRecord,STUDY_ID,SCOPE,require
from .finalize_jspace_qwen_affine import ROOT,logger,read,sha,write
from .finalize_jspace_single_layer import require_committed
from .finalize_jspace_parallel_single_layer import identity

DEFAULT=ROOT/"data/experiments/jspace_stacked_projection_oracle/20261114_v1"
REPORT=ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-stacked-projection-oracle-results.md"
REGISTRY=ROOT/"data/hypothesis_registry/jspace_stacked_projection_oracle_20261114.json"
TYPED=ROOT/"src/neural_architecture_lab/jspace_stacked_oracle_record.py"
TEST=ROOT/"tests/neural_architecture_lab/test_jspace_stacked_oracle.py"
ANALYSIS="stacked_oracle_analysis.json"


def audit(root):
    from experiments.structure_net import jspace_stacked_projection_oracle as producer
    root=Path(root).resolve();plan=producer.verify(root)
    manifest=read(root/"evaluation/manifest.json");result=read(root/"evaluation/result.json")
    require(manifest["status"]=="completed" and manifest["fingerprint"]==plan["fingerprint"]
            and manifest["result_sha256"]==sha(root/"evaluation/result.json"),"Completed matching oracle result required")
    require(manifest["peak_reserved_bytes"]<=manifest["allocator"]["allocator_limit_bytes"],"Oracle exceeded reservation cap")
    artifacts={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or digest==expected,"Changed artifact: "+str(path));artifacts[str(path)]=digest
    for name in ("plan.json","evaluation/manifest.json","evaluation/result.json"):pin(root/name)
    for path,digest in plan["source_sha256"].items():
        pin(path,digest);pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    parent=producer.assembly.verify(plan["assembly_root"])
    require(parent["fingerprint"]==plan["assembly_fingerprint"],"Changed assembly provenance")
    require(plan["basis_paths"]==[parent["blocks"][str(i)]["fit_path"] for i in range(36)],"Oracle did not use the declared36 bases")
    for path in plan["basis_paths"]:
        require(plan["input_sha256"][path]==parent["input_sha256"][path],"Basis hash provenance differs")
    native_root=Path(plan["native_cache_root"]);native_plan=producer.cache.verify(native_root,completed=False)
    native_manifest=read(native_root/"cache/manifest.json")
    require(native_plan["fingerprint"]==plan["native_cache_fingerprint"] and native_manifest["status"]=="completed"
            and native_manifest["fingerprint"]==native_plan["fingerprint"],"Authenticated native cache lineage required")
    bank_names={f"cache/boundary_{i:02d}.npy" for i in range(37)}
    require(set(native_manifest["artifact_sha256"])==bank_names|{"cache/result.json","items.json"},"All37 original native boundaries required")
    for name,digest in native_manifest["artifact_sha256"].items():
        require(plan["input_sha256"][str(native_root/name)]==digest,"Native boundary hash provenance differs")
    for path in (native_root/"plan.json",native_root/"cache/manifest.json",native_root/"cache/result.json",native_root/"items.json"):
        pin(path,plan["input_sha256"].get(str(path)))
    assembly_root=Path(plan["assembly_root"])
    student=producer.assembly.completed_arm(assembly_root,parent,"all_compact")
    for name in ("plan.json","all_compact/manifest.json","all_compact/result.json"):pin(assembly_root/name)
    original=producer.recipe.verify(plan["source"])
    require(original["fingerprint"]==plan["source_fingerprint"],"Original model provenance changed")
    prior={path:dict(sha256=digest,identity=identity(path)) for path,digest in plan["input_sha256"].items()}
    value=dict(status="completed",confirmed=False,plan=plan,evaluation=result,evaluation_manifest=manifest,
        native_reference=read(native_root/"cache/result.json"),separate_student=student,
        separate_student_path=str(assembly_root/"all_compact/result.json"),artifact_sha256=artifacts,
        prior_authenticated_inputs=prior,
        input_authentication="The frozen producer preparation fully authenticated all 36 fit files and all 37 native FP32 boundary banks. Closeout matches their SHA256 pins to the assembly/cache manifests and verifies immutable file identities using existing producer validation, then records current device/inode/size/mtime/ctime. Small result/control metadata and live/archived sources are rehashed. Large bank/model payloads are not repeatedly reread.")
    return value,JSpaceStackedOracleRecord.from_evidence(value)


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(__file__).with_name("finalize_jspace_qwen_affine.py"),
        Path(__file__).with_name("finalize_jspace_single_layer.py"),Path(__file__).with_name("finalize_jspace_parallel_single_layer.py"),
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py"]


def analyze(root):
    root=Path(root);require(not (root/ANALYSIS).exists(),"Analysis already exists")
    value,record=audit(root);archive=root/"stacked_oracle_analysis_source"
    require(not archive.exists(),"Source archive exists")
    for path in sources():
        target=archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    value.update(schema="jspace.stacked-oracle-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",classification=record.classification,analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root=Path(root);value=read(root/ANALYSIS);record=JSpaceStackedOracleRecord.from_evidence(value)
    require(value["record"]==record.to_dict(),"Typed diagnostic differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path)==digest,"Changed diagnostic metadata/source: "+path)
    for path,pin in value["prior_authenticated_inputs"].items():
        require(identity(path)==pin["identity"],"Previously authenticated input changed: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/"stacked_oracle_analysis_source"/Path(path).relative_to(ROOT))==digest,"Archived closeout producer changed")
    return value,record


def report(root,report_path):
    root,report_path=Path(root),Path(report_path);value,r=expected(root);pair=r.teacher_guided["paired_to_native"]
    lines=["# Teacher-guided stacked projection oracle","",f"Study **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**","",
        f"The projected trajectory preserved **{round(r.primary_metric*64)}/64 native decisions ({100*r.primary_metric:.1f}% agreement)**. Both the native control and the teacher-guided oracle answered **{r.native['correct']}/64** correctly; there were **{pair['gains']} gains and {pair['losses']} losses**. This is a teacher-guided diagnostic, not accuracy of a learned compact model.",
        "","The positive result is that the fixed subspaces admit this projected teacher trajectory on these 64 cases while preserving every native decision. The geometry therefore does not inevitably destroy these decisions when supplied with suitable updates. It remains unproven whether the compact attention/MLP branches can learn those updates from their own states, or whether the result generalizes.",
        "","At each layer the calculation receives the ORIGINAL model's next hidden state and applies:","", "```text",
        "h_next = h + ((native_next - h) @ U) @ U.T", "```", "",
        "Here h is the accumulated oracle state, so each correction accounts for previous projection errors. The next native state is supplied at every layer; the learned attention/MLP branches do not produce it. The original readout scores the final trajectory. This is neither deployable inference nor a task-accuracy upper bound: other learned feature trajectories could make different decisions.",
        "","| Family | Native correct /16 | Teacher-guided correct /16 | Separate real student /16 |",
        "|---|---:|---:|---:|"]
    for family in sorted(r.native["by_family"]):
        lines.append(f"| {family} | {r.native['by_family'][family]['correct']} | {r.teacher_guided['by_family'][family]['correct']} | {r.separate_student['by_family'][family]['correct']} |")
    lines += ["",f"The final teacher-relative RMS discrepancy was **{r.final_relative_rms['valid_token_relative_rms']:.8f}** over valid tokens (equal-case aggregation) and **{r.final_relative_rms['answer_relative_rms']:.8f}** at answer positions. States and logits need not be identical for decisions to agree. The native control reproduced its reference with maximum logit difference **{r.native_control['logit_max_abs_delta']:.8g}**.",
        "","The valid-token metric is `sqrt(mean_cases(mean_valid_tokens(||oracle-native||² / max(||native||², 1e-20))))`; the answer metric uses each case's final valid row. All 37 boundary measurements are preserved in the analysis artifact.",
        "",f"For context only, the real assembled student before joint adaptation scored **27/64** in the separate artifact `{Path(value['separate_student_path']).relative_to(ROOT)}`. That student result and this oracle answer different questions. Ongoing joint-training epochs are excluded from this stabilized diagnostic.",
        "","The bases were fixed before this evaluation. The population contains the original 64 repeatedly inspected development questions, 16 from each family; no untouched confirmation examples were used. This study trains no parameters and measures no deployment speed or compression gain. Its primary registry metric is `teacher_guided_native_decision_agreement`, a fraction; 1.0 means native decision agreement, not 100% task accuracy.",
        "",f"Evidence: `{root.relative_to(ROOT)}/{ANALYSIS}`; SHA-256 `{sha(root/ANALYSIS)}`. {value['input_authentication']}",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(report_path),sha256=sha(report_path))


def metadata(root,r):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=r.classification,
        primary_metric=r.primary_metric,primary_metric_name="teacher_guided_native_decision_agreement",primary_metric_unit="fraction",
        teacher_guided_native_decision_agreement=r.primary_metric,teacher_guided_correct=r.teacher_guided["correct"],
        native_correct=50,count=64,learned_model=False,deployable=False,task_accuracy_upper_bound=False,independent_confirmations=0,
        scope=SCOPE,aggregate_sha256=sha(Path(root)/ANALYSIS))


def readback(root,registry,report_path,key=None):
    _,r=expected(root)
    if key is None:
        saved=read(registry)
        require(saved["aggregate_sha256"]==sha(Path(root)/ANALYSIS) and saved["report_sha256"]==sha(report_path),"Registry/report provenance mismatch")
        key=saved["storage"]["result_hash"]
    db=logger();wanted=metadata(root,r);actual=db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(actual["ids"]==[key] and all(actual["metadatas"][0].get(k)==v for k,v in wanted.items()),"Exact diagnostic readback mismatch")
    h=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(h["ids"]==[STUDY_ID],"Hypothesis missing")
    row=h["metadatas"][0]
    require(row.get("tested") is True and row.get("confirmed") is False and row.get("evidence_count")==1
            and row.get("confirmation_status")==r.classification
            and json.loads(row["direct_result_hashes_json"])==[key],"Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    value,r=expected(root);require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis=dict(id=STUDY_ID,name="Teacher-guided decision retention through stacked fixed update subspaces",category="interpretability",
        tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=r.classification,
        tested_scope=SCOPE,description="Original next-layer targets supplied at every projection; a geometric diagnostic, not learned recovery.")
    experiment=ExperimentResult(experiment_id="jspace-stacked-oracle-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(teacher_guided_native_decision_agreement=r.primary_metric,teacher_guided_correct=r.teacher_guided["correct"],
            native_correct=50,**r.final_relative_rms),primary_metric=r.primary_metric,
        model_architecture=[],model_parameters=0,training_time=0.,observations=[SCOPE,r.classification,
            "Canonical model_parameters and training_time are zero because no model is trained; these are not deployment size/speed measurements."])
    db=logger();prior=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis);key=db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,r)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt=readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root,registry,report_path):
    root=Path(root);receipt=read(root/"stacked_oracle_registry_readback.json")
    require(receipt==readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value=dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],
        registry_readback=receipt,report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_append="Delegated to root; no concurrent log mutation")
    write(root/"stacked_oracle_finalization.json",value);return value


if __name__=="__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Hide CUDA for CPU finalization")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT);parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--registry",type=Path,default=REGISTRY);args=parser.parse_args()
    if args.stage=="analyze":output=analyze(args.root)
    elif args.stage=="report":output=report(args.root,args.report)
    else:
        output={"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage=="readback":write(args.root/"stacked_oracle_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
