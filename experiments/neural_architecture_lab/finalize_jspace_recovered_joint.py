"""CPU-only closeout after joint recovery AND independent verification finish."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch

from neural_architecture_lab.jspace_recovered_joint_record import JSpaceRecoveredJointRecord,STUDY_ID,SCOPE,require,scores
from .finalize_jspace_qwen_affine import ROOT,logger,read,sha,write
from .finalize_jspace_single_layer import native_inventory,tensor_inventory,require_committed
from .finalize_jspace_parallel_single_layer import identity

DEFAULT = ROOT/"data/experiments/jspace_recovered_joint/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-recovered-joint-distillation-results.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_recovered_joint_20261114.json"
TYPED = ROOT/"src/neural_architecture_lab/jspace_recovered_joint_record.py"
TEST = ROOT/"tests/neural_architecture_lab/test_jspace_recovered_joint.py"
ANALYSIS = "recovered_joint_analysis.json"


def selected_inventory(directory,model_path):
    manifest = read(Path(directory)/"manifest.json");native = native_inventory(model_path)
    expected = {f"block_{i:02d}.pt" for i in range(36)}|{"norm_gain.pt"}
    require(manifest["blocks"] == 36 and set(manifest["artifacts"]) == expected,"All36 selected blocks and final norm required")
    names = {"wrapped.original."+n for n in
        ("qkv_weight","qkv_bias","attention_output","gate_up_weight","mlp_output","qkv_offset","gate_up_offset")}
    counts = [];parameters = 0
    for layer in range(36):
        state = torch.load(Path(directory)/f"block_{layer:02d}.pt",map_location="cpu",weights_only=True,mmap=True)
        require(state["wrapped._extra_state"] == dict(version=1,mode="softmax",metric_active=False),"Selected block architecture changed")
        require(names <= set(state) and all(v.dtype == torch.float32 for v in state.values() if isinstance(v,torch.Tensor)),"FP32 compact parameters and geometry required")
        require(state["wrapped.original.basis"].shape == (2048,1024),"Fixed half-rank geometry required")
        inventory = tensor_inventory(state);parameters += sum(state[n].numel() for n in names)
        counts.append(dict(layer=layer,tensor_bytes=inventory["tensor_bytes"],tensor_count=inventory["tensor_count"]))
    norm = torch.load(Path(directory)/"norm_gain.pt",map_location="cpu",weights_only=True,mmap=True)
    require(norm.shape == (2048,) and norm.dtype == torch.float32,"FP32 selected final norm required")
    embedding = native["tensors"]["model.embed_tokens.weight"]["elements"]
    return dict(all_selected_fp32=True,selected_blocks=36,tied_embedding_readout_counted_once=True,
        model_parameters=parameters+norm.numel()+embedding,trainable_parameters=parameters+norm.numel(),
        parameter_tensor_count=36*7+1,unpacked_persistent_tensor_bytes=sum(c["tensor_bytes"] for c in counts)+4*(norm.numel()+embedding),
        block_inventories=counts,embedding_parameters=embedding,native_parameter_count=native["parameter_count"],
        scope="Actual selected UNPACKED block tensors plus final norm and original tiedE once, allFP32. Includes saved basis/geometry; excludes transient rotary tables, activations, KV cache and optimizer. No packed export or speed measurement.")


def audit(root):
    from experiments.structure_net import jspace_recovered_joint as producer
    from experiments.structure_net import jspace_recovered_teacher as teacher_module
    root = Path(root).resolve();plan = producer.verify(root)
    producer.verify_pilot(root,plan);train = producer.completed_training(root,plan)
    artifacts,large = {},{}
    def pin(path,expected=None):
        path = Path(path).resolve();digest = sha(path)
        require(expected is None or digest == expected,"Changed artifact: "+str(path))
        if path.stat().st_size > 64*1024**2:large[str(path)] = dict(sha256=digest,identity=identity(path))
        else:artifacts[str(path)] = digest
        return digest
    for name in ("plan.json","items.json","events.jsonl","train/initial_control.json"):
        pin(root/name)
    initial_control = read(root/"train/initial_control.json")
    require(initial_control["passed"] is True and initial_control["correct"] == 27,"Scientific warm start control failed")
    stage_manifests = {}
    for stage in ("pilot","train","verification"):
        directory = root/stage;manifest = read(directory/"manifest.json")
        pin(directory/"manifest.json");pin(directory/"result.json",manifest["result_sha256"])
        require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"],"Matching completed stage required")
        require(manifest["allocator"]["allocation"] == manifest["allocation"],"Allocator/lease mismatch")
        for row in manifest["allocator"]["settings"]:
            peak = manifest["device_peaks"][row["physical_uuid"]]
            require(peak["peak_reserved_bytes"] <= row["allocator_limit_bytes"] < row["lease_bytes"],"GPU peak exceeded reservation cap")
        stage_manifests[stage] = manifest
    require(stage_manifests["verification"]["passed"] is True,"Fresh verification not completed successfully")
    selected = root/"train/selected";selected_manifest = read(selected/"manifest.json")
    pin(selected/"manifest.json",train["selected_manifest_sha256"])
    require(selected_manifest["metadata"] == dict(epoch=train["selected_epoch"],development=train["selected"]),"Selected metadata changed")
    for name,digest in selected_manifest["artifacts"].items():pin(selected/name,digest)
    epochs = []
    for row in train["history"]:
        path = root/"train"/f"epoch_{row['epoch']:02d}_development.json";pin(path,row["development_sha256"])
        epochs.append(dict(epoch=row["epoch"],**read(path)))
    teacher_root = Path(plan["teacher_root"]);teacher_plan = teacher_module.verify(teacher_root,completed=True)
    require(teacher_plan["fingerprint"] == plan["teacher_fingerprint"] and teacher_plan["items_digest"] == plan["items_digest"],"Teacher identity/provenance changed")
    teacher_manifest = read(teacher_root/"cache/manifest.json")
    for name in ("plan.json","cache/manifest.json"):pin(teacher_root/name)
    # verify(completed=True) above already hashed the full logit bank.
    for name,digest in teacher_manifest["artifact_sha256"].items():
        path = teacher_root/name
        if path.stat().st_size > 64*1024**2:large[str(path)] = dict(sha256=digest,identity=identity(path))
        else:pin(path,digest)
    assembly_root = Path(plan["assembly_root"]);assembly_plan = producer.assembly.verify(assembly_root)
    require(assembly_plan["fingerprint"] == plan["assembly_fingerprint"],"Assembly lineage changed")
    assembly = {}
    for arm in producer.assembly.ARMS:
        assembly[arm] = producer.assembly.completed_arm(assembly_root,assembly_plan,arm)
        for name in ("manifest.json","result.json"):pin(assembly_root/arm/name)
    pin(assembly_root/"plan.json");pin(assembly_root/"paired_result.json")
    require(assembly["native"]["control"]["passed"] is True and assembly["accepted_two"]["control"]["passed"] is True
            and assembly["all_compact"]["accuracy"]["correct"] == 27,"Completed native/accepted/initializer controls required")
    for path,digest in plan["source_sha256"].items():
        pin(path,digest);pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    storage = selected_inventory(selected,plan["model"]["path"])
    updates = [r for r in map(json.loads,(root/"events.jsonl").read_text().splitlines()) if r.get("kind") == "update" and r.get("stage") == "train"]
    require([r["update"] for r in updates] == list(range(1,97))
            and [r["epoch"] for r in updates] == [epoch for epoch in (1,2,3) for _ in range(32)]
            and all(r["cases"] == 16 and r["gradient_parameter_count"] == storage["parameter_tensor_count"]
                and r["gradient_scalar_count"] == storage["trainable_parameters"] for r in updates),
            "All96 training updates must cover every trainable block/norm tensor")
    value = dict(status="completed",confirmed=False,plan=plan,training=train,pilot=read(root/"pilot/result.json"),
        verification=read(root/"verification/result.json"),stage_manifests=stage_manifests,epochs=epochs,
        teacher=read(teacher_root/"cache/result.json"),teacher_plan=teacher_plan,teacher_manifest=teacher_manifest,
        assembly_controls={arm:dict(correct=v["accuracy"]["correct"],count=64) for arm,v in assembly.items()},
        storage=storage,artifact_sha256=artifacts,authenticated_large_artifacts=large,
        update_audit=dict(count=96,gradient_parameter_tensors=storage["parameter_tensor_count"],gradient_scalars=storage["trainable_parameters"]),
        input_authentication="Selected37 model files and teacher logit bank authenticated by fullSHA256. Stage, history and live/archived producer bytes authenticated. Prior assembly/teacher/native inputs retain original pins and immutable file-identity checks. Subsequent closeout processes recheck small hashes and unchanged identities of already-hashed large artifacts.")
    return value,JSpaceRecoveredJointRecord.from_evidence(value)


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(__file__).with_name("finalize_jspace_single_layer.py"),
        Path(__file__).with_name("finalize_jspace_parallel_single_layer.py"),Path(__file__).with_name("finalize_jspace_qwen_affine.py"),
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py"]


def analyze(root):
    root = Path(root);require(not (root/ANALYSIS).exists(),"Analysis already exists")
    value,record = audit(root);archive = root/"recovered_joint_analysis_source"
    require(not archive.exists(),"Analysis source archive exists")
    for path in sources():
        target = archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    value.update(schema="jspace.recovered-joint-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",classification=record.classification,analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root = Path(root);value = read(root/ANALYSIS);record = JSpaceRecoveredJointRecord.from_evidence(value)
    require(value["record"] == record.to_dict(),"Typed record differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path) == digest,"Changed closeout artifact/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items():
        require(identity(path) == pin["identity"],"Previously authenticated artifact changed: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/"recovered_joint_analysis_source"/Path(path).relative_to(ROOT)) == digest,"Changed archived closeout source")
    return value,record


def report(root,report_path):
    root,report_path = Path(root),Path(report_path);value,r = expected(root)
    lines = ["# Joint recovery of the recovered Qwen assembly","",f"Study **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**","",
        f"The selected full-model checkpoint scored **{r.selected['correct']}/64 ({100*r.primary_metric:.4f}%)**, compared with **27/64** for its recovered-but-unadapted assembly and **50/64** for the original FP32 model. The fixed development gate was **47/64**; it {'passed' if r.quality_pass else 'did not pass'}. These are actual assembled-model task scores on the original64 development cases.",
        "",f"All3 epochs and96 optimizer updates completed on512 training cases. Selection chose epoch{r.selected_epoch}, using highest task accuracy then lowest conditional-choice CE. The initializer remained eligible and {'was selected' if r.initializer_selected else 'was not selected'}. The separate one-update pilot was discarded. Every scientific update reported gradients for all36 compact blocks and final RMS gain; the pilot independently showed a parameter change in every block while fixed geometry stayed unchanged.",
        "","| Epoch | Correct /64 | Accuracy | Choice CE | Choice KL | Full KL | Answer-label mass | Teacher agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in r.epochs:
        m = row["metrics"]
        lines.append(f"| {row['epoch']} | {row['correct']} | {100*row['accuracy']:.4f}% | {m['choice_ce']:.7g} | {m['choice_kl']:.7g} | {m['full_kl']:.7g} | {m['valid_label_probability']:.7g} | {100*m['teacher_agreement']:.4f}% |")
    lines += ["","| Family | Native /16 | Initial /16 | Selected /16 |","|---|---:|---:|---:|"]
    for family in sorted(r.native["by_family"]):
        lines.append(f"| {family} | {r.native['by_family'][family]['correct']} | {r.initial['by_family'][family]['correct']} | {r.selected['by_family'][family]['correct']} |")
    pair = r.selected["paired_to_initial"];native_pair = r.selected["paired_to_native"]
    lines += ["",f"Against initialization, the selected checkpoint gained{pair['gains']} correct answers and lost{pair['losses']}. Against native, it gained{native_pair['gains']} and lost{native_pair['losses']}. Choice margin changed from{r.initial['metrics']['choice_margin']:.7g} to{r.selected['metrics']['choice_margin']:.7g}. Full-vocabulary answer CE is recorded separately from the optimized conditional-choice CE.",
        "","The matched original teacher was captured with original BF16-stored weights promoted to FP32, using full FP32 forward/readout and no TF32. It reproduced native50/64 with exact predictions and the frozen logit tolerance. Cached targets contain full vocabulary logits at each prompt's midpoint and answer. They provide supervision only: every student minibatch starts from raw tokens and recomputes all36 blocks.",
        "","The fixed objective was `.2 T² KL_full + .5 T² KL_choice + .3 CE_choice`, with T=2. Full KL covers midpoint and answer; choice KL/CE use the answer position and its valid choices. Rank1024 updates, native2048 residual carry,128-dimensional heads, softmax and11008 MLP activation width remained unchanged. Original tiedE, bases, anchors and other geometry were frozen; all block parameters and final RMS gain trained in FP32. Blocks0–23 plus tiedE/norm ran on RTX3060; blocks24–35 ran on RTX2060 SUPER. Gradients crossed the partition and nonreentrant activation checkpoints recomputed student blocks.",
        "",f"Fresh-process verification authenticated the selected checkpoint, strictly reloaded it and reproduced all64 predictions plus metrics/logits within the frozen tolerances, with fixed state unchanged. The selected unpacked model contains{r.storage['model_parameters']:,} parameters and{r.storage['unpacked_persistent_tensor_bytes']:,} saved FP32 tensor bytes including basis/geometry and tiedE once. These counts are not a speed result or a packed deployment measurement. The earlier BF16 stored model is not used as a matched-FP32 size denominator here.",
        "","This is one seed on repeatedly inspected development cases. It does not establish held-out quality, unrestricted generation quality or throughput. No untouched confirmation data were opened, and no packing-parity or inference-speed test is claimed. The selected checkpoint and completed failure outcomes, if any, are preserved; changing the training schedule or objective requires a separate continuation.",
        "",f"Evidence: `{root.relative_to(ROOT)}/{ANALYSIS}`; SHA-256 `{sha(root/ANALYSIS)}`. {value['input_authentication']}",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(report_path),sha256=sha(report_path))


def metadata(root,r):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=r.classification,
        primary_metric=r.primary_metric,primary_metric_name="selected_full_model_development_accuracy",accuracy=r.primary_metric,
        correct=r.selected["correct"],count=64,initial_correct=27,native_correct=50,minimum_correct=47,
        quality_pass=r.quality_pass,selected_epoch=r.selected_epoch,initializer_selected=r.initializer_selected,
        updates=96,trained_blocks=36,independent_confirmations=0,speed_measured=False,packing_validated=False,
        scope=SCOPE,aggregate_sha256=sha(Path(root)/ANALYSIS))


def readback(root,registry,report_path,key=None):
    _,r = expected(root)
    if key is None:
        saved = read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path),"Registry/report provenance mismatch")
        key = saved["storage"]["result_hash"]
    db = logger();wanted = metadata(root,r);actual = db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(actual["ids"] == [key] and all(actual["metadatas"][0].get(k) == v for k,v in wanted.items()),"Exact experiment readback mismatch")
    h = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(h["ids"] == [STUDY_ID],"Hypothesis missing")
    row = h["metadatas"][0]
    require(row.get("tested") is True and row.get("confirmed") is False and row.get("evidence_count") == 1
            and row.get("confirmation_status") == r.classification
            and json.loads(row["direct_result_hashes_json"]) == [key],"Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    value,r = expected(root);require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="Joint task recovery from the selected compact Qwen assembly",category="compression",
        tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=r.classification,
        tested_scope=SCOPE,description="Three-epoch joint FP32 output/choice distillation of the actual recovered36-block assembly.")
    experiment = ExperimentResult(experiment_id="jspace-recovered-joint-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(accuracy=r.primary_metric,initial_accuracy=27/64,native_accuracy=50/64,
            choice_ce=r.selected["metrics"]["choice_ce"],choice_kl=r.selected["metrics"]["choice_kl"],full_kl=r.selected["metrics"]["full_kl"]),
        primary_metric=r.primary_metric,model_architecture=[2048]*36,model_parameters=r.storage["model_parameters"],
        training_time=value["stage_manifests"]["train"]["elapsed_seconds"],
        training_history=[dict(epoch=row["epoch"],accuracy=row["accuracy"],choice_ce=row["metrics"]["choice_ce"]) for row in r.epochs],
        model_checkpoint=value["training"]["selected_directory"],observations=[SCOPE,r.classification])
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
    root = Path(root);receipt = read(root/"recovered_joint_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value = dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],
        registry_readback=receipt,report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_append="Delegated to root; no concurrent log mutation")
    write(root/"recovered_joint_finalization.json",value);return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Hide CUDA for CPU finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT);parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--registry",type=Path,default=REGISTRY);args = parser.parse_args()
    if args.stage == "analyze":output = analyze(args.root)
    elif args.stage == "report":output = report(args.root,args.report)
    else:
        output = {"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback":write(args.root/"recovered_joint_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
