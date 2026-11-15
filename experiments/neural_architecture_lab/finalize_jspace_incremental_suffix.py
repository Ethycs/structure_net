"""CPU-only, append-only closeout of a one-block extension of the accepted suffix."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil

import torch

from neural_architecture_lab.jspace_incremental_suffix_record import (
    JSpaceIncrementalSuffixRecord, STUDY_ID, SCOPE, LAYERS, ATOL, RTOL, quality_gates, require,
)
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import native_inventory, tensor_inventory, require_committed
from .finalize_jspace_recovered_assembly import identity

ROOT, read, sha, write, logger = common.ROOT, common.read, common.sha, common.write, common.logger
DEFAULT = ROOT / "data/experiments/jspace_incremental_suffix/20261115_v1"
REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-15_qwen-incremental-suffix-results.md"
REGISTRY = ROOT / "data/hypothesis_registry/jspace_incremental_suffix_20261115.json"
TYPED = ROOT / "src/neural_architecture_lab/jspace_incremental_suffix_record.py"
TEST = ROOT / "tests/neural_architecture_lab/test_jspace_incremental_suffix.py"
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-15_qwen-incremental-suffix-recovery.md"
ANALYSIS, ARCHIVE = "incremental_suffix_analysis.json", "incremental_suffix_analysis_source"


def sources():
    return [Path(__file__).resolve(), TYPED, TEST, PROTOCOL, Path(common.__file__).resolve(),
            ROOT / "src/neural_architecture_lab/jspace_single_layer_record.py",
            ROOT / "experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
            ROOT / "experiments/neural_architecture_lab/finalize_jspace_recovered_assembly.py"]


def selected_inventory(checkpoint, model_path, packed_bytes, *, rotary_bytes=0):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    require(set(state) == {"stack", "norm"}, "Selected checkpoint must contain the suffix and final norm")
    flat = {f"{part}.{key}": value for part in ("stack", "norm") for key,value in state[part].items()}
    inventory = tensor_inventory(flat); native = native_inventory(model_path)
    prefix = {k:v for k,v in native["tensors"].items() if k == "model.embed_tokens.weight"
              or any(k.startswith(f"model.layers.{i}.") for i in range(29))}
    require(all(v["dtype"] == "BF16" for v in prefix.values()), "Native prefix/readout must retain original BF16 storage")
    names = {"qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "qkv_offset",
             "gate_up_offset", "polynomial_output", "polynomial_bias"}
    parameters = sum(v.numel() for k,v in state["stack"].items()
                     if isinstance(v,torch.Tensor) and k.rsplit(".",1)[-1] in names)
    parameters += sum(v.numel() for v in state["norm"].values() if isinstance(v,torch.Tensor))
    return dict(tied_embedding_readout_counted_once=True, native_prefix_layers=list(range(29)),
        compact_layers=LAYERS, selected_suffix_tensor_bytes=inventory["tensor_bytes"],
        selected_checkpoint_file_bytes=Path(checkpoint).stat().st_size, packed_suffix_tensor_bytes=packed_bytes,
        native_prefix_and_embedding_bytes=sum(v["bytes"] for v in prefix.values()),
        whole_model_tensor_bytes=sum(v["bytes"] for v in prefix.values())+packed_bytes+rotary_bytes,
        native_model_tensor_bytes=native["source_tensor_bytes"]+rotary_bytes,rotary_bytes=rotary_bytes,
        model_parameters=sum(v["elements"] for v in prefix.values())+parameters,
        selected_suffix_parameters=parameters,
        scope="Persistent packed suffix and trained norm, plus 29 frozen native blocks and tied embedding/readout once. Includes fixed geometric buffers; excludes activations, KV cache and optimizer. Checkpoint file bytes are separate. No throughput measurement.")


def audit(root):
    from experiments.structure_net import jspace_incremental_suffix as producer
    root=Path(root).resolve();plan=producer.verify(root)
    artifacts,large,manifests,raw={},{},{},{}
    def pin(path,digest=None,*,authenticated=False):
        path=Path(path).resolve()
        checksum=digest if authenticated and digest is not None else sha(path)
        require(digest is None or checksum == digest,"Changed closeout input: "+str(path))
        if path.stat().st_size > 64*1024**2:large[str(path)]=dict(sha256=checksum,identity=identity(path))
        else:artifacts[str(path)]=checksum
    for name in ("plan.json","items.json","progress.jsonl"):pin(root/name)
    require(plan["confirmed"] is False and plan["logit_atol"] == ATOL and plan["logit_rtol"] == RTOL
            and plan["metric_atol"] == 1e-7 and plan["metric_rtol"] == 1e-6,
            "Exploratory protocol and distinct logit/metric tolerances changed")
    for stage in ("pilot","train","verification"):
        raw[stage]=producer.checked(root,stage,plan);manifest=read(root/stage/"manifest.json");manifests[stage]=manifest
        pin(root/stage/"manifest.json")
        for name,digest in manifest["artifacts"].items():pin(root/stage/name,digest,authenticated=True)
        allocation=manifest["allocation"];allocator=manifest["allocator"]
        require(allocation == {allocator["physical_uuid"]:8192} and allocator["physical_uuid"].startswith("GPU-0370fdb0")
            and allocator["lease_bytes"] == 8192*1024**2 and allocator["allocator_limit_bytes"] == 7936*1024**2
            and 0 <= manifest["peak_reserved_bytes"] <= allocator["allocator_limit_bytes"],
            "Reserved3060 peak or allocator limit failed")
    pilot,train,fresh=(raw[k] for k in ("pilot","train","verification"))
    require(all(v["confirmed"] is False for v in raw.values()) and manifests["verification"]["passed"] is True
        and fresh["passed"] is True and fresh["metrics_pass"] is True and fresh["packing_validated"] is True,
        "Fresh packed verification is required even when scientific quality misses")
    require(pilot["controls"]["passed"] is True and pilot["checkpoint_strict_load"] is True
        and pilot["fixed_state"] == train["fixed_state"],"Pilot lifecycle or fixed geometry changed")
    # verify() checks every immutable identity against the original CPU prepare;
    # the prepare hashed all consumed banks. Do not reread multi-GB banks per closeout.
    for path,digest in plan["input_sha256"].items():
        pin(path,digest,authenticated=Path(path).stat().st_size > 64*1024**2)
    for path,digest in plan["source_sha256"].items():
        pin(path,digest);pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    require(plan["label_ids"] == [32,33,34,35],"Cached control label IDs changed")
    data=producer.dataset(plan)
    try:native_reference=producer.rows_from_cache(data,plan["label_ids"])
    finally:data.close()
    parent_reference=read(plan["parent_records"])
    controls={}
    for name in ("native","parent","seam"):
        c=pilot["controls"][name]
        require(c["atol"] == ATOL and c["rtol"] == RTOL,"Control tolerances changed")
        controls[name]=dict(passed=c["passed"],records=c["records"],
            metrics={k:c[k] for k in ("correct","count","accuracy")},
            reference_records=native_reference if name == "native" else parent_reference)
    diagnostic=pilot["controls"]["teacher_cache_diagnostic"]
    require(diagnostic["predictions_exact"] is True and diagnostic["correct"] == 51
        and diagnostic["logit_gate_applied"] is False,"Historical teacher-cache distinction was lost")
    inherited=pilot["inheritance"]
    require(inherited == dict(passed=True,unchanged_layers=list(range(30,36)),interior_bridges_preserved=5,
                             new_layer=29,new_bridge=[29,30],norm_exact=True),"Parent inheritance changed")
    update=pilot["update"]
    require(update["gradients_present_by_block"] == [True]*7 and math.isfinite(update["optimizer"]["unclipped_gradient_norm"])
            and update["optimizer"]["unclipped_gradient_norm"] > 0
            and pilot["parameter_change_gate"] == "FP32 optimizer masters; BF16 stored changes are separate rounding diagnostics",
            "Pilot gradient or optimizer-master evidence missing")
    normalized_pilot=dict(systems_only=pilot["systems_only"],updates=pilot["updates"],
        master_changed_layers=pilot["changed_layers"],norm_master_changed=pilot["norm_changed"],
        stored_changed_layers=pilot["bf16_changed_layers"],stored_norm_changed=pilot["bf16_norm_changed"],
        gradient_layers=[layer for layer,present in zip(LAYERS,update["gradients_present_by_block"]) if present],
        norm_gradient=pilot["norm_changed"],fixed_state_unchanged=pilot["fixed_state_unchanged"],
        parent_inheritance_exact=inherited["passed"],initial_records=pilot["initial_records"])
    require(train["completed_epochs"] == 12 and train["config"] == plan["config"],"Scientific schedule changed")
    checkpoint=root/"train/selected.pt";pin(checkpoint,train["selected_checkpoint_sha256"],authenticated=True)
    epochs=[]
    for row in train["history"]:
        path=root/"train"/f"epoch_{row['epoch']:02d}_development.json";pin(path,row["development_sha256"])
        epochs.append(dict(epoch=row["epoch"],**read(path)))
    replay=read(root/"train/initializer_replay.json")
    require(replay["passed"] is True,"Scientific initialization differed from discarded pilot")
    require(train["quality"] == producer.gates(train["selected"]["correct"])
        and train["quality_pass"] is train["quality"]["passed"] and fresh["quality"] == train["quality"],
        "Producer dual quality gates changed")
    normalized_train=dict(train,gates=quality_gates(train["selected"]["correct"],48,51))
    require(all(fresh["comparison"][k] is True for k in ("passed","predictions_exact","logits_pass")),"Fresh packed comparison failed")
    require(fresh["comparison"]["atol"] == ATOL and fresh["comparison"]["rtol"] == RTOL,"Fresh logit tolerances changed")
    normalized_fresh=dict(fresh,fingerprint=manifests["verification"]["fingerprint"],packed=fresh["packing_validated"],
                          quality_pass=fresh["quality"]["passed"])
    deployed=fresh["storage"]
    storage=selected_inventory(checkpoint,plan["model"]["path"],deployed["packed_suffix_and_norm_bytes"],rotary_bytes=deployed["rotary_bytes"])
    require(storage["native_prefix_and_embedding_bytes"] == deployed["frozen_native_prefix_bytes"]+deployed["tied_embedding_readout_bytes"]
        and storage["whole_model_tensor_bytes"] == deployed["full_model_tensor_bytes"]
        and storage["native_model_tensor_bytes"] == deployed["native_full_tensor_bytes"],"Complete deployed tensor accounting differs")
    updates=[v for v in map(json.loads,(root/"progress.jsonl").read_text().splitlines()) if v.get("kind") == "update"]
    require([v["update"] for v in updates] == list(range(1,385))
        and [v["epoch"] for v in updates] == [e for e in range(1,13) for _ in range(32)]
        and all(v["cases"] == 16 and v["gradients_present_by_block"] == [True]*7
            and v["optimizer"]["updated_scalars"] == storage["selected_suffix_parameters"]
            and all(math.isfinite(v[k]) for k in ("loss","kl","ce"))
            and math.isfinite(v["optimizer"]["unclipped_gradient_norm"]) for v in updates),
        "All384 finite updates must include every block and final norm")
    value=dict(status="completed",confirmed=False,plan=plan,controls=controls,pilot=normalized_pilot,
        training=normalized_train,epochs=epochs,verification=normalized_fresh,storage=storage,raw_stages=raw,
        stage_manifests=manifests,teacher_cache_diagnostic=diagnostic,selected_checkpoint=str(checkpoint),
        update_audit=dict(count=384,all_blocks_and_norm=True,updated_scalars=storage["selected_suffix_parameters"]),
        artifact_sha256=artifacts,authenticated_large_artifacts=large,
        input_authentication="Prepared source/input SHA closure, immutable large-bank identities, all stage and epoch hashes, exact selected checkpoint, and live/archived producers authenticated. No dataset fitting or GPU operation occurs in closeout.")
    return value,JSpaceIncrementalSuffixRecord.from_evidence(value)


def analyze(root):
    root=Path(root).resolve()
    require(not (root/ANALYSIS).exists() and not (root/ARCHIVE).exists(), "Closeout already attempted")
    value,record=audit(root)
    for path in sources():
        target=root/ARCHIVE/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    value.update(schema="jspace.incremental-suffix-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),analysis_source_sha256={str(p):sha(p) for p in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",classification=record.classification,correct=record.selected["correct"],analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root=Path(root); value=read(root/ANALYSIS); record=JSpaceIncrementalSuffixRecord.from_evidence(value)
    require(value["record"] == record.to_dict(), "Typed record differs from authenticated evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path) == digest, "Changed evidence/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items():
        require(identity(path) == pin["identity"], "Changed authenticated large input: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/ARCHIVE/Path(path).relative_to(ROOT)) == digest, "Changed archived closeout source")
    return value,record


def report(root,report_path):
    root,report_path=Path(root),Path(report_path); value,r=expected(root)
    lines=["# One-block expansion of the recovered Qwen suffix", "",
        f"Study **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**", "",
        f"The selected seven-block suffix scored **{r.selected['correct']}/64 ({100*r.primary_metric:.2f}%)**, versus {r.initial['correct']}/64 at initialization, {r.parent['correct']}/64 for its accepted six-block parent, and {r.native['correct']}/64 for matched native BF16 execution. The dual five-percentage-point gate requires **{r.gates['minimum_correct']}/64** and {'passed' if r.gates['quality_pass'] else 'failed'}.", "",
        f"Epoch {r.selected_epoch} won by correct count, then full-vocabulary CE, with the earlier epoch retained for exact ties. "+("The initializer won; its accuracy does not demonstrate a training gain." if r.initializer_selected else "Initialization remained eligible."), "",
        "This is the historical rank-1024, head-64, polynomial-MLP suffix with fixed reduced-coordinate bridges, BF16 trainable parameters and the original geometry precision. It adds only block 29 to recovered blocks 30–35, jointly training all seven and the final norm. It differs from the later full-head, native-residual all-block repair experiment.", "",
        "The frozen native prefix is supplied through authenticated BF16 boundary-29 caches. The complete trainable suffix is recomputed on current student activations; raw-token prefix execution is not rerun on every batch.", "",
        "| Epoch | Correct /64 | Full-vocabulary KL | Full-vocabulary CE | Loss | Native agreement | Parent agreement |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in r.epochs:
        m=row["metrics"]
        lines.append(f"| {row['epoch']} | {row['correct']} | {m['kl']:.6f} | {m['ce']:.6f} | {m['loss']:.6f} | {100*row['native_agreement']:.2f}% | {100*row['parent_agreement']:.2f}% |")
    lines += ["", "| Model | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 |", "|---|---:|---:|---:|---:|"]
    for name,row in (("Native BF16",r.native),("Accepted six-block parent",r.parent),("Initial seven-block",r.initial),("Selected seven-block",r.selected)):
        lines.append(f"| {name} | "+" | ".join(str(row["by_family"][f]["correct"]) for f in ("arc","boolq","hellaswag","winogrande"))+" |")
    for label,key in (("native","paired_to_native"),("parent","paired_to_parent"),("initializer","paired_to_initial")):
        p=r.selected[key];lines += ["",f"Against the {label}, selected predictions gained {p['gains']} and lost {p['losses']} correct answers (net {p['net_correct']:+d})."]
    lines += ["",f"The most frequent gold label scores {r.selected['best_constant_label_correct']}/64 as a constant-label baseline on these reused development cases.",
        "","All 384 updates across 12 epochs completed on the original 512 training cases. The objective stayed .7T² full-vocabulary KL to the accepted parent plus .3 gold next-token CE, T=2, at four existing positions. No hidden-state auxiliary was added. The discarded pilot established finite nonzero gradients and changed FP32 optimizer masters in all seven blocks and norm while the readout and geometry remained fixed. Stored BF16 changes are a separate diagnostic; an individual update can round away at BF16 precision.",
        "","Native, packed-parent and boundary-seam controls passed. The historical Torch teacher-cache versus packed-parent logit difference remains a separate diagnostic; these paths are not required to be bitwise identical. Fresh selected-state verification requires exact predictions and the fixed BF16 tolerance 0.03125 + abs(reference)/128.",
        "",f"The selected suffix checkpoint contains {r.storage['selected_suffix_tensor_bytes']:,} tensor bytes; the packed suffix and norm contain {r.storage['packed_suffix_tensor_bytes']:,}. Including all 29 frozen native blocks and the tied embedding/readout once gives **{r.storage['whole_model_tensor_bytes']:,} persistent tensor bytes**, versus {r.storage['native_model_tensor_bytes']:,} in the original native model. This includes fixed geometry and does not measure throughput.",
        "","This is one development-selected expansion. No untouched confirmation, speed benchmark, next-block expansion or whole-model 38%/1.4× result is established. A quality miss is retained as a completed negative experiment.",
        "",f"Evidence: `{root/ANALYSIS}`, SHA256 `{sha(root/ANALYSIS)}`.",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(report_path),report_sha256=sha(report_path))


def metadata(root,r):
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=r.classification,
        primary_metric=r.primary_metric,primary_metric_name="selected_seven_block_suffix_development_accuracy",primary_metric_unit="fraction",
        correct=r.selected["correct"],count=64,accuracy=r.primary_metric,native_correct=r.native["correct"],
        parent_correct=r.parent["correct"],initial_correct=r.initial["correct"],quality_pass=r.gates["quality_pass"],
        minimum_correct=r.gates["minimum_correct"],selected_epoch=r.selected_epoch,initializer_selected=r.initializer_selected,
        compressed_layer_count=7,updates=384,packed_prediction_parity=True,confirmed_independent_count=0,speed_measured=False,
        aggregate_sha256=sha(Path(root)/ANALYSIS),scope=SCOPE)


def readback(root,registry,report_path,key=None):
    _,r=expected(root)
    if key is None:
        saved=read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path), "Registry/report provenance mismatch")
        key=saved["storage"]["result_hash"]
    db,wanted=logger(),metadata(root,r); row=db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(row["ids"] == [key] and all(row["metadatas"][0].get(k) == v for k,v in wanted.items()), "Experiment readback mismatch")
    h=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"]);require(h["ids"] == [STUDY_ID],"Hypothesis missing")
    value=h["metadatas"][0]
    require(value.get("tested") is True and value.get("confirmed") is False and value.get("evidence_count") == 1
        and value.get("confirmation_status") == r.classification and json.loads(value["direct_result_hashes_json"]) == [key],"Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    value,r=expected(root);require_committed([*sources(),Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis=dict(id=STUDY_ID,name="One-block expansion of recovered compact Qwen suffix",category="compression",tested=True,
        confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=r.classification,tested_scope=SCOPE,
        description="Joint output-only recovery of seven compact blocks, retaining the accepted six-block initializer and fixed native prefix.")
    experiment=ExperimentResult(experiment_id="jspace-incremental-suffix-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(accuracy=r.primary_metric,initial_accuracy=r.initial["accuracy"],parent_accuracy=r.parent["accuracy"],
                     native_accuracy=r.native["accuracy"],full_vocabulary_ce=r.selected["metrics"]["ce"]),
        primary_metric=r.primary_metric,model_architecture=[2048]*29+[1024]*7,model_parameters=r.storage["model_parameters"],
        training_time=value["stage_manifests"]["train"]["elapsed_seconds"],model_checkpoint=value["selected_checkpoint"],
        training_history=[dict(epoch=row["epoch"],accuracy=row["accuracy"],ce=row["metrics"]["ce"]) for row in r.epochs],
        observations=[SCOPE,r.classification,"Suffix checkpoint requires 29 frozen native blocks and original tied embedding/readout."])
    db=logger();prior=db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis);key=db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,r)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt=readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root,registry,report_path):
    root=Path(root);receipt=read(root/"incremental_suffix_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value=dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],
        registry_readback=receipt,report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_modified=False)
    write(root/"incremental_suffix_finalization.json",value);return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Hide CUDA for CPU finalization")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT);parser.add_argument("--report",type=Path,default=REPORT);parser.add_argument("--registry",type=Path,default=REGISTRY)
    args=parser.parse_args()
    if args.stage == "analyze":output=analyze(args.root)
    elif args.stage == "report":output=report(args.root,args.report)
    else:
        output={"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback":write(args.root/"incremental_suffix_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
