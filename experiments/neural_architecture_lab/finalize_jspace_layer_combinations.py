"""CPU-only authenticated closeout of the fixed layer-combination search."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_layer_combinations_record import (
    JSpaceLayerCombinationsRecord,STUDY_ID,SCOPE,PRIMARY_METRIC_NAME,require,policy)
from . import finalize_jspace_qwen_affine as common
from .finalize_jspace_single_layer import require_committed
from .finalize_jspace_recovered_assembly import identity

ROOT,read,sha,write,logger = common.ROOT,common.read,common.sha,common.write,common.logger
DEFAULT = ROOT/"data/experiments/jspace_layer_combinations/20261114_v1"
REPORT = ROOT/"docs/08 - Analysis/jspace/2026-11-14_qwen-layer-combinations-results.md"
REGISTRY = ROOT/"data/hypothesis_registry/jspace_layer_combinations_20261114.json"
TYPED = ROOT/"src/neural_architecture_lab/jspace_layer_combinations_record.py"
TEST = ROOT/"tests/neural_architecture_lab/test_jspace_layer_combinations.py"
ANALYSIS = "layer_combinations_analysis.json"
ARCHIVE = "layer_combinations_analysis_source"


def sources():
    return [Path(__file__).resolve(),TYPED,TEST,Path(common.__file__).resolve(),Path(policy.__file__).resolve(),
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_single_layer.py",
        ROOT/"experiments/neural_architecture_lab/finalize_jspace_recovered_assembly.py",
        ROOT/"src/neural_architecture_lab/jspace_single_layer_record.py"]


def audit(root):
    from experiments.structure_net import jspace_layer_combination_search as search
    root = Path(root).resolve(); proof = search.verified_selection(root); plan = search.verify(root)
    result = read(root/"search/result.json"); artifacts,large,manifests = {},{},{}
    def pin(path,expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected,"Changed search closeout input: "+str(path))
        artifacts[str(path)] = digest
    pin(root/"plan.json")
    for stage in ("pilot","search"):
        manifest = read(root/stage/"manifest.json"); manifests[stage] = manifest
        pin(root/stage/"manifest.json"); pin(root/stage/"result.json",manifest["result_sha256"])
        require(manifest["status"] == "completed" and manifest["passed"] is True and manifest["fingerprint"] == plan["fingerprint"],
            "Completed matching reserved pilot and search required")
        require(manifest["settings"]["allocation"] == manifest["allocation"],"Allocator and lease allocations differ")
        require(len(manifest["settings"]["settings"]) == 3 and sorted(manifest["allocation"].values()) == [5120,7168,11264],
            "Frozen three-card reservation required")
        for row in manifest["settings"]["settings"]:
            peak = manifest["device_peaks"][row["physical_uuid"]]
            require(0 <= peak["peak_reserved_bytes"] <= row["allocator_limit_bytes"] < row["lease_bytes"],"Reservation cap failed")
    manifest = manifests["search"]
    for name,key in (("selection","selection_sha256"),("screen","screen_sha256"),("candidates","candidates_sha256")):
        pin(root/"search"/(name+".json"),manifest[key])
    pin(root/"search/controls.json")
    require(read(root/"search/controls.json") == result["controls"],"Saved control report differs")
    require(read(root/"search/candidates.json") == result["summary"]["candidate_record"],"Saved candidate report differs")
    screen = read(root/"search/screen.json")
    require(set(screen["results"]) == set(policy.screen_masks())
        and all(row == result["results"][key] for key,row in screen["results"].items()),"Complete original screen must remain unchanged in final results")
    for path,digest in plan["source_sha256"].items():
        pin(path,digest); pin(root/"producer"/Path(path).relative_to(ROOT),digest)
    for path,digest in plan["input_sha256"].items():
        if Path(path).stat().st_size <= 64*1024**2: pin(path,digest)
        else: large[path] = dict(sha256=digest,identity=identity(path))
    references = {arm:read(path) for arm,path in plan["references"].items()}
    for path in plan["references"].values(): pin(path,plan["input_sha256"][path])
    parent = read(Path(plan["assembly_root"])/"plan.json")
    config_path = Path(parent["model"]["path"])/"config.json"; config = read(config_path)
    pin(config_path,parent["model"]["files"]["config.json"])
    require(config["tie_word_embeddings"] is True and config["hidden_size"] == 2048 and config["num_hidden_layers"] == 36,
        "Original36-layer tied-vocabulary architecture required")
    inventory = dict(endpoints=config["vocab_size"]*config["hidden_size"]+config["hidden_size"])
    for key,arm in (("native_by_layer","native"),("compact_by_layer","all_compact")):
        rows = references[arm]["metrics"]["layers"]
        require([r["layer"] for r in rows] == list(range(36)) and all(r["parameter_bytes"]%4 == 0 for r in rows),"CompleteFP32 parameter inventory required")
        inventory[key] = [r["parameter_bytes"]//4 for r in rows]
    for name in ("events.jsonl","launch_receipt.json"):
        if (root/name).is_file(): pin(root/name)
    value = dict(status="completed",confirmed=False,plan=plan,search=result,pilot=read(root/"pilot/result.json"),
        stage_manifests=manifests,references=references,selection_proof=proof,parameter_inventory=inventory,
        artifact_sha256=artifacts,authenticated_large_artifacts=large,
        input_authentication="Full successful search, mask completeness, deterministic selection, per-case scores and archived/live source bytes authenticated. Small inputs rehashed; large fixed weights retain producerSHA256 plus immutable file identities, and each selected state was hashed during bank loading.")
    return value,JSpaceLayerCombinationsRecord.from_evidence(value)


def analyze(root):
    root = Path(root).resolve(); require(not (root/ANALYSIS).exists() and not (root/ARCHIVE).exists(),"Closeout already attempted")
    value,record = audit(root)
    for path in sources():
        target = root/ARCHIVE/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    value.update(schema="jspace.layer-combinations-analysis.v1",created_at=datetime.now(timezone.utc).isoformat(),
        record=record.to_dict(),analysis_source_sha256={str(path):sha(path) for path in sources()})
    write(root/ANALYSIS,value)
    return dict(status="analyzed",classification=record.classification,primary_metric=record.primary_metric,
        primary_metric_name=PRIMARY_METRIC_NAME,analysis_sha256=sha(root/ANALYSIS))


def expected(root):
    root = Path(root); value = read(root/ANALYSIS); record = JSpaceLayerCombinationsRecord.from_evidence(value)
    require(value["record"] == record.to_dict(),"Typed search record differs from evidence")
    for path,digest in {**value["artifact_sha256"],**value["analysis_source_sha256"]}.items():
        require(sha(path) == digest,"Changed closeout artifact/source: "+path)
    for path,pin in value["authenticated_large_artifacts"].items(): require(identity(path) == pin["identity"],"Changed fixed weight identity: "+path)
    for path,digest in value["analysis_source_sha256"].items():
        require(sha(root/ARCHIVE/Path(path).relative_to(ROOT)) == digest,"Changed archived closeout source")
    return value,record


def pair_plot(root, interactions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(root); image = root/"pair_excess_rescue.png"; table = root/"pair_excess_rescue.csv"
    require(not image.exists() and not table.exists(),"Pair artifacts already exist")
    matrix = np.full((36,36),np.nan)
    with table.open("x",newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["layer_i","layer_j","restored_correct","restoration_gain","excess_rescue"])
        for row in interactions:
            i,j = row["layers"]; matrix[i,j] = matrix[j,i] = row["excess_rescue"]
            writer.writerow([i,j,row["restored_correct"],row["restoration_gain"],row["excess_rescue"]])
    limit = max(1,float(np.nanmax(np.abs(matrix)))); fig,axis = plt.subplots(figsize=(8,7),layout="constrained")
    heat = axis.imshow(matrix,cmap="RdBu_r",vmin=-limit,vmax=limit,origin="lower",interpolation="nearest")
    axis.set(xlabel="Restored native layer j",ylabel="Restored native layer i",title="Conditional pair excess rescue · original64 development cases")
    axis.set_xticks(range(0,36,5)); axis.set_yticks(range(0,36,5))
    fig.colorbar(heat,ax=axis,label="Additional correct answers beyond summed singleton rescue")
    fig.savefig(image,dpi=180); plt.close(fig)
    return dict(image=str(image),image_sha256=sha(image),csv=str(table),csv_sha256=sha(table))


def report(root,report_path):
    root,report_path = Path(root),Path(report_path); value,r = expected(root); most = r.most_compressed; target = r.selected_repair_target
    require(not report_path.exists(),"Report already exists")
    plot = pair_plot(root,r.pair_interactions)
    lines = ["# Qwen recovered-layer compatibility search","",f"Study **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**","",
        f"The most compressed **tested** model meeting the47/64 development gate uses **{most['compressed_count']}/36 compact blocks** and scores **{most['correct']}/64 ({100*most['accuracy']:.2f}%)**. The original native control scored50/64. This layer count is the primary metric; it is not model accuracy or a claim of global optimality.",
        "",f"The search evaluated **{r.unique_masks} unique masks** on the same64 previously inspected development cases. All36 compact blocks reproduce27/64; compact blocks0/1 with a native suffix reproduce49/64; restoring native12/15 reproduces24/64. No weights changed, no repair training occurred, and the untouched confirmation set was not opened.",
        "",f"Most-compressed passing mask: `{most['mask_id']}`. Compact layers: `{most['compressed_layers']}`. Conditional-choice CE: {most['choice_ce']:.6f}; native decision agreement: {100*most['teacher_agreement']:.2f}%.",
        "","| Background outside six candidates | Compact blocks | Tested at count | Best /64 | Worst /64 | Best switched group | Worst switched group |",
        "|---|---:|---:|---:|---:|---|---|"]
    for row in r.same_count:
        lines.append(f"| {row['context']} | {row['compressed_count']} | {row['tested']} | {row['best_correct']} | {row['worst_correct']} | {row['best_group']} | {row['worst_group']} |")
    lines += ["","In `outside_native`, switched candidate layers are compact; in `outside_compact`, switched layers are restored to native. Each best/worst comparison holds the number of compact blocks constant. Ties follow correct count, native agreement, lower conditional CE, then lexical layer indices.",
        "",f"The deterministic candidate set was `{r.candidates['layers']}` ({r.candidates['method']}). The selected three-layer repair target is **{target['layers']}**: restoring those layers to native in the otherwise compact model scores **{target['restored_correct']}/64**, a change of **{target['rescue_correct']:+d}** from27/64. Paired changes: {r.selected_restoration_paired['gained']} gained and {r.selected_restoration_paired['lost']} lost correct answers. This conditional native-restoration result does not demonstrate that training the compact group can learn the rescue.",
        "","| Selected example | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gains vs native | Losses vs native |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,row in (("Most compressed passing",most),("Chosen triple restored",target["score"])):
        pair = row["paired_to_native"]
        lines.append(f"| {name} | "+" | ".join(str(row["by_family"][f]["correct"]) for f in ("arc","boolq","hellaswag","winogrande"))+f" | {pair['gains']} | {pair['losses']} |")
    lines += ["",f"![All630 pair excess-rescue values]({os.path.relpath(plot['image'],report_path.parent)})","",
        "Every point uses the all-compact background. Excess rescue is correct(i,j)−correct(i)−correct(j)+27; it is a descriptive, background-dependent interaction count. The symmetric plot contains630 unique pairs, not independent replications or a causal decomposition.",
        "","Shared-prefix execution was checked against independent paths, all four controls passed exact predictions and numerical logit tolerance, and all resident parameters/buffers remained unchanged. Runtime timings and decoder-call savings describe search execution only; there is no deployment token-throughput claim. Repeated development selection does not supply independent confirmation.",
        "",f"Evidence: `{root/ANALYSIS}`, SHA256 `{sha(root/ANALYSIS)}`. Pair data: `{plot['csv']}`, SHA256 `{plot['csv_sha256']}`. {value['input_authentication']}",""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as handle: handle.write("\n".join(lines))
    receipt = dict(report=str(report_path),report_sha256=sha(report_path),**plot); write(root/"layer_combinations_report.json",receipt)
    return receipt


def metadata(root,record):
    most,target = record.most_compressed,record.selected_repair_target
    return dict(hypothesis_id=STUDY_ID,confirmed=False,confirmation_status=record.classification,
        primary_metric=record.primary_metric,primary_metric_name=PRIMARY_METRIC_NAME,primary_metric_unit="layers",
        most_compressed_passing_layers=most["compressed_count"],most_compressed_passing_mask=most["mask_id"],
        correct=most["correct"],count=64,accuracy=most["accuracy"],minimum_correct=47,unique_masks=record.unique_masks,
        selected_repair_layers_json=json.dumps(target["layers"]),selected_native_restoration_correct=target["restored_correct"],
        selected_native_restoration_gain=target["rescue_correct"],learned_repair_evaluated=False,training_performed=False,
        independent_confirmations=0,complete_global_search=False,speed_measured=False,
        aggregate_sha256=sha(Path(root)/ANALYSIS),scope=SCOPE)


def plot_receipt(root,report_path):
    value = read(Path(root)/"layer_combinations_report.json")
    require(Path(value["report"]).resolve() == Path(report_path).resolve() and value["report_sha256"] == sha(report_path)
        and sha(value["image"]) == value["image_sha256"] and sha(value["csv"]) == value["csv_sha256"],
        "Report, pair figure or pair data changed")
    return value


def readback(root,registry,report_path,key=None):
    _,record = expected(root)
    plots = plot_receipt(root,report_path)
    if key is None:
        saved = read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/ANALYSIS) and saved["report_sha256"] == sha(report_path),"Registry/report provenance mismatch")
        require(saved["pair_artifacts"] == plots,"Stored pair artifacts differ from report")
        key = saved["storage"]["result_hash"]
    db,wanted = logger(),metadata(root,record); row = db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(row["ids"] == [key] and all(row["metadatas"][0].get(k) == v for k,v in wanted.items()),"Exact search experiment readback mismatch")
    hypothesis = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(hypothesis["ids"] == [STUDY_ID],"Search hypothesis missing")
    value = hypothesis["metadatas"][0]
    require(value.get("tested") is True and value.get("confirmed") is False and value.get("evidence_count") == 1
        and value.get("confirmation_status") == record.classification and json.loads(value["direct_result_hashes_json"]) == [key],"Search hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def store(root,registry,report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(),"Registry exists or report missing")
    _,record = expected(root); require_committed([*sources(),Path(report_path)])
    plots = plot_receipt(root,report_path)
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="Conditional compatibility of recovered Qwen layers",category="compression",tested=True,
        confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=record.classification,tested_scope=SCOPE,
        description="Frozen conditional mask screen and two-context refinement on reused development cases; no repair training.")
    most = record.most_compressed
    experiment = ExperimentResult(experiment_id="jspace-layer-combinations-"+sha(Path(root)/ANALYSIS)[:16],hypothesis_id=STUDY_ID,
        metrics=dict(most_compressed_tested_gate_passing_layer_count=record.primary_metric,accuracy=most["accuracy"],
            correct=most["correct"],unique_masks=record.unique_masks,conditional_choice_ce=most["choice_ce"]),
        primary_metric=record.primary_metric,model_architecture=[2048]*36,model_parameters=record.model_parameters,training_time=0.,
        observations=[SCOPE,record.classification,"Primary metric is a layer count; accuracy is recorded separately."])
    db = logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True,"Refusing to overwrite tested search")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,record)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path))])
    receipt = readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/ANALYSIS),report_path=str(report_path),report_sha256=sha(report_path),pair_artifacts=plots))
    return receipt


def closeout(root,registry,report_path):
    root = Path(root); receipt = read(root/"layer_combinations_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True,"Readback changed")
    value = dict(status="completed",study_id=STUDY_ID,confirmed=False,classification=receipt["confirmation_status"],registry_readback=receipt,
        report_path=str(report_path),report_sha256=sha(report_path),aggregate_sha256=sha(root/ANALYSIS),
        recorded_at=datetime.now(timezone.utc).isoformat(),research_log_modified=False)
    write(root/"layer_combinations_finalization.json",value); return value


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Hide CUDA for CPU finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout")); parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--report",type=Path,default=REPORT); parser.add_argument("--registry",type=Path,default=REGISTRY); args = parser.parse_args()
    if args.stage == "analyze": output = analyze(args.root)
    elif args.stage == "report": output = report(args.root,args.report)
    else:
        output = {"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback": write(args.root/"layer_combinations_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
