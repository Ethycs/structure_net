"""CPU-only, append-only closeout of compact timing plus fresh quality validation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

from .finalize_jspace_qwen_affine import ROOT, LOG, logger, paired, read, sha, write

STUDY = "JSPACE-COMPACT-PERFORMANCE-01"
DEFAULT = ROOT / "data/experiments/jspace_fast_recovery/20261113_v2"
REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-13_jspace-compact-performance.md"
SCOPE = ("Exploratory single-final-block execution of a recovered Qwen model; one original training seed. "
         "Accuracy is 512 fresh multiple-choice tasks, not generation quality. Timing uses eight development "
         "prompts, batch one, 32 greedy tokens including prefill; graph setup is amortized. "
         "No all-layer, equivalence, or independent-confirmation claim.")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def local_timings(bench):
    """Equal-prompt mean call latency; never mix eager, graph, prefill, or decode."""
    result = {}
    for mode in ("prefill", "decode"):
        pairs = [p for p in bench["pairs"] if p["mode"] == mode]
        require(bool(pairs), "Missing local timing mode")
        require(len({p["identity"] for p in pairs}) == len(pairs), "Duplicate local timing prompt")
        for execution in ("eager", "graph"):
            seconds = {arm: statistics.mean(p["summary"][execution][arm]["wall_seconds"] for p in pairs)
                       for arm in ("reference", "candidate")}
            require(all(math.isfinite(v) and v > 0 for v in seconds.values()), "Invalid local timing")
            result[mode + "/" + execution] = {**seconds, "speed_ratio_to_compact": seconds["reference"] / seconds["candidate"],
                "prompts": len(pairs), "scope": "ratio of equal-prompt mean median wall latency; descriptive, not confidence interval"}
    return result


def native_timings(plan, native_plan, evidence):
    """Audit the actual paired native comparator, distinct from the compact control."""
    require(native_plan["parent_root"] == plan["parent_root"] and
            native_plan["candidate_options"] == plan["fast_options"], "Different native-comparison structure")
    require(evidence["candidate"] == plan["source_arm"] + "/" + plan["candidate"] and
            evidence["reference"] == "original native block35", "Different native comparison")
    require(evidence["quality_selection"] is False and evidence["final_norm_in_timing"] is False,
            "Changed native timing scope")
    require(len(set(native_plan["case_identities"])) == 8, "Eight unique native comparison prompts required")
    output = {}
    for mode in ("prefill", "decode"):
        pairs = [p for p in evidence["pairs"] if p["mode"] == mode]
        require([p["identity"] for p in pairs] == native_plan["case_identities"], "Native comparison inputs changed")
        require(all(p[k]["graph_error"]["equal"] is True for p in pairs for k in ("native_graph", "fast_graph")),
                "Native or fast graph changed its own output")
        totals = {metric: {"native": 0., "fast": 0.} for metric in ("wall_seconds", "cuda_event_seconds")}
        ratios = []
        for pair in pairs:
            rows = pair["timings"]
            repeats = native_plan["config"]["paired_repeats"]
            for repeat in range(repeats):
                for order in (["reference", "candidate"], ["candidate", "reference"]):
                    selected = [r for r in rows if r["repeat"] == repeat and r["order"] == order]
                    require([r["variant"] for r in selected] == order, "Unbalanced paired order")
            require(len(rows) == repeats * 4, "Wrong number of paired observations")
            for metric in totals:
                values = {arm: statistics.median(r[metric] for r in rows if r["variant"] == variant)
                          for arm, variant in (("native", "reference"), ("fast", "candidate"))}
                require(all(math.isfinite(v) and v > 0 for v in values.values()), "Invalid native timing")
                for arm, value in values.items():
                    require(math.isclose(value, pair["summary"][metric][arm], rel_tol=1e-12), "Native pair summary mismatch")
                    totals[metric][arm] += value
                if metric == "wall_seconds":
                    ratios.append(values["native"] / values["fast"])
        output[mode] = {}
        for metric, total in totals.items():
            speedup = total["native"] / total["fast"]
            require(math.isclose(speedup, evidence["summary"][mode][metric]["speedup"], rel_tol=1e-12), "Native aggregate mismatch")
            output[mode][metric] = {"native_mean_seconds": total["native"] / len(pairs),
                "fast_mean_seconds": total["fast"] / len(pairs), "speed_ratio_to_native": speedup}
        output[mode].update(prompts=len(pairs), prompts_faster_than_native=sum(v > 1 for v in ratios),
            per_prompt_speed_ratio_range=[min(ratios), max(ratios)], input_tokens=[p["tokens"] for p in pairs])
    return {"measurements": output, "applies_to_exported_tensors": not plan["recover"],
        "reference": "original native Hugging Face final transformer block, both executions in CUDA graphs",
        "scope": "isolated full final block; final model norm excluded; no separately custom-fused native baseline; repeated timing samples are not independently trained models",
        "summary_method": "ratio of sums of per-prompt median graph wall latencies; CUDA-event measurements reported separately"}


def summarize(plan, result, deployment, reload, records, runtime, bench):
    """Validate the evidence boundary before deriving searchable metrics."""
    require(result["status"] == "completed" and result["confirmed"] is False, "Exploratory completed result required")
    require(result["fingerprint"] == plan["fingerprint"], "Mismatched validation fingerprint")
    require(result["fresh_evaluation_count"] == 512, "Fresh evaluation count changed")
    require(result["recover"] == plan["recover"], "Recovery selection changed")
    require(result["source_arm"] == plan["source_arm"] and result["candidate"] == plan["candidate"], "Candidate changed")
    provenance = result["timing_provenance"]
    expected_stage = "runtime" if plan["recover"] else "full_" + plan["source_arm"] + "_" + plan["candidate"]
    require(provenance["stage"] == expected_stage, "Wrong timing stage")
    if not plan["recover"]:
        require(provenance["tensor_identity_verified"] is True and provenance["root"] == plan["performance_root"],
                "Untested tensors cannot inherit parent timing")
    for arm in ("native", "control", "fast"):
        rows, reported = records[arm], result["results"][arm]
        require(len(rows) == 512 and len({r["identity"] for r in rows}) == 512, "Invalid fresh rows")
        correct = sum(r["prediction"] == r["answer"] for r in rows)
        acc = reported["accuracy"]
        require(acc["count"] == 512 and acc["correct"] == correct and acc["overall"] == correct / 512, "Accuracy/record mismatch")
        require(reported["tensor_bytes"] > 0 and reported["parameter_count"] > 0, "Missing deployment footprint")
        require(reported["trained_in_this_study"] == (plan["recover"] and arm != "native"), "Wrong training attribution")
        if arm != "native":
            exported = deployment["checkpoints"][arm]
            require(exported["trained"] == plan["recover"], "Export training attribution changed")
            require(exported["tensor_bytes"] == reported["tensor_bytes"] and exported["parameter_count"] == reported["parameter_count"], "Export footprint mismatch")
            require(reload[arm]["records_exact"] is True and reload[arm]["accuracy"] == acc, "Exact reload missing")
            if not plan["recover"]:
                require(exported["timing_tensors_match_parent_candidate"] is True, "Parent timing tensor identity absent")
    comparison = {"control_to_native": paired(records["native"], records["control"]),
                  "fast_to_native": paired(records["native"], records["fast"]),
                  "fast_to_control": paired(records["control"], records["fast"])}
    names = {"native": "native", "control": "control" if plan["recover"] else "reference",
             "fast": "fast" if plan["recover"] else "candidate", "native_repeat": "native_repeat"}
    if not plan["recover"]:
        names["control_repeat"] = "reference_repeat"
    throughput = {arm: {mode: runtime[key]["timings"][mode]["tokens_per_second"]
                        for mode in ("hf", "cuda_graph")} for arm, key in names.items()}
    require(all(math.isfinite(v) and v > 0 for t in throughput.values() for v in t.values()), "Invalid whole-model timing")
    ratios = {mode: {"to_" + arm: throughput["fast"][mode] / values[mode]
                    for arm, values in throughput.items() if arm != "fast"} for mode in ("hf", "cuda_graph")}
    local = local_timings(bench)
    if plan["recover"]:
        # Screening microbenchmarks timed an earlier checkpoint, so do not attribute them to the recovered export.
        current = {arm: statistics.mean(r["seconds_per_call"] for r in runtime[names[arm]]["isolated_block_prefill"])
                   for arm in ("native", "control", "fast")}
        require(all(math.isfinite(v) and v > 0 for v in current.values()), "Invalid recovered local timing")
        local_gain = current["control"] / current["fast"]
    else:
        current, local_gain = None, local["prefill/graph"]["speed_ratio_to_compact"]
    accuracies = {a: result["results"][a]["accuracy"] for a in records}
    return {"confirmed": False, "classification": ("local_speedup_with_measured_quality" if local_gain > 1 else "no_local_speedup_with_measured_quality"),
        "classification_scope": "descriptive exploratory label; local gain is versus prior compact, not native; not a retrospective confirmatory gate",
        "scope": SCOPE, "accuracy": accuracies, "paired": comparison, "whole_model_throughput": throughput,
        "whole_model_ratios": ratios, "local_screen_timings": local,
        "local_screen_applies_to_exported_tensors": not plan["recover"], "recovered_export_prefill_seconds": current,
        "quality_screen": {"native_maximum_drop_pp": 1., "compact_maximum_drop_pp": .5,
            "fast_minus_native_pp": 100 * (accuracies["fast"]["overall"] - accuracies["native"]["overall"]),
            "fast_minus_control_pp": 100 * (accuracies["fast"]["overall"] - accuracies["control"]["overall"]),
            "scope": "observed point estimates only; 512 cases do not establish equivalence"}}


def audit(root, stages):
    plan = read(root / "plan.json")
    require(plan["fingerprint"] == digest({k: v for k, v in plan.items() if k != "fingerprint"}), "Changed frozen plan")
    hashes, manifests = {}, {}
    for name in stages:
        manifest = read(root / (name + "_manifest.json"))
        require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"], "Incomplete stage: " + name)
        manifests[name] = manifest
        for relative, expected in manifest["artifact_sha256"].items():
            path = str(root / relative)
            require(path not in hashes or hashes[path] == expected, "Conflicting artifact hashes")
            hashes[path] = expected
    for path, expected in plan["source_sha256"].items():
        hashes[path] = expected
        hashes[str(root / "producer" / Path(path).relative_to(ROOT))] = expected
    hashes.update(plan["input_sha256"])
    for path, expected in hashes.items():
        require(sha(path) == expected, "Changed artifact/input/source: " + path)
    return plan, manifests, hashes


def analyze(run, supplements, native_root=None):
    plan = read(run / "plan.json")
    stages = (["train_control", "train_fast"] if plan["recover"] else []) + ["evaluation", "reload"] + (["runtime"] if plan["recover"] else [])
    plan, manifests, hashes = audit(run, stages)
    perf = Path(plan["performance_root"])
    full = "full_" + plan["source_arm"] + "_" + plan["candidate"]
    pplan, pmanifests, phashes = audit(perf, ["profile", "bench", full])
    require(plan["performance_fingerprint"] == pplan["fingerprint"], "Different performance campaign")
    selection = read(perf / "selection.json")
    require(selection["heldout_inspected"] is False and selection["source_arm"] == plan["source_arm"] and selection["candidate"] == plan["candidate"], "Selection inspected heldout or changed candidate")
    require(selection["bench_result_sha256"] == sha(perf / "bench/result.json") and selection["validation_mode"] == ("recovery" if plan["recover"] else "direct"), "Selection changed after screening")
    result, deployment = read(run / "result.json"), read(run / "deployment.json")
    require(result["timing_provenance"]["root"] == str(run if plan["recover"] else perf), "Timing root mismatch")
    for arm in ("control", "fast"):
        require(sha(run / (arm + ".pt")) == deployment["checkpoints"][arm]["checkpoint_sha256"], "Wrong exported checkpoint")
    tasks = read(run / "tasks.json")
    require(digest(tasks) == plan["task_digest"], "Changed task split")
    require(len(tasks["splits"]["train"]) == 512 and len(tasks["splits"]["development"]) == 64,
            "Frozen train/development counts changed")
    evaluation_ids = {r["identity"] for r in tasks["splits"]["evaluation"]}
    exclusions = [read(Path(plan["parent_root"]) / "tasks.json")]
    exclusions.extend(read(p) for p in plan["input_sha256"] if "jspace_qwen_layerkd" in p and p.endswith("tasks.json"))
    seen = {r["identity"] for t in exclusions for rows in t["splits"].values() for r in rows}
    seen.update(r["identity"] for name in ("train", "development") for r in tasks["splits"][name])
    require(len(evaluation_ids) == 512 and not evaluation_ids.intersection(seen), "Fresh split is contaminated")
    records = {a: read(run / "evaluation" / (a + "_fresh.json")) for a in ("native", "control", "fast")}
    require(all([r["identity"] for r in rows] == [r["identity"] for r in tasks["splits"]["evaluation"]] for rows in records.values()), "Fresh result/task order mismatch")
    require(all(r["answer"] == task["answer"] and r["family"] == task["family"]
                for rows in records.values() for r, task in zip(rows, tasks["splits"]["evaluation"])), "Fresh labels/families changed")
    runtime = read((run / "runtime" if plan["recover"] else perf / full) / "result.json")
    bench = read(perf / "bench/result.json")[plan["source_arm"] + "/" + plan["candidate"]]
    summary = summarize(plan, result, deployment, read(run / "reload/verification.json"), records, runtime, bench)
    native_manifests, native_hashes = {}, {}
    if native_root is not None:
        nplan, native_manifests, native_hashes = audit(native_root, ["native_compare"])
        require(nplan["parent_fingerprint"] == pplan["parent_fingerprint"], "Native comparison used another parent")
        native_evidence = read(native_root / "native_compare/result.json")
        summary["native_graph_comparison"] = native_timings(plan, nplan, native_evidence)
        summary["native_graph_comparison"].update(root=str(native_root), fingerprint=nplan["fingerprint"],
            result_sha256=sha(native_root / "native_compare/result.json"))
        if not plan["recover"]:
            native_win = all(v["wall_seconds"]["speed_ratio_to_native"] > 1
                             for v in summary["native_graph_comparison"]["measurements"].values())
            summary["classification"] = "local_speedup_with_measured_quality" if native_win else "no_local_native_speedup_with_measured_quality"
            summary["classification_scope"] = "descriptive local graph gain versus original native Hugging Face final block; not a confirmatory or whole-model speed claim"
    else:
        summary["native_graph_comparison"] = {"status": "not_included", "reason": "no native comparison root supplied"}
    extra = {str(p): {"sha256": sha(p), "value": read(p), "scope": "supplementary descriptive evidence; no automatic promotion to primary result"} for p in supplements}
    sources = [Path(__file__), Path(__file__).with_name("finalize_jspace_qwen_affine.py"),
               ROOT / "tests/neural_architecture_lab/test_jspace_compact_performance_closeout.py"]
    source_hashes = {str(p): sha(p) for p in sources}
    archive = run / "postprocessing_source"
    archive.mkdir(exist_ok=False)
    for path in sources:
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    value = {"schema_version": "jspace.compact-performance.analysis.v1", "study_id": STUDY, "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(), **summary,
        "validation_root": str(run), "performance_root": str(perf), "validation_fingerprint": plan["fingerprint"],
        "performance_fingerprint": pplan["fingerprint"], "recover": plan["recover"], "source_arm": plan["source_arm"], "candidate": plan["candidate"],
        "result": result, "result_sha256": sha(run / "result.json"), "deployment": deployment,
        "manifests": manifests, "performance_manifests": pmanifests, "selection": selection, "supplements": extra,
        "native_comparison_manifests": native_manifests,
        "artifact_sha256": {**hashes, **phashes, **native_hashes}, "analysis_source_sha256": source_hashes,
        "integrity": {"independent_confirmations": 0, "fresh_process_reload_exact": True,
            "native_timing_control": True, "fresh_split_disjoint": True, "heldout_used_for_selection": False}}
    write(run / "campaign_results.json", value)
    return {"aggregate": str(run / "campaign_results.json"), "sha256": sha(run / "campaign_results.json"), "classification": value["classification"]}


def expected(run):
    value = read(run / "campaign_results.json")
    require(value["confirmed"] is False and value["result_sha256"] == sha(run / "result.json"), "Aggregate/result mismatch")
    for path, expected_hash in value["analysis_source_sha256"].items():
        require(sha(path) == expected_hash, "Analysis source changed")
    return value


def metadata(run, value):
    accuracy = value["accuracy"]["fast"]
    result = {"hypothesis_id": STUDY, "confirmed": False, "confirmation_status": value["classification"],
        "accuracy": accuracy["overall"], "primary_metric": accuracy["overall"], "correct": accuracy["correct"], "count": accuracy["count"],
        "aggregate_sha256": sha(run / "campaign_results.json"), "accuracy_scope": SCOPE,
        "graph_tokens_per_second": value["whole_model_throughput"]["fast"]["cuda_graph"],
        "graph_speed_ratio_to_native_repeat": value["whole_model_ratios"]["cuda_graph"]["to_native_repeat"]}
    native = value["native_graph_comparison"]
    if native.get("applies_to_exported_tensors") is True:
        result.update({"isolated_graph_" + mode + "_speed_ratio_to_native": measurements["wall_seconds"]["speed_ratio_to_native"]
                       for mode, measurements in native["measurements"].items()})
    return result


def readback(run, registry, key=None):
    value = expected(run)
    if key is None:
        stored = read(registry); key = stored["storage"]["result_hash"]
        require(stored["aggregate_sha256"] == sha(run / "campaign_results.json"), "Registry aggregate changed")
        require(stored["report_sha256"] == sha(stored["report_path"]), "Registered report changed")
    db = logger(); experiment = db.experiments_collection.get(ids=[key], include=["metadatas"])
    require(experiment["ids"] == [key] and all(experiment["metadatas"][0].get(k) == v for k, v in metadata(run, value).items()), "Experiment readback mismatch")
    hypothesis = db.hypotheses_collection.get(ids=[STUDY], include=["metadatas"])
    require(hypothesis["ids"] == [STUDY], "Missing hypothesis")
    h = hypothesis["metadatas"][0]
    require(h.get("tested") is True and h.get("confirmed") is False and h.get("confirmation_status") == value["classification"], "Hypothesis readback mismatch")
    require(h.get("evidence_count") == 1 and json.loads(h["direct_result_hashes_json"]) == [key], "Incorrect evidence link/count")
    return {"readback_verified": True, "result_hash": key, **metadata(run, value)}


def store(run, registry):
    require(not registry.exists(), "Registry already exists")
    require(REPORT.is_file(), "Write the human-readable report from the aggregate before registration")
    value = expected(run)
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = {"id": STUDY, "name": "Recovery-aware execution of Qwen benchmark-derived compact final block", "category": "compression",
        "tested": True, "confirmed": False, "evidence_count": 1, "direct_experiment_count": 1,
        "confirmation_status": value["classification"], "tested_scope": SCOPE,
        "description": "BF16 geometry and fused execution selected on development; independent fresh choices and exact exported-checkpoint reload. Local gain and full-model throughput recorded separately."}
    metrics = {"accuracy": value["accuracy"]["fast"]["overall"], "native_accuracy": value["accuracy"]["native"]["overall"],
        "compact_control_accuracy": value["accuracy"]["control"]["overall"], "graph_tokens_per_second": value["whole_model_throughput"]["fast"]["cuda_graph"],
        "graph_speed_ratio_to_native_repeat": value["whole_model_ratios"]["cuda_graph"]["to_native_repeat"]}
    metrics.update({k: v for k, v in metadata(run, value).items() if k.startswith("isolated_graph_")})
    experiment = ExperimentResult(experiment_id="jspace-compact-performance-" + value["validation_fingerprint"][:16], hypothesis_id=STUDY,
        metrics=metrics, primary_metric=metrics["accuracy"], model_architecture=[2048] * 35 + [1536],
        model_parameters=value["result"]["results"]["fast"]["parameter_count"],
        training_time=sum(m["elapsed_seconds"] for k, m in value["manifests"].items() if k.startswith("train_")),
        model_checkpoint=str(run / "fast.pt"), observations=[SCOPE, value["classification"],
            "Checkpoint is a packed block-plus-final-norm delta requiring the pinned native model.",
            "Runtime claims apply only to the explicitly measured device, shapes, execution mode, and tensors."])
    db = logger(); prior = db.hypotheses_collection.get(ids=[STUDY], include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to replace tested hypothesis")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key], metadatas=[metadata(run, value)])
    db.hypotheses_collection.update(ids=[STUDY], metadatas=[{"direct_result_hashes_json": json.dumps([key]),
        "aggregate_sha256": sha(run / "campaign_results.json"), "report_path": str(REPORT)}])
    receipt = readback(run, registry, key)
    write(registry, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "storage": receipt,
                    "aggregate_sha256": sha(run / "campaign_results.json"), "analysis_source_sha256": sha(__file__),
                    "report_path": str(REPORT), "report_sha256": sha(REPORT)})
    return receipt


def closeout(run, registry):
    value, receipt = expected(run), read(run / "registry_readback.json")
    require(receipt["readback_verified"] is True and receipt == readback(run, registry), "Independent readback changed")
    event = {"schema_version": "jspace.research-log.v1", "id": "jspace-compact-performance-" + value["validation_fingerprint"],
        "event": "compact_performance_fresh_validation_completed", "study_id": STUDY, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "confirmed": False, "classification": value["classification"], "scope": SCOPE, "accuracy": value["accuracy"],
        "whole_model_ratios": value["whole_model_ratios"], "paired": value["paired"], "new_experimental_evidence": True,
        "native_graph_comparison": value["native_graph_comparison"],
        "independent_confirmations": 0, "registry_result_id": receipt["result_hash"],
        "aggregate_sha256": sha(run / "campaign_results.json"), "registry_sha256": sha(registry),
        "report_path": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT)}
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()), "Research event already logged")
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    write(run / "finalization.json", {"study_id": STUDY, "status": "completed", "confirmed": False,
        "classification": value["classification"], "registry_readback": receipt, "research_event": event,
        "analysis_source_sha256": sha(__file__), "aggregate_sha256": sha(run / "campaign_results.json")})
    return {"finalized": True, "registry": str(registry), "finalization_sha256": sha(run / "finalization.json")}


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Start Python with CUDA_VISIBLE_DEVICES='' for CPU-only finalization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "store", "readback", "closeout"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/hypothesis_registry/jspace_compact_performance_20261113.json")
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--native-comparison-root", type=Path)
    args = parser.parse_args()
    if args.stage == "analyze":
        output = analyze(args.root, args.supplement, args.native_comparison_root)
    else:
        output = {"store": store, "readback": readback, "closeout": closeout}[args.stage](args.root, args.registry)
    if args.stage == "readback":
        write(args.root / "registry_readback.json", output)
    print(json.dumps(output, allow_nan=False))
