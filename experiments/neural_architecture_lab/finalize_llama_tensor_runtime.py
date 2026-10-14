"""Apply frozen runtime gates, audit evidence and verify registry readback."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.neural_architecture_lab.analyze_llama_tensor_runtime import audit, summarize
from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit as audit_native
from experiments.structure_net.llama_tensor_runtime import HYPOTHESIS_ID, PLAN, FIXTURES, digest
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_tensor_runtime_hypothesis import RuntimeEvidence, classification


def review_counts(rows, examples, path):
    payload = json.loads(path.read_text())
    reviews = {(r["arm"], r["id"], r["text_sha256"]):r for r in payload["items"]}
    expected = {(r["arm"], r["id"], digest(r["text"])) for r in rows}
    if len(reviews) != len(payload["items"]) or set(reviews) != expected:
        raise ValueError("all distinct outputs require paired explicit reviews")
    counts = {}
    for arm in {r["arm"] for r in rows}:
        count = 0
        for identifier, e in examples.items():
            decisions = [r for (a, i, _),r in reviews.items() if (a,i) == (arm,identifier)]
            if not decisions:
                raise ValueError("missing arm/case source review")
            for r in decisions:
                if r["source_sha256"] != digest(e["fragments"]) or type(r["passed"]) is not bool or not r["rationale"]:
                    raise ValueError("source identity or review decision missing")
                quotes = [{"id":f["id"], "quote":f["reference_fact"]} for f in e["fragments"]]
                if r["source_quotes"] != quotes or any(f["reference_fact"] not in f["source"] for f in e["fragments"]):
                    raise ValueError("source quotations differ")
            count += int(all(r["passed"] for r in decisions))
        counts[arm] = count
    return counts


def development(root, reviews):
    examples = {e["id"]:e for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] == "development"}
    rows, producer = audit(root, examples)
    if set(producer["arms"]) != {"reference", "eager", "graph", "graph_lookup8"} or producer["repeats"] != 3:
        raise ValueError("fixed development design required")
    counts = review_counts(rows, examples, reviews)
    summaries = summarize(rows, examples)
    eligible = [a for a in ("eager", "graph", "graph_lookup8") if counts[a] == 8]
    priority = {"graph":0, "eager":1, "graph_lookup8":2}
    chosen = min(eligible, key=lambda a:(summaries[a]["sum_median_seconds"],priority[a])) if eligible else None
    return chosen, counts, summaries, len(rows)


def select(args):
    chosen, counts, summary, _ = development(args.development, args.reviews)
    if args.output.exists():
        raise ValueError("immutable selection path required")
    write_json(args.output, {"hypothesis_id":HYPOTHESIS_ID, "selected_arm":chosen,
        "development_pass_counts":counts, "development_summaries":summary,
        "development_rows_sha256":sha256(args.development / "rows.json"), "reviews_sha256":sha256(args.reviews),
        "rule":"8/8 source reviews, lowest summed median complete latency; tie graph, eager, lookup"})
    print(json.dumps({"selected_arm":chosen, "pass_counts":counts,
        "seconds":{a:v["sum_median_seconds"] for a,v in summary.items()}}))


def finalize(args):
    if args.output.exists():
        raise ValueError("fresh aggregate required")
    chosen, counts, dev_summary, direct = development(args.development, args.development_reviews)
    selection = json.loads(args.selection.read_text())
    if selection["selected_arm"] != chosen or selection["reviews_sha256"] != sha256(args.development_reviews):
        raise ValueError("selection provenance mismatch")
    paths = [PLAN, Path("docs/07 - Status Reports/2026-10-14_llama32-tensor-runtime-warmup-amendment.md"),
             FIXTURES, args.development_reviews, args.development / "rows.json", args.development / "producer.json",
             args.development / "result.json", args.selection, args.pilot / "result.json", args.pilot / "producer.json"]
    pilot = json.loads((args.pilot / "result.json").read_text())
    if pilot["status"] != "passed" or pilot["label"] != "systems_lifecycle_only_not_quality_evidence":
        raise ValueError("CUDA validity gate failed")
    result = {"schema":"nal.llama-tensor-runtime-campaign.v1", "hypothesis_id":HYPOTHESIS_ID,
              "status":"completed_exploratory", "selected_arm":chosen,
              "development_pass_counts":counts, "development_summary":dev_summary,
              "scheduler":{"physical_gpu":0, "slots":1, "isolated":True},
              "scientific_boundary":"one fixed checkpoint, twenty fresh synthetic requests, five technical repetitions, assistant source review without independent judge",
              "application_accuracy_tested":False, "cross_request_cache_reuse":False,
              "failed_cells":0, "retried_cells":0, "reused_cells":0,
              "excluded_native_warmup_cells":5 if chosen else 0, "pilot":pilot}
    if chosen is None:
        result["classification"] = "runtime_development_quality_failed"
        result["heldout_tested"] = False
    else:
        examples = {e["id"]:e for e in json.loads(args.fixtures.read_text())["examples"]}
        if len(examples) != 20 or {e["seed"] for e in examples.values()} != set(range(24000,24020)):
            raise ValueError("fresh fixed final seed set required")
        rows, producer = audit(args.heldout, examples)
        if set(producer["arms"]) != {"reference",chosen} or producer["repeats"] != 5:
            raise ValueError("final student design mismatch")
        raw, native_producer, native_examples = audit_native(args.native)
        if native_producer["model_sha256"] != "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff" or raw["config"]["drafts"] != [8] or raw["config"]["repeats"] != 5:
            raise ValueError("native control mismatch")
        for identifier, e in examples.items():
            if native_examples[identifier]["prompt_ids"] != e["prompt_ids"] or native_examples[identifier]["cap"] != e["cap"]:
                raise ValueError("final native workload mismatch")
        native_rows = [dict(r, arm="q4_lookup8", output_ids=r["generated_ids"], eos=r["stopped_eos"])
                       for r in raw["rows"] if r["id"] != "excluded_warmup_longest"]
        if len(native_rows) != 100 or len(raw["rows"]) != 105:
            raise ValueError("native final/warmup cardinality mismatch")
        all_rows = rows+native_rows
        quality = review_counts(all_rows, examples, args.heldout_reviews)
        reviewed = json.loads(args.heldout_reviews.read_text())["items"]
        factual_counts = {arm:sum(all(r.get("factual_passed",r["passed"]) for r in reviewed
                                    if (r["arm"],r["id"]) == (arm,identifier)) for identifier in examples)
                          for arm in quality}
        summary = summarize(all_rows, examples)
        old_ratio = summary["reference"]["sum_median_seconds"]/summary[chosen]["sum_median_seconds"]
        native_ratio = summary["q4_lookup8"]["sum_median_seconds"]/summary[chosen]["sum_median_seconds"]
        tps_ratio = summary[chosen]["end_to_end_tokens_per_second"]/summary["q4_lookup8"]["end_to_end_tokens_per_second"]
        verdict = classification(quality[chosen],quality["reference"],quality["q4_lookup8"],old_ratio,native_ratio,tps_ratio)
        result.update(classification=verdict, heldout_tested=True, heldout_pass_counts=quality,
                      exploratory_factual_pass_counts=factual_counts,
                      heldout_summary=summary, prototype_latency_speedup=old_ratio,
                      native_latency_speedup=native_ratio, native_tokens_per_second_ratio=tps_ratio,
                      quality_gate=quality[chosen] >= max(19,quality["reference"],quality["q4_lookup8"]),
                      native_double_speed_gate=min(native_ratio,tps_ratio) >= 2)
        direct += len(all_rows)
        paths += [args.fixtures, args.heldout_reviews, args.heldout / "rows.json", args.heldout / "producer.json",
                  args.heldout / "result.json", args.native / "raw.json", args.native / "producer.json", args.native / "result.json"]
    result["direct_experiment_count"] = direct
    result["artifacts"] = {str(p):{"path":str(p),"sha256":sha256(p)} for p in paths}
    source_paths = [Path(__file__), Path("experiments/neural_architecture_lab/analyze_llama_tensor_runtime.py"),
                    Path("src/neural_architecture_lab/llama_tensor_runtime_hypothesis.py")]
    snapshots = args.output.parent / "analysis_source"
    snapshots.mkdir(parents=True)
    for p in source_paths:
        shutil.copyfile(p,snapshots / p.name)
    result["analysis_source_hashes"] = {str(p):sha256(p) for p in source_paths}
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ("classification","heldout_pass_counts","prototype_latency_speedup","native_latency_speedup","native_tokens_per_second_ratio","direct_experiment_count")}))


def register(args):
    campaign = json.loads(args.campaign.read_text())
    for item in campaign["artifacts"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("evidence changed after aggregation")
    record = RuntimeEvidence(str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),
                             campaign["classification"],campaign["direct_experiment_count"]).record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue",sent_dir="data/experiment_sent",rejected_dir="data/experiment_rejected",
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=["metadatas"])
    if exact["ids"] != [HYPOTHESIS_ID] or any(exact["metadatas"][0][k] != record[k]
            for k in ("tested","confirmed","confirmation_status","direct_experiment_count")):
        raise ValueError("exact registry readback failed")
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]],n_results=10)
    if HYPOTHESIS_ID not in semantic["ids"][0]:
        raise ValueError("semantic registry readback failed")
    receipt = {"status":"registered_and_read_back","record":record,"exact_readback":exact["metadatas"][0],
               "semantic_readback_ids":semantic["ids"][0]}
    for path in (args.campaign.parent / "registry_readback.json",Path("data/hypothesis_registry") / (HYPOTHESIS_ID+".json")):
        if path.exists():
            raise ValueError("immutable registry receipt required")
        write_json(path,receipt)
        if json.loads(path.read_text())["record"] != record:
            raise ValueError("full JSON registry readback failed")
    print(json.dumps(record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    s = sub.add_parser("select")
    for name in ("development","reviews","output"):
        s.add_argument("--"+name,type=Path,required=True)
    f = sub.add_parser("finalize")
    for name in ("development","development-reviews","selection","pilot","output"):
        f.add_argument("--"+name,type=Path,required=True)
    for name in ("fixtures","heldout","heldout-reviews","native"):
        f.add_argument("--"+name,type=Path)
    r = sub.add_parser("register")
    for name in ("campaign","report"):
        r.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    {"select":select,"finalize":finalize,"register":register}[args.command](args)


if __name__ == "__main__":
    main()
