"""Audit preserved training, fresh-request outputs and explicit source reviews."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from experiments.structure_net.llama_neighbourhood_tensorization import Config, digest
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit as audit_native
from neural_architecture_lab.llama_neighbourhood_tensorization_hypothesis import NeighbourhoodEvidence


def audit_producer(root):
    p = json.loads((root / "producer.json").read_text())
    if digest({k: v for k, v in p.items() if k != "fingerprint"}) != p["fingerprint"]:
        raise ValueError("producer fingerprint differs")
    snapshots = {sha256(path) for path in (root / "producer_source").iterdir()}
    if not set(p["source_hashes"].values()) <= snapshots:
        raise ValueError("producer source snapshot missing or changed")
    return p


def audit_training(root):
    p = audit_producer(root)
    t = json.loads((root / "training.json").read_text())
    if t["status"] != "completed" or t["stage"] != "training" or t["config"] != asdict(Config()):
        raise ValueError("training configuration mismatch")
    if t["producer_fingerprint"] != p["fingerprint"] or [g["group"] for g in t["groups"]] != list(range(14)):
        raise ValueError("missing or unpaired neighbourhoods")
    for g in t["groups"]:
        if not all(g[k] for k in ("checkpoint_reload_exact", "teacher_frozen", "all_factors_have_gradients")):
            raise ValueError("training validity gate failed")
        if not 0 <= g["dense_control_nmse"] <= 1e-6:
            raise ValueError("dense teacher boundary parity failed")
        if [r["step"] for r in g["history"]] != list(range(97)):
            raise ValueError("incomplete update history")
        for kind in ("initial", "selected"):
            if sha256(g[f"{kind}_checkpoint"]) != g[f"{kind}_sha256"]:
                raise ValueError("factor checkpoint changed")
        expected = min(r["development_nmse"] for r in g["history"] if "development_nmse" in r)
        if not math.isclose(g["selected_development_nmse"], expected, rel_tol=1e-12):
            raise ValueError("development selection changed")
    return t


def audit_evaluation(root, examples):
    audit_producer(root)
    terminal = json.loads((root / "result.json").read_text())
    if terminal["status"] != "completed":
        raise ValueError("incomplete evaluation")
    for name in ("outputs", "runtime"):
        if sha256(root / f"{name}.json") != terminal[f"{name}_sha256"]:
            raise ValueError("evaluation artifact changed")
    outputs = json.loads((root / "outputs.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    rows = outputs["rows"]
    if len(rows) != len(examples) or {r["id"] for r in rows} != set(examples):
        raise ValueError("missing or duplicated development requests")
    for r in rows:
        if r["prompt_tokens"] != len(examples[r["id"]]["prompt_ids"]) or r["cap"] != examples[r["id"]]["cap"]:
            raise ValueError("unpaired workload")
        if not 0 < len(r["output_ids"]) <= r["cap"] or not 0 < r["ttft_seconds"] <= r["wall_seconds"]:
            raise ValueError("invalid generation record")
        if not r["eos"] and len(r["output_ids"]) != r["cap"]:
            raise ValueError("generation silently truncated")
    expected = {(key, repeat) for key in list(examples)[:4] for repeat in range(3)}
    if {(r["id"], r["repeat"]) for r in runtime["rows"]} != expected or len(runtime["rows"]) != 12:
        raise ValueError("incomplete fixed-token timing")
    if any(r["cap"] != 32 or not 0 < r["ttft_seconds"] <= r["wall_seconds"] for r in runtime["rows"]):
        raise ValueError("invalid fixed-token runtime")
    return outputs, runtime


def finalize(args):
    if args.output.exists():
        raise ValueError("preserve completed aggregate")
    frozen = json.loads(args.fixtures.read_text())
    examples = {e["id"]: e for e in frozen["examples"] if e["split"] == "development"}
    hashes = [digest(e["prompt_ids"]) for e in frozen["examples"]]
    if len(set(hashes)) != len(hashes):
        raise ValueError("split overlap")
    training = audit_training(args.training)
    results = {arm: audit_evaluation(getattr(args, arm), examples) for arm in ("dense", "initial", "distilled")}
    native, native_producer, native_examples = audit_native(args.native_quality)
    timing_native, _, timing_examples = audit_native(args.native_runtime)
    if native["config"]["drafts"] != [8] or native_producer["model_sha256"] != "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff":
        raise ValueError("wrong deployment baseline")
    for key, example in examples.items():
        if native_examples[key]["prompt_ids"] != example["prompt_ids"]:
            raise ValueError("native baseline has different input")
    for key, example in timing_examples.items():
        if example["prompt_ids"] != examples[key]["prompt_ids"] or example["cap"] != 32 or example["stop_eos"]:
            raise ValueError("fixed-token control differs")
    review = json.loads(args.reviews.read_text())
    if review["fixtures_sha256"] != sha256(args.fixtures):
        raise ValueError("review input identity mismatch")
    if sha256(review["protocol"]) != review["protocol_sha256"]:
        raise ValueError("review protocol changed")
    reviewed = {(r["arm"], r["id"]): r for r in review["items"]}
    if len(reviewed) != 32 or len(review["items"]) != 32:
        raise ValueError("explicit review required for all 32 arm/request pairs")
    arms = {}
    for arm in ("dense", "initial", "distilled", "q4_lookup8"):
        rows = native["rows"] if arm == "q4_lookup8" else results[arm][0]["rows"]
        count = 0
        for r in rows:
            v = reviewed[(arm, r["id"])]
            if v["text_sha256"] != digest(r["text"]) or not v["rationale"] or type(v["passed"]) is not bool:
                raise ValueError("unmatched or incomplete review")
            if v["source_sha256"] != digest(examples[r["id"]]["fragments"]):
                raise ValueError("review source identity differs")
            expected_sources = examples[r["id"]]["fragments"]
            if len(v["source_quotes"]) != len(expected_sources) or any(
                q["id"] != f["id"] or q["quote"] != f["reference_fact"] or q["quote"] not in f["source"]
                for q, f in zip(v["source_quotes"], expected_sources)):
                raise ValueError("review source quotes differ")
            count += int(v["passed"])
        timings = [r for r in timing_native["rows"] if r["draft"] == 8] if arm == "q4_lookup8" else results[arm][1]["rows"]
        arms[arm] = {"development_passed": count, "development_total": 8,
                     "fixed32_seconds": sum(r["wall_seconds"] for r in timings),
                     "fixed32_tokens_per_second": 32 * len(timings) / sum(r["wall_seconds"] for r in timings),
                     "mean_natural_request_seconds": statistics.mean(r["wall_seconds"] for r in rows),
                     "weight_bytes": native["model_weight_bytes"] if arm == "q4_lookup8" else results[arm][0]["parameter_bytes"]}
    parameter_ratio = arms["distilled"]["weight_bytes"] / arms["dense"]["weight_bytes"]
    # This finalizer deliberately implements the prospective development stop.
    if arms["distilled"]["development_passed"] == 8:
        raise ValueError("development passed: perform the conditional held-out stage before finalizing")
    classification = "tensorized_student_development_quality_failed"
    paths = {"fixtures": args.fixtures, "reviews": args.reviews, "training": args.training / "training.json"}
    paths["review_protocol"] = Path(review["protocol"])
    for arm in ("dense", "initial", "distilled"):
        paths[f"{arm}_outputs"] = getattr(args, arm) / "outputs.json"
        paths[f"{arm}_runtime"] = getattr(args, arm) / "runtime.json"
    paths.update(native_quality=args.native_quality / "raw.json", native_runtime=args.native_runtime / "raw.json")
    result = {"schema": "nal.llama-neighbourhood-campaign.v1", "hypothesis_id": training["hypothesis_id"],
              "status": "completed_exploratory", "classification": classification,
              "artifacts": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
              "arms": arms, "parameter_ratio": parameter_ratio, "source_review_completed": True,
              "judge": "assistant source-grounded review and structural/token checks; no independent judge model",
              "trained_neighbourhoods": 14, "optimizer_updates": 14 * 96,
              "direct_experiment_count": 14 + 32 + 60,
              "trained_factor_parameters": sum(g["trained_parameters"] for g in training["groups"]),
              "local_error_reduction_fraction": [1 - g["selected_development_nmse"] / g["initial_development_nmse"] for g in training["groups"]],
              "selected_steps": [g["selected_step"] for g in training["groups"]],
              "training_seconds": sum(g["seconds"] for g in training["groups"]),
              "prototype_fixed32_speedup_over_dense": arms["distilled"]["fixed32_tokens_per_second"] / arms["dense"]["fixed32_tokens_per_second"],
              "prototype_fixed32_speedup_over_native_q4_lookup": arms["distilled"]["fixed32_tokens_per_second"] / arms["q4_lookup8"]["fixed32_tokens_per_second"],
              "heldout_tested": False, "application_accuracy_tested": False,
              "unexecuted_conditional_stages": ["held-out quality", "five-repeat final latency campaign", "native factor-kernel integration", "factor quantization"],
              "literal_line_format_pass_counts": {arm: sum(bool(v["literal_line_format_passed"]) for v in review["items"] if v["arm"] == arm) for arm in arms},
              "limitations": ["one trained model", "synthetic repeated final-decision dialogues", "development boundary-error selection",
                              "PyTorch prototype has no CUDA graphs or lookup; native deployment baseline does",
                              "fixed32 timing includes special tokens and is systems evidence, not summary quality",
                              "14 independently trained groups receive teacher inputs; assembly changes their input distribution"]}
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("artifacts", "limitations")}), flush=True)


def register(args):
    c = json.loads(args.campaign.read_text())
    for item in c["artifacts"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("campaign evidence changed")
    e = NeighbourhoodEvidence(str(args.campaign), sha256(args.campaign), str(args.report), sha256(args.report),
            c["trained_neighbourhoods"], c["arms"]["distilled"]["development_passed"], 8, c["parameter_ratio"])
    record = e.record()
    if record["confirmation_status"] != c["classification"]:
        raise ValueError("classification mismatch")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    assert exact["ids"] == [record["id"]]
    for key in ("tested", "confirmed", "confirmation_status", "direct_experiment_count"):
        if exact["metadatas"][0][key] != record[key]:
            raise RuntimeError("exact registry readback mismatch")
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise RuntimeError("semantic readback missing")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact_readback": exact["metadatas"][0], "semantic_readback_ids": semantic["ids"][0]}
    for path in (args.campaign.parent / "registry_readback.json", Path("data/hypothesis_registry") / f"{record['id']}.json"):
        if path.exists():
            raise ValueError("preserve existing registry receipt")
        write_json(path, receipt)
        if json.loads(path.read_text())["record"] != record:
            raise RuntimeError("full JSON registry readback mismatch")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("finalize")
    for name in ("training", "fixtures", "dense", "initial", "distilled", "native-quality", "native-runtime", "reviews", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("register")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    (finalize if args.command == "finalize" else register)(args)


if __name__ == "__main__":
    main()
