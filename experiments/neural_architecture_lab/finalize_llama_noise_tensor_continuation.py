"""Audit paired continuation and retain conservative searchable evidence."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from experiments.structure_net.llama_noise_tensor_continuation import Config, FIXTURES, HYPOTHESIS_ID, PARENT
from experiments.structure_net.llama_neighbourhood_tensorization import digest
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.neural_architecture_lab.finalize_llama_neighbourhood_tensorization import audit_producer, audit_evaluation
from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit as audit_native
from neural_architecture_lab.llama_noise_tensor_continuation_hypothesis import ContinuationEvidence, heldout_gate
from experiments.structure_net.campaign_llama_noise_tensor_heldout import select_arm


def training(root, arm, stage):
    producer = audit_producer(root)
    t = json.loads((root / "training.json").read_text())
    if t["status"] != "completed" or t["stage"] != stage or t["arm"] != arm or t["config"] != asdict(Config()):
        raise ValueError("training contract mismatch")
    if producer["hypothesis_id"] != HYPOTHESIS_ID or t["producer_fingerprint"] != producer["fingerprint"]:
        raise ValueError("producer mismatch")
    if [g["group"] for g in t["groups"]] != list(range(14)):
        raise ValueError("all 14 groups required")
    steps = Config().local_steps if stage == "local_continuation" else Config().global_steps
    for g in t["groups"]:
        if g["updates"] != steps or not all(g[k] for k in ("all_factors_have_gradients", "frozen_weights_unchanged", "checkpoint_reload_exact")):
            raise ValueError("training validity failed")
        expected = list(range(steps+1)) if stage == "local_continuation" else list(range(1, steps+1))
        if [r["step"] for r in g["history"]] != expected:
            raise ValueError("incomplete updates")
        if sha256(g["selected_checkpoint"]) != g["selected_sha256"]:
            raise ValueError("checkpoint changed")
        if stage == "local_continuation":
            for kind in ("initial", "parent"):
                if sha256(g[kind+"_checkpoint"]) != g[kind+"_sha256"]:
                    raise ValueError("continuation identity changed")
            if g["optimizer_state_resumed"] or not g["teacher_frozen"]:
                raise ValueError("invalid resume/teacher contract")
            validations = [r for r in g["history"] if "development_balanced_loss" in r]
            selected = min(validations, key=lambda r: r["development_balanced_loss"])
            if selected["step"] != g["selected_step"]:
                raise ValueError("development selection mismatch")
            for r in g["history"][1:]:
                noise = arm == "noise" and r["step"] % 2 == 0
                if r["input_noise_applied"] != noise or r["teacher_target_recomputed"] != noise:
                    raise ValueError("noisy input/target pairing mismatch")
        elif g["selection"] != "final_fixed_step":
            raise ValueError("global best-of-run selection forbidden")
    return t


def review_count(rows, arm, examples, reviews):
    count = 0
    for row in rows:
        r = reviews[(arm, row["id"])]
        source = examples[row["id"]]["fragments"]
        if r["text_sha256"] != digest(row["text"]) or r["source_sha256"] != digest(source):
            raise ValueError("unpaired source review")
        if type(r["passed"]) is not bool or not r["rationale"]:
            raise ValueError("explicit review decision required")
        expected = [{"id": f["id"], "quote": f["reference_fact"]} for f in source]
        if r["source_quotes"] != expected or any(f["reference_fact"] not in f["source"] for f in source):
            raise ValueError("source quotation mismatch")
        count += int(r["passed"])
    return count


def finalize(args):
    if args.output.exists():
        raise ValueError("new aggregate required")
    examples = {e["id"]: e for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] == "development"}
    stages = {}
    results = {}
    paths = {"fixtures": FIXTURES, "parent_training": PARENT, "reviews": args.reviews}
    parent_campaign_path = Path("data/experiments/llama_neighbourhood_tensorization/20261014_registered/campaign_results.json")
    parent_campaign = json.loads(parent_campaign_path.read_text())
    if parent_campaign["artifacts"]["fixtures"]["sha256"] != sha256(FIXTURES):
        raise ValueError("reused parent controls use different fixtures")
    for item in parent_campaign["artifacts"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("parent control evidence changed")
    paths["parent_campaign"] = parent_campaign_path
    for label in ("local_noise_pilot_v1", "global_memory_pilot_v1"):
        root = args.root / ("20261014_"+label)
        audit_producer(root)
        pilot = json.loads((root / "training.json").read_text())
        if pilot["status"] != "completed" or pilot["stage"] != "systems_lifecycle_only_not_quality_evidence":
            raise ValueError("systems pilot failed")
        paths[label] = root / "training.json"
    identity = args.root / "20261014_teacher_identity_v1.json"
    verified = json.loads(identity.read_text())
    if verified["status"] != "passed" or not all(verified["checks"].values()):
        raise ValueError("original HF teacher identity check failed")
    paths["teacher_identity"] = identity
    prepared_root = args.root / "20261014_prepared_v1"
    audit_producer(prepared_root)
    paths["prepared"] = prepared_root / "prepared.json"
    for label in ("clean_integrity_v1", "noise_initial_integrity_v1", "size_arithmetic_v1"):
        path = args.root / ("20261014_"+label+".json")
        if label != "size_arithmetic_v1" and json.loads(path.read_text())["status"] != "passed":
            raise ValueError("factor initialization audit failed")
        paths[label] = path
    for arm in ("clean", "noise"):
        local_root = args.root / f"20261014_{arm}_local_v2"
        global_root = args.root / f"20261014_{arm}_global_v2"
        eval_root = args.root / f"20261014_{arm}_development_v2"
        local = training(local_root, arm, "local_continuation")
        global_ = training(global_root, arm, "global_recovery")
        if global_["local_root"] != str(local_root) or global_["local_training_sha256"] != sha256(local_root / "training.json"):
            raise ValueError("global/local checkpoint lineage mismatch")
        stages[arm] = (local, global_)
        results[arm] = audit_evaluation(eval_root, examples)
        for name, path in (("local", local_root / "training.json"), ("global", global_root / "training.json"),
                           ("outputs", eval_root / "outputs.json"), ("runtime", eval_root / "runtime.json")):
            paths[arm+"_"+name] = path
    for stage in (0, 1):
        for c, n in zip(stages["clean"][stage]["groups"], stages["noise"][stage]["groups"]):
            if [(r["step"], r.get("example")) for r in c["history"]] != [(r["step"], r.get("example")) for r in n["history"]]:
                raise ValueError("arms received different update schedules")
    native, producer, timing_examples = audit_native(args.native_runtime)
    if producer["model_sha256"] != "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff":
        raise ValueError("wrong native Q4 baseline")
    if native["config"]["drafts"] != [8] or len(native["rows"]) != 12:
        raise ValueError("fresh paired Q4+lookup timing required")
    for key, e in timing_examples.items():
        if e["prompt_ids"] != examples[key]["prompt_ids"] or e["cap"] != 32 or e["stop_eos"]:
            raise ValueError("timing workload mismatch")
    paths["native_runtime"] = args.native_runtime / "raw.json"
    review = json.loads(args.reviews.read_text())
    if review["fixtures_sha256"] != sha256(FIXTURES) or sha256(review["protocol"]) != review["protocol_sha256"]:
        raise ValueError("review protocol/input changed")
    paths["review_protocol"] = Path(review["protocol"])
    reviewed = {(r["arm"], r["id"]): r for r in review["items"]}
    if len(reviewed) != 16 or len(review["items"]) != 16:
        raise ValueError("review all paired development outputs")
    arms = {}
    for arm in ("clean", "noise"):
        outputs, runtime = results[arm]
        passed = review_count(outputs["rows"], arm, examples, reviewed)
        local, global_ = stages[arm]
        arms[arm] = {"development_passed": passed, "development_total": 8,
                     "initial_global_development_kl": global_["initial_development"]["mean_kl"],
                     "final_global_development_kl": global_["final_development"]["mean_kl"],
                     "final_global_development_ce": global_["final_development"]["mean_ce"],
                     "local_selected_steps": [g["selected_step"] for g in local["groups"]],
                     "local_updates": sum(g["updates"] for g in local["groups"]),
                     "global_updates": sum(g["updates"] for g in global_["groups"]),
                     "training_seconds": sum(g["seconds"] for t in (local, global_) for g in t["groups"]),
                     "peak_training_cuda_bytes": max(g["peak_cuda_bytes"] for t in (local, global_) for g in t["groups"]),
                     "fixed32_tokens_per_second": runtime["aggregate_tokens_per_second"],
                     "weight_bytes": outputs["parameter_bytes"]}
    native_speed = 32*len(native["rows"])/sum(r["wall_seconds"] for r in native["rows"])
    heldout = None
    if max(a["development_passed"] for a in arms.values()) == 8:
        if args.heldout is None:
            raise ValueError("development passed: complete conditional heldout before finalization")
        heldout = json.loads(args.heldout.read_text())
        if heldout["status"] != "completed_source_reviewed" or not heldout["source_review_completed"]:
            raise ValueError("final source review incomplete")
        for item in heldout["artifacts"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("final-test evidence changed")
        chosen = select_arm({a:v["development_passed"] for a,v in arms.items()},
                            {a:v["final_global_development_kl"] for a,v in arms.items()})
        if heldout["selected_arm"] != chosen or heldout["quality_gate"] != heldout_gate(
            heldout["pass_counts"][chosen], heldout["pass_counts"]["dense"], heldout["pass_counts"]["q4_lookup8"]):
            raise ValueError("final selection or quality gate mismatch")
        paths["heldout"] = args.heldout
    direct_count = 56 + 16 + 24 + 12 + (heldout["direct_experiment_count"] if heldout else 0)
    evidence = ContinuationEvidence(str(args.output), "0"*64, "report_pending", "0"*64,
        arms["clean"]["development_passed"], arms["noise"]["development_passed"], direct_count,
        heldout is not None, heldout["quality_gate"] if heldout else False)
    campaign = {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID, "status": "completed_exploratory",
                "classification": evidence.record()["confirmation_status"], "config": asdict(Config()),
                "arms": arms, "native_q4_lookup_fixed32_tokens_per_second": native_speed,
                "prototype_speed_ratio_to_native_q4_lookup": {a: v["fixed32_tokens_per_second"]/native_speed for a,v in arms.items()},
                "noise_benefit_observed": arms["noise"]["development_passed"] > arms["clean"]["development_passed"],
                "reused_parent_quality_controls": {a: parent_campaign["arms"][a]["development_passed"]
                    for a in ("dense", "distilled", "q4_lookup8")},
                "systems_pilots": 2, "systems_pilot_failures": 0,
                "direct_experiment_count": direct_count, "local_groups_completed": 28, "global_groups_completed": 28,
                "total_additional_optimizer_updates": 2*14*(Config().local_steps+Config().global_steps),
                "heldout_tested": heldout is not None, "heldout": heldout, "application_accuracy_tested": False,
                "speed_target_met": False, "source_review_completed": True,
                "judge": "assistant source-grounded review and structural/token checks; no independent judge model",
                "artifacts": {k: {"path": str(p), "sha256": sha256(p)} for k,p in paths.items()},
                "limitations": ["one paired parent/noise seed", "adaptive synthetic development set",
                                "noise package combines weight and input perturbations",
                                "block-coordinate global optimization, not simultaneous full-student optimizer",
                                "FP16 PyTorch factors versus native Q4 with CUDA graphs and prompt lookup",
                                "fixed32 speed diagnostic does not measure quality-matched useful-summary throughput"],
                "unexecuted_conditional_stages": ([] if heldout else ["heldout quality"]) +
                    ["native factor kernels", "factor quantization", "five-repeat quality-matched final speed"]}
    analysis_paths = [Path(__file__), Path("src/neural_architecture_lab/llama_noise_tensor_continuation_hypothesis.py"),
                      Path("experiments/structure_net/campaign_llama_noise_tensor_heldout.py")]
    snapshots = args.output.parent / "analysis_source"
    snapshots.mkdir(parents=True, exist_ok=False)
    for p in analysis_paths:
        shutil.copyfile(p, snapshots / p.name)
    campaign["analysis_source_hashes"] = {str(p):sha256(p) for p in analysis_paths}
    write_json(args.output, campaign)
    print(json.dumps({k:v for k,v in campaign.items() if k not in ("artifacts", "limitations", "heldout")}), flush=True)


SCHEMA_VERSION = "nal.llama-noise-tensor-continuation-campaign.v1"


def finalize_heldout(args):
    if args.output.exists():
        raise ValueError("new final-test summary required")
    selection = json.loads((args.root / "selection.json").read_text())
    chosen = select_arm(selection["development_pass_counts"], selection["development_kl"])
    if chosen != selection["selected_arm"]:
        raise ValueError("final arm selection differs")
    examples = {e["id"]:e for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] == "heldout"}
    candidate, _ = audit_evaluation(args.root / "candidate", examples)
    dense, _ = audit_evaluation(args.root / "dense", examples)
    native, producer, native_examples = audit_native(args.root / "q4_lookup8")
    if producer["model_sha256"] != "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff":
        raise ValueError("wrong Q4 final-test control")
    if native["config"]["drafts"] != [8] or native["config"]["repeats"] != 1 or len(native["rows"]) != 20:
        raise ValueError("final-test control cell mismatch")
    for key, e in examples.items():
        if native_examples[key]["prompt_ids"] != e["prompt_ids"] or native_examples[key]["cap"] != e["cap"]:
            raise ValueError("final controls have different workload")
    reviews = json.loads(args.reviews.read_text())
    if reviews["fixtures_sha256"] != sha256(FIXTURES) or sha256(reviews["protocol"]) != reviews["protocol_sha256"]:
        raise ValueError("final review identity mismatch")
    reviewed = {(r["arm"],r["id"]):r for r in reviews["items"]}
    if len(reviewed) != 60 or len(reviews["items"]) != 60:
        raise ValueError("all 60 final source reviews required")
    counts = {a:review_count(rows, a, examples, reviewed) for a,rows in
              ((chosen,candidate["rows"]),("dense",dense["rows"]),("q4_lookup8",native["rows"]))}
    paths = {"selection":args.root/"selection.json", "reviews":args.reviews,
             "candidate_outputs":args.root/"candidate/outputs.json", "candidate_runtime":args.root/"candidate/runtime.json",
             "dense_outputs":args.root/"dense/outputs.json", "dense_runtime":args.root/"dense/runtime.json",
             "q4_quality":args.root/"q4_lookup8/raw.json", "protocol":Path(reviews["protocol"])}
    write_json(args.output, {"status":"completed_source_reviewed", "selected_arm":chosen, "pass_counts":counts,
        "total":20, "quality_gate":heldout_gate(counts[chosen],counts["dense"],counts["q4_lookup8"]),
        "source_review_completed":True, "direct_experiment_count":84,
        "artifacts":{k:{"path":str(p),"sha256":sha256(p)} for k,p in paths.items()},
        "boundary":"one synthetic final-decision proxy test, not measured application accuracy"})
    print(json.dumps({"pass_counts":counts,"quality_gate":heldout_gate(counts[chosen],counts["dense"],counts["q4_lookup8"])}))


def register(args):
    c = json.loads(args.campaign.read_text())
    for item in c["artifacts"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("campaign evidence changed")
    e = ContinuationEvidence(str(args.campaign), sha256(args.campaign), str(args.report), sha256(args.report),
        c["arms"]["clean"]["development_passed"], c["arms"]["noise"]["development_passed"],
        c["direct_experiment_count"], c["heldout_tested"], c["heldout"]["quality_gate"] if c["heldout"] else False)
    record = e.record()
    if record["confirmation_status"] != c["classification"]:
        raise ValueError("registry classification mismatch")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    if exact["ids"] != [record["id"]]:
        raise ValueError("exact hypothesis ID readback failed")
    for key in ("tested", "confirmed", "confirmation_status", "direct_experiment_count"):
        if exact["metadatas"][0][key] != record[key]:
            raise ValueError("exact registry fields mismatch")
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise ValueError("semantic readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact_readback": exact["metadatas"][0], "semantic_readback_ids": semantic["ids"][0]}
    for path in (args.campaign.parent / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS_ID+".json")):
        if path.exists():
            raise ValueError("preserve prior receipt")
        write_json(path, receipt)
        if json.loads(path.read_text())["record"] != record:
            raise ValueError("full JSON record readback mismatch")
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("finalize")
    for name in ("root", "native-runtime", "reviews", "output"):
        f.add_argument("--"+name, type=Path, required=True)
    f.add_argument("--heldout", type=Path)
    r = sub.add_parser("register")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--report", type=Path, required=True)
    h = sub.add_parser("heldout")
    for name in ("root", "reviews", "output"):
        h.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    {"finalize":finalize, "register":register, "heldout":finalize_heldout}[args.command](args)


if __name__ == "__main__":
    main()
