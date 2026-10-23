"""Seal matched generated policies, then grade each complete development set."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT
from experiments.structure_net.ncm_paper_grade import grade

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-decoding-policy-plan.md")
OUTPUT = ROOT / "20261022_decode_outputs_v1"
SEAL = ROOT / "20261022_decode_output_manifest_v1.json"
POLICIES = ("local", "paper", "domain")


def seal():
    if SEAL.exists():
        raise ValueError("fresh immutable generation seal required")
    result = q.read(OUTPUT / "result.json")
    if result["status"] != "generated_pending_task_grading" or result["cases"] != 64 or not result["policies_share_exact_logits"]:
        raise ValueError("complete matched64-case generation required")
    producer = q.read(OUTPUT / "producer.json")
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("generation producer changed")
    ids = None
    for policy in POLICIES:
        rows = q.read(OUTPUT / (policy+".json"))
        signature = [(row["id"], row["source_sha256"]) for row in rows]
        if ids is not None and ids != signature:
            raise ValueError("policy source mismatch")
        ids = signature
    paths = [path for path in OUTPUT.glob("*") if path.is_file()]
    write_json(SEAL, {"status": "sealed_complete_generation", "producer_fingerprint": producer["fingerprint"],
        "artifacts": {str(path): file_hash(path) for path in paths}, "task_accuracy": None,
        "sealer_sha256": file_hash(__file__), "plan_sha256": file_hash(PLAN)})
    print(json.dumps({"status": "sealed_complete_generation", "policies": len(POLICIES)}), flush=True)


def evaluate(role):
    record = q.read(SEAL)
    for path, expected in record["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("sealed generation artifact changed")
    pilot = ROOT / ("20261022_recovered_judge_pilot_"+role+"_v1/result.json")
    previous = q.read(pilot)
    if previous["status"] != "completed" or previous["invalid"] or not previous["systems_only"]:
        raise ValueError("passed recovered-judge systems pilot required")
    concurrency_inputs = []
    if role == "primary":
        teacher_result = ROOT / "20261022_expanded_teacher_v2/result.json"
        if teacher_result.exists():
            concurrency_inputs.append(teacher_result)
        else:
            combined = ROOT / "20261022_combined_primary_pilot_v1/result.json"
            if not combined.exists() or q.read(combined)["status"] != "passed":
                raise ValueError("primary grading waits for teacher completion or a passed combined-concurrency pilot")
            concurrency_inputs.append(combined)
    results = {}
    for policy in POLICIES:
        source = OUTPUT / (policy+".json")
        destination = ROOT / ("20261022_decode_"+policy+"_grade_"+role+"_v1")
        results[policy] = grade(q.read(source), destination, [source, SEAL, pilot, PLAN, Path(__file__), *concurrency_inputs], role=role)
    receipt = ROOT / ("20261022_decode_quality_"+role+"_v1.json")
    if receipt.exists():
        raise ValueError("fresh policy evaluation receipt required")
    write_json(receipt, {"role": role, "judge": q.JUDGES[role], "results": results,
        "final_test_opened": False, "speed_tested": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["seal", "primary", "secondary"])
    command = parser.parse_args().command
    seal() if command == "seal" else evaluate(command)
