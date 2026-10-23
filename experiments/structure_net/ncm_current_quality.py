"""Grade the fixed final NCM endpoint and an original-parent reference."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import PREP, ROOT, verify
from experiments.structure_net.ncm_paper_grade import grade

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-current-quality-execution.md")
TEACHER = ROOT / "20261022_expanded_teacher_v2"
PILOT = ROOT / "20261022_combined_primary_pilot_v1"
ARMS = {
    "ncm": (ROOT / "20261022_full_train_v1", "development_step5120.json", "step5120.pt"),
    "parent": (ROOT / "20261022_parent_outputs_v1", "outputs.json", None),
}


def combined_pilot():
    if PILOT.exists() or (TEACHER / "result.json").exists():
        raise ValueError("fresh combined pilot while the teacher is live required")
    source = ROOT / "20261022_expanded_teacher_pilot_v2/train.json"
    before = q.read(TEACHER / "progress.json")
    cases = [{**row, "id": "combined-primary-systems-"+row["id"], "text": row["target"]}
             for row in q.read(source)[:4]]
    results = grade(cases, ROOT / "20261022_combined_primary_judgments_v1",
                    [source, PLAN, Path(__file__), TEACHER / "producer.json"],
                    role="primary", systems_only=True)
    after = q.read(TEACHER / "progress.json")
    valid = (results["status"] == "completed" and results["valid"] == 4
             and after["completed_batches"] > before["completed_batches"])
    PILOT.mkdir()
    receipt = {"status": "passed" if valid else "failed", "systems_only": True,
               "judge_workers": 4, "teacher_workers": 4, "teacher_before": before,
               "teacher_after": after, "judgments": results, "task_accuracy": None,
               "semantic_outcomes_not_an_availability_gate": True,
               "inputs": {str(p): file_hash(p) for p in
                          (PLAN, Path(__file__), source, TEACHER / "producer.json")}}
    write_json(PILOT / "result.json", receipt)
    print(json.dumps(receipt), flush=True)


def seal(arm):
    root, name, checkpoint = ARMS[arm]
    destination = ROOT / ("20261022_current_"+arm+"_seal_v1.json")
    if destination.exists():
        raise ValueError("fresh immutable seal required")
    result = q.read(root / "result.json")
    expected = "trained_pending_development_grading" if arm == "ncm" else "generated_pending_task_grading"
    if result["status"] != expected:
        raise ValueError("completed generation required")
    producer = q.read(root / "producer.json")
    verify(producer)
    rows = q.read(root / name)
    fixed = q.read(PREP / "development_sources.json")
    if len(rows) != 64 or [(r["id"], r["source_sha256"]) for r in rows] != [
            (r["id"], r["source_sha256"]) for r in fixed]:
        raise ValueError("fixed full64-case source order required")
    paths = [root / name, root / "result.json", root / "producer.json", PLAN, Path(__file__)]
    if checkpoint:
        paths.append(root / checkpoint)
    write_json(destination, {"status": "sealed_complete_generation", "arm": arm,
        "artifacts": {str(p): file_hash(p) for p in paths}, "task_accuracy": None})
    print(json.dumps({"status": "sealed_complete_generation", "arm": arm}), flush=True)


def evaluate(arm, role):
    root, name, _ = ARMS[arm]
    sealed = ROOT / ("20261022_current_"+arm+"_seal_v1.json")
    for path, expected in q.read(sealed)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("sealed producer changed: "+path)
    pilot = ROOT / ("20261022_recovered_judge_pilot_"+role+"_v1/result.json")
    r = q.read(pilot)
    if r["status"] != "completed" or r["invalid"] or not r["systems_only"]:
        raise ValueError("recovered judge pilot required")
    inputs = [sealed, pilot, PLAN, Path(__file__), root / name]
    if role == "primary":
        completed = TEACHER / "result.json"
        if completed.exists():
            inputs.append(completed)
        else:
            combined = PILOT / "result.json"
            if q.read(combined)["status"] != "passed":
                raise ValueError("combined concurrency pilot required")
            inputs.append(combined)
    grade(q.read(root / name), ROOT / ("20261022_current_"+arm+"_grade_"+role+"_v1"), inputs, role=role)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["combined-pilot", "seal", "grade"])
    parser.add_argument("--arm", choices=ARMS, default="ncm")
    parser.add_argument("--role", choices=q.JUDGES, default="secondary")
    args = parser.parse_args()
    if args.command == "combined-pilot":
        combined_pilot()
    elif args.command == "seal":
        seal(args.arm)
    else:
        evaluate(args.arm, args.role)
