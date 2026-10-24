"""Fixed all16 semantic TRAIN diagnosis of full-native NCM learnability."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_lm1b_native_capacity as control
from experiments.structure_net import ncm_additional_teacher_qualification as previous
from experiments.structure_net.ncm_memory_data import file_hash, write_json

q, native, ROOT = control.q, control.native, control.ROOT
reference, semantic, oracle = previous.reference, previous.semantic, previous.oracle
HYPOTHESIS = "ncm-lm1b-native16-TRAIN-semantic-v1"
MODULE = "experiments.structure_net.ncm_lm1b_native_capacity_semantic"
AUDITOR_MODULE = "experiments.neural_architecture_lab.finalize_ncm_lm1b_native_capacity_semantic"
AUDITOR = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_native_capacity_semantic.py")
TEST = Path("tests/structure_net/test_ncm_lm1b_native_capacity_semantic.py")
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-capacity-semantic-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-native-capacity-semantic.md")
WORKFLOW = ROOT / "20261024_lm1b_native16_semantic_workflow_v1"
PREP = ROOT / "20261024_lm1b_native16_semantic_prepare_v1"
PILOT = ROOT / "20261024_lm1b_native16_semantic_pilot_v1"
GRADE = ROOT / "20261024_lm1b_native16_semantic_grade_v1"
RESULTS = ROOT / "20261024_lm1b_native16_semantic_results_v1"
CONTROL_AUDIT = ROOT / "20261024_lm1b_native16_capacity_audit_v1"
CACHE = previous.GRADE / "judgment_cache.json"
CACHE_SHA = "7bafa3ac6cb3a54ad3e29290a80ec9fc0e782bf576903022646edd929067f7c6"
CAMPAIGN_SHA = "b8cbbbfef48e33e6781ba145009b6a383afd6b1966732dbc53dff49b449b64f6"
ARMS = ("reference", "native")
IDENTITY = ("id", "source_id", "group_id", "source_sha256", "source", "spans")


def configure():
    previous.configure()


def base_cache():
    path = previous.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(previous.RESULTS / (previous.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != CAMPAIGN_SHA or file_hash(CACHE) != CACHE_SHA
            or campaign["canonical_first_judgments"] != 2920 or campaign["artifacts"].get(str(CACHE)) != CACHE_SHA
            or receipt["record"]["campaign_sha256"] != CAMPAIGN_SHA or receipt["exact"]["campaign_sha256"] != CAMPAIGN_SHA
            or q.read(previous.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0
            or q.read(previous.RESULTS / "backend_exact_readback.json")["status"] != "independent_backend_exact_readback_passed"):
        raise ValueError("latest registered2920 canonical first judgments required")
    cache = q.read(CACHE)
    if len(cache) != 2920:
        raise ValueError("all2920 preceding firsts required")
    semantic.audit_cache(cache)
    return cache


def cases_for(rows, outputs):
    if (len(rows) != 16 or len(outputs) != 16 or len({r["id"] for r in rows}) != 16
            or [r["id"] for r in rows] != [r["id"] for r in outputs]):
        raise ValueError("complete ordered16 TRAIN sources and fixed-final outputs required")
    result = []
    for row, value in zip(rows, outputs):
        if row["source_sha256"] != value["source_sha256"]:
            raise ValueError("TRAIN output/source binding differs")
        result.append({**{k: row[k] for k in IDENTITY},
            "texts": {"reference": row["target_text"], "native": value["text"]}})
    return result


def packets_for(cases):
    # Judges see full source and one anonymous summary, never another arm or target.
    return {arm: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"][arm]}])
        for r in cases] for arm in ARMS}


def metric(scores, arm):
    result = oracle.metric(scores)
    result["TRAIN_semantic_pass_rate"] = result.pop("oracle_semantic_pass_rate")
    result.update(task_accuracy=None, source_disjoint_task_accuracy=None, oracle_target_codes=False,
        source_prediction=arm == "native", reference_summary=arm == "reference", TRAIN_only=True)
    return result


def joint(secondary, primary):
    if [r["id"] for r in secondary] != [r["id"] for r in primary]:
        raise ValueError("same16 cases required across both judges")
    return {"cases": len(secondary), "both_valid": sum(a["valid"] and b["valid"] for a, b in zip(secondary, primary)),
        "both_passed": sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))}


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(control.__file__), control.PLAN,
        control.AUDITOR, Path(previous.__file__), Path(oracle.__file__), Path(semantic.__file__),
        Path(reference.first.__file__), reference.first.AMENDMENT, Path(reference.profile.__file__), reference.profile.PLAN]
    common = [CACHE, previous.RESULTS / "campaign_results.json", previous.RESULTS / (previous.HYPOTHESIS + "_registry_readback.json"),
        previous.RESULTS / "backend_exact_readback.json", previous.WORKFLOW / "tool_terminal_receipt.json",
        CONTROL_AUDIT / "result.json", CONTROL_AUDIT / "actual_tool_terminal_receipt.json",
        control.WORKFLOW / "tool_terminal_receipt.json", control.TRAIN / "result.json", control.TRAIN / "outputs.json",
        control.DATA / "cases.json", control.DATA / "selection.json"]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {"hypothesis_id": HYPOTHESIS,
        "TRAIN_only": True, "source_disjoint_task_accuracy": None, "task_accuracy": None,
        "all16_fixed_final256_outputs": True, "arms": ARMS, "both_judges_on_every_case": True,
        "rubric": q.RUBRIC, "judges": q.JUDGES, "canonical_prior_firsts": 2920,
        "no_semantic_retries": True, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"native16_semantic_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    configure()
    base_cache()
    if q.read(CONTROL_AUDIT / "actual_tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual independent CPU TRAIN-control audit required")
    audit = q.read(CONTROL_AUDIT / "result.json")
    for path, expected in audit["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("audited TRAIN control changed:" + path)
    cases = cases_for(q.read(control.DATA / "cases.json"), q.read(control.TRAIN / "outputs.json"))
    packets = packets_for(cases)
    proof = reference.verify_public(cases)
    p = producing(PREP, "fixed16_TRAIN_anonymous_source_only_semantic_packets", [reference.PUBLIC_FILE])
    write_json(PREP / "cases.json", cases)
    write_json(PREP / "packets.json", packets)
    write_json(PREP / "public_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256, "exact_sources": proof})
    native.vendor.verify(p)
    write_json(PREP / "result.json", {"status": "all16_TRAIN_packets_prepared", "cases": 16, "task_accuracy": None})


def grade(pilot=False):
    configure()
    native.vendor.terminal(PREP)
    cases, packets = q.read(PREP / "cases.json"), q.read(PREP / "packets.json")
    cache = base_cache()
    inputs = [PREP / n for n in ("cases.json", "packets.json", "public_proof.json", "worker_terminal_receipt.json")]
    if not pilot:
        native.vendor.terminal(PILOT)
        prior = q.read(PILOT / "judgment_cache.json")
        if any(prior.get(k) != v for k, v in cache.items()) or q.read(PILOT / "result.json")["status"] != "TRAIN_semantic_pilot_valid_only":
            raise ValueError("actual valid pilot and unchanged preceding firsts required")
        cache = prior
        inputs += [PILOT / n for n in ("judgment_cache.json", "result.json", "worker_terminal_receipt.json")]
    inherited = dict(cache)
    root = PILOT if pilot else GRADE
    p = producing(root, "TRAIN_judge_validity_only" if pilot else "all16_TRAIN_two_judge_semantic_diagnosis", inputs)
    write_json(root / "judgment_cache.json", cache)
    selected = q.read(control.DATA / "selection.json")["pilot_indices"] if pilot else list(range(16))
    chosen = [cases[i] for i in selected]
    values = {arm: [packets[arm][i] for i in selected] for arm in ARMS}
    metrics, intersections, scores = {}, {}, {}
    for arm in ARMS:
        metrics[arm], scores[arm] = {}, {}
        for role in (("secondary",) if pilot else ("secondary", "primary")):
            oracle.grade_arm(root, arm, role, chosen, values, cache)
            rows = q.read(root / f"{role}_{arm}_case_scores.json")
            scores[arm][role] = rows
            metrics[arm][role] = {**metric(rows, arm), "judge": q.JUDGES[role]}
            write_json(root / f"{role}_{arm}_result.json", metrics[arm][role])
        if not pilot:
            intersections[arm] = joint(scores[arm]["secondary"], scores[arm]["primary"])
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError("inherited first judgment changed")
    semantic.audit_cache(cache)
    valid = all(r["valid"] for values in scores.values() for rows in values.values() for r in rows)
    status = ("TRAIN_semantic_pilot_valid_only" if valid else "TRAIN_semantic_pilot_invalid") if pilot else "all16_TRAIN_semantic_diagnosis_completed"
    native.vendor.verify(p)
    write_json(root / "result.json", {"status": status, "metrics": metrics, "same_case_joint": intersections,
        "canonical_first_judgments": len(cache), "inherited_first_judgments": len(inherited),
        "source_disjoint_task_accuracy": None, "task_accuracy": None, "TRAIN_only": True,
        "final_test_opened": False, "speed_tested": False})
    if pilot and not valid:
        raise ValueError("invalid TRAIN judge pilot preserved")


def workflow():
    configure()
    base_cache()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GRADE, RESULTS)):
        raise ValueError("fresh append-only native16 semantic roots required")
    p = producing(WORKFLOW, "supervise_fixed16_TRAIN_semantic_diagnosis")
    receipts = []
    for module, stage, root in ((MODULE, "prepare", PREP), (MODULE, "pilot", PILOT), (MODULE, "grade", GRADE),
            (AUDITOR_MODULE, "finalize", RESULTS), (AUDITOR_MODULE, "register", None)):
        native.vendor.verify(p)
        command = [sys.executable, "-u", "-m", module, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / "progress.json", {"active_stage": stage, "child_pid": child.pid, "stages": receipts})
        code = child.wait()
        receipt = {"stage": stage, "exit_code": code, "command": command, "child_pid": child.pid,
            "observer_pid": os.getpid(), "observer": "subprocess.wait"}
        write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
        if root is not None and root.exists():
            write_json(root / "worker_terminal_receipt.json", receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError("native16 semantic diagnosis failed:" + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / "result.json", {"status": "native16_TRAIN_semantic_workflow_completed", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "grade", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": lambda: grade(True), "grade": grade, "workflow": workflow}[args.stage]()
