"""Semantic qualification of the fixed64 new TRAIN labels, not model accuracy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_additional_teacher_pilot as teacher
from experiments.structure_net import ncm_lm1b_word_gradient_development as parent
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = parent.native, parent.q, parent.ROOT
reference, semantic, oracle = parent.reference, parent.semantic, parent.base.oracle
HYPOTHESIS = "ncm-additional64-teacher-semantic-qualification-v1"
MODULE = "experiments.structure_net.ncm_additional_teacher_qualification"
AUDITOR_MODULE = "experiments.neural_architecture_lab.finalize_ncm_additional_teacher_qualification"
AUDITOR = Path("experiments/neural_architecture_lab/finalize_ncm_additional_teacher_qualification.py")
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-additional-teacher-qualification-plan.md")
TEST = Path("tests/structure_net/test_ncm_additional_teacher_qualification.py")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-additional-teacher-qualification.md")
WORKFLOW = ROOT / "20261024_additional64_teacher_qualification_workflow_v1"
PREP = ROOT / "20261024_additional64_teacher_qualification_prepare_v1"
PILOT = ROOT / "20261024_additional64_teacher_qualification_pilot_v1"
GRADE = ROOT / "20261024_additional64_teacher_qualification_grade_v1"
RESULTS = ROOT / "20261024_additional64_teacher_qualification_results_v1"
RAW_AUDIT = ROOT / "20261024_additional64_teacher_raw_audit_v1"
CACHE = parent.GRADE / "judgment_cache.json"
CACHE_SHA = "42ba6bf5a85c9b850a79a50ced43eaba2f55b3305a9c8ce462ce22dd48583c62"
CAMPAIGN_SHA = "067ade9b2c24680c9564dec3a2847e02cdbb805ca7053fb60b7f08d6a6edcfe2"
LABEL_SHA = "e8855022ff805f34a1f30aedd82f3e8f3edc5f0aef983e6ac3f2bfaf5d157a46"


def configure():
    native.vendor.corpus.quality.dependencies()
    reference.profile.install()
    semantic.require_judges()


def base_cache():
    path = parent.RESULTS / "campaign_results.json"
    receipt = q.read(parent.RESULTS / (parent.HYPOTHESIS + "_registry_readback.json"))
    state = q.read(path)
    if (file_hash(path) != CAMPAIGN_SHA or file_hash(CACHE) != CACHE_SHA
            or state["classification"] != "full128_source_only_DEV_quality_failed"
            or state["artifacts"].get(str(CACHE)) != CACHE_SHA or state["canonical_first_judgments"] != 2856
            or receipt["record"]["campaign_sha256"] != CAMPAIGN_SHA or receipt["exact"]["campaign_sha256"] != CAMPAIGN_SHA
            or q.read(parent.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0
            or q.read(parent.RESULTS / "backend_exact_readback.json")["status"] != "independent_backend_exact_readback_passed"):
        raise ValueError("latest completed registered2856 canonical firsts required")
    cache = q.read(CACHE)
    if len(cache) != 2856:
        raise ValueError("all2856 preceding first judgments required")
    semantic.audit_cache(cache)
    return cache


def ordered_cases(labels):
    if (len(labels) != 64 or len({r["source_component"] for r in labels}) != 64
            or any(not r["teacher_valid"] or not r["capacity_fit"] for r in labels)):
        raise ValueError("all64 original fitting first TRAIN candidates required")
    probes = [*[r for r in labels if r["bin"] == 0][:2], next(r for r in labels if r["bin"] == 1),
        min(labels, key=lambda r: (-r["source_cl100k_tokens"], r["id"]))]
    if len({r["id"] for r in probes}) != 4:
        raise ValueError("four distinct source-only short/long validity probes required")
    chosen = {r["id"] for r in probes}
    ordered = [*probes, *[r for r in labels if r["id"] not in chosen]]
    return [{**{k: r[k] for k in ("id", "source_id", "group_id", "source_sha256", "source_component", "source", "spans", "bin")},
        "texts": {"labels": r["target_text"]}} for r in ordered]


def metric(scores):
    value = oracle.metric(scores)
    value["teacher_label_semantic_pass_rate"] = value.pop("oracle_semantic_pass_rate")
    value.update(task_accuracy=None, oracle_target_codes=False, source_prediction=False, teacher_label_quality=True)
    return value


def producing(root, stage, inputs=(), extra=None):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(teacher.__file__), Path(oracle.__file__), Path(semantic.__file__),
        Path(reference.first.__file__), reference.first.AMENDMENT, Path(reference.profile.__file__), reference.profile.PLAN]
    common = [CACHE, parent.RESULTS / "campaign_results.json", parent.RESULTS / (parent.HYPOTHESIS + "_registry_readback.json"),
        parent.RESULTS / "backend_exact_readback.json", parent.WORKFLOW / "tool_terminal_receipt.json",
        teacher.GENERATE / "labels.json", RAW_AUDIT / "result.json", RAW_AUDIT / "actual_tool_terminal_receipt.json"]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {**(extra or {}), "hypothesis_id": HYPOTHESIS,
        "teacher_label_quality_only": True, "task_accuracy": None, "student_training_admitted": False,
        "canonical_prior_firsts": 2856, "rubric": q.RUBRIC, "judges": q.JUDGES,
        "all64_cases": True, "joint_minimum": 61, "all_valid_required": True, "no_semantic_retries": True,
        "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"teacher_qualification_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    configure()
    base_cache()
    if (q.read(RAW_AUDIT / "actual_tool_terminal_receipt.json")["exit_code"] != 0
            or file_hash(teacher.GENERATE / "labels.json") != LABEL_SHA):
        raise ValueError("actual independently replayed fixed64 first teacher candidates required")
    raw = q.read(RAW_AUDIT / "result.json")
    for path, expected in raw["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("raw first-response audit changed:" + path)
    cases = ordered_cases(q.read(teacher.GENERATE / "labels.json"))
    packets = {"labels": [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"]["labels"]}]) for r in cases]}
    proof = reference.verify_public(cases)
    p = producing(PREP, "fixed64_exact_public_TRAIN_label_packets", [reference.PUBLIC_FILE])
    write_json(PREP / "cases.json", cases)
    write_json(PREP / "packets.json", packets)
    write_json(PREP / "public_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256, "exact_sources": proof})
    native.vendor.verify(p)
    write_json(PREP / "result.json", {"status": "fixed64_teacher_packets_prepared", "cases": 64,
        "cases_sha256": file_hash(PREP / "cases.json"), "packets_sha256": file_hash(PREP / "packets.json"), "task_accuracy": None})


def grade(pilot=False):
    configure()
    native.vendor.terminal(PREP)
    cases, packets = q.read(PREP / "cases.json"), q.read(PREP / "packets.json")
    cache = base_cache()
    inputs = [PREP / n for n in ("cases.json", "packets.json", "public_proof.json", "worker_terminal_receipt.json")]
    if not pilot:
        native.vendor.terminal(PILOT)
        prior = q.read(PILOT / "judgment_cache.json")
        if any(prior.get(k) != v for k, v in cache.items()) or q.read(PILOT / "result.json")["status"] != "four_teacher_judgments_valid_only":
            raise ValueError("all prior firsts and actual four-case validity pilot required")
        cache = prior
        inputs += [PILOT / n for n in ("judgment_cache.json", "result.json", "worker_terminal_receipt.json")]
    inherited = dict(cache)
    root = PILOT if pilot else GRADE
    p = producing(root, "teacher_judge_validity_only" if pilot else "fixed64_teacher_semantic_quality", inputs, {"pilot": pilot})
    write_json(root / "judgment_cache.json", cache)
    metrics = {}

    def judge(role):
        oracle.grade_arm(root, "labels", role, cases[:4] if pilot else cases, packets, cache)
        scores = q.read(root / f"{role}_labels_case_scores.json")
        metrics[role] = {**metric(scores), "judge": q.JUDGES[role]}
        write_json(root / f"{role}_labels_result.json", metrics[role])
        write_json(root / "judgment_cache.json", cache)
        return scores

    secondary = judge("secondary")
    admitted = not pilot and oracle.qualifies(metrics["secondary"])
    joint = parent.joint_gate(secondary, judge("primary")) if admitted else None
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError("an inherited first judgment changed")
    semantic.audit_cache(cache)
    valid_pilot = all(r["valid"] for r in secondary)
    qualified = admitted and joint["qualified"]
    status = ("four_teacher_judgments_valid_only" if valid_pilot else "teacher_judge_pilot_invalid") if pilot else (
        "fixed64_teacher_labels_semantically_qualified" if qualified else "fixed64_teacher_label_quality_failed")
    native.vendor.verify(p)
    write_json(root / "result.json", {"status": status, "metrics": metrics, "joint": joint, "primary_admitted": admitted,
        "teacher_labels_qualified": bool(qualified), "cases": 4 if pilot else 64,
        "canonical_first_judgments": len(cache), "inherited_first_judgments": len(inherited),
        "student_task_accuracy": None, "student_trained": False, "final_test_opened": False, "speed_tested": False})
    if pilot and not valid_pilot:
        raise ValueError("teacher judgment validity pilot failed; all firsts preserved")


def workflow():
    configure()
    base_cache()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GRADE, RESULTS)):
        raise ValueError("fresh teacher semantic qualification roots required")
    p = producing(WORKFLOW, "supervise_fixed64_teacher_semantic_qualification")
    receipts = []
    for module, stage, root in ((MODULE, "prepare", PREP), (MODULE, "pilot", PILOT), (MODULE, "grade", GRADE),
            (AUDITOR_MODULE, "finalize", RESULTS), (AUDITOR_MODULE, "register", None)):
        native.vendor.verify(p)
        command = [sys.executable, "-u", "-m", module, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / "progress.json", {"active_stage": stage, "child_pid": child.pid, "stages": receipts})
        code = child.wait()
        receipt = {"stage": stage, "exit_code": code, "command": command, "child_pid": child.pid, "observer_pid": os.getpid(), "observer": "subprocess.wait"}
        write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
        if root is not None and root.exists():
            write_json(root / "worker_terminal_receipt.json", receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError("teacher semantic qualification failed:" + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / "result.json", {"status": "teacher_semantic_qualification_workflow_completed", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "grade", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": lambda: grade(True), "grade": grade, "workflow": workflow}[args.stage]()
