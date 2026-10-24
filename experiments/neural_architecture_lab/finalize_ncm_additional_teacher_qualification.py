"""Replay fixed TRAIN-label qualification without reporting student accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_additional_teacher_qualification as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def finalize():
    run.configure()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh append-only teacher qualification aggregate required")
    for root in (run.PREP, run.PILOT, run.GRADE):
        run.native.vendor.terminal(root)
    q = run.q
    cases = run.ordered_cases(q.read(run.teacher.GENERATE / "labels.json"))
    packets = {"labels": [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"]["labels"]}]) for r in cases]}
    proof = {"public_file_sha256": run.reference.PUBLIC_SHA256, "exact_sources": run.reference.verify_public(cases)}
    if (q.read(run.PREP / "cases.json") != cases or q.read(run.PREP / "packets.json") != packets
            or q.read(run.PREP / "public_proof.json") != proof
            or run.file_hash(run.teacher.GENERATE / "labels.json") != run.LABEL_SHA):
        raise ValueError("all64 original source/first-label/packet bindings differ")
    old = run.base_cache()
    pilot = q.read(run.PILOT / "judgment_cache.json")
    cache = q.read(run.GRADE / "judgment_cache.json")
    if any(pilot.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot.items()):
        raise ValueError("a preceding or pilot canonical first changed")
    run.semantic.audit_cache(cache)

    def replay(root, role, n, decisions):
        expected = replay_scores(cases[:n], {"labels": packets["labels"][:n]}, decisions, "labels", role)
        metric = {**run.metric(expected), "judge": q.JUDGES[role]}
        if (q.read(root / f"{role}_labels_case_scores.json") != expected
                or q.read(root / f"{role}_labels_result.json") != metric):
            raise ValueError("actual first-score or teacher-only metric replay differs")
        return expected, metric

    pilot_scores, _ = replay(run.PILOT, "secondary", 4, pilot)
    secondary, secondary_metric = replay(run.GRADE, "secondary", 64, cache)
    if not all(r["valid"] for r in pilot_scores):
        raise ValueError("actual four-case judgment validity required")
    admitted = run.oracle.qualifies(secondary_metric)
    metrics, joint = {"secondary": secondary_metric}, None
    if admitted:
        primary, metrics["primary"] = replay(run.GRADE, "primary", 64, cache)
        # Independent same-case intersection, not just two marginal pass counts.
        other = {r["id"]: r for r in primary}
        if len(other) != 64 or set(other) != {r["id"] for r in secondary}:
            raise ValueError("both judges must score the same64 cases")
        valid = all(r["valid"] and other[r["id"]]["valid"] for r in secondary)
        count = sum(r["passed"] is True and other[r["id"]]["passed"] is True for r in secondary)
        joint = {"cases": 64, "joint_passed": count, "all_valid": valid, "qualified": valid and count >= 61}
    elif (run.GRADE / "primary_labels_case_scores.json").exists():
        raise ValueError("primary ran before secondary admission")
    qualified = admitted and joint["qualified"]
    expected_status = "fixed64_teacher_labels_semantically_qualified" if qualified else "fixed64_teacher_label_quality_failed"
    status = q.read(run.GRADE / "result.json")
    if (status["status"] != expected_status or status["metrics"] != metrics or status["joint"] != joint
            or status["teacher_labels_qualified"] != bool(qualified) or status["primary_admitted"] != admitted
            or status["student_task_accuracy"] is not None or status["student_trained"]):
        raise ValueError("locked same-case label qualification differs")
    allowed = {run.reference.first.packet_key(p, role) for p in packets["labels"]
        for role in (("secondary", "primary") if admitted else ("secondary",))}
    pilot_allowed = {run.reference.first.packet_key(p, "secondary") for p in packets["labels"][:4]}
    if set(cache) - set(old) - allowed or set(pilot) - set(old) - pilot_allowed:
        raise ValueError("first judgments outside the fixed64 scope appeared")
    for root, current, inherited in ((run.PILOT, pilot, old), (run.GRADE, cache, pilot)):
        state = q.read(root / "result.json")
        if state["canonical_first_judgments"] != len(current) or state["inherited_first_judgments"] != len(inherited):
            raise ValueError("canonical first counts differ")
    artifacts = {}
    for root in (run.PREP, run.PILOT, run.GRADE):
        for path in root.rglob("*.json"):
            if "producer_source" not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    artifacts[str(run.CACHE)] = run.CACHE_SHA
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    for entry in cache.values():
        for path in entry["row"]["attempts"]:
            artifacts[path] = run.file_hash(path)
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": expected_status,
        "teacher_labels_qualified": bool(qualified), "primary_admitted": admitted, "metrics": metrics, "joint": joint,
        "canonical_first_judgments": len(cache), "teacher_label_quality_only": True,
        "student_task_accuracy": None, "student_trained": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    run.write_json(path, campaign)
    rows = ["| Judge | Semantic label passes | Valid |", "| --- | ---: | ---: |"]
    for role, metric in metrics.items():
        rows.append(f"| {role} | {metric['passed']}/64 | {metric['valid']}/64 |")
    run.REPORT.write_text(
        "# Additional TRAIN-label semantic qualification\n\n"
        f"Hypothesis `{run.HYPOTHESIS}`. Classification **{expected_status}**.\n"
        f"Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        "These are the fixed64 first teacher candidates on newly reserved TRAIN sources,\n"
        "with original source text,128stored-token cap and no exact-summary matching.\n"
        "The four-case pilot admitted judgment validity only. Every candidate and\n"
        "canonical first judgment was retained; no label was regenerated or filtered.\n\n"
        + "\n".join(rows) + "\n\n"
        + f"Same-case joint result: `{joint}`. Preceding2856firsts retained; final cache{len(cache)}.\n\n"
        "This measures teacher-label quality. It is not student accuracy, a95%NCM\n"
        "result, codec qualification or speed evidence. The completed student still\n"
        "scored0/128on source-disjoint DEV. Final sources remain closed and unhashed.\n")
    print(json.dumps({"classification": expected_status, "teacher_labels_qualified": bool(qualified),
        "campaign_sha256": run.file_hash(path), "student_task_accuracy": None}), flush=True)


def register():
    path = run.RESULTS / "campaign_results.json"
    state = run.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Fixed64 new training-label semantic qualification",
        "description": "All64 first teacher labels assessed against complete original new TRAIN sources with the128-token semantic rubric, exact canonical-first lineage and same-case two-judge61gate; not student accuracy.",
        "classification": state["classification"], "teacher_labels_qualified": state["teacher_labels_qualified"],
        "teacher_label_quality_only": True, "canonical_first_judgments": state["canonical_first_judgments"],
        "student_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
