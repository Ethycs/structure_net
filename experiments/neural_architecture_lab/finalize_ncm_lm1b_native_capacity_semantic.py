"""Independent raw-first replay of the fixed16 native-NCM TRAIN diagnosis."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_native_capacity_semantic as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def finalize():
    run.configure()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh TRAIN semantic aggregate required")
    for root in (run.PREP, run.PILOT, run.GRADE):
        run.native.vendor.terminal(root)
    q = run.q
    cases = run.cases_for(q.read(run.control.DATA / "cases.json"), q.read(run.control.TRAIN / "outputs.json"))
    packets = run.packets_for(cases)
    if (q.read(run.PREP / "cases.json") != cases or q.read(run.PREP / "packets.json") != packets
            or q.read(run.PREP / "public_proof.json") != {"public_file_sha256": run.reference.PUBLIC_SHA256,
                "exact_sources": run.reference.verify_public(cases)}):
        raise ValueError("complete original TRAIN source/summary/anonymous-packet binding differs")
    old = run.base_cache()
    pilot = q.read(run.PILOT / "judgment_cache.json")
    cache = q.read(run.GRADE / "judgment_cache.json")
    if any(pilot.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot.items()):
        raise ValueError("a preceding canonical first changed")
    run.semantic.audit_cache(cache)
    selected = q.read(run.control.DATA / "selection.json")["pilot_indices"]
    probes = [cases[i] for i in selected]
    pilot_packets = {arm: [packets[arm][i] for i in selected] for arm in run.ARMS}
    metrics, intersections = {}, {}
    for root, rows, payloads, decisions, is_pilot in ((run.PILOT, probes, pilot_packets, pilot, True),
            (run.GRADE, cases, packets, cache, False)):
        observed_metrics, observed_joint = {}, {}
        for arm in run.ARMS:
            observed_metrics[arm], scores = {}, {}
            for role in (("secondary",) if is_pilot else ("secondary", "primary")):
                scores[role] = replay_scores(rows, payloads, decisions, arm, role)
                expected = {**run.metric(scores[role], arm), "judge": q.JUDGES[role]}
                if (q.read(root / f"{role}_{arm}_case_scores.json") != scores[role]
                        or q.read(root / f"{role}_{arm}_result.json") != expected):
                    raise ValueError("actual raw first decisions or TRAIN-only metrics differ")
                observed_metrics[arm][role] = expected
                if is_pilot and not all(r["valid"] for r in scores[role]):
                    raise ValueError("actual valid semantic pilot required")
            if not is_pilot:
                secondary, primary = scores["secondary"], scores["primary"]
                if [r["id"] for r in secondary] != [r["id"] for r in primary]:
                    raise ValueError("same ordered TRAIN cases required for independent intersection")
                observed_joint[arm] = {"cases": 16,
                    "both_valid": sum(a["valid"] and b["valid"] for a, b in zip(secondary, primary)),
                    "both_passed": sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))}
        state = q.read(root / "result.json")
        expected_status = "TRAIN_semantic_pilot_valid_only" if is_pilot else "all16_TRAIN_semantic_diagnosis_completed"
        if (state["status"] != expected_status or state["metrics"] != observed_metrics
                or state["same_case_joint"] != observed_joint or state["canonical_first_judgments"] != len(decisions)
                or state["inherited_first_judgments"] != len(old if is_pilot else pilot)
                or state["task_accuracy"] is not None or state["source_disjoint_task_accuracy"] is not None):
            raise ValueError("fixed16 TRAIN aggregate or lineage differs")
        if not is_pilot:
            metrics, intersections = observed_metrics, observed_joint
    allowed = {run.reference.first.packet_key(p, role) for values in packets.values() for p in values
        for role in ("secondary", "primary")}
    pilot_allowed = {run.reference.first.packet_key(p, "secondary") for values in pilot_packets.values() for p in values}
    if set(cache) - set(old) - allowed or set(pilot) - set(old) - pilot_allowed:
        raise ValueError("canonical additions outside fixed16 TRAIN scope appeared")
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__)), str(run.CACHE): run.CACHE_SHA}
    for root in (run.PREP, run.PILOT, run.GRADE):
        for path in root.rglob("*"):
            if path.is_file() and "producer_source" not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    for entry in cache.values():
        for path in entry["row"]["attempts"]:
            artifacts[path] = run.file_hash(path)
    train = q.read(run.control.TRAIN / "result.json")
    state = {"hypothesis_id": run.HYPOTHESIS, "classification": "native16_TRAIN_semantic_diagnosis_completed",
        "TRAIN_only": True, "metrics": metrics, "same_case_joint": intersections,
        "canonical_prior_firsts": 2920, "canonical_first_judgments": len(cache),
        "source_disjoint_task_accuracy": None, "task_accuracy": None, "confirmed": False,
        "training_checkpoint_sha256": train["checkpoint_sha256"], "trainable_parameters": train["trainable_parameters"],
        "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    run.write_json(path, state)
    table = ["| Arm | Terra | Sonnet | Same-case passes |", "| --- | ---: | ---: | ---: |"]
    for arm in run.ARMS:
        table.append(f"| {arm} | {metrics[arm]['secondary']['passed']}/16 | {metrics[arm]['primary']['passed']}/16 | {intersections[arm]['both_passed']}/16 |")
    run.REPORT.write_text(
        "# Full-native NCM: sixteen-summary TRAIN learnability diagnosis\n\n"
        f"Hypothesis `{run.HYPOTHESIS}`; campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        "The authors' pretrained generator, domain word decoder and source attention\n"
        "were trained together:127,732,736parameters,256fixed updates,16TRAIN cases.\n"
        "The native encoder/VQ and original Llama body stayed frozen. Both pilot and\n"
        "full control reset the identical warm initializer; pilot weights were discarded.\n\n"
        "Both calibrated judges assessed EVERY final source-only TRAIN summary against\n"
        "the complete source, with paraphrases allowed and128stored-token cap. The\n"
        "reference arm diagnoses target quality and is never shown alongside model output.\n"
        "All16selected cases and all prior first judgments were retained.\n\n"
        + "\n".join(table) + "\n\n"
        + f"Descriptive word-ID fit: `{train['word_ID_fit_diagnostic_only']}`; this is not semantic accuracy.\n"
        + f"Final checkpoint: `{train['checkpoint_sha256']}`. Canonical cache: `{len(cache)}` firsts.\n\n"
        "These are seen TRAIN cases. They do not establish source-disjoint accuracy or\n"
        "the95%goal. The last qualified source-disjoint student remains0/128.\n"
        "Final320sources remain closed/unhashed. Speed has not been measured.\n")
    print(json.dumps({k: v for k, v in state.items() if k != "artifacts"}), flush=True)


def register():
    path = run.RESULTS / "campaign_results.json"
    state = run.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full-native NCM fixed16 TRAIN semantic learnability diagnosis",
        "description": "Source-only summaries on all16 seen TRAIN cases after256 native generator/decoder/attention updates, both calibrated semantic judges, anonymous full-source packets and canonical firsts; no heldout accuracy claim.",
        "classification": state["classification"], "TRAIN_only": True,
        "canonical_first_judgments": state["canonical_first_judgments"], "student_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
