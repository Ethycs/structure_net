"""Replay fixed-output baseline scores without changing earlier rubric results."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_budget128_fixed_baselines as b
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_source_readout_grade as first
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_budget128_baselines_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-budget128-fixed-baselines.md")


def replay(condition, role, outputs, endpoint, cache, pilot_cache):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    root = b.GRADING / (role + "_" + condition)
    verify(q.read(root / "producer.json"))
    scores = q.read(root / "case_scores.json")
    packets = q.read(root / "packets.json")
    if len(scores) != 64 or len(packets) != 64:
        raise ValueError("complete64-case endpoint required")
    counts = {k: 0 for k in ("valid", "passed", "faithful", "salient", "readable", "stored_overflow")}
    for output, score, packet in zip(outputs, scores, packets):
        expected = q.packet(output["id"], output["source"], [{"candidate_id": "summary", "text": output["text"]}])
        if packet != expected or score["id"] != output["id"] or score["text"] != output["text"]:
            raise ValueError("frozen output or source changed")
        key = first.packet_key(packet, role)
        entry = cache[key]
        first.audit_row(packet, role, entry["row"])
        if key in pilot_cache and entry != pilot_cache[key]:
            raise ValueError("first pilot judgment replaced")
        if score["packet_key"] != key or score["first_origin"] != entry["origin"]:
            raise ValueError("first judgment origin changed")
        row = entry["row"]
        valid = row["status"] == "valid"
        tokens = len(encoding.encode(output["text"], disallowed_special=()))
        if score["tokens"] != tokens:
            raise ValueError("actual stored token count changed")
        passed = q.passes(row["evaluations"][0], output["text"], tokens) if valid else None
        if score["valid"] != valid or score["passed"] != passed:
            raise ValueError("whole-summary decision changed")
        counts["valid"] += valid
        counts["passed"] += passed is True
        counts["stored_overflow"] += score["tokens"] > 128
        if valid:
            for component in ("faithful", "salient", "readable"):
                counts[component] += row["evaluations"][0][component] is True
    if (counts["valid"], counts["passed"]) != (endpoint["valid"], endpoint["passed"]):
        raise ValueError("raw first judgments and endpoint disagree")
    return counts


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    for root in (b.PILOT, b.GRADING):
        if q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("both judge workers must be observed terminal")
        verify(q.read(root / "producer.json"))
    _, outputs = b.inputs_and_cases()
    result = q.read(b.GRADING / "result.json")
    cache = q.read(b.GRADING / "judgment_cache.json")
    pilot_cache = q.read(b.PILOT / "judgment_cache.json")
    old_roots = {"native_3b": ROOT / "20261023_native_parent_results_v1",
                 "native_1b": ROOT / "20261023_intact_decoder_results_v1"}
    old_endpoints = {}
    old_score_paths = {
        "native_3b": ROOT / "20261023_native_parent_grading_v1/secondary_native_parent/case_scores.json",
        "native_1b": ROOT / "20261023_intact_decoder_grading_v1/secondary_native_readout/case_scores.json",
    }
    paired_changes = {}
    for name, root in old_roots.items():
        old = q.read(root / "campaign_results.json")
        endpoint = old["grading"]["secondary"] if name == "native_3b" else old["endpoint"]
        if endpoint["passed"] != b.OLD_PASSES[name] or endpoint["cases"] != 64:
            raise ValueError("original old-profile endpoint changed")
        old_endpoints[name] = endpoint
        old_scores = q.read(old_score_paths[name])
        new_scores = q.read(b.GRADING / ("secondary_" + name) / "case_scores.json")
        if (len(old_scores) != 64 or len(new_scores) != 64
                or any((a["id"], a["text"]) != (c["id"], c["text"]) for a, c in zip(old_scores, new_scores))
                or sum(s["passed"] is True for s in old_scores) != b.OLD_PASSES[name]):
            raise ValueError("paired original outputs/old scores changed")
        paired_changes[name] = {
            "old_fail_new_pass": sum(a["passed"] is False and c["passed"] is True for a, c in zip(old_scores, new_scores)),
            "old_pass_new_fail": sum(a["passed"] is True and c["passed"] is False for a, c in zip(old_scores, new_scores)),
        }
    components = {}
    for name, evaluation in result["results"].items():
        secondary, primary = evaluation["secondary"], evaluation["primary"]
        if (primary is not None) != b.quality_gate(secondary):
            raise ValueError("independent primary judge gate order changed")
        roles = {"secondary": replay(name, "secondary", outputs[name], secondary, cache, pilot_cache)}
        if primary is not None:
            roles["primary"] = replay(name, "primary", outputs[name], primary, cache, pilot_cache)
        if evaluation["both_judges_quality_gate"] != (primary is not None and b.quality_gate(primary)):
            raise ValueError("joint baseline decision changed")
        components[name] = roles
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    paths += [root / "campaign_results.json" for root in old_roots.values()]
    paths += list(old_score_paths.values())
    for root in (b.PILOT, b.GRADING):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    for root in b.BASELINES.values():
        paths += [root / n for n in ("producer.json", "result.json", "outputs.json")]
    aggregate = {"schema": "nal.ncm-budget128-fixed-baselines.v1",
        "classification": "fixed_baseline_re_evaluation_complete_student_quality_separate",
        "new_profile": result, "original_old_profile": old_endpoints, "components": components,
        "paired_profile_decision_changes": paired_changes,
        "first_judgments_replayed": True, "model_outputs_unchanged": True,
        "NCM_model": False, "student_quality_measured": False, "NCM_success_established": False,
        "confirmed": False, "speed_tested": False, "final_test_opened": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir()
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = []
    for name, values in result["results"].items():
        e = values["secondary"]
        c = components[name]["secondary"]
        table.append(f"| {name} | {b.OLD_PASSES[name]}/64 | {e['passed']}/64 | {e['valid']}/64 | "
                     f"{c['faithful']}/64 | {c['salient']}/64 | {c['readable']}/64 | {c['stored_overflow']} |")
    rows = "\n".join(table)
    REPORT.write_text(f'''# Fixed native baselines evaluated with the application budget

Date: 2026-10-23. Exploratory `ncm-budget128-fixed-baselines-v1`.
Classification: `{aggregate['classification']}`.
This compares the same frozen outputs under two explicit rubrics. No model
weights, prompts, generation settings or candidate texts changed.

| Baseline | Old-rubric whole pass | New-rubric whole pass | New valid | Faithful | Salient | Readable | Storage overflow |
| --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

The new qualified rubric removes an inherited approximately 32-word salience
target while retaining the 128-token storage cap, source-grounded faithfulness,
coverage and readability. Both judge aliases previously passed all 50 controls.
These are fallible LLM assessments, not independent human gold.
Paired whole-decision changes are `{paired_changes}`. These fresh judgments
cannot isolate a rubric effect from judge variability and do not improve model
weights. No failed source is removed from either comparison.

All 64 original development sources per baseline were retained in order.
Over-cap, empty and poor-quality outputs remain in the denominator. First
new-profile pilot judgments, including negatives, were reused exactly; no
old-profile decision was passed off as a new-profile result. Raw request/parser
and first-valid decisions were replayed before finalization. Old aggregates
and caches remain unchanged. Wilson intervals and conditional independent
Sonnet outcomes are in the aggregate; Sonnet runs only after 64 valid Terra
judgments and at least 61 whole passes for that baseline.

These intact AR baselines are controls for the current frozen-body NCM study.
They cannot establish student accuracy, a working NCM replacement, compression
or speedup. The student must generate and pass its own source-disjoint summaries.
The final 256 cases remain sealed and the short-source prototype does not prove
the complete 2,048-token fragment / 6,300-token batch application range.

Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`. Registry confirmed=false.
''')
    print(json.dumps({"status": "finalized", "components": components}), flush=True)


def register():
    aggregate = q.read(DEST / "campaign_results.json")
    counts = {name: values["secondary"]["passed"] for name, values in aggregate["new_profile"]["results"].items()}
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-budget128-fixed-baselines-v1": {
            "name": "Unchanged native baseline outputs under application128-token rubric",
            "description": f"Same64sources and frozen texts: old3B49/64,1B50/64; new qualified128-token Terra3B{counts['native_3b']}/64,1B{counts['native_1b']}/64. Old judgments preserved; no student NCM or speed claim.",
            "confirmation_status": aggregate["classification"], "task_accuracy_measured": True,
            "student_quality_measured": False, "NCM_model": False, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
