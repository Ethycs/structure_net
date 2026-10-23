"""Record intact-model language competence separately from summary quality."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_intact_language_control as c
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_source_readout_grade import audit_row, packet_key
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_intact_decoder_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-intact-language-control.md")


def gates(valid, passed, readable):
    return {"language_component": valid == 64 and readable >= 61,
            "secondary_summary_quality": valid == 64 and passed >= 61,
            "NCM_success_established": False}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh intact-language finalization required")
    roots = [c.PILOT, c.OUTPUTS, c.JUDGE_PILOT, c.GRADING]
    for root in roots:
        verify(q.read(root / "producer.json"))
        receipt = q.read(root / "worker_terminal_receipt.json")
        if root == c.GRADING and receipt["status"] == "completed_result_session_unavailable":
            if receipt["exit_code"] is not None or receipt["exact_module_processes_present"] != 0:
                raise ValueError("missing trustworthy terminal evidence")
        elif receipt["exit_code"] != 0:
            raise ValueError("nonterminal or failed worker")
    result = q.read(c.GRADING / "result.json")
    endpoint = result["endpoint"]
    scores = q.read(c.GRADING / "secondary_native_readout/case_scores.json")
    packets = q.read(c.GRADING / "secondary_native_readout/packets.json")
    outputs = q.read(c.OUTPUTS / "outputs.json")
    cache = q.read(c.GRADING / "judgment_cache.json")
    pilot_cache = q.read(c.JUDGE_PILOT / "judgment_cache.json")
    if result["status"] != "completed" or len(scores) != 64 or len(packets) != 64 or len(outputs) != 64:
        raise ValueError("complete fixed64 endpoint required")
    counts = dict.fromkeys(("faithful", "salient", "readable"), 0)
    passed = valid = 0
    for score, packet, output in zip(scores, packets, outputs):
        key = packet_key(packet, "secondary")
        entry = cache[key]
        row = entry["row"]
        audit_row(packet, "secondary", row)
        if (score["id"] != output["id"] or score["text"] != output["text"]
                or score["packet_key"] != key or score["first_origin"] != entry["origin"]):
            raise ValueError("output or first judgment changed")
        if key in pilot_cache and entry != pilot_cache[key]:
            raise ValueError("first pilot judgment replaced")
        ok = q.passes(row["evaluations"][0], score["text"], score["tokens"]) if row["status"] == "valid" else None
        if ok != score["passed"]:
            raise ValueError("whole-summary decision changed")
        valid += ok is not None
        passed += ok is True
        if ok is not None:
            for name in counts:
                counts[name] += row["evaluations"][0][name] is True
    if (passed, valid) != (endpoint["passed"], endpoint["valid"]):
        raise ValueError("endpoint reconstruction changed")
    decision = gates(valid, passed, counts["readable"])
    classification = ("intact_language_pass_summary_quality_failed" if decision["language_component"]
                      and not decision["secondary_summary_quality"] else "intact_language_other_result")
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-intact-language-control.v1", "classification": classification,
        "endpoint": endpoint, "component_passing_counts": counts, "gates": decision,
        "first_judgments_replayed": True, "NCM_model": False, "confirmed": False,
        "underpowered": True, "final_test_opened": False, "speed_tested": False,
        "worker_terminal_limitation": "grading exit output unavailable; completed result, expired session and exact process absence verified; exit code not invented",
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Intact language passes; summary quality remains below target

Date: 2026-10-23. `ncm-intact-language-control-v1`, **{classification}**.
The untouched Llama-3.2-1B-Instruct scored **{passed}/64 = {100*passed/64:.1f}%**
whole-summary accuracy on the fixed development sources under qualified Terra.
All {valid}/64 judgments were valid. Component passes: faithful {counts['faithful']}/64,
salient {counts['salient']}/64, readable {counts['readable']}/64. The necessary
61/64 readable language gate passes; the complete-summary gate fails. This model
has no NCM attached and does not establish the 95% task objective.

The official cached revision and all six files were verified against pinned
Hugging Face LFS/Git identities. Its native vocabulary and all64 prompt IDs match
the 3B parent. All16 pretrained blocks and the tied native word head remained
unchanged. Greedy output repetition, serialization and frozen-weight checks
passed. The measured peak allocation was 2,634,609,152 bytes on GPU2 in FP16.
This is memory evidence, not a matched speed comparison with the BF16/GPU0
five-block readout, which scored 0/64 and failed every readability judgment.

The fixed64-source outputs were frozen before judging. Four first pilot
judgments (including negatives) were reused and60 new judgments completed the
endpoint. No semantic retries, clipping, output repair or post-hoc exclusions.
Sonnet was not run after the failed secondary summary-quality gate. The final256
remain sealed. Current short-source evaluation does not establish the full
2,048-token fragment / 6,300-token batch operating range.

The contrast supports retaining a usable pretrained language path. It does not
isolate pruning, bridge initialization, model size or training-data effects.
Faithfulness is still the main failure: {64-counts['faithful']} cases fail that
component. A new NCM training protocol must qualify its labels, preserve native
initialization and evaluate actual source-disjoint summaries. Training fit and
an intact AR baseline cannot substitute for that result.

The grading worker's terminal output was unavailable after context rollover.
Its result is complete, the exec session expired and an exact process check
found no live worker; its unknown exit code is recorded as null. Other worker
exit codes were directly observed. All source fingerprints and raw judgments
were replayed before finalization.

Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`. Confirmed=false;
no NCM-success, final-test or speed claim. Reproduce the read-only evidence
validation with `python -m experiments.neural_architecture_lab.finalize_ncm_intact_language`;
fresh finalization is intentionally refused once this aggregate exists.
''')
    print(json.dumps({"classification": classification, "passed": passed, "cases": 64,
                      "component_passing_counts": counts, "gates": decision}), flush=True)


def register():
    a = q.read(DEST / "campaign_results.json")
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-intact-language-control-v1": {
        "name": "Intact pretrained language component control",
        "description": "Intact 1B native model: 50/64 whole-summary passes, 53/64 faithful, 61/64 salient, 64/64 readable. Language component passes; summary quality fails. No NCM or speed success.",
        "confirmation_status": a["classification"], "task_accuracy_measured": True,
        "NCM_model": False, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
