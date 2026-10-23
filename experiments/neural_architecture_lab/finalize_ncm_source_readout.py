"""Finalize full-summary source-readout feasibility with first-judgment replay."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net import ncm_source_readout as r
from experiments.structure_net.ncm_source_readout_grade import DEST as GRADING, PILOT as JUDGE_PILOT, audit_row, packet_key
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_source_readout_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-source-grounded-readout.md")


def passed_gate(value):
    return value is not None and value["status"] == "completed" and value["cases"] == 64 and value["passed"] >= 61


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh completed comparison required")
    result = q.read(GRADING / "result.json")
    if result["status"] != "completed":
        raise ValueError("terminal full readout grading required")
    terminal = q.read(GRADING / "worker_terminal_receipt.json")
    if terminal["status"] != "observed_worker_terminal" or terminal["exit_code"] != 0:
        raise ValueError("authoritative grading-worker termination required")
    roots = [r.DATA, r.PILOT, r.MEASURE, JUDGE_PILOT, GRADING]
    for root in roots:
        verify(q.read(root / "producer.json"))
    for path, expected in q.read(r.MEASURE / "result.json")["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("frozen readout selection changed")
    cache = q.read(GRADING / "judgment_cache.json")
    for key, entry in cache.items():
        if packet_key(entry["packet"], entry["role"]) != key:
            raise ValueError("cache packet identity differs")
        audit_row(entry["packet"], entry["role"], entry["row"])
    outputs = q.read(r.MEASURE / "outputs.json")
    condition_scores, flags, joint = {}, {}, []
    for condition in (*r.CONDITIONS, "reference_query"):
        expected_n = 20 if condition == "reference_query" else 64
        summary = result["results"][condition]
        if summary["primary"] is not None and (condition == "reference_query" or not passed_gate(summary["secondary"])):
            raise ValueError("primary ran after a failed ordered gate")
        for role in ("secondary", "primary"):
            endpoint = summary[role]
            if endpoint is None:
                if role != "primary" or (condition != "reference_query" and passed_gate(summary["secondary"])):
                    raise ValueError("required primary endpoint missing")
                continue
            root = GRADING / (role+"_"+condition)
            roots.append(root); verify(q.read(root / "producer.json"))
            scores, packets = q.read(root / "case_scores.json"), q.read(root / "packets.json")
            if len(scores) != expected_n or [s["id"] for s in scores] != [s["id"] for s in outputs[condition]]:
                raise ValueError("condition denominator/source order changed")
            count, invalid, failures = 0, 0, Counter()
            for score, packet, output in zip(scores, packets, outputs[condition]):
                key = packet_key(packet, role)
                if key != score["packet_key"] or score["first_origin"] != cache[key]["origin"]:
                    raise ValueError("first judgment provenance changed")
                row = cache[key]["row"]
                value = q.passes(row["evaluations"][0], score["text"], score["tokens"]) if row["status"] == "valid" else None
                if value != score["passed"] or score["text"] != output["text"]:
                    raise ValueError("summary score changed")
                chosen = output["selected"]
                if chosen and (output["source"][chosen["start"]:chosen["end"]] != chosen["raw"] or
                               r.normalized(chosen["raw"]) != output["text"] or output["tokens"] > 128):
                    raise ValueError("copying span/length identity failed")
                count += value is True; invalid += value is None
                if value is not None:
                    failures.update(k for k in ("faithful", "salient", "readable") if not row["evaluations"][0][k])
            if count != endpoint["passed"] or invalid != endpoint["invalid"] or endpoint["cases"] != expected_n:
                raise ValueError("endpoint reconstruction differs")
            condition_scores[role+"/"+condition] = scores
            flags[role+"/"+condition] = dict(failures)
        if condition != "reference_query" and passed_gate(summary["secondary"]) and passed_gate(summary["primary"]):
            joint.append(condition)
    for packet in q.read(JUDGE_PILOT / "packets.json"):
        key = packet_key(packet, "secondary")
        if cache[key]["row"] != q.read(JUDGE_PILOT / "rows" / (packet["case_id"]+".json")):
            raise ValueError("pilot first judgment was replaced")
    centroid = condition_scores["secondary/source_centroid"]
    ncm = condition_scores["secondary/ncm_query"]
    if [row["id"] for row in centroid] != [row["id"] for row in ncm]:
        raise ValueError("paired selector cases are misaligned")
    pairs = Counter((a["passed"] is True, b["passed"] is True) for a,b in zip(centroid,ncm))
    paired = {"both_pass": pairs[(True,True)], "centroid_only": pairs[(True,False)],
              "ncm_only": pairs[(False,True)], "both_fail_or_invalid": pairs[(False,False)]}
    reference = result["results"]["reference_query"]["secondary"]
    oracle_gate = reference["status"] == "completed" and reference["passed"] >= 19
    classification = "source_copy_readout_feasible" if joint else "source_copy_readout_task_gate_failed"
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-source-readout.v1", "classification": classification,
        "grading": result, "joint_passing_conditions": joint, "reference_projection_gate_passed": oracle_gate,
        "failed_flags": flags, "paired_secondary_centroid_vs_ncm": paired,
        "unique_judgments_including_pilot": len(cache), "first_valid_replay_verified": True,
        "NCM_trained": False, "NCM_success_established": False, "student_training_admitted": False,
        "confirmed": False, "underpowered": True, "speed_tested": False, "final_test_opened": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    lines = []
    for condition, values in result["results"].items():
        for role, endpoint in values.items():
            if endpoint is None:
                lines.append(f"|{condition}|{role}|not run: ordered gate|—|—|")
            else:
                ci = endpoint["wilson_95"]
                interval = f"{100*ci[0]:.2f}–{100*ci[1]:.2f}%" if ci else "invalid judgments; bounds retained"
                lines.append(f"|{condition}|{role}|{endpoint['passed']}/{endpoint['cases']}={100*endpoint['all_case_pass_lower_bound']:.4f}%|{interval}|{endpoint['invalid']}|")
    table = "\n".join(lines)
    REPORT.write_text(f'''# Source-grounded readout: whole-summary feasibility

Date: 2026-10-23. `ncm-source-grounded-readout-feasibility-v1` is complete,
classified **{classification}**. Jointly passing non-oracle conditions:
`{joint}`. No new NCM head was trained; the final task test stays sealed and
the95% final NCM objective is not established by this comparison.

These are **whole-summary** pass counts under the unchanged faithful, salient,
readable, nonempty and128-cl100k-token criterion, not codec reconstruction or
exact-reference accuracy. Secondary is Terra; primary is Sonnet.

| Readout | Judge role | Complete summaries passing | Wilson95% | Invalid |
| --- | --- | --- | --- | --- |
{table}

Lead uses the earliest complete eligible sentence. Source centroid uses mean
normalized BGE sentence vectors. NCM/parent query conditions use the preserved
draft as a semantic query; they copy source words, never repair those drafts.
Reference-query20 uses the separate teacher summary and is target-aware,
unusable at inference. Its19/20 projection gate passed=`{oracle_gate}`.

Every condition uses the same64 original source-disjoint development cases;
reference20 is a separate source-disjoint diagnostic. Candidates are one to
three contiguous source sentences, normalized only for whitespace and capped
at128tokens. No source/case exclusion, truncation, rewritten word or LLM fallback.
All selected raw character spans and rendered text identities were verified.
This output budget is the application's128-token cap; the old free-text NCM
used a48-GPT2-token canvas, so this is not a matched generation-speed comparison.

Exact source-summary/rubric packets share their first judgment within a judge
role. The amendment was fixed before main grading; pilot negatives and exhausted
invalids cannot get another semantic draw. All four pilot judgments replay
exactly, cache requests/role/model/payloads replay, and denominators stay64.
There are{len(cache)} unique judgments including the pilot, with each condition's
cache hits and raw origins retained. Role-separated caches preserve independent
Sonnet judgment. Primary is run only after61/64 on the valid secondary endpoint.
No positive/negative cherry-picking or alternate checkpoint selection occurred.

On the same secondary cases, centroid versus NCM-query paired counts are
`{paired}`. This identifies whether the NCM draft helps or hurts this selector;
it is not proof of a superior trained NCM mechanism. A BGE/lead baseline passing
would not count as NCM success. The original free-text attachment's prior0/64
is preserved; this experiment changes the readout and has no new head training.

BGE-M3 public revision is5617a9f61b028005a4858fdac845db406aefb181, verified
against official metadata and LFS hashes. Strict safe state loading, FP16,
SDPA, batch8 on GPU2, normalized CLS readout; no model weights changed. Real
worst-input memory pilot, finite vectors and exact embedding/selection reload
passed. pySBD0.3.4 supplies exact sentence character spans. Embedding preparation
and cached old drafts are offline experimental work, not inference speed evidence.

The independent broader-label audit remains113/128 and fails its training-data
quality gate. Therefore this report admits no scaled student run. A subsequent
NCM pointer/concept head needs a newly qualified training-label representation,
its own source-disjoint complete-summary evaluation and mechanism ablations.
If this copying family fails, use a new declared abstractive final-token/pointer
study; do not weaken the rubric or return to closed frozen-decoder data sweeps.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. All fixed outputs,
source spans, embeddings, first raw responses, exact reuse origins, producers
and source snapshots remain append-only. Registry confirmed=false,
NCM_success_established=false, final_test_opened=false and speed_tested=false.
''')
    print(json.dumps({"status": classification, "joint_passing": joint, "report": str(REPORT)}), flush=True)


def register():
    a = q.read(DEST / "campaign_results.json")
    description = "; ".join(f"{condition}Terra{values['secondary']['passed']}/{values['secondary']['cases']}"
        for condition,values in a["grading"]["results"].items())
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        r.HYPOTHESIS: {"name": "Whole-summary source-grounded readout feasibility",
            "description": description+". Fixed first-judgment-preserved source-copying comparison; no new NCM head, final test or speed claim.",
            "confirmation_status": a["classification"], "task_accuracy_measured": True,
            "NCM_trained": False, "NCM_success_established": False},
    })
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
