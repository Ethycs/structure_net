"""Archive native-query systems controls and the separate failed teacher gate."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_source_readout_grade import audit_row, packet_key
from experiments.neural_architecture_lab.ncm_result_registry import register_records


DEST = ROOT / "20261023_native_query_and_label_controls_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-native-query-and-label-controls.md")


def completed(name):
    root = ROOT / name
    if q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("observed successful worker termination required")
    verify(q.read(root / "producer.json"))
    return root, q.read(root / "result.json")


def joint_teacher_gate(passed, valid, cases, labels):
    if cases != 128 or len(labels) != cases or not 0 <= passed <= valid <= cases:
        raise ValueError("complete fixed128-case teacher cohort required")
    capacity = all(row["text"].strip() and row["claims"] and row["stored_tokens"] <= 128
                   and row["native_tokens_including_eos"] <= 128 for row in labels)
    return capacity, capacity and valid == cases and passed >= 122


def replay_grade(root, source, condition):
    labels = q.read(source / "labels.json")
    scores = q.read(root / ("secondary_" + condition) / "case_scores.json")
    cache = q.read(root / "judgment_cache.json")
    if len(labels) != len(scores):
        raise ValueError("all labels and first scores required")
    valid = passed = 0
    for label, score in zip(labels, scores):
        packet = q.packet(label["id"], label["source"], [{"candidate_id": "summary", "text": label["text"]}])
        key = packet_key(packet, "secondary")
        entry = cache[key]
        if entry["packet"] != packet or entry["role"] != "secondary" or score["packet_key"] != key:
            raise ValueError("first packet identity changed")
        row = entry["row"]
        audit_row(packet, "secondary", row)
        ok = row["status"] == "valid"
        decision = q.passes(row["evaluations"][0], label["text"], label["stored_tokens"]) if ok else None
        if score["passed"] != decision or score["valid"] != ok or score["id"] != label["id"]:
            raise ValueError("first whole-summary decision changed")
        valid += ok
        passed += decision is True
    result = q.read(root / "result.json")["endpoint"]
    if (valid, passed, len(labels)) != (result["valid"], result["passed"], result["cases"]):
        raise ValueError("raw judgments and aggregate disagree")
    return labels, result


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    cache_root, cache = completed("20261023_native_query_cache_pilot_v2")
    concept_root, concept = completed("20261023_native_query_concept_pilot_v1")
    pilot_root, pilot = completed("20261023_native_query_neighborhood_pilot_v1")
    failed_cache = ROOT / "20261023_native_query_cache_pilot_v1"
    if q.read(failed_cache / "worker_terminal_receipt.json")["exit_code"] != 1:
        raise ValueError("original OOM must remain a failed terminal run")
    for result in (cache, concept, pilot):
        if result["status"] != "passed" or result["task_accuracy"] is not None or not result["original_weights_frozen"]:
            raise ValueError("systems-only result identity differs")
    if not all(cache[k] for k in ("source_cache_immutable", "reload_exact")):
        raise ValueError("cache lifecycle failed")
    if not all(concept[k] for k in ("books_unchanged", "source_cache_immutable", "free_repeat_and_reload_exact")):
        raise ValueError("soft-input NCM lifecycle failed")
    if (pilot["updates"] != 4 or pilot["fixed_parallel_passes"] != 9
            or pilot["uses_AR_fallback_or_external_verifier"] or not pilot["checkpoint_generator_reload_exact"]):
        raise ValueError("mixed-task pilot contract differs")
    audit_root, audit = completed("20261023_revised_labels_main_audit_v2")
    revision_root, revision = completed("20261023_revised_labels_main_revision_v2")
    critique_root, critique = completed("20261023_revised_labels_main_critique_v2")
    draft_root, draft = completed("20261023_revised_labels_main_draft_v1")
    profile.install()
    labels, endpoint = replay_grade(audit_root, revision_root, "audit")
    _, draft_endpoint = replay_grade(critique_root, draft_root, "critique")
    capacity, gate = joint_teacher_gate(endpoint["passed"], endpoint["valid"], endpoint["cases"], labels)
    if capacity != audit["all_final_capacity_passed"] or gate != audit["label_quality_gate"] or gate:
        raise ValueError("preserved failed joint teacher gate differs")
    if revision["status"] != "failed_format_or_capacity" or draft["status"] != "failed_format_or_capacity":
        raise ValueError("original capacity failures must remain failures")
    failures = [{"id": r["id"], "stored_tokens": r["stored_tokens"],
                 "native_tokens_including_eos": r["native_tokens_including_eos"]}
                for r in labels if r["stored_tokens"] > 128 or r["native_tokens_including_eos"] > 128]
    roots = [cache_root, concept_root, pilot_root, failed_cache, audit_root, revision_root, critique_root, draft_root]
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-native-query-controls.v1",
        "classification": "systems_passed_teacher_joint_capacity_gate_failed",
        "native_query_cache": cache, "soft_input_NCM_lifecycle": concept,
        "mixed_task_neighborhood_lifecycle": pilot, "original_cache_OOM_preserved": True,
        "teacher_first_drafts": draft_endpoint, "teacher_single_revision": endpoint,
        "teacher_capacity_failures": failures, "teacher_joint_gate": gate,
        "selected_training_labels_are_a_separate_declared_study": True,
        "student_task_accuracy": None, "NCM_success_established": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Native-query lifecycle works; student quality remains unmeasured

Date: 2026-10-23. Exploratory, `systems_passed_teacher_joint_capacity_gate_failed`.
The95%student summary goal remains unmet. These controls are neither a paper
training replication nor a speed benchmark.

The shared-query3B control retains all28native blocks and freezes every original
weight. Its initial true-previous-word numerical comparison reached99.21875%top1
agreement and relativeL2error0.01966365. That denominator includes256query
positions, including padding; it is **not summary accuracy or exact identity**.
The original backward OOM remains failed, with its completed update count unknown.
The explicit v2 memory recovery passed two updates, immutable sourceKV, frozen
weights and exact new-weight reload at8,908,633,088bytes peak allocation.

The actual five-concept-level NCM soft-input pilot passed two real updates and
frozen weights/books, conditional-residual, source-cache and free-output reload
checks. It adds8,637,696trainable parameters to the original model. Its text is
unusable after two updates; passing lifecycle checks establishes no competence.
Peak allocation was8,787,542,528bytes. No additional1Bdecoder or parent verifier
was used. The full soft-input-only training configuration was deferred.

The revised native-word neighborhood pilot passed four updates covering raw
language and summary inputs, coarse codes, native-center and generated-history
word arms. Fixed primary inference is five concept passes plus four parallel
native-word refinements, without AR fallback. Frozen originals/books/caches and
exact checkpoint/generator reload passed; peak allocation8,656,379,904bytes.
This four-update pilot is systems evidence only; held-out student quality and
throughput require separate measurement from the fixed trained checkpoint.

The independently scheduled teacher pipeline's128first drafts scored
{draft_endpoint['passed']}/128whole-summary passes. After one critique-informed
revision per draft, {endpoint['passed']}/128passed({endpoint['task_accuracy']:.5%}),
with128valid first judgments and Wilson95%interval
[{endpoint['wilson_95'][0]:.5%},{endpoint['wilson_95'][1]:.5%}].
Fifteen exact first judgments were reused;113new requests were made. Terra both
critiqued and audited; this is not independent human gold or student accuracy.
All raw requests and first decisions were replayed during finalization.

The original joint teacher gate still **failed**: one final text exceeded the
128stored-token cap and two exceeded128native tokens including EOS. The latter
counts overlap. The stricter all-case capacity rule is not repaired by the
96.09%whole-summary point estimate. No original full-population training
admission is inferred. A separately preregistered143-label exploratory study
uses111individually accepted first drafts plus32older accepted labels, without
choosing the better draft/revision or relabeling this failed gate.

Aggregate `{campaign}`, SHA256`{file_hash(campaign)}`. Registry records retain
confirmed=false; no95%student, compression or speedup claim. Original64development
cases stay the student gate and final256cases remain sealed. The short768-token
prompt prototype cannot establish the full6300-token application range.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"],
                      "teacher_passed": endpoint["passed"], "student_task_accuracy": None}), flush=True)


def register():
    specs = {
        "ncm-native-query-systems-controls-v1": {
            "name": "Frozen full-native query and NCM lifecycle controls",
            "description": "Native28-block3B cache and NCM pilots passed frozen/cache/reload checks; original OOM retained, mixed-task four-update native-word pilot passed. No student summary accuracy or speed established.",
            "confirmation_status": "systems_only_original_failure_preserved", "task_accuracy_measured": False},
        "ncm-revised-teacher-label-quality-v2": {
            "name": "One-revision teacher summaries with preserved capacity failures",
            "description": "128fixed teacher drafts111whole passes; one revision each123/128Terra passes,128valid. Joint gate failed: stored overflow1,native includingEOSoverflow2. Separate accepted-label student study cannot rescue the failed full-population protocol.",
            "confirmation_status": "teacher_point_endpoint_passed_joint_capacity_gate_failed", "task_accuracy_measured": True}}
    for spec in specs.values():
        spec.update(student_task_accuracy_measured=False, NCM_success_established=False)
    register_records(DEST / "campaign_results.json", REPORT, DEST, specs)
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
