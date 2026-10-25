"""CPU replay of the full training record and all source-only DEV judgments."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_level_conditioning_development as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def audit_training():
    result, saved = run.checkpoint_contract()
    model = run.training.lifecycle.mechanism.enable(run.head.from_domain(device="cpu"))
    model.load_state_dict(saved["model"], strict=True)
    run.model_contract(model, result)
    full_hash = run.native.vendor.module_hash(model)
    model.close()
    audit = run.q.read(run.TRAIN_AUDIT / "result.json")
    return {"updates": 1024, "presentations": 16384, "source_only_history_microbatches": 1024,
        "train_cases": 655, "trainable_parameters": 127732736,
        "independent_trace_audit_sha256": file_hash(run.TRAIN_AUDIT / "result.json"),
        "fixed_final_checkpoint_sha256": result["checkpoint_sha256"], "full_model_sha256": full_hash,
        "training_precision": result["precision"], "physical_GPU": result["physical_GPU"],
        "scaler": result["scaler"], "new_random_trainable_weights": False,
        "pilot_or_native16_weights_reused": False,
        "conditioning_corruption": "same-book linear.75to.25 including root; targets clean",
        "frozen_geometry": result["protected_codec_hashes"],
        "actual_training_audit": audit["results"]["full"]}


def replay_outputs(values, rows, manifest, tokenizer):
    if [r["id"] for r in values] != [r["id"] for r in rows]:
        raise ValueError("one output per complete ordered source required")
    for value, row in zip(values, rows):
        decoded = run.decode(value["raw_word_ids"], tokenizer)
        entry = manifest[row["id"]]
        if (any(value[k] != v for k, v in decoded.items())
                or value["source_sha256"] != row["source_sha256"] or value["split"] != row["split"]
                or value["source_positions"] != entry["source_tokens"] or value["feature_sha256"] != entry["sha256"]
                or len(value["codes"]) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in value["codes"])):
            raise ValueError("complete raw source-only prediction or decoding differs")
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("original full-source feature bytes changed")


def finalize():
    run.configure()
    run.semantic.require_judges()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh append-only DEV aggregate and report required")
    q = run.q
    roots = (run.METADATA, run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE)
    for root in roots:
        run.native.vendor.terminal(root)
    cases, manifest = run.source_inputs()
    training = audit_training()
    proof = run.training.split_guard(q.read(run.training.DATA / "cases.json"), cases)
    if (q.read(run.METADATA / "cases.json") != cases or q.read(run.METADATA / "split_proof.json") != proof
            or file_hash(run.METADATA / "cases.json") != run.SOURCE_SHA):
        raise ValueError("source-only metadata and source exclusion replay differ")
    public = run.reference.verify_public([r for rows in cases.values() for r in rows])
    public_proof = {"public_file_sha256": run.reference.PUBLIC_SHA256, "exact_sources": public}
    if q.read(run.METADATA / "public_proof.json") != public_proof:
        raise ValueError("metadata exact public source proof differs")
    values, pilot = q.read(run.OUTPUTS / "outputs.json"), q.read(run.PILOT / "outputs.json")
    archived = run.archive_outputs()
    if values != archived:
        raise ValueError("all128 predictions must reproduce the first independent numeric archive")
    run.assert_archive_match(pilot, archived)
    rows = [r for c in run.COHORTS for r in cases[c]]
    tokenizer = run.native.tokenizer()
    replay_outputs(values, rows, manifest, tokenizer)
    replay_outputs(pilot, run.inference_probes(cases, manifest), manifest, tokenizer)
    by_id = {r["id"]: r for r in values}
    if any(by_id[r["id"]] != r for r in pilot):
        raise ValueError("fresh full inference changed a retained pilot")
    for root, size in ((run.PILOT, 2), (run.OUTPUTS, 128)):
        state = q.read(root / "result.json")
        producer = q.read(root / "producer.json")
        if (state["cases"] != size or state["checkpoint_sha256"] != training["fixed_final_checkpoint_sha256"]
                or not state["model_unchanged"] or state["model_sha256"] != training["full_model_sha256"] or state["outputs_sha256"] != file_hash(root / "outputs.json")
                or state["model_sha256"] != q.read(run.PILOT / "result.json")["model_sha256"]
                or producer["config"]["precision"] != "FP32" or producer["config"]["TF32"]
                or producer["config"]["oracle_target_codes"] or not producer["config"]["source_only"]
                or not state["first_numeric_archive_exact_match"]
                or state["first_numeric_archive_sha256"] != file_hash(run.archive.OUTPUTS / "outputs.json")):
            raise ValueError("fixed checkpoint, inference state or precision differs")
    packet_cases = {c: [{**r, "texts": {"student": by_id[r["id"]]["text"]}} for r in cases[c]] for c in run.COHORTS}
    packets = {c: {"student": [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"]["student"]}])
        for r in packet_cases[c]]} for c in run.COHORTS}
    if (q.read(run.PACKETS / "cases.json") != packet_cases or q.read(run.PACKETS / "exact_packets.json") != packets
            or q.read(run.PACKETS / "public_proof.json") != public_proof
            or q.read(run.PACKETS / "result.json")["exact_packets_sha256"] != file_hash(run.PACKETS / "exact_packets.json")):
        raise ValueError("all128 exact source packets differ")
    old = run.base_cache()
    pilot_cache, cache = q.read(run.JUDGE_PILOT / "judgment_cache.json"), q.read(run.GRADE / "judgment_cache.json")
    if any(pilot_cache.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot_cache.items()):
        raise ValueError("a canonical or pilot first judgment changed")
    run.semantic.audit_cache(cache)
    score_tables, metrics = {}, {c: {} for c in run.COHORTS}

    def replay(root, cohort, role, n, decisions):
        expected = replay_scores(packet_cases[cohort][:n],
            {"student": packets[cohort]["student"][:n]}, decisions, "student", role)
        if q.read(root / cohort / f"{role}_student_case_scores.json") != expected:
            raise ValueError("source-only first-judgment score replay differs")
        metric = run.task_metric(expected)
        actual = q.read(root / cohort / f"{role}_student_result.json")
        if any(actual.get(k) != v for k, v in metric.items()) or "oracle_semantic_pass_rate" in actual:
            raise ValueError("actual source-only task metric differs")
        return expected, actual

    for cohort in run.COHORTS:
        pilot_scores, _ = replay(run.JUDGE_PILOT, cohort, "secondary", 2, pilot_cache)
        if any(not r["valid"] for r in pilot_scores):
            raise ValueError("four-case systems judgment validity required")
        score_tables[cohort, "secondary"], metrics[cohort]["secondary"] = replay(run.GRADE, cohort, "secondary", 64, cache)
    # Apply the locked decision independently of the producer's admission flag.
    admitted = all(metrics[c]["secondary"]["valid"] == 64 and metrics[c]["secondary"]["passed"] >= 61 for c in run.COHORTS)
    joints = None
    if admitted:
        joints = {}
        for cohort in run.COHORTS:
            score_tables[cohort, "primary"], metrics[cohort]["primary"] = replay(run.GRADE, cohort, "primary", 64, cache)
            secondary, primary = score_tables[cohort, "secondary"], score_tables[cohort, "primary"]
            if [r["id"] for r in secondary] != [r["id"] for r in primary]:
                raise ValueError("same ordered cases required")
            valid = all(a["valid"] and b["valid"] for a, b in zip(secondary, primary))
            joint = sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))
            joints[cohort] = {"cases": 64, "joint_passed": joint, "all_valid": valid, "qualified": valid and joint >= 61}
    else:
        if any((run.GRADE / c / "primary_student_case_scores.json").exists() for c in run.COHORTS):
            raise ValueError("primary grading ran before secondary admission")
    allowed_pilot = {run.reference.first.packet_key(p, "secondary") for c in run.COHORTS for p in packets[c]["student"][:2]}
    allowed = {run.reference.first.packet_key(p, role) for c in run.COHORTS for p in packets[c]["student"]
        for role in (("secondary", "primary") if admitted else ("secondary",))}
    if set(pilot_cache) - set(old) - allowed_pilot or set(cache) - set(old) - allowed:
        raise ValueError("new first judgments outside admitted complete DEV scope")
    for root, expected_count, inherited in ((run.JUDGE_PILOT, len(pilot_cache), len(old)), (run.GRADE, len(cache), len(pilot_cache))):
        result = q.read(root / "result.json")
        if result["canonical_first_judgments"] != expected_count or result["inherited_first_judgments"] != inherited:
            raise ValueError("canonical first-decision counts differ")
    qualified = admitted and all(joints[c]["qualified"] for c in run.COHORTS)
    classification = "full128_source_only_DEV_qualified" if qualified else "full128_source_only_DEV_quality_failed"
    result = q.read(run.GRADE / "result.json")
    if (result["status"] != classification or result["metrics"] != metrics or result["joint"] != joints
            or result["primary_admitted"] != admitted or result["DEV_qualified"] != qualified
            or result["final_test_opened"] or result["speed_tested"]):
        raise ValueError("locked complete source-only DEV decision differs")
    artifacts = {}
    for root in (*roots, run.training.WORKFLOW, run.training.TRAIN):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "producer_source" not in path.parts and path.suffix in (".json", ".jsonl", ".pt"):
                artifacts[str(path)] = file_hash(path)
    artifacts[str(run.OLD_CACHE)] = run.CACHE_SHA
    for entry in cache.values():
        for path in entry["row"]["attempts"]:
            artifacts[path] = file_hash(path)
    campaign = {"schema": run.SCHEMA, "hypothesis_id": run.HYPOTHESIS,
        "status": "completed", "classification": classification, "tested": True, "confirmed": False,
        "underpowered": True, "training_audit": training, "metrics": metrics, "joint": joints,
        "DEV_qualified": qualified, "primary_admitted": admitted, "canonical_first_judgments": len(cache),
        "source_prediction": True, "oracle_target_codes": False, "all128_cases_retained": True,
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "final_task_accuracy_status": "unmeasured; sealed final sources unopened", "artifacts": artifacts}
    run.RESULTS.mkdir(parents=True)
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = ["| Cohort | Judge | Complete-summary passes | Valid |", "| --- | --- | ---: | ---: |"]
    for c in run.COHORTS:
        for role, value in metrics[c].items():
            table.append(f"| {c} | {role} | {value['passed']}/64 | {value['valid']}/64 |")
    run.REPORT.write_text(
        "# FP16 level-conditioning NCM source-only development result\n\n"
        f"Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`. Classification **{classification}**.\n"
        f"Campaign `{path}`, SHA256 `{file_hash(path)}`.\n\n"
        "The released97Mconcept generator, domain word decoder and source attention\n"
        "were adapted together:127,732,736parameters,1024fixed updates on all655\n"
        "unchanged original TRAIN targets. The35revised targets were excluded.\n"
        "Both pilot and full run reset the same warm initializer; neither pilot\n"
        "weights nor the16-case overfit checkpoint were reused. Native encoder/VQ\n"
        "geometry and the original Llama body/MLPs remained frozen.\n"
        "The actual independent CPU training audit replayed all16,384presentations,\n"
        "1024source-only history microbatches, complete noise, scaler and checkpoint state.\n"
        "Training used FP16 autocast, FP32 master weights, GradScaler128, micro1\n"
        "and accumulation16 on physicalGPU2. History corruption decreased linearly\n"
        "from.75to.25 across levels, including the generated root; targets were clean.\n"
        "These endpoints are our choices, not a verified author training recipe.\n"
        "The prior native655 comparison also changes precision, microbatches and seeds;\n"
        "this experiment cannot isolate a causal benefit of conditioning noise.\n"
        "The qualified primary judge is Sol and secondary is Terra; all3350 preceding\n"
        "mixed-model first judgments retain their original identities and scores.\n"
        "The hard-forward/biased-soft-backward word objective is an experimental\n"
        "conditional adaptation, not a matched paper replication. Full FP32\n"
        "source-only inference loaded the ENTIRE trained native checkpoint, retained\n"
        "every source position, and received no target words or codes.\n\n"
        + "\n".join(table) + "\n\n"
        + f"Both-cohort same-case DEV qualified: **{qualified}**. Primary admitted: **{admitted}**. "
        + f"Canonical first judgments retained: {len(cache)}.\n\n"
        "The semantic judges assess supported claims, salience and readability; exact\n"
        "summary matches are not required. Nonempty output and128stored-token cap\n"
        "remain mandatory. Invalids are retained. Every raw output, failure and prior\n"
        "first judgment is preserved. Previous old-student0/128 and codec failures\n"
        "remain separate evidence. The final320sources remain closed and unhashed;\n"
        "final task accuracy and speed are unmeasured. This DEV result does not\n"
        "complete the95%task objective.\n")
    print(json.dumps({"classification": classification, "DEV_qualified": qualified, "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.RESULTS / "campaign_results.json"
    data = run.q.read(campaign)
    register_records(campaign, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "FP16 level-conditioning NCM: complete source-only DEV",
        "description": "Fixed1024 FP16 level-conditioning generator/decoder/source-attention adaptation on655 unchanged targets, scored on all128 complete source-disjoint sources; semantic complete-summary accuracy, source-only hard generation, no oracle target codes.",
        "classification": data["classification"], "DEV_qualified": data["DEV_qualified"],
        "canonical_first_judgments": data["canonical_first_judgments"], "source_only": True,
        "all128_cases_retained": True, "final_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("audit-training", "finalize", "register"))
    args = parser.parse_args()
    if args.stage == "audit-training":
        print(json.dumps(audit_training()), flush=True)
    else:
        {"finalize": finalize, "register": register}[args.stage]()
