"""CPU replay of the full training record and all source-only DEV judgments."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_grounded_development as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def audit_training():
    training, q = run.training, run.q
    result, saved = run.checkpoint_contract()
    for root in (training.DATA, training.CACHE_PILOT, training.CACHE, training.PILOT, training.TRAIN):
        run.native.vendor.terminal(root)
    whole = q.read(training.WORKFLOW / "result.json")
    if ([r["stage"] for r in whole["stages"]] != ["prepare", "cache-pilot", "cache", "pilot", "train"]
            or any(r["exit_code"] != 0 for r in whole["stages"])):
        raise ValueError("all actual full-training child exits required")
    rows = q.read(training.DATA / "cases.json")
    manifest = q.read(training.DATA / "manifest.json")
    codes = {r["id"]: r["codes"] for r in q.read(training.CACHE / "outputs.json")}
    if len(rows) != 690 or set(codes) != {r["id"] for r in rows}:
        raise ValueError("complete690 training labels required")
    if (result["initial_attention_sha256"] != q.read(training.PILOT / "result.json")["initial_attention_sha256"]
            or q.read(training.TRAIN / "initial_probes.json") != q.read(training.PILOT / "initial_probes.json")
            or result["initial_attention_sha256"] == result["final_attention_sha256"]):
        raise ValueError("fresh full-training initialization and actual attention update required")
    sampler = torch.Generator().manual_seed(training.Config().sampler_seed)
    neighbor = torch.Generator().manual_seed(training.Config().neighbor_seed)
    count, histories = 0, 0
    with (training.TRAIN / "trace.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            trace = json.loads(line)
            if (trace["step"] != count or trace["lr"] != training.rate(count)
                    or not math.isfinite(trace["gradient_norm"]) or trace["gradient_norm"] < 0
                    or len(trace["microbatches"]) != training.Config().accumulation):
                raise ValueError("contiguous fixed-budget finite training trace required")
            for micro_index, micro in enumerate(trace["microbatches"]):
                indices = torch.randint(len(rows), (4,), generator=sampler).tolist()
                selected = [rows[i] for i in indices]
                if micro["case_ids"] != [r["id"] for r in selected]:
                    raise ValueError("uniform complete690 sampler differs")
                if micro["source_positions"] != max(manifest[r["id"]]["source_tokens"] for r in selected):
                    raise ValueError("full-source microbatch length differs")
                native_codes = torch.tensor([codes[r["id"]] for r in selected], dtype=torch.long)
                _, proof = training.objective.corrupt_history(native_codes, count, neighbor)
                if micro["losses"]["corruption"] != proof:
                    raise ValueError("actual native neighborhood noise stream differs")
                names = {"clean_concept_loss", "clean_word_loss", "noisy_concept_loss", "noisy_word_loss", "corruption"}
                expected_history = count > 256 and count % 4 == 0 and micro_index == 0
                if expected_history:
                    names |= {"self_concept_loss", "self_word_loss", "source_only_history_codes"}
                    history = micro["losses"]["source_only_history_codes"]
                    if (len(history) != 4 or any(len(h) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in h) for h in history)):
                        raise ValueError("complete generated source-only histories required")
                    histories += 1
                if set(micro["losses"]) != names or any(not math.isfinite(v) or v < 0
                        for k, v in micro["losses"].items() if k.endswith("_loss")):
                    raise ValueError("finite exact clean/noisy/self-history objective required")
    if (count != 4096 or histories != 960 or not torch.equal(sampler.get_state(), saved["sampler_rng"])
            or not torch.equal(neighbor.get_state(), saved["neighbor_rng"])):
        raise ValueError("complete fixed4096 sampler/noise replay and saved RNG states differ")
    return {"updates": count, "sampled_case_presentations": count * 16, "source_only_history_microbatches": histories,
        "uniform_sampler_replayed": True, "neighborhood_stream_replayed": True,
        "fixed_final_checkpoint_sha256": result["checkpoint_sha256"], "new_attention_updated": True,
        "frozen_native_states_unchanged": result["frozen_native_states_unchanged"]}


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
    proof = run.training.disjoint(q.read(run.training.DATA / "cases.json"), cases)
    if (q.read(run.METADATA / "cases.json") != cases or q.read(run.METADATA / "split_proof.json") != proof
            or file_hash(run.METADATA / "cases.json") != run.SOURCE_SHA):
        raise ValueError("source-only metadata and source exclusion replay differ")
    public = run.reference.verify_public([r for rows in cases.values() for r in rows])
    public_proof = {"public_file_sha256": run.reference.PUBLIC_SHA256, "exact_sources": public}
    if q.read(run.METADATA / "public_proof.json") != public_proof:
        raise ValueError("metadata exact public source proof differs")
    values, pilot = q.read(run.OUTPUTS / "outputs.json"), q.read(run.PILOT / "outputs.json")
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
                or not state["model_unchanged"] or state["outputs_sha256"] != file_hash(root / "outputs.json")
                or state["model_sha256"] != q.read(run.PILOT / "result.json")["model_sha256"]
                or producer["config"]["precision"] != "FP32" or producer["config"]["TF32"]
                or producer["config"]["oracle_target_codes"] or not producer["config"]["source_only"]):
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
    campaign = {"schema": "nal.ncm-lm1b-grounded-source-only-dev.v1", "hypothesis_id": run.HYPOTHESIS,
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
        "# Pretrained NCM source-only development result\n\n"
        f"Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`. Classification **{classification}**.\n"
        f"Campaign `{path}`, SHA256 `{file_hash(path)}`.\n\n"
        "The authors' released LM1B generator and task-adapted codec were frozen while\n"
        "11,796,480 new source-attention parameters trained for4096fixed updates on690\n"
        "examples. The independent CPU auditor replayed all65,536 sampled examples,\n"
        "all neighborhood corruption and960source-only self-history microbatches.\n"
        "Full precision source-only inference used every unchanged permanent DEV source,\n"
        "without target words or codes. This is conditional task adaptation, not a paper replication.\n\n"
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
        "name": "Released NCM with learned source attention: complete source-only DEV",
        "description": "Fixed4096 pretrained NCM adaptation scored on all128 source-disjoint full sources; semantic complete-summary task accuracy, not oracle reconstruction.",
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
