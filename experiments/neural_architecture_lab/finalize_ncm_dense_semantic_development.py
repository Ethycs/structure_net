"""Audit complete semantic development judgments of the fixed dense student."""
import argparse
import json
import math
from pathlib import Path

from experiments.structure_net import ncm_dense_semantic_development as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def joint_counts(secondary, primary):
    if primary is None:
        return None
    if [r["id"] for r in secondary] != [r["id"] for r in primary] or not secondary:
        raise ValueError("same ordered cases required for joint semantic counts")
    n = len(secondary)
    valid = sum(a["valid"] and b["valid"] for a, b in zip(secondary, primary))
    passed = sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))
    z = 1.96
    point = passed / n
    center = (point + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(point * (1 - point) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"cases": n, "valid": valid, "invalid": n - valid, "passed": passed,
        "task_accuracy": point if valid == n else None,
        "all_case_pass_lower_bound": point, "all_case_pass_upper_bound": (passed + n - valid) / n,
        "wilson_95": [max(0., center - half), min(1., center + half)] if valid == n else None}


def replay_outputs(values, expected, tokenizer):
    reference = run.reference
    reference.require_outputs(values, expected)
    for value, row in zip(values, expected):
        text, end = reference.decode_words(value["raw_word_ids"], tokenizer)
        levels = value["coarse_codes"]
        if (value["text"] != text or value["global_EOS_position"] != (end if end < 192 else None)
                or value["source_tokens"] != len(row["source_ids"]) or value["source_frame"] != row["source_frame"]
                or len(value["latent_sha256"]) != 64 or len(levels) != 8
                or any(len(level) != width or any(type(c) is not int or not 0 <= c < 4096 for c in level)
                       for level, width in zip(levels, (16, 24, 40, 56, 80, 104, 128, 160)))):
            raise ValueError("raw complete word/concept/source decoding differs")


def finalize():
    run.configure()
    cases, _ = run.inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh dense semantic aggregate/report required")
    reference = run.reference
    for root in (run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE):
        reference.data.terminal(root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(reference.data.GPT2, local_files_only=True)
    expected = [r for rows in cases.values() for r in rows]
    values = reference.q.read(run.OUTPUTS / "outputs.json")
    replay_outputs(values, expected, tokenizer)
    pilot_outputs = reference.q.read(run.PILOT / "outputs.json")
    replay_outputs(pilot_outputs, run.inference_probes(cases), tokenizer)
    by_id = {r["id"]: r for r in values}
    if any(by_id[r["id"]] != r for r in pilot_outputs):
        raise ValueError("retained inference pilot outputs differ")
    train_rows, _, expected_train = run.training_parity_inputs()
    train_outputs = reference.q.read(run.PILOT / "train_parity_outputs.json")
    replay_outputs(train_outputs, train_rows, tokenizer)
    run.parity_check(train_outputs, expected_train)
    for root, is_pilot in ((run.PILOT, True), (run.OUTPUTS, False)):
        result = reference.q.read(root / "result.json")
        if (result["cases"] != (2 if is_pilot else 128)
                or result["selected_checkpoint_sha256"] != run.CHECKPOINT
                or result["output_sha256"] != file_hash(root / "outputs.json")
                or not result["all_parameters_and_codec_unchanged"] or not result["fresh_reload_or_TRAIN_parity"]
                or result["peak_allocated_bytes"] > 8_000_000_000 or result["final_test_opened"]
                or (not is_pilot and not result["retained_pilot_exact"])):
            raise ValueError("frozen source-only generation, reload, memory or output evidence differs")
    packets = reference.q.read(run.PACKETS / "exact_packets.json")
    if packets != {split: [reference.q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}])
                          for r in values if r["split"] == split] for split in reference.COHORTS}:
        raise ValueError("anonymous single-candidate public packets differ")
    reference.verify_public(values)
    baseline = run.base_cache()
    prior = run.prior_cache()
    if any(prior.get(k) != v for k, v in baseline.items()):
        raise ValueError("all1669 first judgments must remain unchanged")
    pilot = reference.q.read(run.JUDGE_PILOT / "result.json")
    if (pilot["status"] != "systems_semantic_judge_path_passed" or pilot["cases"] != 4 or pilot["valid"] != 4
            or pilot["quality_gate"] or not pilot["all1669_prior_first_judgments_preserved"]):
        raise ValueError("valid schema-only pilot required without a model-quality gate")
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    pilot_passes = 0
    for row in run.judge_probes(values):
        packet = reference.q.packet(row["id"], row["source"], [{"candidate_id": "summary", "text": row["text"]}])
        key = reference.first.packet_key(packet, "secondary")
        if key not in prior or prior[key]["row"]["status"] != "valid":
            raise ValueError("every judge pilot first decision must be retained")
        pilot_passes += reference.q.passes(prior[key]["row"]["evaluations"][0], row["text"],
            len(encoding.encode(row["text"], disallowed_special=()))) is True
    if pilot["passed"] != pilot_passes or pilot["unique_first_judgments"] != len(prior):
        raise ValueError("raw semantic pilot pass/cache counts differ")
    grade, failures, count = run.legacy_audit.replay_grade(values, prior)
    cache = reference.q.read(run.GRADE / "judgment_cache.json")
    joint, scores = {}, {}
    for split in reference.COHORTS:
        scores[split] = {}
        for role in ("secondary", "primary"):
            endpoint = grade["results"][split][role]
            if endpoint is None:
                scores[split][role] = None
                continue
            rows = reference.q.read(run.GRADE / f"{split}_{role}_case_scores.json")
            for row in rows:
                if row["evaluations"] != cache[row["packet_key"]]["row"].get("evaluations"):
                    raise ValueError("summary components differ from the raw first judgment")
            summary = reference.previous.study.teacher.old.summarize_scores(rows)
            if any(endpoint.get(k) != v for k, v in summary.items()):
                raise ValueError("full endpoint/component/token/Wilson summary differs")
            scores[split][role] = rows
        joint[split] = joint_counts(scores[split]["secondary"], scores[split]["primary"])
    if grade["prior_first_judgments_preserved"] != len(prior):
        raise ValueError("complete inherited judge-pilot cache size differs")
    invalid = any(endpoint["invalid"] for r in grade["results"].values()
                  for endpoint in (r["secondary"], r["primary"]) if endpoint is not None)
    joint_gate = all(r is not None and r["cases"] == r["valid"] == 64 and r["passed"] >= 61 for r in joint.values())
    gate = grade["development_gate"] and joint_gate
    classification = ("dense_semantic_development_judgments_invalid" if invalid else
        "dense_semantic_development_passed_final_unproven" if gate else "dense_semantic_development_quality_failed")
    receipts = [run.WORKFLOW / "stage_receipts" / (stage + ".json") for stage in
        ("pilot", "generate", "packets", "judge-pilot", "grade")]
    if any(reference.q.read(p)["exit_code"] != 0 for p in receipts):
        raise ValueError("five actual inference/grading worker zero exits required")
    artifacts = [Path(run.__file__), Path(__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts,
        run.training.TRAIN / "model.pt", run.training.RESULTS / "campaign_results.json", run.OLD_CACHE]
    for root in (run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE):
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    campaign = {"schema": "nal.ncm-dense-semantic-development.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "selected_checkpoint_sha256": run.CHECKPOINT,
        "semantic_results": grade["results"], "same_case_joint": joint,
        "per_judge_development_gate": grade["development_gate"], "same_case_joint_gate": joint_gate,
        "semantic_development_gate": gate, "parent_TRAIN_fitting_classification": "dense_source_TRAIN_fit_failed",
        "source_disjoint_development_cases": 128, "all1669_prior_first_judgments_preserved": True,
        "all_judge_pilot_first_decisions_preserved": True, "canonical_first_judgments": count,
        "reused_development_cohorts": True, "model_training_executed": False,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    write_json(run.RESULTS / "case_failures.json", failures)
    lines = []
    for split, result in grade["results"].items():
        for role in ("secondary", "primary"):
            endpoint = result[role]
            if endpoint is not None:
                point = f"{endpoint['passed'] / 64:.2%}" if endpoint["valid"] == 64 else "unmeasured"
                lines.append(f"| {split} | {role} | {endpoint['passed']}/64 | {endpoint['valid']}/64 | {point} |")
        if joint[split] is not None:
            endpoint = joint[split]
            point = f"{endpoint['passed'] / 64:.2%}" if endpoint["valid"] == 64 else "unmeasured"
            lines.append(f"| {split} | same case joint | {endpoint['passed']}/64 | {endpoint['valid']}/64 | {point} |")
    run.REPORT.write_text(f"""# Semantic development accuracy of the dense NCM student

Date: 2026-10-23. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Reused development cohort | Judge | Complete summary passes | Valid | Semantic task accuracy |
| --- | --- | --- | --- | --- |
""" + "\n".join(lines) + """

The fixed final2048 student is evaluated on the unchanged64 short and64 long
source-disjoint development cases. Each summary must be faithful, salient,
readable, nonempty and at most128 stored cl100k tokens. Judges inspect source
claims and evidence, allowing paraphrases; exact target-text matching is absent.
Qualified Terra is secondary and Sonnet primary. Sonnet is admitted separately
for a cohort only after64 valid Terra decisions and at least61 passes.
The new joint gate additionally requires at least61 of the SAME64 cases to pass
both judges, with all decisions valid, in both cohorts. Judges are models;
these are not independent human ratings or a95% population lower-bound claim.

The student's32-case TRAIN token agreement was99.66%, with one subgroup missing
its literal99.5% fitting gate by one token. That failed study stays failed.
This separately specified exploratory semantic evaluation measures the actual
summary task and does not retroactively pass the parent numerical gate.

The original body/MLPs and decoder stay frozen. A fixed source-only forecaster
runs eight concept levels, then the dense word branch predicts latents and
decodes all192 positions, using one global EOS. Full public sources and frozen
feature bytes are retained. No oracle labels, output repair, clipping, lookup,
autoregressive fallback or best-of choice enters prediction. Constructor parity
against the registered TRAIN outputs, fresh reload, retained development pilot,
model/codec identity and memory checks passed. All1669 prior first judgments,
pilot decisions, raw responses, claim citations, budget counts and Wilson
intervals replayed, with no semantic rerolls or excluded cases.

This reused development evaluation does not establish the final95% goal. Both
sealed final cohorts remain unopened; passing supports a separately specified
final evaluation. No inference speed or compression improvement is claimed.
""")


def register():
    run.configure()
    campaign = run.reference.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Source disjoint semantic development accuracy of the dense NCM student",
        "description": "Fixed2048 dense student, unchanged64short+64long, first Terra/conditionalSonnet decisions, same-case joint counts. Sealed final closed.",
        "confirmation_status": campaign["classification"],
        "task_accuracy_measured": campaign["classification"] != "dense_semantic_development_judgments_invalid",
        "semantic_development_gate": campaign["semantic_development_gate"], "NCM_success_established": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
