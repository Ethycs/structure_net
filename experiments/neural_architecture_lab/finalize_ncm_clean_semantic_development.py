"""Replay fresh source exclusions and complete-summary first judgments."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_clean_semantic_development as run
from experiments.neural_architecture_lab import finalize_ncm_dense_semantic_development as prior
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

joint_counts = prior.joint_counts
replay_outputs = prior.replay_outputs


def finalize():
    run.configure()
    cases, _ = run.inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh clean semantic aggregate/report required")
    reference, q = run.reference, run.q
    for root in (run.SOURCES, run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE):
        reference.data.terminal(root)
    if cases != run.metadata_inputs():
        raise ValueError("full fresh source metadata or actual TRAIN exclusion replay differs")
    expected = [r for values in cases.values() for r in values]
    proof = reference.verify_public(expected)
    if q.read(run.SOURCES / "public_proof.json") != {"exact_sources": proof, "public_file_sha256": reference.PUBLIC_SHA256}:
        raise ValueError("exact complete public source proof differs")
    if q.read(run.SOURCES / "split_audit.json") != {"TRAIN_cases": 690, "development_cases": 128,
            "source_components": 128, "TRAIN_development_conflicts": 0, "within_development_conflicts": 0,
            "informative_full_turn_minimum_cl100k_tokens": 192, "full_text_and_family_guards": True,
            "sealed_final_content_opened": False}:
        raise ValueError("complete actual source-family/text exclusion evidence differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(reference.data.GPT2, local_files_only=True)
    values = q.read(run.OUTPUTS / "outputs.json")
    replay_outputs(values, expected, tokenizer)
    pilot = q.read(run.PILOT / "outputs.json")
    replay_outputs(pilot, run.inference_probes(cases), tokenizer)
    by_id = {r["id"]: r for r in values}
    if any(by_id[r["id"]] != r for r in pilot):
        raise ValueError("retained inference pilot predictions differ")
    train_rows, _, expected_train = run.training_parity_inputs()
    train_outputs = q.read(run.PILOT / "train_parity_outputs.json")
    replay_outputs(train_outputs, train_rows, tokenizer)
    run.parity_check(train_outputs, expected_train)
    for root, is_pilot in ((run.PILOT, True), (run.OUTPUTS, False)):
        result = q.read(root / "result.json")
        if (result["cases"] != (2 if is_pilot else 128) or result["selected_checkpoint_sha256"] != run.CHECKPOINT
                or result["output_sha256"] != file_hash(root / "outputs.json")
                or not result["all_parameters_and_codec_unchanged"] or not result["fresh_reload_or_TRAIN_parity"]
                or result["peak_allocated_bytes"] > 8_000_000_000 or result["final_test_opened"]
                or (not is_pilot and not result["retained_pilot_exact"])):
            raise ValueError("fixed selected candidate, exact reload or memory evidence differs")
    packets = q.read(run.PACKETS / "exact_packets.json")
    if packets != {split: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}])
            for r in values if r["split"] == split] for split in reference.COHORTS}:
        raise ValueError("anonymous single-candidate source-only packets differ")
    for name, digest in q.read(run.PACKETS / "result.json")["hashes"].items():
        if file_hash(run.PACKETS / name) != digest:
            raise ValueError("exact public payload proof bytes changed")
    baseline, inherited = run.base_cache(), run.prior_cache()
    if any(inherited.get(k) != v for k, v in baseline.items()):
        raise ValueError("a canonical prior first judgment changed")
    judge_pilot = q.read(run.JUDGE_PILOT / "result.json")
    if (judge_pilot["status"] != "systems_semantic_judge_path_passed" or judge_pilot["cases"] != 4
            or judge_pilot["valid"] != 4 or judge_pilot["quality_gate"]
            or not judge_pilot["all_prior_first_judgments_preserved"]):
        raise ValueError("schema-only pilot must preserve all four first decisions")
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    pilot_passes = 0
    for row in run.judge_probes(values):
        packet = q.packet(row["id"], row["source"], [{"candidate_id": "summary", "text": row["text"]}])
        key = reference.first.packet_key(packet, "secondary")
        if key not in inherited or inherited[key]["row"]["status"] != "valid":
            raise ValueError("every pilot first judgment must be valid and retained")
        pilot_passes += q.passes(inherited[key]["row"]["evaluations"][0], row["text"],
            len(encoding.encode(row["text"], disallowed_special=()))) is True
    if judge_pilot["passed"] != pilot_passes or judge_pilot["unique_first_judgments"] != len(inherited):
        raise ValueError("raw pilot pass/cache counts differ")
    grade, failures, count = run.legacy_audit.replay_grade(values, inherited)
    cache = q.read(run.GRADE / "judgment_cache.json")
    joint, scores = {}, {}
    for split in reference.COHORTS:
        scores[split] = {}
        for role in ("secondary", "primary"):
            endpoint = grade["results"][split][role]
            if endpoint is None:
                scores[split][role] = None
                continue
            rows = q.read(run.GRADE / f"{split}_{role}_case_scores.json")
            if any(row["evaluations"] != cache[row["packet_key"]]["row"].get("evaluations") for row in rows):
                raise ValueError("semantic components differ from raw first judgments")
            summary = reference.previous.study.teacher.old.summarize_scores(rows)
            if any(endpoint.get(k) != v for k, v in summary.items()):
                raise ValueError("complete endpoint/component/token/Wilson summary differs")
            scores[split][role] = rows
        joint[split] = joint_counts(scores[split]["secondary"], scores[split]["primary"])
    if grade["prior_first_judgments_preserved"] != len(inherited):
        raise ValueError("complete inherited pilot cache required")
    invalid = any(e["invalid"] for r in grade["results"].values() for e in (r["secondary"], r["primary"]) if e is not None)
    joint_gate = all(r is not None and r["cases"] == r["valid"] == 64 and r["passed"] >= 61 for r in joint.values())
    gate = grade["development_gate"] and joint_gate
    classification = ("clean_full690_semantic_judgments_invalid" if invalid else
        "clean_full690_semantic_development_passed_final_unproven" if gate else "clean_full690_semantic_development_quality_failed")
    stages = ("sources", "pilot", "generate", "packets", "judge-pilot", "grade")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in stages]
    if any(q.read(p)["exit_code"] or q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("six actual successful source/inference/grading children required")
    admission = q.read(run.WORKFLOW / "admission.json")
    workflow = q.read(run.WORKFLOW / "producer.json")
    if (not admission["same_word_kernel_terminal"]
            or admission["upstream_kernel_identity"] != workflow["config"]["upstream_kernel_identity"]
            or admission["selected_checkpoint_sha256"] != run.CHECKPOINT):
        raise ValueError("fixed candidate and actual upstream termination observation required")
    reference.data.verify(workflow)
    artifacts = [Path(run.__file__), Path(__file__), run.PLAN, run.WORKFLOW / "producer.json",
        run.WORKFLOW / "admission.json", *receipts, run.word.TRAIN / "model.pt", run.word.RESULTS / "campaign_results.json"]
    hashes = {str(p): file_hash(p) for p in artifacts}
    hashes.update(workflow["input_hashes"])
    for root in (run.SOURCES, run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE):
        hashes.update(q.read(root / "producer.json")["input_hashes"])
        hashes.update({str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification,
        "selected_checkpoint_sha256": run.CHECKPOINT, "semantic_results": grade["results"], "same_case_joint": joint,
        "per_judge_development_gate": grade["development_gate"], "same_case_joint_gate": joint_gate,
        "semantic_development_gate": gate, "source_disjoint_development_cases": 128,
        "canonical_prior_firsts": len(baseline), "canonical_first_judgments": count,
        "all_prior_first_judgments_preserved": True, "all_pilot_first_decisions_preserved": True,
        "fresh_development_cohorts": True, "model_training_executed": False,
        "parent_TRAIN_fit_gate": admission["word_TRAIN_fit_gate"], "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False, "artifacts": hashes}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    write_json(run.RESULTS / "case_failures.json", failures)
    lines = []
    for split, results in grade["results"].items():
        for role, endpoint in [(role, results[role]) for role in ("secondary", "primary")] + [("same case joint", joint[split])]:
            if endpoint is not None:
                point = f"{endpoint['passed'] / 64:.2%}" if endpoint["valid"] == 64 else "unmeasured"
                lines.append(f"| {split} | {role} | {endpoint['passed']}/64 | {endpoint['valid']}/64 | {point} |")
    run.REPORT.write_text(f"""# Full summary semantic accuracy on fresh development sources

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Fresh development cohort | Judge | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
""" + "\n".join(lines) + f"""

The fixed final32768-update word student trained on all690 approved summaries.
All64 short and64 long sources come from fresh reserved components, with actual
TRAIN/family/text/turn exclusion replayed. No cases or first judgments were
removed. Judges permit paraphrases and assess faithfulness, salience, readability
and the128-token storage cap against the source. Exact reference-summary matching
is absent. Terra and conditionally admitted Sonnet retain their50/50 controls.

At least61 of the same64 cases must pass both judges in each cohort, with all
decisions valid. Reported rates and Wilson intervals describe model judgments;
they are not human ratings or a95% population confidence lower bound. The TRAIN
token-fit diagnostic does not determine semantic admission or task accuracy.

Original frozen Llama features, fixed source-only concept forecasts and one dense
word pass produce each complete192-position frame with global EOS. Raw TRAIN
constructor parity, fresh reload, retained inference pilots, unchanged models
and codec, memory bounds and exact anonymous public packets replayed. All1797
baseline first judgments and all new pilot decisions remain unchanged, with
{count} canonical firsts after this study. No negative was rerolled.

This fresh development result does not establish final95% accuracy. Sealed final
sources remain closed until a separately specified eligible final evaluation and
original frozen-body integration check. No inference speed gain is established.
""")
    print(json.dumps({"classification": classification, "semantic_results": grade["results"],
        "same_case_joint": joint, "canonical_first_judgments": count, "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full690 NCM student semantic accuracy on fresh source components",
        "description": "Fixed final word student on all128 fresh source-disjoint DEV cases, qualified first Terra/conditionalSonnet judgments and same-case joint gates; final320 sealed.",
        "classification": campaign["classification"], "semantic_development_gate": campaign["semantic_development_gate"],
        "task_accuracy_measured": campaign["classification"] != "clean_full690_semantic_judgments_invalid"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
