"""Audit sealed-source consensus accuracy with truthful final-evaluation metadata."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_sealed_final_quality as final
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = final.ROOT / "20261023_sealed_final_results_v1"
WORKFLOW = final.ROOT / "20261023_sealed_final_workflow_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-sealed-final-quality.md")


def replay_judgments(outputs):
    reference = final.candidate.reference
    reference.profile.install()
    prior = final.q.read(final.candidate.GRADE / "judgment_cache.json")
    cache = final.q.read(final.GRADE / "judgment_cache.json")
    if any(cache.get(key) != value for key, value in prior.items()):
        raise ValueError("a prior first judgment changed")
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical packet/profile/model identity differs")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    endpoint = final.q.read(final.GRADE / "result.json")
    failures = []
    for split, count in final.COUNTS.items():
        rows = [row for row in outputs if row["split"] == split]
        saved = endpoint["results"][split]
        score_sets = {}
        for role in ("secondary", "primary"):
            stats = saved[role]
            if stats is None:
                if role != "primary" or final.qualifies(saved["secondary"], count):
                    raise ValueError("required qualified judge endpoint missing")
                score_sets[role] = None
                continue
            scores = final.q.read(final.GRADE / f"{split}_{role}_case_scores.json")
            if len(scores) != count or len(rows) != count:
                raise ValueError("full fixed final denominator required")
            for row, score in zip(rows, scores):
                packet = reference.q.packet(row["id"], row["source"], [{"candidate_id": "summary", "text": row["text"]}])
                key = reference.first.packet_key(packet, role)
                entry = cache[key]
                valid = entry["row"]["status"] == "valid"
                tokens = len(encoding.encode(row["text"], disallowed_special=()))
                passed = reference.q.passes(entry["row"]["evaluations"][0], row["text"], tokens) if valid else None
                if (score["id"] != row["id"] or score["source_id"] != row["source_id"]
                        or score["text"] != row["text"] or score["tokens"] != tokens
                        or score["valid"] != valid or score["passed"] != passed
                        or score["packet_key"] != key or score["first_origin"] != entry["origin"]):
                    raise ValueError("raw first final-summary decision differs")
                if passed is not True:
                    failures.append({"cohort": split, "role": role, "output": row, "score": score})
            summary = reference.previous.study.teacher.old.summarize_scores(scores)
            if any(stats[key] != value for key, value in summary.items()):
                raise ValueError("whole-summary endpoint or confidence interval differs")
            score_sets[role] = scores
        joint = final.consensus(score_sets["secondary"], score_sets["primary"])
        if joint != saved["consensus"]:
            raise ValueError("two-judge consensus count or gate differs")
    met = all(value["consensus"]["qualified"] for value in endpoint["results"].values())
    if endpoint["quality_gate"] != met:
        raise ValueError("final joint quality target gate differs")
    return endpoint, failures, len(prior), len(cache)


def finalize():
    final.require_admission()
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only final aggregate required")
    workflow = final.q.read(WORKFLOW / "evaluation_complete.json")
    if workflow["status"] != "completed_sealed_final_evaluation" or any(r["exit_code"] != 0 for r in workflow["stages"]):
        raise ValueError("completed durable final workflow and actual child exits required")
    final.base.verify(final.q.read(WORKFLOW / "producer.json"))
    roots = [final.DATA, final.FEATURE_PILOT, final.FEATURES, final.PILOT, final.OUTPUTS, final.PACKETS, final.GRADE]
    for root in roots:
        final.base.terminal(root)
    cases = final.sources()
    expected = [r for split in final.COUNTS for r in cases[split]]
    train, _ = final.study.require_data()
    dev = [r for rows in final.candidate.reference.sources().values() for r in rows]
    final.base.prior.check_disjoint(expected, [*train, *dev])
    features = final.q.read(final.FEATURES / "manifest.json")
    feature_result = final.q.read(final.FEATURES / "result.json")
    if (file_hash(final.FEATURES / "manifest.json") != feature_result["manifest_sha256"]
            or feature_result["parent_body_hash_unchanged"] != final.base.EXPECTED_BODY
            or not feature_result["all_source_tokens_retained"]
            or feature_result["peak_allocated_bytes"] > 8_600_000_000
            or set(features) != {r["id"] for r in expected}):
        raise ValueError("unchanged complete original-body source feature proof required")
    feature_paths = []
    for row in expected:
        item = features[row["id"]]
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("final source feature bytes changed")
        cached = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (cached["id"] != row["id"] or cached["source_sha256"] != row["source_sha256"]
                or cached["alignment_sha256"] != row["alignment_sha256"]
                or cached["native_prompt_sha256"] != row["native_prompt_sha256"]
                or cached["features"].shape != (len(row["source_ids"]), 3072)
                or cached["features"].dtype != torch.float32 or not torch.isfinite(cached["features"]).all()):
            raise ValueError("complete finite aligned final source features required")
        feature_paths.append(path)
    outputs = final.q.read(final.OUTPUTS / "outputs.json")
    final.candidate.reference.require_outputs(outputs, expected)
    generated = final.q.read(final.OUTPUTS / "result.json")
    if (not generated["all_parameters_and_codec_unchanged"]
            or file_hash(final.OUTPUTS / "outputs.json") != generated["outputs_sha256"]):
        raise ValueError("single fixed unchanged-model final output set required")
    pilot = final.q.read(final.PILOT / "outputs.json")
    indexed = {r["id"]: r for r in outputs}
    if [r["id"] for r in pilot] != final.q.read(final.DATA / "selection.json")["inference_pilot_ids"]:
        raise ValueError("fixed first/maximum final inference pilot identities changed")
    if any(indexed.get(r["id"]) != r for r in pilot):
        raise ValueError("final inference pilot outputs were not retained exactly")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(final.base.GPT2, local_files_only=True)
    for row in outputs:
        text, end = final.candidate.reference.decode_words(row["raw_word_ids"], tokenizer)
        if row["text"] != text or row["global_EOS_position"] != (end if end < 192 else None):
            raise ValueError("single global-EOS final word decode differs")
        if len(row["coarse_codes"]) != 8 or any(len(v) != n or any(type(c) is not int or not 0 <= c < 4096 for c in v)
                for v, n in zip(row["coarse_codes"], (16, 24, 40, 56, 80, 104, 128, 160))):
            raise ValueError("full predicted coarse hierarchy required")
    grade, failures, prior_count, cache_count = replay_judgments(outputs)
    invalid = any(stats["invalid"] for value in grade["results"].values()
        for stats in (value["secondary"], value["primary"]) if stats is not None)
    met = grade["quality_gate"] and not invalid
    classification = ("sealed_final_quality_judgments_invalid" if invalid else
        "sealed_final_consensus_quality_target_met" if met else "sealed_final_consensus_quality_target_failed")
    artifacts = [Path(__file__), final.PLAN, *feature_paths]
    for root in roots:
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    artifacts += [WORKFLOW / "producer.json", WORKFLOW / "evaluation_complete.json"]
    artifacts += list((WORKFLOW / "stage_receipts").glob("*.json"))
    aggregate = {"schema": "nal.ncm-sealed-final-quality.v1", "hypothesis_id": final.HYPOTHESIS,
        "classification": classification, "goal_target_met": met, "results": grade["results"],
        "cohort_counts": final.COUNTS, "source_disjoint": True, "all_sources_complete": True,
        "all_first_judgments_replayed": True, "prior_first_judgments_preserved": prior_count,
        "canonical_judgments": cache_count, "model_weights_unchanged_since_development": True,
        "original_body_and_MLPs_frozen": True, "final_test_opened": True,
        "human_rated": False, "confirmed": False, "underpowered": True,
        "paper_replication_established": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    DEST.mkdir()
    write_json(DEST / "campaign_results.json", aggregate)
    write_json(DEST / "case_failures.json", failures)
    table = []
    for split, value in grade["results"].items():
        for role in ("secondary", "primary"):
            stats = value[role]
            if stats is not None:
                accuracy = f"{stats['passed']/stats['cases']:.2%}" if stats['valid'] == stats['cases'] else "unmeasured"
                table.append(f"| {split} | {role} | {stats['passed']}/{stats['cases']} | {stats['valid']}/{stats['cases']} | {accuracy} |")
        joint = value['consensus']
        accuracy = f"{joint['task_accuracy']:.2%}" if joint['task_accuracy'] is not None else "unmeasured"
        passes = f"{joint['joint_passed']}/{joint['cases']}" if joint['primary_admitted'] else "not admitted"
        table.append(f"| {split} | both judges | {passes} | {joint['valid_both']}/{joint['cases']} | {accuracy} |")
    campaign = DEST / "campaign_results.json"
    REPORT.write_text(f'''# Sealed complete summary accuracy for the frozen NCM candidate

Date: 2026-10-23. Hypothesis `{final.HYPOTHESIS}`.
Classification **{classification}**. The fixed95%consensus target
{'passed' if met else 'did not pass'}. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Final cohort | Judge role | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
'''+"\n".join(table)+'''

Final success requires at least244/256short and61/64long summaries passing both
qualified judges on the same cases, with every judgment valid. Each pass needs
supported material claims, salience, readability, nonempty text and at most128
stored cl100k tokens. Separate95%judge rates cannot pass the consensus rule.
All cases remain in the denominator. Wilson intervals, component failures,
overflows, empty outputs and invalids remain in the aggregate. These are held-out
sample accuracies under qualified model judges, not human ratings or a95%
population lower confidence bound.

Both sealed source files were accessed only after the unchanged model passed
both registered development gates. All original256short and64long identities,
source hashes and public spans were retained; the long cohort keeps16cases per
original source-length band. Final sources are disjoint from expanded training
and both development cohorts by case, session, connected group and source hash.
All source tokens and aligned frozen-body features were retained. Original
Llama weights and MLPs stayed frozen. No model selection or update occurred
after final admission.

Inference used the fixed concept forecaster followed by the trained independent
word branch. All192raw word IDs and eight predicted concept levels are retained.
Texts use one decode at the first global EOS. First/maximum-frame inference
pilot outputs were retained exactly. No oracle/gold history, lookup, output
repair, cap truncation, autoregressive fallback, reroll or best-of selection
entered evaluation. Every new and prior first judgment, including negatives,
replayed from raw responses. All individual failures remain available.

This is an NCM adaptation evaluated for the summarization task; it does not
replicate the paper's full training experiment. All earlier failures remain
failures. No throughput, latency, memory-compression or speedup claim is made.
''')
    print(json.dumps({"status": "finalized", "classification": classification,
        "goal_target_met": met, "campaign_sha256": file_hash(campaign)}), flush=True)


def register():
    campaign = DEST / "campaign_results.json"
    aggregate = final.q.read(campaign)
    for path, expected in aggregate["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed final evidence changed")
    receipt_path = DEST / (final.HYPOTHESIS+"_registry_readback.json")
    json_path = Path("data/hypothesis_registry") / (final.HYPOTHESIS+".json")
    if receipt_path.exists() or json_path.exists():
        raise ValueError("fresh final hypothesis registration required")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("final hypothesis registry unavailable")
    record = {"id": final.HYPOTHESIS, "name": "Sealed complete-summary consensus quality for frozen-body NCM",
        "description": "Fixedfrozenconceptforecasterandindependentwordbranch evaluatedonce on256short+64longsealedpublicsources afterdualjudgedevelopmentqualification. Bothjudgesmustpasssame244/256and61/64cases. Rawoutputs,source/featureidentity,allfirstjudgmentsandno-updateinvariant audited. No speedclaim.",
        "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
        "human_rated": False, "final_test_opened": True, "speed_tested": False,
        "goal_target_met": aggregate["goal_target_met"], "confirmation_status": aggregate["classification"],
        "campaign": str(campaign), "campaign_sha256": file_hash(campaign),
        "report": str(REPORT), "report_sha256": file_hash(REPORT)}
    logger.log_hypothesis(record)
    exact = collection.get(ids=[final.HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[final.HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[final.HYPOTHESIS], include=["metadatas"])
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if any(exact["metadatas"][0].get(k) != v for k, v in record.items()) or final.HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("truthful final-evaluation registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
        "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for path in (receipt_path, json_path):
        write_json(path, receipt)
        if final.q.read(path) != receipt:
            raise ValueError("final JSON registry readback differs")
    print(json.dumps({"status": "registered_and_read_back", "goal_target_met": aggregate["goal_target_met"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
