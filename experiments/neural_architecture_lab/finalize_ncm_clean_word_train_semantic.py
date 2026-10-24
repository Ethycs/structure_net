"""Audit first judgments and raw bindings for the TRAIN-only word semantic study."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_clean_word_train_semantic as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def replay(observations, packets, cache, prior, pilot_cache, saved, encoding):
    if (tuple(observations) != run.CONDITIONS or set(saved["results"]) != set(run.CONDITIONS)
            or any(len(rows) != 32 for rows in observations.values())):
        raise ValueError("all four complete32-case conditions required")
    if any(cache.get(key) != entry for key, entry in {**prior, **pilot_cache}.items()):
        raise ValueError("prior or pilot first judgments changed")
    allowed = set(prior)
    results = {}
    for condition in run.CONDITIONS:
        rows, values = observations[condition], packets[condition]
        if values != run.packets_for(rows):
            raise ValueError("source-only anonymous packet binding differs")
        allowed.update(run.first.packet_key(packet, "secondary") for packet in values)
        secondary = run.scores_for(rows, values, "secondary", cache, encoding)
        if run.q.read(run.GRADE / (condition + "_secondary_scores.json")) != secondary:
            raise ValueError("raw secondary scores differ")
        secondary_metric = run.metric(secondary)
        summary = saved["results"][condition]
        if summary["secondary"] != secondary_metric:
            raise ValueError("secondary fixed denominators or semantic passes differ")
        primary = None
        joint = None
        if run.qualifies(secondary_metric):
            allowed.update(run.first.packet_key(packet, "primary") for packet in values)
            primary_scores = run.scores_for(rows, values, "primary", cache, encoding)
            if run.q.read(run.GRADE / (condition + "_primary_scores.json")) != primary_scores:
                raise ValueError("raw primary scores differ")
            primary = run.metric(primary_scores)
            joint = run.joint_count(secondary, primary_scores)
        elif (run.GRADE / (condition + "_primary_scores.json")).exists():
            raise ValueError("primary condition was not admitted by secondary31/32")
        if summary["primary"] != primary or summary["joint"] != joint:
            raise ValueError("conditional primary or same-case joint counts differ")
        results[condition] = {"secondary": secondary_metric, "primary": primary, "joint": joint}
    if set(cache) != allowed:
        raise ValueError("extra requests or missing complete first judgments")
    return results


def finalize():
    run.configure()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh semantic diagnostic archive and report required")
    selected, expected_observations = run.inputs_and_observations()
    observations, packets = run.prepared()
    if observations != expected_observations:
        raise ValueError("saved raw model outputs, histories or source identities differ")
    for destination in (run.PREP, run.PILOT, run.GRADE):
        run.clean.recovery.study.data.terminal(destination)
    receipts = [run.WORKFLOW / "stage_receipts" / (stage + ".json") for stage in ("prepare", "pilot", "grade")]
    if any(run.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("three actual successful CPU children required")
    prior = run.require_prior()
    checkpoints = {run.MODELS[0]: run.q.read(run.clean.diagnostic.DEST / "result.json")["model_checkpoint_sha256"],
        run.MODELS[1]: run.q.read(run.clean.FIT / "result.json")["checkpoint_sha256"]}
    if run.q.read(run.PREP / "result.json")["model_checkpoints"] != checkpoints:
        raise ValueError("exact two fixed model checkpoint identities required")
    if run.q.read(run.PREP / "judgment_cache.json") != prior:
        raise ValueError("entire registered1572-entry initial cache required")
    pilot_cache = run.q.read(run.PILOT / "judgment_cache.json")
    cache = run.q.read(run.GRADE / "judgment_cache.json")
    run.audit_cache(pilot_cache)
    run.audit_cache(cache)
    pilot = run.q.read(run.PILOT / "result.json")
    pilot_ids = run.q.read(run.PREP / "pilot_case_ids.json")
    if pilot_ids != [row["id"] for row in run.clean.training_probes(selected)]:
        raise ValueError("exact first and longest TRAIN pilot sources required")
    pilot_packets = [packet for values in packets.values() for packet in values if packet["case_id"] in set(pilot_ids)]
    keys = {run.first.packet_key(packet, "secondary") for packet in pilot_packets}
    if (pilot["status"] != "systems_TRAIN_semantic_judge_path_passed" or not pilot["systems_only"]
            or pilot["observations"] != 8 or pilot["valid"] != 8
            or set(pilot_cache) != set(prior) | keys
            or any(pilot_cache[key]["row"]["status"] != "valid" for key in keys)
            or any(pilot_cache.get(key) != entry for key, entry in prior.items())):
        raise ValueError("pilot exact first judgments or lifecycle validity differs")
    saved = run.q.read(run.GRADE / "result.json")
    if (saved["status"] != "TRAIN_semantic_diagnostic_completed"
            or saved["prior_and_pilot_first_judgments_preserved"] != len(pilot_cache)
            or saved["unique_first_judgments"] != len(cache)
            or saved["heldout_accuracy_measured"] or saved["task_accuracy"] is not None
            or saved["NCM_success_established"] or saved["final_test_opened"]):
        raise ValueError("TRAIN-only study scope or full canonical cache size differs")
    import tiktoken
    metrics = replay(observations, packets, cache, prior, pilot_cache, saved,
        tiktoken.get_encoding("cl100k_base"))
    classification = ("TRAIN_semantic_diagnostic_measured" if all(
        row[role] is None or row[role]["invalid"] == 0 for row in metrics.values() for role in ("secondary", "primary"))
        else "TRAIN_semantic_diagnostic_invalid_judgments_preserved")
    artifacts = [Path(__file__), Path(run.__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts,
        run.clean.RESULTS / "campaign_results.json", run.clean.RESULTS / (run.clean.HYPOTHESIS + "_registry_readback.json")]
    for destination in (run.PREP, run.PILOT, run.GRADE):
        artifacts += [path for path in destination.rglob("*")
                      if path.is_file() and "producer_source" not in path.parts]
    campaign = {"schema": "nal.ncm-clean-word-TRAIN-semantic.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "results": metrics,
        "model_checkpoints": checkpoints,
        "training_cases": 32, "summary_observations": 128,
        "prior_first_judgments_preserved": len(prior), "pilot_first_judgments_preserved": len(pilot_cache),
        "unique_first_judgments": len(cache), "training_semantic_quality_measured": True,
        "heldout_accuracy_measured": False, "task_accuracy": None, "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(path): file_hash(path) for path in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    lines = []
    for condition in run.CONDITIONS:
        summary = metrics[condition]
        secondary = summary["secondary"]
        primary = summary["primary"]
        other = f"{primary['passed']}/32 ({primary['valid']} valid)" if primary else "not admitted"
        groups = secondary["groups"]
        lines.append(f"| {condition} | {secondary['passed']}/32 ({secondary['valid']} valid) | "
            f"{groups['original_16']['passed']}/16 | {groups['additional_16']['passed']}/16 | {other} |")
    run.REPORT.write_text(f"""# Semantic diagnosis of saved TRAIN summaries

Date:2026-10-23. Hypothesis {run.HYPOTHESIS}.
Classification **{classification}**. Campaign {path}; SHA256 {file_hash(path)}.

| Saved model and history | Terra whole-summary passes | Original TRAIN16 | Additional TRAIN16 | Sonnet |
| --- | --- | --- | --- | --- |
""" + "\n".join(lines) + f"""

Every row uses the same32 TRAIN sources, not unseen evaluation data. Correct
histories supply oracle target-derived coarse concepts. Actual histories use
the existing252-case forecaster. This study measures no new source-disjoint
accuracy and does not reclassify the token-fitting gate or admit a queued GPU
study. A high TRAIN or oracle semantic pass rate cannot satisfy the95% goal.

The saved192-position raw outputs were decoded without repair or clipping and
paired with complete public sources. No model generation or training occurred.
Judges saw one anonymous candidate and source only, using the unchanged qualified
128-token semantic rubric with claim-level evidence citations. Paraphrases are
allowed. Primary Sonnet was admitted separately for each condition only after
32valid secondary Terra judgments and at least31passes. All invalid judgments
stay in their fixed denominators. Semantic pass fractions are reported only
when every case in that condition is valid. Judges are models, not human gold.

All{len(prior)} prior first judgments and all pilot first judgments were
preserved exactly. Identical packets reused the first decision, including
failures. Conditional primary admissions, raw requests/responses, complete
source/output bindings, token budgets, subgroup and same-case joint counts
replayed. The independent semantic development result remains0/64short+0/64long
for the preceding candidate. Both sealed final cohorts remain unopened; this
TRAIN diagnostic establishes neither NCM success nor speed/compression gains.
""")
    print(json.dumps({"status": "finalized", "classification": classification,
        "campaign_sha256": file_hash(path), "task_accuracy": None}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {
        run.HYPOTHESIS: {"name": "Semantic TRAIN diagnosis of correct and predicted NCM histories",
            "description": "Two fixed word checkpoints and two histories on32TRAIN sources; semantic first judgments preserved; no unseen accuracy or speed claim.",
            "confirmation_status": campaign["classification"], "training_semantic_quality_measured": True,
            "task_accuracy_measured": False, "heldout_accuracy_measured": False, "NCM_success_established": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
