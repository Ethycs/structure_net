"""Replay first teacher candidates, capacities and all semantic first decisions."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_capacity_revision as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def replay_teacher(case, row):
    return run.replay_teacher(case, row)


def finalize():
    run.configure()
    import tiktoken
    if run.RESULTS.exists() or run.REPORT.exists(): raise ValueError("fresh label aggregate required")
    roots = (run.DATA, run.DRAFT_PILOT, run.DRAFT, run.PACKETS, run.JUDGE_PILOT, run.GRADE)
    for root in roots: run.native.vendor.terminal(root)
    q = run.q
    prior, reviews = run.review_selection()
    if q.read(run.DATA / "prior_candidates.json") != prior or q.read(run.DATA / "revision_selection.json") != reviews:
        raise ValueError("exact12 revision/23 retained selection differs")
    original = q.read(run.native.CASES)["train"]
    tokenizer, storage = run.native.tokenizer(), tiktoken.get_encoding("cl100k_base")
    capacities = {r["id"]: run.capacity(r["target_text"], tokenizer, storage) for r in original}
    selected = sorted([r for r in original if not capacities[r["id"]]["fits"]],
        key=lambda r: q.digest("capacity-labels-89142:" + r["id"]))
    cases = [{k: r[k] for k in run.reference.IDENTITY} for r in selected]
    if len(cases) != 35 or q.read(run.DATA / "cases.json") != cases or q.read(run.DATA / "original_capacities.json") != capacities:
        raise ValueError("all35 capacity-only selected cases and unchanged original690 required")
    proof = run.reference.verify_public(cases)
    if q.read(run.DATA / "public_proof.json") != {"public_file_sha256": run.reference.PUBLIC_SHA256,
            "exact_sources": proof, "private_data_in_packets": False}:
        raise ValueError("public teacher source proof differs")
    outputs, pilot = q.read(run.DRAFT / "outputs.json"), q.read(run.DRAFT_PILOT / "outputs.json")
    if [r["id"] for r in outputs] != [c["id"] for c in cases] or outputs[:4] != pilot:
        raise ValueError("all35 ordered first candidates and retained pilot required")
    for c, row in zip(cases, outputs):
        text = replay_teacher(c, row)
        if row["capacity"] != run.capacity(text, tokenizer, storage): raise ValueError("teacher candidate capacity differs")
    packet_cases = [{**c, "texts": {"raw": row["text"], "bert_roundtrip": row["capacity"]["bert_roundtrip"]}}
        for c, row in zip(cases, outputs)]
    packets = {arm: [q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["texts"][arm]}])
        for c in packet_cases] for arm in run.ARMS}
    if q.read(run.PACKETS / "cases.json") != packet_cases or q.read(run.PACKETS / "exact_packets.json") != packets:
        raise ValueError("all35 exact raw/BERT candidate packets differ")
    if q.read(run.PACKETS / "public_proof.json") != {"public_file_sha256": run.reference.PUBLIC_SHA256,
            "exact_sources": proof, "exact_packets_sha256": file_hash(run.PACKETS / "exact_packets.json"), "private_data_in_packets": False}:
        raise ValueError("semantic public proof differs")
    old, pilot_cache, cache = run.base_cache(), q.read(run.JUDGE_PILOT / "judgment_cache.json"), q.read(run.GRADE / "judgment_cache.json")
    if any(pilot_cache.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot_cache.items()):
        raise ValueError("a prior canonical or pilot first changed")
    run.semantic.audit_cache(cache)
    allowed = {run.reference.first.packet_key(p, role) for payload in packets.values() for p in payload for role in run.ROLES}
    if set(cache) - set(old) - allowed: raise ValueError("extra semantic decision outside35 candidates appeared")
    grade, judge_pilot = q.read(run.GRADE / "result.json"), q.read(run.JUDGE_PILOT / "result.json")
    results = {}
    for arm in run.ARMS:
        results[arm] = {}
        for role in run.ROLES:
            for root, cs, payload, decisions in ((run.JUDGE_PILOT, packet_cases[:4], {arm: packets[arm][:4]}, pilot_cache),
                    (run.GRADE, packet_cases, packets, cache)):
                scores = replay_scores(cs, payload, decisions, arm, role)
                metrics = run.label_metric(run.oracle.metric(scores))
                if q.read(root / f"{role}_{arm}_case_scores.json") != scores:
                    raise ValueError("first-decision semantic case scores differ")
                actual = q.read(root / f"{role}_{arm}_result.json")
                if any(actual[k] != v for k, v in metrics.items()): raise ValueError("label semantic aggregate differs")
                if root == run.GRADE: results[arm][role] = actual
    if (grade["canonical_first_judgments"] != len(cache) or grade["prior_first_judgments_preserved"] != len(pilot_cache)
            or judge_pilot["canonical_first_judgments"] != len(pilot_cache) or judge_pilot["status"] != "systems_label_judges_valid"):
        raise ValueError("canonical counts or pilot validity differ")
    fit = all(r["capacity"]["fits"] for r in outputs)
    valid = all(v[role]["valid"] == 35 for v in results.values() for role in run.ROLES)
    passed = valid and all(v[role]["passed"] == 35 for v in results.values() for role in run.ROLES)
    admitted = fit and passed
    classification = ("capacity35_critique_label_judgments_invalid" if not valid else
        "capacity35_critique_revisions_qualified_for_TRAIN_only" if admitted else "capacity35_critique_revisions_quality_or_capacity_failed")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in
        ("prepare", "draft-pilot", "draft", "packets", "judge-pilot", "grade")]
    if any(q.read(p)["exit_code"] != 0 or q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("six actual successful producing children required")
    run.RESULTS.mkdir()
    exported = None
    if admitted:
        dataset = run.merge(original, {r["id"]: r["text"] for r in outputs})
        if any(not run.capacity(r["target_text"], tokenizer, storage)["fits"] for r in dataset):
            raise ValueError("all690 exported labels must fit both representations")
        if sum(a["target_text"] != b["target_text"] for a, b in zip(original, dataset)) != 35:
            raise ValueError("exactly35 actual revised texts and655 unchanged targets required")
        write_json(run.RESULTS / "cases.json", {"train": dataset})
        exported = {"path": str(run.RESULTS / "cases.json"), "sha256": file_hash(run.RESULTS / "cases.json"),
            "cases": 690, "revised": 35, "unchanged": 655, "old_word_or_concept_targets_carried": False}
    artifacts = {str(p): file_hash(p) for p in [Path(__file__), Path(run.__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts]}
    artifacts.update(q.read(run.WORKFLOW / "producer.json")["input_hashes"])
    for root in roots:
        artifacts.update(q.read(root / "producer.json")["input_hashes"])
        artifacts.update({str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    if exported: artifacts[exported["path"]] = exported["sha256"]
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "label_dataset_admitted": admitted,
        "final_candidate_cases": 35, "new_teacher_revisions": 12, "retained_first_candidates": 23, "capacity_fit_candidates": sum(r["capacity"]["fits"] for r in outputs),
        "semantic_results": results, "exported_dataset": exported, "canonical_prior_firsts": len(old),
        "canonical_first_judgments": len(cache), "all_prior_and_pilot_firsts_preserved": True,
        "teacher_model": q.JUDGES["primary"], "teacher_also_one_judge": True,
        "model_training_executed": False, "task_accuracy": None, "source_only_accuracy_measured": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    path = run.RESULTS / "campaign_results.json"; write_json(path, campaign)
    lines = [f"| {arm} | {role} | {v[role]['passed']}/35 | {v[role]['valid']}/35 |" for arm, v in results.items() for role in run.ROLES]
    run.REPORT.write_text(f"""# Native NCM capacity-compatible training label revisions

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`, SHA256 `{file_hash(path)}`.

The original 35-case capacity population is retained. The 23 first candidates
that passed capacity and both judges in both wordings were kept exactly. The
other 12 received one genuine critique-guided revision from the full source,
previous candidate, BERT rendering and recorded issues. The new prompt aims
at 30-40 words and spells out essential units/numbers. The first four pilot
candidates were reused exactly. Capacity/semantic misses never triggered
schema retries. Unchanged failed text shares its original negative decision.

| Candidate wording | Judge | Passes | Valid |
| --- | --- | ---: | ---: |
""" + "\n".join(lines) + f"""

Capacity-compatible candidates: {campaign['capacity_fit_candidates']}/35. Dataset
admitted: **{admitted}**. Admission requires all 35 candidates to fit 126 BERT
lexical positions and 128 storage tokens before and after BERT rendering, and
all 35 raw and rendered summaries to pass BOTH qualified judges. Sonnet also
generated the candidates; Terra provides the second model's assessment. Both
judges use full exact public sources and accept paraphrases.

All {len(old)} earlier first decisions and all pilot firsts remain unchanged;
the resulting canonical cache contains {len(cache)} decisions. Valid negatives
and exhausted invalids were not rerolled. If admitted, all 690 cases are retained,
655 previous targets stay unchanged, and stale word/concept target arrays are
removed before new native encoding. Previous labels and failures are preserved.

This is TRAIN-label preparation, not source-only model accuracy. The native
task-domain codec's separate held-out result remains 52/64; the previous
conditional student remains 0/128. No model trained here, no final source was
opened, and neither the full 95% task goal nor a speedup is established.
""")
    print(json.dumps({"classification": classification, "admitted": admitted, "capacity_fit": campaign["capacity_fit_candidates"],
        "results": results, "canonical_firsts": len(cache), "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Capacity35 critique-guided revisions with23 retained first passes",
        "description": "One critique-guided revision for12 failed candidates and23 retained first passes; original35 population maintained; all35 require capacity and both judges on raw/BERT text before690-case export. No student accuracy or speed claim.",
        "classification": campaign["classification"], "label_dataset_admitted": campaign["label_dataset_admitted"],
        "source_only_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args(); (finalize if args.stage == "finalize" else register)()
