"""Replay generated-history training and complete development summary judgments."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_generated_history_study as study
from experiments.structure_net import ncm_generated_history_eval as evaluation
from experiments.structure_net import ncm_generated_history_workflow as workflow
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = study.ROOT / "20261023_generated_history_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-generated-history-word.md")
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-generated-history-audit-plan.md")


def replay_training(rows, histories, trace, state):
    config = study.Config()
    if len(trace) != config.updates or state["step"] != config.updates or state["config"] != asdict(config):
        raise ValueError("only the full fixed final word checkpoint can be finalized")
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    counts = Counter()
    for step, entry in enumerate(trace, 1):
        row = rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        if (entry["step"] != step or entry["id"] != row["id"]
                or entry["source_frame"] != row["source_frame"] or entry["lr"] != study.rate(step)
                or entry["history_sha256"] != histories[row["id"]]["history_sha256"]):
            raise ValueError("training identity, complete frame, own history or rate replay differs")
        if any(not math.isfinite(entry[key]) or entry[key] < 0
                for key in ("own_word_CE", "gold_word_CE", "gradient_norm")):
            raise ValueError("finite own/gold loss and gradient evidence required")
        counts[row["id"]] += 1
    if (not torch.equal(sampler.get_state(), state["sampler_rng"])
            or not state["optimizer"]["state"]
            or any(int(value["step"]) != config.updates for value in state["optimizer"]["state"].values())):
        raise ValueError("final sampler RNG or optimizer steps differ")
    return {"updates": len(trace), "sample_counts": dict(sorted(counts.items())),
        "all_sources_frames_histories_rates_replayed": True, "final_sampler_and_optimizer_steps_exact": True}


def replay_grade(outputs, prior):
    reference = evaluation.reference
    cache = study.q.read(evaluation.GRADE / "judgment_cache.json")
    if any(cache.get(key) != value for key, value in prior.items()):
        raise ValueError("a prior first judgment, including a negative, changed")
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical packet/rubric/model identity differs")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    result = study.q.read(evaluation.GRADE / "result.json")
    failures = []
    for split in reference.COHORTS:
        rows = [row for row in outputs if row["split"] == split]
        for role in ("secondary", "primary"):
            endpoint = result["results"][split][role]
            if endpoint is None:
                if role != "primary" or reference.previous.study.qualifies(result["results"][split]["secondary"]):
                    raise ValueError("required qualified judge endpoint is missing")
                continue
            scores = study.q.read(evaluation.GRADE / f"{split}_{role}_case_scores.json")
            if len(scores) != len(rows) or len(rows) != 64:
                raise ValueError("complete64case denominator required for each endpoint")
            valid_count, passes = 0, 0
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
                    raise ValueError("raw first-judgment summary score replay differs")
                valid_count += valid
                passes += passed is True
                if passed is not True:
                    failures.append({"split": split, "role": role, "output": row, "score": score})
            if endpoint["cases"] != 64 or endpoint["valid"] != valid_count or endpoint["passed"] != passes:
                raise ValueError("complete-summary endpoint counts differ")
        values = result["results"][split]
        passed = values["primary"] is not None and reference.previous.study.qualifies(values["primary"])
        if values["passed"] != passed:
            raise ValueError("development joint gate changed")
    if result["development_gate"] != all(v["passed"] for v in result["results"].values()):
        raise ValueError("overall development gate differs")
    return result, failures, len(cache)


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    evaluation.configure()
    evaluation.reference.profile.install()
    parent = study.q.read(workflow.DEST / "result.json")
    if parent["status"] != "completed_generated_history_development_candidate" or any(r["exit_code"] != 0 for r in parent["stages"]):
        raise ValueError("completed durable candidate workflow required")
    study.data.verify(study.q.read(workflow.DEST / "producer.json"))
    roots = [study.CACHE_PILOT, study.CACHE, study.PILOT, study.TRAIN,
        evaluation.PILOT, evaluation.OUTPUTS, evaluation.PACKETS, evaluation.GRADE]
    for root in roots:
        study.data.terminal(root)
    rows, _ = study.require_data()
    histories = study.histories_for(rows)
    forecaster = study.require_forecaster()
    trained = study.q.read(study.TRAIN / "result.json")
    if (trained["status"] != "generated_history_word_trained_pending_task_evaluation"
            or not trained["reload_exact"] or not trained["source_only_composed_reload_exact"]
            or trained["peak_allocated_bytes"] > 8_000_000_000
            or file_hash(study.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("completed unchanged final word model required")
    state = torch.load(study.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if (state["producer_fingerprint"] != trained["producer_fingerprint"]
            or state["forecaster_checkpoint_sha256"] != forecaster["checkpoint_sha256"]):
        raise ValueError("selected word checkpoint producing identity differs")
    trace = [json.loads(line) for line in (study.TRAIN / "trace.jsonl").open()]
    replay = replay_training(rows, histories, trace, state)
    expected_snapshots = {f"step_{step:06d}.pt" for step in range(4096, study.Config().updates, 4096)}
    if {p.name for p in (study.TRAIN / "checkpoints").glob("*.pt")} != expected_snapshots:
        raise ValueError("all fixed recovery snapshots must remain present")
    initial = torch.load(study.bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    frozen = {name: value for name, value in state["model"].items()
        if name == "source_head.generator.l0_queries" or name.startswith("source_head.generator.lm_heads.")}
    if (not frozen or any(not torch.equal(value, initial[name]) for name, value in frozen.items())
            or study.base.tensor_digest(frozen) != trained["frozen_word_branch_parameter_sha256"]):
        raise ValueError("unused word-branch roots or concept heads changed")
    probes = study.q.read(study.TRAIN / "source_only_reload_probe.json")
    if [r["id"] for r in probes] != [r["id"] for r in study.probes(rows)]:
        raise ValueError("complete source-only training reload probes required")
    if any(probe["coarse_codes"] != [histories[probe["id"]]["coarse_codes"]] for probe in probes):
        raise ValueError("composed inference forecaster differs from its actual training histories")
    sources = evaluation.reference.sources()
    expected = [r for cohort in sources.values() for r in cohort]
    outputs = study.q.read(evaluation.OUTPUTS / "outputs.json")
    evaluation.reference.require_outputs(outputs, expected)
    if (file_hash(evaluation.OUTPUTS / "outputs.json") != study.q.read(evaluation.OUTPUTS / "result.json")["output_sha256"]
            or outputs[:2] != study.q.read(evaluation.PILOT / "outputs.json")):
        raise ValueError("fixed outputs or retained inference pilot differ")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    for row in outputs:
        text, end = evaluation.reference.decode_words(row["raw_word_ids"], tokenizer)
        if row["text"] != text or row["global_EOS_position"] != (end if end < 192 else None):
            raise ValueError("single global-EOS text decode differs")
    grade, failures, count = replay_grade(outputs, evaluation.prior_cache())
    invalid = any(endpoint["invalid"] for value in grade["results"].values()
        for endpoint in (value["secondary"], value["primary"]) if endpoint is not None)
    classification = ("generated_history_development_judgments_invalid" if invalid else
        "generated_history_development_passed_final_quality_unproven" if grade["development_gate"] else
        "generated_history_development_quality_failed")
    artifacts = [Path(__file__), PLAN, study.PLAN]
    for root in [*roots, workflow.DEST]:
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-generated-history-word.v1", "hypothesis_id": study.HYPOTHESIS,
        "classification": classification, "config": asdict(study.Config()), "training": trained,
        "training_cases": len(rows), "replay": replay, "development": grade,
        "canonical_judgments": count, "all_first_judgments_replayed": True,
        "final_test_opened": False, "confirmed": False, "NCM_success_established": False,
        "speed_tested": False, "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    DEST.mkdir()
    write_json(DEST / "campaign_results.json", aggregate)
    write_json(DEST / "case_failures.json", failures)
    lines = []
    for split, values in grade["results"].items():
        for role in ("secondary", "primary"):
            endpoint = values[role]
            if endpoint is not None:
                accuracy = f"{endpoint['passed']/64:.2%}" if endpoint["valid"] == 64 else "unmeasured"
                lines.append(f"| {split} | {role} | {endpoint['passed']}/64 | {endpoint['valid']}/64 | {accuracy} |")
    campaign = DEST / "campaign_results.json"
    REPORT.write_text(f'''# Complete summary quality after generated history word training

Date: 2026-10-23. Hypothesis `{study.HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Development cohort | Judge role | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
'''+"\n".join(lines)+f'''

Every summary requires supported material claims, useful salient content,
readability, nonempty text and at most128stored cl100k tokens. Sonnet is admitted
only after64valid Terra judgments and at least61passes in that cohort. All64
cases remain in each denominator. These reused development cohorts are source
disjoint from training; both final cohorts remain closed. Wilson intervals,
component failures, overflows and invalid judgments remain in the aggregate.

The original Llama body and MLPs stayed frozen. The registered16,384-step concept
forecaster generated its own eight coarse levels from source inputs only.
A separate word branch trained for exactly32,768updates on{len(rows)}complete
vetted TRAINsummaries, using actual generated histories plus a0.25clean-history
auxiliary. The forecaster, original codec and unused word-branch concept heads
stayed fixed. Word gradients did not alter the forecaster. This is an NCM
adaptation with two new-head parameter sets, not a paper training replication.

All sampled sources, full frames, code-history hashes, learning rates, optimizer
steps and sampler state replayed. Actual CUDA pilots, model/optimizer/RNG and
composed inference reloads passed before downstream stages. Only the fixed final
word checkpoint was selected. The two first inference outputs were retained
exactly. Every192-position raw word frame and predicted concept level remains
available; texts use one decode at the first global EOS. No oracle concepts,
gold history, output repair, lookup, autoregressive fallback or best-of selection
entered evaluation. All prior first judgments and new negative decisions remain
exact and were replayed from raw responses. Individual failures are retained.

This development result does not establish95%final task accuracy. All earlier
failed models remain failures. No end-to-end throughput, latency or compression
benefit has been measured. The overall goal remains unmet pending qualifying
quality on the sealed source-disjoint final evaluation.
''')
    print(json.dumps({"status": "finalized", "classification": classification,
        "campaign_sha256": file_hash(campaign)}), flush=True)


def register():
    aggregate = study.q.read(DEST / "campaign_results.json")
    register_records(DEST / "campaign_results.json", REPORT, DEST, {study.HYPOTHESIS: {
        "name": "Complete summaries from frozen NCM generated concepts",
        "description": "Frozen originalLlama body and registeredconceptforecaster; separatewordbranch trained32768updates onexpandedvettedwhole summaries atactualgeneratedhistories plus0.25goldaux. Source-only64short+64longdevelopment complete-summary grading; rawtraining and allfirstjudgments replayed. Finaltaskquality andspeedunproven.",
        "confirmation_status": aggregate["classification"], "task_accuracy_measured": aggregate["classification"] != "generated_history_development_judgments_invalid",
        "NCM_success_established": False, "development_gate": aggregate["development"]["development_gate"]}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
