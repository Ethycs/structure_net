"""Close the completed noisy-pair campaign with actual summary accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT

HYPOTHESIS = "ncm-paper-noisy-pairs-v1"
DESTINATION = ROOT / "20261022_noisy_quality_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-paper-regime-task-accuracy.md")


def finalize():
    if DESTINATION.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    training_path = ROOT / "20261022_noisy_campaign_v1/campaign_results.json"
    training = q.read(training_path)
    for path, expected in training["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed training evidence changed")
    scores = {}; artifacts = [training_path, Path(__file__)]
    for arm in ("native", "hybrid"):
        scores[arm] = {}
        for role in ("primary", "secondary"):
            root = ROOT / ("20261022_noisy_"+arm+"_grade_"+role+"_v1")
            result = q.read(root / "result.json")
            if result["status"] != "completed" or result["invalid"] or result["cases"] != 64 or result["systems_only"]:
                raise ValueError("all4complete64-case task evaluations required")
            producer = q.read(root / "producer.json")
            for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
                if file_hash(path) != expected:
                    raise ValueError("grading producer/input changed: "+path)
            scores[arm][role] = result
            artifacts.extend(path for path in root.rglob("*.json") if "producer_source" not in path.parts)
    passed = {arm: all(s["passed"] >= 61 for s in roles.values()) for arm, roles in scores.items()}
    classification = "development_gate_passed" if any(passed.values()) else "development_gate_failed"
    DESTINATION.mkdir(parents=True)
    aggregate = {"schema": "nal.ncm-noisy-quality.v1", "hypothesis_id": HYPOTHESIS,
        "classification": classification, "training": training, "whole_summary_accuracy": scores,
        "development_admission": passed, "unqualified_training_labels": True,
        "final_test_opened": False, "speed_tested": False, "human_rated": False,
        "confirmed": False, "underpowered": True,
        "artifacts": {str(path): file_hash(path) for path in artifacts}}
    write_json(DESTINATION / "campaign_results.json", aggregate)
    table = "\n".join(f"|{arm}|{q.JUDGES[role]}|{result['passed']}/64|{100*result['task_accuracy']:.2f}%|{result['invalid']}|"
        for arm, roles in scores.items() for role, result in roles.items())
    REPORT.write_text(f'''# Actual summary accuracy after the NCM paper-style training run

Date:2026-10-22. Hypothesis `{HYPOTHESIS}`; exploratory, one model seed.
Classification **`{classification}`**. These are complete-summary judgments,
not token accuracy, code prediction, ROUGE or training loss.

|Model|Independent judge|Passes|Accuracy|Invalid|
|---|---|---|---|---|
{table}

All64fixed source-disjoint development passages were evaluated under each
judge. Passing requires source-grounded faithfulness, salience, readability,
nonempty text and <=128cl100ktokens. The61/64admission threshold was fixed before
grading. Each zero result has Wilson95%interval0–5.66%, excluding judge error.
Neither arm qualifies for the sealed256-case final test. No speedup is measured.
The user's95% goal remains unmet by these models.

The native arm used NCM source embeddings; the hybrid used frozen full-Llama
hidden states projected through a new bridge. Both shared the same512-update
unpaired checkpoint and received2,048conditional updates, batch8xaccumulation2.
Hybrid additionally received512bridge-only updates. Both used the exact newly
trained codec and identical conditional minibatch/augmentation schedules. All
1,999labels were used with disclosed label noise: the independent audit passed
30/32. The diagnostic was explicitly declared to test noisy-pair learning and
did not retroactively rescue the failed quality-gated earlier studies.

The full codec encoder, codebooks and projections were frozen during generator
training; original Llama layers, including MLPs, were never trained. Targets
were reencoded from the accepted codec, and tied-head codebook buffers matched
that codec. Loss fell from4.24to2.03(native) and4.21to2.23(hybrid), but the output
still invented names and scrambled wording. Training loss did not predict
useful task quality. The released native baseline on these same64sources had
also scored0/64under the qualified secondary judge.

Both original judge endpoints initially failed transport/infrastructure checks.
After recovery, each passed a fresh4-concurrent-request systems pilot before
full grading. Systems decisions were not pooled into task evaluation. The
original48-control qualification, source-grounded rubric and conservative parser
were retained. Each request contained one candidate; all raw payloads/replies
and valid negative judgments remain preserved. No semantic failure was retried
to obtain a pass. Controls are source-reviewed by Codex, not human gold.

Separate follow-ups have demonstrated32-case training competence after a
coarse-code curriculum and improved full-cutoff decoder reconstruction. Those
results are in the [competence report](2026-10-22_ncm-training-competence.md),
and cannot reclassify this failed full-data campaign. A separate full-data
curriculum run is underway and must earn its own held-out result.

This is an adaptation of the paper's tokenizer-first, frozen-codec,
unpaired-then-conditional staging, at a much smaller training budget. It does
not establish that the published NCM architecture cannot work. The prototype
covers only512GPT2source tokens and48target tokens, rather than the full
application input/batch range. [NCM paper](https://arxiv.org/html/2610.14162v1).

Aggregate:`{DESTINATION / 'campaign_results.json'}`;
SHA256`{file_hash(DESTINATION / 'campaign_results.json')}`.
Reproduce with `ncm_noisy_pairs_campaign` and `ncm_recovered_grade_campaign`
under `experiments.structure_net`. Checkpoints, optimizer/RNG states, source
snapshots and64-case raw scores remain in the declared roots. Registry keeps
confirmed=false, underpowered=true and final_test_opened=false.
''')
    print(json.dumps({"classification": classification, "report": str(REPORT)}), flush=True)


def register():
    campaign = DESTINATION / "campaign_results.json"
    result = q.read(campaign)
    for path, expected in result["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed task evidence changed")
    registry = Path("data/hypothesis_registry") / (HYPOTHESIS+".json")
    receipt_path = DESTINATION / "registry_readback.json"
    if registry.exists() or receipt_path.exists():
        raise ValueError("fresh registry record required")
    record = {"id": HYPOTHESIS, "name": "Actual NCM task accuracy after staged noisy-pair training",
        "description": "Native and frozen-Llama NCM generators received staged unpaired/conditional training on1999public summary pairs; both failed whole-summary development quality under both independent judges.",
        "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
        "confirmation_status": result["classification"], "final_test_opened": False,
        "speed_tested": False, "human_rated": False, "unqualified_training_labels": True,
        "campaign": str(campaign), "campaign_sha256": file_hash(campaign),
        "report": str(REPORT), "report_sha256": file_hash(REPORT)}
    for arm, roles in result["whole_summary_accuracy"].items():
        for role, score in roles.items():
            record[arm+"_"+role+"_passes"] = score["passed"]
            record[arm+"_"+role+"_cases"] = score["cases"]
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    logger.log_hypothesis(record)
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if any(exact["metadatas"][0].get(k) != v for k, v in record.items()) or HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record, "exact": exact["metadatas"][0],
               "semantic_ids": semantic["ids"][0]}
    for path in (registry, receipt_path):
        write_json(path, receipt)
        if q.read(path) != receipt:
            raise ValueError("JSON readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
