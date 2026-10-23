"""Replay the completed failed native NCM task endpoint without new requests."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_native_query_first_grade as grade
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, verify
from experiments.neural_architecture_lab import finalize_ncm_budget128_baselines as baseline
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_native_query_neighborhood_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-native-query-neighborhood-failure.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh result root required")
    roots = (run.TRAIN, run.OUTPUTS, grade.GRADE)
    for root in roots:
        if run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("training, generation and grading must be observed terminal")
        verify(run.q.read(root / "producer.json"))
    training = run.q.read(run.TRAIN / "result.json")
    generated = run.q.read(run.OUTPUTS / "result.json")
    grading = run.q.read(grade.GRADE / "result.json")
    if (training["updates"] != 2048 or not training["original_weights_frozen"]
            or not training["books_unchanged"] or not training["checkpoint_generator_reload_exact"]
            or file_hash(run.TRAIN / "model.pt") != training["checkpoint_sha256"]):
        raise ValueError("complete original fixed training run required")
    if file_hash(run.OUTPUTS / "outputs.json") != generated["output_sha256"]:
        raise ValueError("frozen development outputs differ")
    outputs = run.q.read(run.OUTPUTS / "outputs.json")
    sources = run.q.read(PREP / "development_sources.json")
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if (len(outputs) != 64 or len(sources) != 64
            or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(outputs, sources))
            or any(r["source_truncated"] or r["AR_fallback"] or r["fixed_forward_passes"] != 9 for r in outputs)):
        raise ValueError("original fixed64 complete-source nine-pass endpoint required")
    train_groups = {c["group_id"] for rows in run.q.read(run.data.DEST / "cases.json").values() for c in rows}
    if train_groups & {c["group_id"] for c in sources}:
        raise ValueError("training/development group overlap")
    prior = grade.load_prior_cache()  # Offline replay, including original negative judgments.
    cache = run.q.read(grade.GRADE / "judgment_cache.json")
    if len(prior) != 128 or any(cache.get(k) != v for k, v in prior.items()):
        raise ValueError("exact prior first judgments changed")
    baseline.b.GRADING = grade.GRADE
    counts = baseline.replay("summary", "secondary", outputs, grading["secondary"], cache, prior)
    passed_first = counts["valid"] == 64 and counts["passed"] >= 61
    if (grading["primary"] is not None) != passed_first:
        raise ValueError("conditional second-judge gate changed")
    if grading["classification"] != "NCM_summary_development_failed" or grading["development_gate"]:
        raise ValueError("this finalizer archives the failed study only")
    paths = [Path(__file__), Path(grade.__file__), Path(run.__file__), run.AMENDMENT, grade.PLAN,
        Path(baseline.__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts
                  and (p.suffix != ".pt" or p.name == "model.pt")]
    paths += [grade.CAMPAIGN, grade.REGISTRY, grade.BASE / "judgment_cache.json", PREP / "development_sources.json"]
    artifacts = {str(p): file_hash(p) for p in dict.fromkeys(paths)}
    for root in roots:
        producer = run.q.read(root / "producer.json")
        artifacts.update(producer["source_hashes"])
    aggregate = {"schema": "nal.ncm-native-query-neighborhood-results.v1",
        "classification": grading["classification"], "training": training, "generation": generated,
        "grading": grading, "components": counts, "task_accuracy_measured": True,
        "task_accuracy": counts["passed"] / 64, "NCM_model": True, "NCM_trained": True,
        "first_judgments_replayed": True, "prior_128_first_judgments_preserved": True,
        "endpoint_annotation_correction": "Inherited endpoint NCM_trained=false is a generic grader placeholder; actual model completed2048updates. Raw endpoint unchanged.",
        "conditional_primary_not_run": not passed_first, "final_test_opened": False,
        "NCM_success_established": False, "speed_tested": False, "confirmed": False,
        "artifacts": artifacts}
    DEST.mkdir()
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# The full frozen-native NCM failed summary generation

Date: 2026-10-23. Hypothesis `ncm-native-query-neighborhood-v1`.
Classification: **NCM_summary_development_failed**. Aggregate:
`{campaign}`, SHA256 `{file_hash(campaign)}`.

The completed 2048-update model scored **{counts['passed']}/64 whole-summary
passes ({100 * counts['passed'] / 64:.1f}%)**, with {counts['valid']}/64 valid
first judgments from qualified Terra. Component passes: faithful
{counts['faithful']}/64, salient {counts['salient']}/64, readable
{counts['readable']}/64. Stored summaries exceeded128tokens in
{counts['stored_overflow']}/64 cases. The Wilson95% interval for whole-summary
accuracy is0–5.66%. Zero readable summaries is a catastrophic generation
failure. Training loss, numerical controls and correct process execution do
not qualify it as a successful model.

All28original Llama3.2-3B blocks and native embedding/output weights remained
frozen. The model trained8,637,696new query-attention, concept-head and hint
parameters in BF16-native/FP32-new precision, using1024language updates and
1024summary updates with language replay. Its five concept levels and four
parallel word refinements were fixed in advance. Every development source
was complete, untruncated, in the original64-case order, and disjoint from
training groups. No AR fallback or external verifier generated these outputs.

All three workers terminated successfully. Frozen body, books and CPU cache
integrity and exact checkpoint/generator reload passed. These are systems
checks. Final generated texts and all raw first-judgment requests were retained;
the offline finalizer replayed packet identities, token lengths, raw response
parsing, first-valid decisions and whole-summary gates. All128earlier baseline
judgments, including negatives, remain exact in the grading cache. The generic
endpoint's inherited `NCM_trained:false` annotation is stale: the actual model
completed2048updates. This correction is explicit here and in the aggregate;
the completed raw endpoint is unchanged.

The required61/64Terra gate failed, so Sonnet was correctly not run. Neither
the original256sealed final cases nor the new64longer-source final cases were
opened. No throughput, compression or speed advantage was measured. Under the
same128-token rubric, unchanged3B and1B AR baselines scored52/64 and47/64;
neither establishes95%, but both substantially exceed this adaptation.

This is another failed custom adaptation, not a faithful replication of an
author-provided NCM trainer: the inspected official NCM repository releases
inference/checkpoints but no training loop. The pinned JacobiForcing trainer
retains a clean AR loss plus a noisy-token consistency loss, uses different
training capacity and decoding, and has not been run locally. Comparing its
code does not transfer its reported results to this model.

The next registered exploratory diagnostic crosses predicted versus oracle
coarse plans with parallel versus teacher-forced word inputs on four training
summaries and four raw-language diagnostic cases. Its recovery rates will be
mechanism evidence only. The0/64primary result stays fixed. Wider vetted
teacher labels are likewise training data quality, never student accuracy.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"], "components": counts}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-native-query-neighborhood-v1": {
            "name": "Full frozen native NCM neighborhood training failed development",
            "description": "Completed2048updates with28frozen Llama3Bblocks and8.64Mnewparameters; fixedninepass source-only inference scored0/64whole summaries,10faithful,4salient,0readable,64valid qualifiedTerra firstjudgments. Independentjudge/finaltest/speed correctly notrun. Allfailurespreserved.",
            "confirmation_status": "NCM_summary_development_failed", "task_accuracy": 0.0,
            "task_accuracy_measured": True, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
