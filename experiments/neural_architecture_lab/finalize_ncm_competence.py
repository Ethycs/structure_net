"""Record passed training-competence gates without claiming held-out task success."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT

DESTINATION = ROOT / "20261022_competence_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-training-competence.md")
STUDIES = {"ncm-coarse-curriculum-v1": "20261022_coarse_train_v1",
           "ncm-decoder-polish-v1": "20261022_decoder_train_v1"}


def finalize():
    if DESTINATION.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    results = {key: q.read(ROOT / name / "result.json") for key, name in STUDIES.items()}
    coarse, decoder = results.values()
    m = coarse["training_diagnostics"]
    if (coarse["status"] != "mechanism_gate_passed" or m["code_accuracy"] < .9 or
        m["nonpad_reconstruction"] < .95 or coarse["coarse_matched_source_advantage"] < .2):
        raise ValueError("coarse competence evidence differs")
    if (decoder["status"] != "reconstruction_gate_passed" or decoder["nonpad_reconstruction"] < .995 or
        decoder["exact_text_cases"]/decoder["cases"] < .9 or
        not decoder["nondecoder_unchanged"] or not decoder["all_summary_codes_unchanged"]):
        raise ValueError("decoder competence evidence differs")
    artifacts = [Path(__file__)]
    for name in STUDIES.values():
        root = ROOT / name
        producer = q.read(root / "producer.json")
        for path, expected in {**producer["input_hashes"], **producer["source_hashes"]}.items():
            if file_hash(path) != expected:
                raise ValueError("competence producer changed: "+path)
        artifacts += [path for path in root.glob("*") if path.is_file()]
    DESTINATION.mkdir(parents=True)
    aggregate = {"schema": "nal.ncm-competence.v1", "classification": "training_competence_demonstrated",
        "results": results, "artifacts": {str(path): file_hash(path) for path in artifacts},
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "underpowered": True, "confirmed": False}
    write_json(DESTINATION / "campaign_results.json", aggregate)
    REPORT.write_text(f'''# NCM training competence after adapting the paper's regime

Date: 2026-10-22. Two separate exploratory, one-seed training diagnostics.
Both local competence gates passed. **Neither establishes95% held-out summary
accuracy.** The256-case final test remains sealed; no speedup is measured.

| Measurement | Result | Interpretation |
| --- | --- | --- |
| Free code accuracy on32training pairs |6,379/6,400=99.6719%|Training memorization only|
| Coarse codes with matched sources |128/128|Training conditioning control|
| Coarse codes with cyclically shifted sources |1/128|Source-sensitive generation|
| Decoded target tokens, original decoder |1,132/1,149=98.5205%|Codec errors persist despite learned codes|
| Exact training texts, original decoder |19/32|Not task accuracy|
| Full-data original decoder reconstruction |68,431/69,682=98.2047%|Fixed baseline|
| Full-data polished decoder reconstruction |69,515/69,682=99.7603%|Training reconstruction only|
| Exact texts after decoder polish |1,856/1,999=92.8464%|Training reconstruction only|

The paper-style adaptation first trained the actual tokenizer encoder,
resampling projections, EMA codebooks and decoder with nested-cutoff losses.
The frozen-codec generator then received unpaired training-source windows,
bridge warmup and conditional source/summary pairs. The completed larger
noisy-pair run still produced invented names and malformed wording; its first
complete development judgment is0/64under Sonnet, with the remaining independent
model/judge results being recorded separately. Low token CE was not accepted
as evidence of task success.

The coarse follow-up continued the native generator on32fixed training pairs.
It preserved the native source embedding channel, added a zero-initialized
global residual from the frozen Llama final valid hidden state, enabled the
released implementation's optional autoregression over the first4codes, and
used coarse-only warmup followed by equal total loss per hierarchy level and
conditioning corruption. The original tokenwise objective gave the first4of200
positions only2%of loss weight. This is one bundled intervention, so the study
does not isolate which individual change caused improvement. Shuffling both
native and body source inputs shows source dependence; it does not isolate the
Llama body's contribution. All original body layers, including their MLPs,
remained frozen. Checkpoint reload and causal coarse-input checks passed.

The decoder follow-up froze every encoder, codebook, residual and resampling
state, then trained only full-cutoff text reconstruction. Every non-decoder
state byte and every one of the1,999summary code assignments stayed unchanged.
This reduces word corruption while retaining the discrete language already
learned by generators. Pairing with a new decoder requires an explicit
compatibility receipt; original checkpoint metadata is preserved.

The original tokenizer, generator and new source interface use released
checkpoint warm starts; the source residual projection starts at zero.
These short adaptations are far smaller than the paper's training budget.
The decoder phase is an engineering addition rather than a claimed reproduction
of the published recipe. [NCM paper](https://arxiv.org/html/2610.14162v1).

Measured training compute was289.0seconds for the32-case curriculum and
59.1seconds for decoder training plus recorded checks, on a sharedRTX3060.
These are not inference benchmarks. Peaked own allocation was3.289GB and1.652GB
respectively. The prototype supports512GPT2source tokens and48target tokens,
which does not cover the full application range. Known teacher-label noise
(30/32audit) and one-seed training-fit limitations remain disclosed.

A separate [full-data continuation](../07%20-%20Status%20Reports/2026-10-22_ncm-full-curriculum-plan.md)
has passed its GPU lifecycle pilot and is running. It must earn its own64-case
development accuracy under both qualified judges. No result from that run is
included here. The user's95% complete-summary objective remains open.

Aggregate: `{DESTINATION / 'campaign_results.json'}`;
SHA256 `{file_hash(DESTINATION / 'campaign_results.json')}`.
Reproduce with `train_ncm_coarse_curriculum` and `train_ncm_decoder_polish`
under `experiments.structure_net`, using the pinned moba environment and GPU0.
The immutable plans, producer source copies, hashes, raw predictions, code
labels, optimizer/RNG checkpoints and memory records remain in the run roots.
Registry records keep confirmed=false and underpowered=true.
''')
    print(json.dumps({"status": aggregate["classification"], "report": str(REPORT)}), flush=True)


def register():
    campaign = DESTINATION / "campaign_results.json"
    aggregate = q.read(campaign)
    for path, expected in aggregate["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed competence evidence changed")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    for hypothesis, name in STUDIES.items():
        receipt_path = DESTINATION / (hypothesis+"_registry_readback.json")
        json_path = Path("data/hypothesis_registry") / (hypothesis+".json")
        if receipt_path.exists() or json_path.exists():
            raise ValueError("fresh hypothesis registration required")
        record = {"id": hypothesis, "name": "NCM "+("coarse training competence" if "coarse" in hypothesis else "fixed-code decoder polish"),
            "description": ("A frozen-body residual and coarse curriculum learn32training summary hierarchies with a shuffled-source control."
                if "coarse" in hypothesis else "Full-cutoff decoder training improves reconstruction while preserving every encoder, book and code assignment."),
            "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
            "confirmation_status": "training_competence_only", "task_accuracy_measured": False,
            "final_test_opened": False, "speed_tested": False, "human_rated": False,
            "campaign": str(campaign), "campaign_sha256": file_hash(campaign),
            "report": str(REPORT), "report_sha256": file_hash(REPORT), "run": str(ROOT / name)}
        logger.log_hypothesis(record)
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        collection.update(ids=[hypothesis], metadatas=[{**exact["metadatas"][0], **record}])
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        semantic = collection.query(query_texts=[record["description"]], n_results=10)
        if any(exact["metadatas"][0].get(key) != value for key, value in record.items()) or hypothesis not in semantic["ids"][0]:
            raise ValueError("registry readback failed")
        receipt = {"status": "registered_and_read_back", "record": record,
                   "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
        for path in (receipt_path, json_path):
            write_json(path, receipt)
            if q.read(path) != receipt:
                raise ValueError("JSON registry readback failed")
    print(json.dumps({"status": "registered_and_read_back", "studies": len(STUDIES)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
