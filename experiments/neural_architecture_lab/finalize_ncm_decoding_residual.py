"""Record failed decoding policies and a target-aware residual capacity probe."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify

DEST = ROOT / "20261022_decoding_residual_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-decoding-and-residual-capacity.md")
STUDIES = {
    "ncm-decoding-policy-v1": {
        "name": "Matched NCM decoding policies",
        "description": "Raw, published repetition-penalized, and domain-prior decoding all fail complete-summary development grading on identical generated codes and logits.",
        "confirmation_status": "development_gate_failed", "task_accuracy_measured": True,
        "runs": ["20261022_decode_outputs_v1", *[
            "20261022_decode_"+p+"_grade_secondary_v1" for p in ("local", "paper", "domain")]],
    },
    "ncm-residual-oracle-v1": {
        "name": "NCM residual repair capacity oracle",
        "description": "A frozen codec recovers over92percent of token losses after nearby coarse-code perturbations when finer codes are recomputed from known target latents. This is a representation oracle, not generator accuracy.",
        "confirmation_status": "representation_oracle_only", "task_accuracy_measured": False,
        "runs": ["20261022_residual_oracle_v1"],
    },
}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    decoding = q.read(ROOT / "20261022_decode_quality_secondary_v1.json")
    oracle = q.read(ROOT / "20261022_residual_oracle_v1/result.json")
    if any(r["status"] != "completed" or r["valid"] != 64 or r["passed"] != 0
           for r in decoding["results"].values()):
        raise ValueError("recorded decoding evidence differs")
    if (oracle["status"] != "completed_representation_oracle" or oracle["cases"] != 1999
            or not oracle["vendor_code_and_latent_parity"] or not oracle["forced_roots_preserved"]
            or not oracle["reload_equal"] or not oracle["oracle_knows_target_text"]):
        raise ValueError("oracle invariants differ")
    paths = [Path(__file__), ROOT / "20261022_decode_output_manifest_v1.json",
             ROOT / "20261022_decode_quality_secondary_v1.json",
             Path("docs/07 - Status Reports/2026-10-22_ncm-decoding-ordered-gates-amendment.md")]
    for spec in STUDIES.values():
        for name in spec["runs"]:
            root = ROOT / name
            verify(q.read(root / "producer.json"))
            paths.extend(p for p in root.glob("*") if p.is_file())
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-decoding-residual.v1", "decoding_secondary": decoding,
        "residual_oracle": oracle, "primary_decoding_judge_run": False,
        "primary_skip_reason": "No policy reached61/64under secondary; ordered-gates amendment applies.",
        "joint_task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "underpowered": True, "confirmed": False,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    write_json(DEST / "campaign_results.json", aggregate)
    REPORT.write_text(f'''# NCM decoding failure and residual-code capacity

Date: 2026-10-22. Two separate exploratory studies. Neither reaches the user's
95% complete-summary target. The final256-case test remains sealed; no speedup
has been established.

| Matched generated decoding policy | Complete summaries passing Terra |
| --- | --- |
| Local raw argmax |0/64=0%|
| Published frequency/adjacency penalties and released prior |0/64=0%|
| Published penalties and domain summary-position prior |0/64=0%|

Every policy used identical generated codes and text logits for each source.
All192 judgments were valid with no retries. Each policy's Wilson95% interval
is[0%,5.6624%]. Under the pre-judgment ordered-gates amendment, none qualified
for primary Sonnet grading. Sonnet was therefore not run for these policies;
no joint-judge accuracy is estimated. This is a diagnostic failure on the
development cohort, not a final-test result.

The candidate pairs the old native noisy-pair generator with the polished
decoder and uses FP32 parameters/FP16 forward on GPU2. This differs from older
BF16/old-decoder results and is not a controlled precision or decoder ablation.
Within this study, codes, logits and precision are matched. A CPU test compared
the new published-policy implementation with the actual released evaluator's
`pick_tokens` function. The published penalties did not repair invented words
and names from this generator. [Official NCM implementation](https://github.com/HazyResearch/ncm).

The separate oracle asks whether frozen finer codebooks have enough capacity
to compensate for a nearby wrong coarse code. For1,999 known training targets,
it substitutes one or all four coarse codes with one of the eight nearest
non-identical codebook vectors. It either replays the original finer codes or
recomputes them against the known desired target latent.

| Oracle condition | Correct non-PAD tokens /69,682 | Exact target texts /1,999 |
| --- | --- | --- |
| Clean hierarchy |69,517=99.7632%|1,858|
| One changed coarse code, old finer codes |68,355=98.0956%|988|
| One changed coarse code, recomputed finer codes |69,432=99.6412%|1,785|
| All four changed, old finer codes |64,898=93.1345%|117|
| All four changed, recomputed finer codes |69,162=99.2538%|1,576|

Recomputing finer codes restores1,077/1,162=92.6850% and4,264/4,619=92.3144%
of token losses relative to the clean baseline. Vendor quantization parity,
preservation of forced coarse roots, frozen codec identity and exact checkpoint
reload passed. Neighbors exclude identical vectors; minimum squared distance
was1.30769. Own peak allocation was862,561,792bytes on GPU2.

**This oracle sees the correct target text. It is not a summarizer, model task
accuracy, distillation success or evidence of generalization.** It supports
testing a new training objective that teaches finer codes to repair actual
coarse-code mistakes. The published training regime uses clean finer labels
with corrupted conditioning; dynamic residual relabeling would be an explicitly
new adaptation, not a reproduction claim. The original frozen Llama body and
all codec parameters remain unchanged in this oracle.

Artifacts, producer hashes, preserved raw judgments and oracle predictions:
`{DEST / 'campaign_results.json'}` (SHA256
`{file_hash(DEST / 'campaign_results.json')}`). Reproduce with
`ncm_decode_policy_campaign`, `ncm_decode_grade_campaign` and
`run_ncm_residual_oracle` under `experiments.structure_net`, following their
immutable plans and recorded environments. Both registry records remain
tested=true, confirmed=false and underpowered=true.
''')
    print(json.dumps({"status": "finalized", "report": str(REPORT)}), flush=True)


def register():
    campaign = DEST / "campaign_results.json"
    for path, expected in q.read(campaign)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed evidence changed: "+path)
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    for hypothesis, spec in STUDIES.items():
        receipt_path = DEST / (hypothesis+"_registry_readback.json")
        json_path = Path("data/hypothesis_registry") / (hypothesis+".json")
        if receipt_path.exists() or json_path.exists():
            raise ValueError("fresh hypothesis registration required")
        record = {"id": hypothesis, **{k: v for k, v in spec.items() if k != "runs"},
            "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
            "final_test_opened": False, "speed_tested": False, "human_rated": False,
            "campaign": str(campaign), "campaign_sha256": file_hash(campaign),
            "report": str(REPORT), "report_sha256": file_hash(REPORT)}
        logger.log_hypothesis(record)
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        collection.update(ids=[hypothesis], metadatas=[{**exact["metadatas"][0], **record}])
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        semantic = collection.query(query_texts=[record["description"]], n_results=10)
        if any(exact["metadatas"][0].get(k) != v for k, v in record.items()) or hypothesis not in semantic["ids"][0]:
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
