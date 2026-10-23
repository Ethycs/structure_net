"""Close the failed v1 controls without closing the user's ongoing goal."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT

HYPOTHESIS = "ncm-paper-regime-recovery-v1"
DESTINATION = ROOT / "20261022_stopped_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-paper-regime-control-failures.md")


def finalize():
    if DESTINATION.exists() or REPORT.exists():
        raise ValueError("fresh finalization root and report required")
    names = {"data": "20261022_data_v1", "teacher": "20261022_teacher_v1",
        "teacher_audit": "20261022_teacher_audit_v1", "codec": "20261022_codec_control_v1",
        "precision": "20261022_codec_precision_v1", "baseline": "20261022_baseline_grade_secondary_v1",
        "features": "20261022_features_v1", "training_pilot": "20261022_training_pilot_v1"}
    results = {key: q.read(ROOT / name / "result.json") for key, name in names.items()}
    if results["teacher_audit"]["status"] != "teacher_quality_gate_failed" or results["codec"]["nonpad_reconstruction"] >= .98:
        raise ValueError("the known stopped-gate state differs")
    if not results["precision"]["prior_reproduced"] or results["baseline"]["invalid"]:
        raise ValueError("precision or baseline evidence invalid")
    for name in ("20261022_unpaired_v1", "20261022_native_v1", "20261022_hybrid_v1"):
        if (ROOT / name).exists():
            raise ValueError("unexpected full adaptation artifacts")
    artifacts = [Path(__file__), Path("docs/07 - Status Reports/2026-10-22_ncm-paper-regime-plan.md"),
        Path("docs/07 - Status Reports/2026-10-22_ncm-codec-precision-diagnostic.md")]
    for name in names.values():
        artifacts.extend(path for path in (ROOT / name).rglob("*.json") if "producer_source" not in path.parts)
    DESTINATION.mkdir(parents=True)
    result = {"schema": "nal.ncm-paper-controls.v1", "hypothesis_id": HYPOTHESIS,
        "status": "stopped", "classification": "stopped_before_generator_adaptation",
        "conformance_profile": "EXPLORATORY", "results": results,
        "failed_gates": ["independent_teacher_audit_30_of_32_below_31", "codec_reconstruction_below_98_percent"],
        "original_native_development_accuracy": results["baseline"]["task_accuracy"],
        "new_full_generator_trained": False, "final_test_opened": False,
        "confirmed": False, "underpowered": True, "human_rated": False, "speed_tested": False,
        "artifacts": {str(path): file_hash(path) for path in artifacts}}
    write_json(DESTINATION / "campaign_results.json", result)
    REPORT.write_text(f'''# NCM paper-regime controls: adaptation stopped before generator training

Date: 2026-10-22. Hypothesis `{HYPOTHESIS}`. Classification:
`stopped_before_generator_adaptation`. Profile EXPLORATORY, one model seed.
Aggregate: `{DESTINATION / 'campaign_results.json'}`;
SHA256 `{file_hash(DESTINATION / 'campaign_results.json')}`.

The released NCM scored **0/64 complete summaries (0%)** on the new conversation
development cohort under the qualified secondary judge. All64 judgments were
valid, with no retries. Wilson95% interval0–5.66%, excluding judge error. This is
a development baseline, not the accuracy of a newly adapted generator. The new
full generator was not trained because both competence gates failed.

| Control | Measured result | Gate |
| --- | --- | --- |
| New teacher labels fitting native canvas |1,999/2,048;49excluded without truncation|Passed minimum1,792|
| Independent teacher-label audit |30/32 complete-summary passes,0invalid|Failed required31/32|
| Released-codec non-PAD reconstruction |67,112/69,682=96.3118%|Failed required98%|
| Released-codec exact decoded texts |676/1,999|Diagnostic only|
| Full FP32 codec diagnostic |67,121/69,682=96.3247%|Does not explain away codec failure|
| GPU generator lifecycle |Unpaired, bridge-only and joint updates; exact reload; codec frozen|Passed systems only|
| Frozen Llama feature extraction |64cases; unchanged body digest|Passed systems only|

The 2,048 training passages,64development passages and256sealed final-test
passages are disjoint by source group, session and source hash. Development and
test selection excluded earlier quality cohorts and calibration sessions; one
group per development/test example. Teacher requests used only source spans
reconstructed from the SHA-verified public LongMemEval release. All512teacher
batches yielded valid JSON;7needed transport/schema retries. The independent
audit retained both valid negative judgments: one unsupported stage-management
role and one overstatement of qualified festival descriptions. No failed label
was retried for a better semantic verdict.

The precision diagnostic fixed the target codes in FP32 and crossed latent and
decoder precision. It exactly reproduced the original BF16/BF16 integer counts.
FP32/FP32 recovered67,121tokens, FP32/BF16 67,111, BF16/FP32 67,122, and
BF16/BF16 67,112. Mixed precision changes a few tokens; it is not the main
reason reconstruction is below the gate. These are token reconstruction rates,
not semantic accuracy. No change of precision rescues the original failed gate.

The implementation adapts the paper's tokenizer-first, frozen-codec,
unpaired-then-conditional staging. It adds a frozen Llama source bridge and
corrects left-padded context placement for NCM. The numerical configuration and
boundaries are in the [plan](../07%20-%20Status%20Reports/2026-10-22_ncm-paper-regime-plan.md)
and its training addendum. Source/tests were frozen in commits ce5142e and
4791fc4; raw producers also retain source copies and hashes. The released
checkpoint's nine-level schedule is retained despite the paper table listing
eight. The authors' XSum result is not a promise of95% task accuracy here.
[Paper](https://arxiv.org/html/2610.14162v1).

The v1 study is closed at its failed controls. The user's95% objective remains
active in a separately declared [v2 study](../07%20-%20Status%20Reports/2026-10-22_ncm-paper-recovery-v2-plan.md):
train the actual codec encoder/EMA books/resampling/decoder with nested dropout,
and screen the paired labels before conditional training. No v2 quality result
is included in this aggregate. The final256sources remain sealed. There is no
new speed claim, human accuracy claim, or full application input-range result;
the native prefix/canvas are512/48GPT-2tokens.

Reproduce with the moba environment Python, offline HF cache and
`LD_LIBRARY_PATH='' OMP_NUM_THREADS=2 PYTHONPATH=src:.`; select physical GPU0
before CUDA. Add memory dependencies and tiktoken cache for grading. Entry
points: `ncm_paper_data`, `ncm_paper_controls`, `ncm_paper_features`,
`ncm_paper_train`, `ncm_paper_grade`, and `ncm_codec_precision` under
`experiments.structure_net`. Pinned roots are declared in source; changed
protocols require new roots. The registry retains confirmed=false.
''')
    print(json.dumps({"classification": result["classification"], "report": str(REPORT)}), flush=True)


@dataclass(frozen=True)
class Evidence:
    def record(self):
        path = DESTINATION / "campaign_results.json"
        result = q.read(path)
        for source, expected in result["artifacts"].items():
            if file_hash(source) != expected:
                raise ValueError("completed control evidence changed")
        return {"id": HYPOTHESIS, "name": "NCM staged-adaptation competence controls",
            "description": "Fresh public-conversation native baseline and independent teacher/codec controls before frozen-Llama NCM adaptation; both competence gates failed.",
            "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
            "confirmation_status": result["classification"], "native_development_passes": 0,
            "native_development_cases": 64, "new_full_generator_trained": False,
            "human_rated": False, "speed_tested": False, "final_test_opened": False,
            "campaign": str(path), "campaign_sha256": file_hash(path),
            "report": str(REPORT), "report_sha256": file_hash(REPORT)}


def register():
    record = Evidence().record()
    paths = [DESTINATION / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS+".json")]
    if any(path.exists() for path in paths):
        raise ValueError("fresh registration required")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if any(exact["metadatas"][0].get(k) != v for k,v in record.items()) or HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for path in paths:
        write_json(path, receipt)
        if q.read(path) != receipt:
            raise ValueError("JSON registry readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
